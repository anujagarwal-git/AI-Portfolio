"""IS RAGAS WORKING? A smoke test with a falsifying condition. No index, no pipeline.

WHY THIS IS NOT "RUN IT ON THE GOLDEN SET AND SEE"
    RAGAS with a local judge fails SILENTLY. The documented failure here
    (2026-08-01, "RAGAS parse-error churn") is that the judge returns text the
    metric cannot parse and the metric yields NaN - or worse, the same number
    for every row. A run that completes and prints a mean is NOT evidence that
    anything worked. A mean of 0.0 and a mean over three NaNs look similar at a
    glance and mean opposite things.

    So this test uses three HAND-MADE rows whose correct answer is known before
    the run, and asserts SEPARATION rather than completion:

      R1 supported    response says only what the context says
                      -> faithfulness HIGH,  context_recall HIGH
      R2 fabricated   response adds a specific figure absent from the context
                      -> faithfulness LOW
      R3 missing      reference states something the context does not contain
                      -> context_recall LOW

    If R1 and R2 come back equal, the judge is not judging - it is emitting a
    constant, and every number a real evaluation produces would be noise.

THE CONDITIONS ARE FIXED HERE, BEFORE THE RUN. Do not soften them afterwards.
    1. no NaN in any cell
    2. faithfulness(R1)   - faithfulness(R2)   >= 0.30
    3. context_recall(R1) - context_recall(R3) >= 0.30

THE JUDGE IS NOT THE GENERATOR, ON PURPOSE. GEN_MODEL is qwen3:1.7b; the judge
below is a different model. Scoring your own output with the model that wrote
it measures agreement with itself, not correctness.

    uv run python scripts/smoke_ragas.py
    uv run python scripts/smoke_ragas.py --model qwen2.5:3b-instruct
"""
from __future__ import annotations

import argparse
import sys
import traceback

W = 78
LINE = "=" * W

# --- the three rows. Regulatory prose, so the judge sees the register it will
# --- actually meet, but self-contained: nothing here touches Qdrant.
CTX_MODEL = (
    "For the purposes of this document, the term model refers to a quantitative "
    "method, system, or approach that applies statistical, economic, financial, "
    "or mathematical theories, techniques, and assumptions to process input data "
    "into quantitative estimates. A model consists of three components: an "
    "information input component, a processing component, and a reporting "
    "component."
)
CTX_VALIDATION = (
    "Model validation is the set of processes and activities intended to verify "
    "that models are performing as expected, in line with their design "
    "objectives and business uses."
)

ROWS = [
    {   # R1 — every claim is in the context, and the context covers the reference
        "user_input": "How does SR 11-7 define a model?",
        "retrieved_contexts": [CTX_MODEL],
        "response": ("A model is a quantitative method, system, or approach that "
                     "applies statistical, economic, financial, or mathematical "
                     "theories to process input data into quantitative estimates. "
                     "It has three components: input, processing, and reporting."),
        "reference": ("A model applies statistical, economic, financial or "
                      "mathematical theories to process input data into "
                      "quantitative estimates, and has three components: an "
                      "information input component, a processing component, and "
                      "a reporting component."),
        "_id": "R1 supported",
    },
    {   # R2 — same context, but the response invents a figure that is not in it
        "user_input": "How does SR 11-7 define a model?",
        "retrieved_contexts": [CTX_MODEL],
        "response": ("A model is a quantitative method with three components. "
                     "SR 11-7 requires every model to be revalidated every 18 "
                     "months and sets a materiality threshold of $50 million "
                     "above which independent validation is mandatory."),
        "reference": ("A model applies statistical, economic, financial or "
                      "mathematical theories to process input data into "
                      "quantitative estimates, and has three components."),
        "_id": "R2 fabricated",
    },
    {   # R3 — the context is about something else, so it cannot support the reference
        "user_input": "What are the three core elements of model validation?",
        "retrieved_contexts": [CTX_VALIDATION],
        "response": "Model validation verifies that models perform as expected.",
        "reference": ("The three core elements of validation are evaluation of "
                      "conceptual soundness, ongoing monitoring including process "
                      "verification and benchmarking, and outcomes analysis "
                      "including back-testing."),
        "_id": "R3 missing context",
    },
]


def preflight(model: str) -> None:
    """Fail loudly and specifically, rather than hanging on a missing model."""
    import requests
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as exc:                                  # noqa: BLE001
        sys.exit(f"  OLLAMA NOT REACHABLE on localhost:11434 ({exc}).\n"
                 f"  Start it, then re-run. Nothing below would mean anything.")
    have = [m["name"] for m in r.json().get("models", [])]
    if not any(h == model or h.startswith(model + ":") for h in have):
        sys.exit(f"  JUDGE MODEL {model!r} IS NOT PULLED.\n"
                 f"  installed: {', '.join(sorted(have)) or '(none)'}\n"
                 f"  ollama pull {model}")
    print(f"  ollama up; judge {model!r} present")


def probe(model: str) -> int:
    """CAN THIS JUDGE DO ATTRIBUTION AT ALL? Ask it directly, no RAGAS.

    Run when context_recall fails the conditions above. context_recall works by
    splitting the REFERENCE into statements and asking, for each, whether the
    retrieved context supports it. If the model says yes to a statement the
    context plainly does not contain, no RAGAS setting can rescue the metric -
    the judge is the defect. This asks that one question directly so the answer
    is OBSERVED rather than inferred from a score.

    TWO ARMS, ONE VARIABLE BETWEEN THEM: plain text vs format="json". JSON mode
    is what makes a small model's output parseable, and it can also degrade its
    reasoning. Running both says which. Do not change the model AND the format
    together and then credit one - that is the 2026-08-23 rule.
    """
    from langchain_ollama import ChatOllama

    CONTEXT = ("Model validation is the set of processes and activities intended "
               "to verify that models are performing as expected, in line with "
               "their design objectives and business uses.")
    CASES = [
        ("SUPPORTED  ", "Model validation verifies that models perform as "
                        "expected.", True),
        ("UNSUPPORTED", "The three core elements of validation are evaluation of "
                        "conceptual soundness, ongoing monitoring, and outcomes "
                        "analysis.", False),
        ("UNSUPPORTED", "Validation must be repeated every eighteen months.",
         False),
    ]
    ASK = ('Context:\n"""{ctx}"""\n\nStatement:\n"""{stmt}"""\n\n'
           'Can the statement be attributed to the context above? Answer only '
           'yes or no.')

    print(LINE)
    print(f"  PROBE — can {model!r} attribute a statement to a context?")
    print(LINE)
    for fmt in (None, "json"):
        label = "plain text" if fmt is None else 'format="json"'
        kw = {"format": fmt} if fmt else {}
        llm = ChatOllama(model=model, temperature=0, **kw)
        print(f"\n  --- {label} ---")
        wrong = 0
        for tag, stmt, want in CASES:
            q = ASK.format(ctx=CONTEXT, stmt=stmt)
            if fmt == "json":
                q += ' Reply as {"answer": "yes"} or {"answer": "no"}.'
            try:
                raw = llm.invoke(q).content.strip().replace("\n", " ")
            except Exception as exc:                          # noqa: BLE001
                print(f"    {tag}  CALL FAILED: {exc}")
                wrong += 1
                continue
            said_yes = "yes" in raw.lower()[:40]
            bad = said_yes != want
            wrong += bad
            print(f"    {tag}  expected {'yes' if want else 'no ':<3}"
                  f"  {'WRONG' if bad else 'ok   '}  raw: {raw[:64]!r}")
        print(f"    -> {wrong} of {len(CASES)} wrong on {label}")

    print("\n" + LINE)
    print("  HOW TO READ THIS")
    print("  wrong in BOTH arms  -> the model cannot do attribution. A bigger")
    print("                         judge is the fix; RAGAS settings are not.")
    print("  wrong only in json  -> JSON mode is degrading it. Try a judge that")
    print("                         does structured output natively.")
    print("  right in both       -> the model is capable and the defect is in")
    print("                         how RAGAS prompts or parses. Inspect there.")
    print(LINE)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct",
                    help="judge model. MUST NOT be the generator model.")
    ap.add_argument("--embed", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--think", action="store_true",
                    help="leave qwen3 reasoning ON. Slow, but context_recall "
                         "has only ever worked on a reasoning judge.")
    ap.add_argument("--probe", action="store_true",
                    help="skip RAGAS; ask the judge the attribution question "
                         "directly. Run this when context_recall fails.")
    ns = ap.parse_args()
    if ns.probe:
        preflight(ns.model)
        return probe(ns.model)

    print(LINE)
    print("RAGAS SMOKE TEST — 3 hand-made rows, no index, no pipeline")
    print(LINE)
    preflight(ns.model)

    try:
        import ragas
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import context_recall, faithfulness
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import ChatOllama
    except Exception:                                         # noqa: BLE001
        traceback.print_exc()
        sys.exit("\n  IMPORT FAILED — the environment, not the metrics. Fix this first.")
    print(f"  ragas {getattr(ragas, '__version__', '?')}")

    # format="json" is what lets a small local judge produce something the
    # metric parser can read. temperature 0 is the project's pinned convention:
    # a judge that varies between runs cannot be compared between runs.
    from regrag.evaluation import judge as judge_mod
    judge = judge_mod.make(ns.model, think=ns.think)
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=ns.embed))

    # ONE WORKER. RAGAS fans out by default; a single local Ollama thrashes
    # under parallel calls and starts timing out, which then looks exactly like
    # a metric failure. Serial is slower and interpretable.
    run_config = None
    try:
        from ragas.run_config import RunConfig
        run_config = RunConfig(timeout=300, max_workers=1)
    except Exception:                                         # noqa: BLE001
        print("  NOTE: RunConfig unavailable on this version; using defaults. "
              "If rows time out, that is why.")

    ds = EvaluationDataset.from_list(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in ROWS])

    print(f"\n  scoring {len(ROWS)} rows on faithfulness + context_recall...")
    print("  (a local 3B judge takes a few minutes; serial by design)\n")
    try:
        kw = {"run_config": run_config} if run_config is not None else {}
        result = evaluate(dataset=ds, metrics=[faithfulness, context_recall],
                          llm=judge, embeddings=emb, **kw)
    except Exception:                                         # noqa: BLE001
        traceback.print_exc()
        sys.exit("\n  evaluate() RAISED. That is the finding — read the trace above,\n"
                 "  do not retry with different settings until you know the cause.")

    try:
        df = result.to_pandas()
    except Exception:                                         # noqa: BLE001
        print("  to_pandas() failed; raw result below. Check the installed API.")
        print(f"  {result}")
        return 1

    # PER ROW, NEVER THE MEAN. A mean over NaN and a mean of zero look alike.
    print(LINE)
    print(f"  {'row':<20}{'faithfulness':>16}{'context_recall':>18}")
    print("  " + "-" * (W - 2))
    got = {}
    for row, (_i, rec) in zip(ROWS, df.iterrows()):
        f = rec.get("faithfulness")
        c = rec.get("context_recall")
        got[row["_id"]] = (f, c)
        fs = "NaN" if f != f else f"{f:.3f}"       # NaN != NaN
        cs = "NaN" if c != c else f"{c:.3f}"
        print(f"  {row['_id']:<20}{fs:>16}{cs:>18}")

    print("\n" + LINE)
    print("  THE CONDITIONS, AS FIXED BEFORE THE RUN")
    print(LINE)
    f1, c1 = got["R1 supported"]
    f2, _ = got["R2 fabricated"]
    _, c3 = got["R3 missing context"]
    nan = any(v != v for pair in got.values() for v in pair)

    checks = [
        ("no NaN in any cell", not nan,
         "the judge's output is not parsing — this is the 2026-08-01 failure"),
        ("faithfulness separates a grounded answer from a fabricated one",
         (not nan) and (f1 - f2) >= 0.30,
         f"R1 {f1} vs R2 {f2} — a constant here means the judge is not judging"),
        ("context_recall separates sufficient context from insufficient",
         (not nan) and (c1 - c3) >= 0.30,
         f"R1 {c1} vs R3 {c3}"),
    ]
    ok = True
    for name, passed, why in checks:
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name}")
        if not passed:
            print(f"          {why}")

    print("\n" + LINE)
    if ok:
        print("  RAGAS IS WORKING. The metrics discriminate, which is the only")
        print("  thing this test claims. It says NOTHING about your pipeline —")
        print("  next step is wiring answer() into a harness (its signature")
        print("  changed: answer(q) -> Answer, contexts live on a.retrieval).")
    else:
        print("  DO NOT RUN AN EVALUATION UNTIL THIS PASSES. Numbers from a")
        print("  judge that cannot discriminate are noise wearing a decimal point.")
        print("  Inspect first: print one raw judge response before changing")
        print("  settings. No guess-and-check.")
    print(LINE)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
