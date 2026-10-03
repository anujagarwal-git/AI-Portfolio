"""STAGE 9 — context recall and faithfulness on the golden set.

WHAT EACH METRIC ACTUALLY ANSWERS, because they are not interchangeable

  CONTEXT RECALL  splits the REFERENCE answer into statements and asks, for
                  each, whether the RETRIEVED CONTEXT supports it. It grades
                  RETRIEVAL, and it is the first metric in this project that
                  can SEE A MISS. Every retrieval number before it - the
                  2026-09-03 relevance study included - was computed on
                  passages retrieval had already delivered, so a chunk that was
                  never fetched had no row and was invisible. Delivery recall
                  was <=66% and that was a CEILING. This is the real number.

  FAITHFULNESS    splits the ANSWER into claims and asks whether each is
                  supported by the context the model was given. It grades
                  GENERATION. A perfectly faithful answer built on the wrong
                  context still scores 1.0, so faithfulness alone says nothing
                  about whether the answer is correct.

  Read them together or not at all. High faithfulness with low context recall
  means the model is loyally summarising the wrong evidence.

WHY THE REFERENCES CANNOT COME FROM RETRIEVAL
  `golden_set_v2.jsonl` was written from the PARSED SOURCE DOCUMENTS, read
  directly, never from what the pipeline returns. Writing a reference out of
  the chunks the system delivers guarantees high context recall and measures
  nothing - the eval-set equivalent of fitting a scaler before the CV split.
  If you extend the set, keep that rule or the metric dies quietly.

  gs-01 and gs-02 are in the set ON PURPOSE: the relevance study suspected both
  facts (Basel RBC's minimum ratios, Basel LEX's 25% limit) were never
  retrieved at all, and confirmed both exist in the corpus. If context recall
  cannot see those two, it is not working.

THE JUDGE — WHAT HAS ACTUALLY BEEN MEASURED, not preferred

    judge                 faithfulness            context_recall
    qwen2.5:3b-instruct   PASSES 1.000 / 0.000    BROKEN - scored a plainly
                          clean separation        unsupported statement 1.000
    qwen3:8b              NaN on every row        completed; gs-02 correctly
                          (timeout, 600s/call)    0.00 on a real miss
    qwen3:4b              UNMEASURED              UNMEASURED

  qwen3:4b is the current default because it sits between two judges that each
  failed one half, NOT because it is known to work. Gate it before trusting a
  run: `uv run python scripts/smoke_ragas.py --model qwen3:4b` scores three
  hand-made rows on both metrics against conditions fixed before the run. If it
  passes, this table gets a third row and the run means something. If it does
  not, use qwen2.5:3b-instruct for --metrics faith and qwen3:8b for
  --metrics recall, which is the measured combination.

  Read speed context from 2026-09-03: qwen3:1.7b reads ~77 tok/s and qwen3:4b
  ~31 tok/s on ~11k-token prompts. Faithfulness verifies every claim against
  the whole context, so its cost scales with claims x context size - that is
  the arithmetic behind the 8b timeouts, and why 4b may clear them.

  The judge is NOT the generator (qwen3:1.7b). Scoring your own
  output with the model that wrote it measures self-agreement. Run
  `scripts/smoke_ragas.py --probe --model qwen3:8b` FIRST: on 2026-09-07
  qwen2.5:3b-instruct passed faithfulness cleanly and scored a plainly
  unsupported statement 1.000 on context recall. A judge that cannot attribute
  turns this whole run into noise wearing a decimal point.

    uv run python scripts/run_golden.py --limit 3      # smoke, ~3 questions
    uv run python scripts/run_golden.py --save
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback
from datetime import datetime, timezone

W = 88
LINE = "=" * W
GOLDEN = pathlib.Path("evaluation") / "golden_set_v2.jsonl"



# Paragraph units, taken from the aligner itself so this script cannot invent a
# second definition of "a paragraph the model read". Returns [] if the aligner
# is unavailable, so a run never dies over its own evidence field.
def _units(passage):
    try:
        from regrag.generation import align as _al
        return _al.paragraph_units(passage)
    except Exception:                                         # noqa: BLE001
        return []


def load_golden(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"  {path} not found.")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_pipeline(golden: list[dict]) -> list[dict]:
    """Answer every question with the SHIPPED pipeline.

    The prototype harness called `answer(q, strategy=..., k_final=...)` and
    unpacked a tuple. That signature no longer exists: `answer(question) ->
    Answer`, and the evidence lives on `a.retrieval`. Reusing the old harness
    would fail at the first question.
    """
    from regrag.generation.answer import answer

    # THE GATE IS RECORDED, NOT ENFORCED. `answer()` does not call it, so every
    # number this harness has ever produced describes a path a real user never
    # takes - ask.py gates first. Running it here and STORING the decision shows
    # which questions the shipped front door would have stopped, without
    # changing what is measured or breaking comparability with earlier runs.
    # Blocking on it would conflate two questions: "can retrieval find this"
    # and "does the gate let it through".
    try:
        from regrag.gate.intent import classify
    except Exception:                                         # noqa: BLE001
        classify = None

    out = []
    for i, item in enumerate(golden, 1):
        t0 = time.perf_counter()
        try:
            a = answer(item["question"])
            err = ""
        except Exception as exc:                              # noqa: BLE001
            traceback.print_exc()
            a, err = None, f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0

        if a is None:
            out.append({**item, "answer": "", "contexts": [], "docs": [],
                        "refused": True, "error": err, "seconds": round(dt, 1)})
            print(f"  {i:2d}/{len(golden)} {item['id']}  ERROR after {dt:4.1f}s")
            continue

        gate_dec = gate_why = ""
        if classify is not None:
            try:
                g = classify(item["question"])
                gate_dec, gate_why = g.decision, g.reason
            except Exception as exc:                          # noqa: BLE001
                gate_dec, gate_why = "ERROR", f"{type(exc).__name__}: {exc}"

        passages = list(a.retrieval.passages) if a.retrieval else []
        out.append({
            **item,
            "answer": a.text,
            # ONE CONTEXT PER DELIVERED PASSAGE, exactly the text the model
            # read. Not the chunk, not the parent - the windowed passage.
            "contexts": [p.text for p in passages],
            "docs": [p.short_name for p in passages],
            "locators": [p.parent_id for p in passages],
            # Per-child hit scores. Omitted on the 2026-09-08 run, which meant
            # the scores behind those numbers had to be RE-DERIVED rather than
            # read. A graded run should carry its own evidence.
            "hits": [[{"locator": h.payload.get("locator"), "facet": h.facet,
                       "dense_rank": h.dense_rank, "lexical_rank": h.lexical_rank,
                       "rrf": round(h.rrf, 6), "rerank": h.rerank}
                      for h in p.children] for p in passages],
            # THE CITATION EVIDENCE (2026-09-10). Layers 1, 2 and 3a of the
            # citation-accuracy metric need each claim's LABEL and LOCATOR, and
            # the set of paragraph locators actually DELIVERED in each window.
            # The 2026-09-08 run saved only `n_claims`, so the metric could not
            # be computed from it at all - the same gap as the omitted per-child
            # scores, one field over. A graded run carries its own evidence.
            "claims": [{"text": c.text, "label": c.label,
                        "citation": c.citation, "locator": c.locator,
                        "overlap": c.overlap, "align_verdict": c.align_verdict}
                       for c in a.claims],
            # S1..Sn exactly as the prompt numbered them, so layer 2 checks a
            # label against the table THAT ANSWER SAW, not a rebuilt guess.
            "labels": [f"S{i}" for i, _ in enumerate(passages, 1)],
            # Per passage, the locators of the paragraphs the model could read.
            # Layer 3a is membership in this set: a claim cannot honestly cite
            # a paragraph that was never in front of the model.
            "unit_locators": [[u.locator for u in _units(pas)]
                              for pas in passages],
            "plan_actual": a.retrieval.plan.mode if a.retrieval else "",
            "gate_decision": gate_dec, "gate_reason": gate_why,
            "refused": a.refused,
            "refusal_reason": a.refusal_reason,
            "config_hash": a.config_hash,
            "n_claims": len(a.claims),
            "n_invalid": len(a.invalid),
            "error": "",
            "seconds": round(dt, 1),
        })
        plan = out[-1]["plan_actual"]
        flag = "" if plan.startswith(item["plan_expected"][:6]) else "  ! plan"
        print(f"  {i:2d}/{len(golden)} {item['id']}  {dt:5.1f}s  "
              f"{len(passages)} psg  {plan:<20}{flag}  {(a.text or '')[:44]}")
    return out


def literal_recall(results: list[dict]) -> None:
    """A JUDGE-FREE RECALL FLOOR. Exact, no model, cannot time out.

    If the fact is in the delivered context, its literal string is there. This
    cannot replace context_recall - it says nothing about paraphrase, and a
    question with no expected_strings gets no score - but it is IMMUNE to the
    two failures that have already bitten this project: a judge that cannot
    attribute (2026-09-07, qwen2.5:3b scored an unsupported statement 1.000)
    and a judge that times out. When the judge and this disagree, believe THIS
    for presence and the judge for meaning.
    """
    for r in results:
        want = r.get("expected_strings") or []
        blob = " ".join(r["contexts"])
        hits = [w for w in want if w.lower() in blob.lower()]
        r["literal_hits"] = hits
        r["literal_recall"] = (len(hits) / len(want)) if want else None


def score(results: list[dict], model: str, embed: str, metrics: str,
          timeout: int, think: bool = False, num_ctx: int = 32768):
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_recall, faithfulness
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_ollama import ChatOllama

    # ONE place builds a judge, and it proves thinking is off before returning.
    from regrag.evaluation import judge as judge_mod
    judge = judge_mod.make(model, think=think, num_ctx=num_ctx)
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embed))
    kw = {}
    try:
        from ragas.run_config import RunConfig
        kw["run_config"] = RunConfig(timeout=timeout, max_workers=1)
    except Exception:                                         # noqa: BLE001
        print("  NOTE: RunConfig unavailable; defaults in use.")

    chosen = {"recall": [context_recall], "faith": [faithfulness],
              "both": [context_recall, faithfulness]}[metrics]
    # SHOW THE LOAD. faithfulness verifies EVERY claim against the WHOLE
    # context, so its cost scales with claims x context size. Printing this is
    # how a timeout stops being a mystery.
    ch = [sum(len(c) for c in r["contexts"]) for r in results]
    print(f"  context per question: median {sorted(ch)[len(ch)//2]:,}c  "
          f"max {max(ch):,}c  (~{max(ch)//4:,} tokens on the biggest)")
    print(f"  metrics: {', '.join(m.name for m in chosen)}   timeout {timeout}s/call")

    ds = EvaluationDataset.from_list([{
        "user_input": r["question"],
        "retrieved_contexts": r["contexts"] or [""],
        "response": r["answer"] or "",
        "reference": r["reference_answer"],
    } for r in results])
    return evaluate(dataset=ds, metrics=chosen, llm=judge, embeddings=emb,
                    **kw).to_pandas()


def report(results: list[dict], df) -> None:
    import statistics

    # A metric that was not requested has no column. Absent and NaN are
    # different things and must not print the same: "-" means not measured,
    # "NaN" means measured and the judge failed.
    for r, (_i, rec) in zip(results, df.iterrows()):
        for m in ("context_recall", "faithfulness"):
            r[m] = rec[m] if m in df.columns else None

    def fmt(v):
        if v is None: return "-"
        if v != v: return "NaN"
        return f"{v:.2f}"

    nan = [r for r in results
           if any(r[m] is not None and r[m] != r[m]
                  for m in ("context_recall", "faithfulness"))]
    print("\n" + LINE)
    print(f"  {'id':<8}{'scenario':<17}{'psg':>4}{'lit':>6}{'recall':>8}"
          f"{'faith':>7}{'plan':>20}{'  docs'}")
    print("  " + "-" * (W - 2))
    for r in results:
        lr = r.get("literal_recall")
        hit = "ok " if set(r["expected_docs"]) & set(r["docs"]) else "MISS"
        print(f"  {r['id']:<8}{r['scenario']:<17}{len(r['contexts']):>4}"
              f"{('-' if lr is None else f'{lr:.2f}'):>6}"
              f"{fmt(r['context_recall']):>8}{fmt(r['faithfulness']):>7}"
              f"{r.get('plan_actual',''):>20}  "
              f"{hit} {','.join(dict.fromkeys(r['docs']))[:22]}")

    ok = [r for r in results if r["context_recall"] is not None
          and r["context_recall"] == r["context_recall"]]
    print("\n" + LINE)
    if nan:
        print(f"  {len(nan)} ROW(S) RETURNED NaN — the judge failed to parse on those.")
        print("  A mean over NaN is not a low score, it is no score. Fix before reading on.")
    if ok:
        print(f"  context recall  mean {statistics.mean(r['context_recall'] for r in ok):.3f}"
              f"   median {statistics.median(r['context_recall'] for r in ok):.3f}")
        fo = [r for r in results if r["faithfulness"] is not None
              and r["faithfulness"] == r["faithfulness"]]
        if fo:
            print(f"  faithfulness    mean {statistics.mean(r['faithfulness'] for r in fo):.3f}"
                  f"   median {statistics.median(r['faithfulness'] for r in fo):.3f}")

    # BY SCENARIO — the mean over 18 hides which SHAPE is failing, and shape is
    # what you would tune. The relevance study's whole finding was per-shape.
    blocked = [r for r in results if r.get("gate_decision")
               not in ("", "ANSWERABLE")]
    if blocked:
        print(f"\n  THE GATE WOULD HAVE STOPPED {len(blocked)} OF {len(results)} "
              f"— recorded, not enforced")
        for r in blocked:
            print(f"    {r['id']}  {r['gate_decision']:<14}{r['gate_reason'][:56]}")
        print("    These scored normally here because answer() does not gate.")
        print("    A user asking them through ask.py gets the gate's message instead.")

    print("\n  BY SCENARIO")
    print(f"  {'scenario':<20}{'n':>3}{'recall':>9}{'faith':>8}{'doc hit':>9}")
    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(r["scenario"], []).append(r)
    for k, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        c = [r["context_recall"] for r in g if r["context_recall"] is not None
             and r["context_recall"] == r["context_recall"]]
        f_ = [r["faithfulness"] for r in g if r["faithfulness"] is not None
              and r["faithfulness"] == r["faithfulness"]]
        dh = sum(1 for r in g if set(r["expected_docs"]) & set(r["docs"])) / len(g)
        print(f"  {k:<20}{len(g):>3}"
              f"{(statistics.mean(c) if c else float('nan')):>9.2f}"
              f"{(statistics.mean(f_) if f_ else float('nan')):>8.2f}{dh:>9.0%}")

    # THE TWO PLANTED MISSES. If recall cannot see these, it is not working.
    print("\n  THE TWO CONFIRMED-MISS QUESTIONS (planted; both facts ARE in the corpus)")
    for gid in ("gs-01", "gs-02"):
        r = next((x for x in results if x["id"] == gid), None)
        if r:
            lr = r.get("literal_recall")
            print(f"    {gid}  judge recall {fmt(r['context_recall'])}"
                  f"   literal {('-' if lr is None else f'{lr:.2f}')}"
                  f"   served: {','.join(dict.fromkeys(r['docs']))[:34] or '(nothing)'}")
            miss = [w for w in (r.get('expected_strings') or [])
                    if w not in (r.get('literal_hits') or [])]
            if miss:
                print(f"           NOT IN THE DELIVERED CONTEXT: {miss}")
    print("    A HIGH score on these two contradicts the relevance study. Believe the")
    print("    study and suspect the judge before you believe a flattering recall here.")
    print(LINE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:7b-instruct", help="judge; NOT the generator")
    ap.add_argument("--metrics", default="recall",
                    choices=["recall", "faith", "both", "none"],
                    help="DEFAULT recall. Run the two SEPARATELY: on 2026-09-07 "
                         "context_recall completed on qwen3:8b while "
                         "faithfulness returned NaN on every row. Splitting "
                         "them means one metric failing no longer costs you "
                         "the other.")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="seconds per judge call (was 600 when faithfulness "
                         "timed out)")
    ap.add_argument("--embed", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--think", action="store_true",
                    help="leave qwen3 reasoning ON (slow; see judge.py table)")
    ap.add_argument("--num-ctx", type=int, default=32768,
                    help="judge context window. Ollama defaults to 4096, which "
                         "TRUNCATES a 20,000-char context and reports a low "
                         "recall for retrieval that actually worked.")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only")
    ap.add_argument("--save", action="store_true")
    # EVAL TRAFFIC IS TAGGED APART FROM HAND TESTING. Forty-eight
    # questions in the same view as the ones being tuned is how "what
    # does a normal query look like" quietly gets answered from the
    # eval set. Same discipline as the golden set itself.
    ap.add_argument("--trace", action="store_true",
                    help="trace this run into the regrag-eval project")
    ap.add_argument("--golden", default=str(GOLDEN))
    ns = ap.parse_args()

    if getattr(ns, "trace", False):
        from regrag.observability import tracing
        tracing.init(project="regrag-eval", verbose=True)

    golden = load_golden(pathlib.Path(ns.golden))
    if ns.limit:
        golden = golden[:ns.limit]
    print(LINE)
    print(f"  GOLDEN RUN — {len(golden)} question(s), judge {ns.model!r}")
    print(LINE)
    print("  answering (retrieval + generation, ~25-125s each)...\n")
    results = run_pipeline(golden)

    bad = [r for r in results if r["error"]]
    if bad:
        print(f"\n  {len(bad)} question(s) raised. Scoring the rest; fix these first.")

    # SAVE THE ANSWERS BEFORE SCORING. The pipeline half costs 25-125s per
    # question; a scoring failure must never throw that away again.
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    # Runs are written into evaluation/runs/ragas/ - inputs (golden_set*.jsonl)
    # stay at the top of evaluation/, where config.GOLDEN_SET_PATH expects them.
    RUNS = pathlib.Path("evaluation") / "runs" / "ragas"
    RUNS.mkdir(parents=True, exist_ok=True)
    raw = RUNS / f"golden_raw_{stamp}.json"
    raw.write_text(json.dumps({"run": stamp, "results": results}, indent=1,
                              ensure_ascii=False), encoding="utf-8")
    print(f"\n  answers saved before scoring -> {raw}")

    literal_recall(results)
    lit = [r for r in results if r.get("literal_recall") is not None]
    if lit:
        print(f"\n  JUDGE-FREE LITERAL RECALL (exact string presence, {len(lit)} scored)")
        for r in lit:
            want = r["expected_strings"]
            miss = [w for w in want if w not in r["literal_hits"]]
            tag = "ok  " if not miss else "MISS"
            print(f"    {r['id']}  {tag} {len(r['literal_hits'])}/{len(want)}"
                  + (f"   absent: {miss}" if miss else ""))

    # *** --metrics none: GENERATE, SAVE, DO NOT SCORE. (2026-09-10, Anuj.) ***
    #
    # Citation accuracy layers 1-3a are string checks over the saved claims. No
    # judge reads anything, so paying four hours of RAGAS scoring to obtain
    # them is four hours spent on a number this run is not asking for. The
    # generation pass alone is roughly 45-90 minutes for 48 questions.
    #
    # THE SAVED FILE IS THE SAME SHAPE either way - same `results`, same
    # citation fields, same `generator_config_hash`. It simply carries no
    # `context_recall` or `faithfulness` columns.
    #
    # SAY SO IN THE FILE, NOT ONLY ON SCREEN. A run whose scores are absent
    # must never be mistaken later for a run whose scores were zero, so
    # `metrics` is recorded in the JSON below.
    if ns.metrics == "none":
        print("\n  --metrics none: generation only. NOT SCORED - this file "
              "carries no\n  context recall or faithfulness, and must not be "
              "compared with a run that has them.")
        df = None
    else:
        print(f"\n  scoring with {ns.model!r} (serial; a local 8B judge is slow)...")
        try:
            df = score(results, ns.model, ns.embed, ns.metrics, ns.timeout,
                       ns.think, ns.num_ctx)
        except Exception:                                     # noqa: BLE001
            traceback.print_exc()
            sys.exit("\n  SCORING RAISED — read the trace. Do not retry with new settings\n"
                     "  until you know the cause.")
        report(results, df)

    if ns.save:
        p = RUNS / f"golden_run_{stamp}.json"
        p.write_text(json.dumps({
            "run": stamp, "judge": ns.model, "embed": ns.embed,
            # WHICH METRICS THIS FILE ACTUALLY CARRIES. "none" means the
            # scoring step did not run - absent, not zero, not failed.
            "metrics": ns.metrics,
            "generator_config_hash": next((r.get("config_hash") for r in results
                                           if r.get("config_hash")), ""),
            "golden_set": ns.golden, "n": len(results), "results": results,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n  saved -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
