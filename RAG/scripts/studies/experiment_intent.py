"""INTENT CLASSIFIER — A TEST, NOT A FEATURE. Changes nothing in the pipeline.

THE QUESTION BEING ASKED
    Can a small local model, placed BEFORE retrieval, sort a question into four
    buckets well enough to (a) refuse out-of-domain, (b) ask for clarification
    when the corpus holds more than one answer, (c) decline out-of-corpus, and
    (d) pass everything else through unchanged?

    It calls `regrag.registry` and `regrag.generation.llm` READ-ONLY. It imports
    nothing from retrieval or generation, writes no config, and touches no
    module. Deleting this file returns the project to exactly its current state.

FOUR ARMS — two axes, crossed, so each can be read alone
    MODEL   qwen3:1.7b            the generator; already resident in Ollama
            qwen2.5:3b-instruct   JUDGE_MODEL; instruct-tuned, never timed
            gemma3:1b             smallest of the three. If it classifies as
                                  well as the others it is the right choice -
                                  a gate runs on EVERY question, so its cost is
                                  paid even by questions that pass straight
                                  through. Cheapest adequate wins here, which
                                  is the opposite of the generator decision.
    PROMPT  A  the model is given the document list and returns THE LABEL.
               This is the design shape that failed on 2026-08-26: a small model
               asked to judge sufficiency. Included so its failure is measured
               here rather than assumed from that run.
            B  the model returns ONLY extraction — subject, jurisdiction, named
               document, and whether the question is specific. CODE then decides
               the label from the registry grid. The model does what models are
               good at (reading a sentence); the corpus fact stays deterministic.

    Arm B exists because ABSENCE IS STRUCTURAL, NOT SCORED. An empty
    (subject x jurisdiction) cell is a fact in registry.yaml. That is how the
    hard negatives were derived in the first place, and it is the one part of
    this that cannot hallucinate.

THE DECISION RULE FOR ARM B IS FIXED HERE, BEFORE THE RUN — see `decide_b`.
Do not move it after seeing results. The 2026-08-26 experiment is trustworthy
because its rule was written first; that is the only reason its verdict stands.

WHAT THIS CANNOT TELL YOU
  * Whether the pipeline ANSWERS better. It measures classification only. A
    correct label still has to be wired to a behaviour, and that is not built.
  * Whether these labels survive questions I did not write. THE 60 QUESTIONS
    BELOW ARE MY CONSTRUCTION. If a prompt is later tuned against them and then
    scored on them, that is eval contamination of exactly the kind already
    recorded in this project. Tune on a different set, or write fresh questions
    before believing a second number.
  * Whether ANSWERABLE questions are actually answerable. They name documents
    that exist in the registry. Whether the specific provision survived parsing
    is NOT verified — IFRS 9 in particular is an extract, not the full standard.

USE
    uv run python scripts/experiment_intent.py            # all 4 arms, 60 q
    uv run python scripts/experiment_intent.py --quick    # 5 per class, 20 q
    uv run python scripts/experiment_intent.py --arms 1.7b:B
    uv run python scripts/experiment_intent.py --list     # print the set, no calls

COST. 60 questions x 6 arms = 360 CPU generations. The prompt is short, so
expect roughly 3-8s each; budget 20-30 minutes for the full run. `--quick` is
80 calls. `--list` is free.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys
import time
from datetime import datetime, timezone

import requests

from regrag import config, registry
from regrag.generation import llm

W = 94
LINE = "=" * W
THIN = "-" * W

MODELS = {"1.7b": "qwen3:1.7b",
          "3b": "qwen2.5:3b-instruct",
          "gemma": "gemma3:1b"}
LABELS = ("OUT_OF_DOMAIN", "INCOMPLETE", "OUT_OF_CORPUS", "ANSWERABLE")
OUTF: pathlib.Path | None = None


# ===========================================================================
# THE QUESTION SET — 15 per class. Labels assigned by CONSTRUCTION, not taste.
# ===========================================================================
# INCOMPLETE is defined precisely: the corpus holds TWO OR MORE DIFFERENT
# answers depending on a facet the question did not give. Every INCOMPLETE
# question below is MODEL_RISK_MANAGEMENT (US 2 docs vs UK 1) or STRESS_TESTING
# (US 6 vs GLOBAL 2). Capital, impairment and data quality are GLOBAL-only in
# this corpus, so a bare question about them is NOT ambiguous and is not here.
#
# OUT_OF_CORPUS is absence BY CONSTRUCTION — an empty cell in the
# (subject x jurisdiction) grid, an absent Basel volume, or the recorded
# 12 CFR 217 known_gap. No judgement call was made about any of them.

OUT_OF_DOMAIN = [
    "What is the capital of France?",
    "Write me a Python function that reverses a string.",
    "How do I fix a leaking kitchen tap?",
    "What is the weather in Mumbai today?",
    "Recommend a good laptop for video editing.",
    "Who won the last football World Cup?",
    "Translate 'good morning' into Japanese.",
    "What is the best way to learn guitar?",
    "How do I cook basmati rice so the grains stay separate?",
    "What are the health benefits of walking thirty minutes a day?",
    "Summarise the plot of Hamlet in three sentences.",
    "What is the square root of 4096?",
    "Book me a flight to Singapore for next Tuesday.",
    "Explain how a diesel engine works.",
    "What should I plant in a north-facing garden?",
]

INCOMPLETE = [
    "What are the model validation requirements?",
    "What is the definition of a model?",
    "How should model risk be governed?",
    "What must a model inventory contain?",
    "Who is responsible for independent model review?",
    "What are the documentation requirements for models?",
    "What does the supervisory guidance say about model limitations?",
    "How should model performance be monitored on an ongoing basis?",
    "What are the expectations for vendor and third-party models?",
    "How should a bank handle model overrides?",
    "What are the stress testing requirements for large banks?",
    "How should stress test scenarios be designed?",
    "What must be disclosed about stress test results?",
    "What are the governance expectations for stress testing?",
    "What role does the board play in model risk management?",
]

OUT_OF_CORPUS = [
    # --- empty (subject x jurisdiction) cells -----------------------------
    "What capital buffers does the PRA require UK banks to hold?",
    "What is the US minimum CET1 ratio requirement under Regulation Q?",
    "How does CECL differ from the incurred loss model under US GAAP?",
    "What does the PRA expect on IFRS 9 provisioning for UK banks?",
    "What are the US regulatory requirements for risk data aggregation?",
    "What does the PRA require on data governance and data lineage?",
    "What is the Basel standard for model risk management?",
    "What are the PRA's annual cyclical scenario stress testing requirements?",
    # --- Basel volumes not held (corpus has CAP CRE LEX RBC SCO SRP32) ----
    "What is the Basel treatment of operational risk capital?",
    "How is the Liquidity Coverage Ratio calculated?",
    "What are the market risk capital requirements under FRTB?",
    "What Pillar 3 disclosures are required?",
    "What is the Net Stable Funding Ratio requirement?",
    # --- the recorded known_gap: 12 CFR 217 (Regulation Q) ---------------
    "What is the CET1 distribution limitation for a US bank holding company?",
    "What does 12 CFR 217.20(b) say about the definition of common equity tier 1?",
]

ANSWERABLE = [
    "What does SR 11-7 say about the three core elements of model risk management?",
    "Under SS1/23, what does Principle 1 require on model identification?",
    "What does IFRS 9 say about the general approach to recognising expected credit losses?",
    "Under BCBS 239, what does Principle 3 require on accuracy and integrity?",
    "What does SR 11-7 say about the use of vendor and third-party models?",
    "What does IFRS 9 require for purchased or originated credit-impaired financial assets?",
    "Under 12 CFR Part 252, what are the company-run stress test requirements?",
    "What does Basel CAP say about the criteria for Common Equity Tier 1 instruments?",
    "What does SS1/23 say about independence in model validation?",
    "What does Basel LEX say about the large exposure limit to a single counterparty?",
    "What does SR 15-18 say about capital planning expectations for large firms?",
    "What does BCBS d450 say about the principles for sound stress testing?",
    "What does Basel SCO say about the scope of application of the framework?",
    "What does 12 CFR 225.8 require regarding a firm's capital plan?",
    "What does Basel CRE say about the standardised approach to credit risk mitigation?",
]

SETS = {"OUT_OF_DOMAIN": OUT_OF_DOMAIN, "INCOMPLETE": INCOMPLETE,
        "OUT_OF_CORPUS": OUT_OF_CORPUS, "ANSWERABLE": ANSWERABLE}

# Questions where a disagreement is INFORMATIVE rather than simply wrong. Read
# these individually before counting them against an arm.
NOTED = {
    "What is the US minimum CET1 ratio requirement under Regulation Q?":
        "Basel RBC states 4.5% and the DFAST table contains the numeral 4.5. "
        "Both look like answers; neither is a US requirement. This is the "
        "known_gap case — false corroboration from two documents at once.",
    "What are the stress testing requirements for large banks?":
        "'large banks' hints US without naming it. If an arm calls this "
        "ANSWERABLE it is guessing a jurisdiction, which is the 2b failure.",
    "What is the Basel standard for model risk management?":
        "Names a real framework and a real subject, but that cell is empty. "
        "The hardest OUT_OF_CORPUS shape: everything about it looks valid.",
}


# ===========================================================================
# THE CORPUS GRID — from registry.yaml, so it cannot drift from the index
# ===========================================================================
def grid() -> tuple[dict, list[str], list[str], dict]:
    """(subject, jurisdiction) -> [short_name], plus the vocabularies."""
    reg = registry.load_cached()
    docs = reg.indexable()      # a METHOD, not a property
    g: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for d in docs:
        su = d.subject.value if hasattr(d.subject, "value") else str(d.subject)
        ju = d.jurisdiction.value if hasattr(d.jurisdiction, "value") else str(d.jurisdiction)
        g[(su, ju)].append(d.short_name)
    subjects = sorted({k[0] for k in g})
    jurisdictions = sorted({k[1] for k in g})
    names = {d.short_name.lower(): d.short_name for d in docs}
    return dict(g), subjects, jurisdictions, names


def print_grid(g, subjects, jurisdictions) -> None:
    print(f"\n  {'subject':26}" + "".join(f"{j:>10}" for j in jurisdictions))
    for su in subjects:
        row = "".join(f"{len(g.get((su, j), [])):>10}" for j in jurisdictions)
        print(f"  {su:26}{row}")
    empty = [(su, j) for su in subjects for j in jurisdictions if not g.get((su, j))]
    print(f"\n  {len(empty)} empty cell(s) — absence BY CONSTRUCTION:")
    for su, j in empty:
        print(f"    {su} x {j}")


# ===========================================================================
# PROMPTS
# ===========================================================================
def prompt_a(g, subjects, jurisdictions, names) -> str:
    docs = "\n".join(f"  - {n}" for n in sorted(names.values()))
    return f"""You sort a user's question into exactly one of four labels. You do not answer it.

The system you guard searches ONLY these documents:
{docs}

Labels:
  OUT_OF_DOMAIN  - not about banking regulation at all.
  INCOMPLETE     - about banking regulation, but the documents above hold more
                   than one different answer and the question does not say which
                   is wanted (for example it does not say US or UK, or does not
                   say which version).
  OUT_OF_CORPUS  - about banking regulation, but the documents above do not
                   cover it. Say this even when the topic sounds familiar.
  ANSWERABLE     - about banking regulation and specific enough that the
                   documents above can answer it.

Reply with JSON only: {{"label": "<one label>", "reason": "<one short sentence>"}}"""


def prompt_b(g, subjects, jurisdictions, names) -> str:
    return f"""You read a user's question and extract facts from it. You do not answer it,
and you do not decide whether it can be answered.

Reply with JSON only, exactly these keys:
  "banking"      true if the question is about banking regulation, else false
  "subject"      one of {subjects} or null if none clearly applies
  "jurisdiction" one of {jurisdictions} or null if the question does not say
  "document"     the name of a regulation or standard the question NAMES,
                 written exactly as the question writes it, or null
  "specific"     true if the question asks about a particular rule, principle
                 or provision; false if it is a broad open question

Do not guess a jurisdiction that is not in the question. If the question does
not say US or UK, jurisdiction is null. That null is useful information, and a
guess destroys it."""


# ===========================================================================
# THE ARM-B DECISION RULE — WRITTEN BEFORE THE RUN. DO NOT EDIT AFTER RESULTS.
# ===========================================================================
def decide_b(x: dict, g, names) -> tuple[str, str]:
    """Extraction -> label, in code. Every branch is a registry fact."""
    if not x.get("banking"):
        return "OUT_OF_DOMAIN", "not banking regulation"

    doc = (x.get("document") or "").strip().lower()
    if doc:
        hit = next((full for low, full in names.items() if low in doc or doc in low), None)
        if hit:
            return "ANSWERABLE", f"names a document we hold ({hit})"
        return "OUT_OF_CORPUS", f"names {doc!r}, which is not in the registry"

    su, ju = x.get("subject"), x.get("jurisdiction")
    if not su:
        return "INCOMPLETE", "banking, but no subject identified"

    if not ju:
        have = [j for (s, j) in g if s == su and g[(s, j)]]
        if not have:
            return "OUT_OF_CORPUS", f"no document at all on {su}"
        if len(have) == 1:
            return "ANSWERABLE", f"{su} exists only for {have[0]} — no ambiguity"
        return "INCOMPLETE", f"{su} exists for {sorted(have)} — which one?"

    if not g.get((su, ju)):
        return "OUT_OF_CORPUS", f"({su} x {ju}) is an empty cell"
    return "ANSWERABLE", f"({su} x {ju}) has {len(g[(su, ju)])} document(s)"


# ===========================================================================
# RUN
# ===========================================================================
def call(model: str, system: str, user: str) -> tuple[dict | None, dict]:
    """One classification. Returns (parsed_or_None, timing)."""
    t0 = time.perf_counter()
    try:
        # `think` is a hybrid-Qwen3 field. Ollama ignores unknown fields
        # silently, which is how a setting looks applied when it is not - so
        # send it only where it means something, and say so here.
        kw = {"think": False} if model.startswith("qwen3") else {}
        out = llm.chat(system, user, model=model, fmt="json",
                       max_tokens=200, **kw)
    except Exception as exc:                                       # noqa: BLE001
        return None, {"error": f"{type(exc).__name__}: {exc}",
                      "seconds": time.perf_counter() - t0}
    t = {"seconds": round(out.seconds, 2),
         "prompt_seconds": round(out.prompt_seconds, 2),
         "eval_seconds": round(out.eval_seconds, 2),
         "prompt_tokens": out.prompt_tokens, "output_tokens": out.output_tokens}
    try:
        return json.loads(out.text), t
    except Exception:                                              # noqa: BLE001
        t["unparsable"] = out.text[:200]
        return None, t


def run_arm(mkey: str, pkey: str, questions, g, subjects, jurisdictions, names) -> dict:
    model = MODELS[mkey]
    system = (prompt_a if pkey == "A" else prompt_b)(g, subjects, jurisdictions, names)
    print(f"\n{LINE}\nARM {mkey}:{pkey}   model {model}   "
          f"prompt {'label' if pkey == 'A' else 'extract + registry'}\n{LINE}")

    rows = []
    for i, (truth, q) in enumerate(questions, 1):
        raw, t = call(model, system, q)
        if raw is None:
            got, why = "UNPARSABLE", t.get("error") or t.get("unparsable", "")
        elif pkey == "A":
            got = str(raw.get("label", "")).strip().upper()
            why = str(raw.get("reason", ""))[:70]
            if got not in LABELS:
                got, why = "UNPARSABLE", f"label {got!r} not in the four"
        else:
            got, why = decide_b(raw, g, names)

        ok = got == truth
        rows.append({"truth": truth, "got": got, "question": q, "why": why,
                     "raw": raw, "ok": ok, **t})
        print(f"  {'ok ' if ok else 'MISS'} [{i:>2}] {truth:<14} -> {got:<14} "
              f"{t.get('seconds', 0):>5.1f}s  {q[:44]}")
        if not ok:
            print(f"        {why}")

    return {"model": model, "prompt": pkey, "rows": rows}


def report(arm: dict) -> None:
    rows = arm["rows"]
    secs = [r["seconds"] for r in rows if "seconds" in r and "error" not in r]
    ok = sum(r["ok"] for r in rows)
    print(f"\n{THIN}\n  ARM {arm['model']} / prompt {arm['prompt']}"
          f"   {ok}/{len(rows)} correct\n{THIN}")

    per = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["truth"]][1] += 1
        per[r["truth"]][0] += r["ok"]
    for lab in LABELS:
        c, n = per[lab]
        if n:
            print(f"    {lab:<15} {c}/{n}")

    print("\n    confusion (truth -> what it said):")
    conf = collections.Counter((r["truth"], r["got"]) for r in rows if not r["ok"])
    for (tr, gt), n in conf.most_common():
        print(f"      {tr:<15} -> {gt:<15} {n}")
    if not conf:
        print("      none")

    if secs:
        rd = [r.get("prompt_seconds", 0) for r in rows if "seconds" in r]
        wr = [r.get("eval_seconds", 0) for r in rows if "seconds" in r]
        print(f"\n    latency  median {statistics.median(secs):.1f}s   "
              f"min {min(secs):.1f}s   max {max(secs):.1f}s   "
              f"total {sum(secs):.0f}s")
        print(f"    of which read {statistics.median(rd):.1f}s   "
              f"write {statistics.median(wr):.1f}s   (medians)")
        print(f"    -> this is what EVERY question would pay before retrieval starts")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5 per class")
    ap.add_argument("--arms", default="", help="e.g. 1.7b:B,3b:B (default: all four)")
    ap.add_argument("--list", action="store_true", help="print the set and exit")
    ns = ap.parse_args()

    g, subjects, jurisdictions, names = grid()

    n = 5 if ns.quick else 15
    questions = [(lab, q) for lab in LABELS for q in SETS[lab][:n]]

    if ns.list:
        print(f"{LINE}\nCORPUS GRID (registry.yaml, indexable rows)\n{LINE}")
        print_grid(g, subjects, jurisdictions)
        print(f"\n{LINE}\nQUESTION SET — {len(questions)}\n{LINE}")
        for lab in LABELS:
            print(f"\n{lab}")
            for i, q in enumerate(SETS[lab][:n], 1):
                print(f"  {i:>2}. {q}")
                if q in NOTED:
                    print(f"      NOTE: {NOTED[q]}")
        return 0

    global OUTF
    OUTF = pathlib.Path("evaluation") / (
        f"intent_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")

    arms = ([tuple(a.split(":")) for a in ns.arms.split(",") if a]
            or [(m, p) for m in MODELS for p in ("A", "B")])

    # Preflight: name WHICH model is missing, not a generic failure.
    try:
        tags = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10).json()
        have = {m["name"] for m in tags.get("models", [])}
    except Exception as exc:                                       # noqa: BLE001
        sys.exit(f"Ollama unreachable at {config.OLLAMA_HOST}: {exc}")
    for mkey, _ in arms:
        if mkey not in MODELS:
            sys.exit(f"unknown model key {mkey!r} — use {list(MODELS)}")
        want = MODELS[mkey]
        if not any(h == want or h.startswith(want + ":") for h in have):
            sys.exit(f"{want} not pulled. Run: ollama pull {want}")

    print(f"{LINE}\nINTENT CLASSIFIER EXPERIMENT — classification only, nothing wired\n{LINE}")
    print_grid(g, subjects, jurisdictions)
    print(f"\n  {len(questions)} questions x {len(arms)} arm(s) = "
          f"{len(questions) * len(arms)} generations")
    print("  The arm-B decision rule is fixed in `decide_b`. Do not edit it after "
          "seeing results.")

    # SAVE AFTER EVERY ARM. A six-arm full run is ~an hour of CPU; writing
    # only at the end means a crash or a Ctrl-C at arm five loses all of it.
    OUTF.parent.mkdir(exist_ok=True)
    results = []
    for mkey, pkey in arms:
        try:
            arm = run_arm(mkey, pkey, questions, g, subjects, jurisdictions, names)
        except KeyboardInterrupt:
            print("\n  interrupted — keeping the arms already finished")
            break
        except Exception as exc:                                   # noqa: BLE001
            # One arm failing (model unloaded, Ollama restarted) must not take
            # down the arms that already cost real time.
            print(f"\n  ARM {mkey}:{pkey} FAILED — {type(exc).__name__}: {exc}")
            continue
        report(arm)
        results.append(arm)
        OUTF.write_text(json.dumps({"tool": "experiment_intent.py",
                                    "complete": False, "arms": results},
                                   indent=1), encoding="utf-8")
        print(f"  [saved {len(results)} arm(s) -> {OUTF}]")
    if not results:
        sys.exit("no arm completed")

    print(f"\n{LINE}\nALL ARMS\n{LINE}")
    print(f"  {'arm':<28}{'correct':>10}{'median s':>11}{'OOC recall':>13}")
    for a in results:
        rows = a["rows"]
        secs = [r["seconds"] for r in rows if "seconds" in r]
        ooc = [r for r in rows if r["truth"] == "OUT_OF_CORPUS"]
        ooc_hit = sum(r["ok"] for r in ooc)
        tp = [r for r in rows if r["truth"] == "ANSWERABLE"]
        tp_kept = sum(r["got"] == "ANSWERABLE" for r in tp)
        print(f"  {a['model'] + '/' + a['prompt']:<28}"
              f"{sum(r['ok'] for r in rows):>4}/{len(rows):<5}"
              f"{statistics.median(secs) if secs else 0:>10.1f}s"
              f"{ooc_hit:>7}/{len(ooc):<5}"
              f"   answerable kept {tp_kept}/{len(tp)}")
    print("\n  READ BOTH COLUMNS TOGETHER. An arm that declines every out-of-corpus")
    print("  question AND blocks answerable ones is the 2026-08-26 arm-A result")
    print("  again — a switch, not a judgement. High OOC recall only counts when")
    print("  'answerable kept' stays at or near full marks.")

    OUTF.write_text(json.dumps({"tool": "experiment_intent.py",
                                "complete": True, "arms": results},
                               indent=1), encoding="utf-8")
    print(f"\n  raw -> {OUTF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
