"""PROMPT VARIANTS FOR ARM B — does better wording fix what we measured?

WHAT IS HELD FIXED, AND WHY THAT IS THE WHOLE POINT
    MODEL          qwen3:1.7b — the winner of 2026-09-02 (46/60, 3.8s median).
    DECISION RULE  `decide_b` is IMPORTED from experiment_intent.py, not copied.
                   A copy drifts; an import cannot. The code that turns an
                   extraction into a label is byte-identical across variants.
    QUESTIONS      one set, all variants, same order.

    So the ONLY thing that differs between variants is the prompt text. That is
    the 2026-08-23 lesson applied on purpose: change exactly one thing, or the
    attribution is worthless.

A NEW QUESTION SET — AND WHY
    SET 2 below is 60 fresh questions. The variants are SELECTED on set 2, so
    set 2's winning score is optimistic by construction — that is what selection
    does. Set 1 (in experiment_intent.py) was never used to choose a prompt, so
    running the winner on `--set 1` afterwards gives a number that IS comparable
    to the 46/60 baseline. Do both. Quote the set-1 number.

THE FOUR VARIANTS — a control and two targeted fixes, plus their combination.
Not four guesses. Each attacks ONE measured failure of the baseline.

    v1_baseline   the exact 2026-09-02 prompt. The control. Without it, a
                  variant's score has nothing to be better THAN.

    v2_issuer     attacks `OUT_OF_CORPUS -> INCOMPLETE` (5 misses).
                  Cause: the baseline says "do not guess a jurisdiction that is
                  not in the question", so the model returned null for "the PRA"
                  and "Basel". Those are not guesses — an issuer NAMES its
                  jurisdiction. v2 gives the mapping explicitly.

    v3_domain     attacks `INCOMPLETE -> OUT_OF_DOMAIN` (6 misses).
                  Cause: the model set banking=false on real regulatory
                  questions that did not contain an obviously financial noun
                  ("What must be included in a model development record?").
                  v3 defines banking wider and shows three worked examples.

    v4_both       v2 + v3. Tests whether two fixes that each help still help
                  together. They may not: a longer prompt is also a harder one
                  for a 1.7B model, and that cost is real.

READ THE COLUMNS TOGETHER, ALWAYS. An arm that lifts OUT_OF_CORPUS while
dropping ANSWERABLE has not improved; it has moved toward the switch that
failed in August. The baseline to beat is 9/15 out-of-corpus WITH 13/15
answerable kept.

USE
    uv run python scripts/experiment_prompts.py --list
    uv run python scripts/experiment_prompts.py                # set 2, 4 variants
    uv run python scripts/experiment_prompts.py --set 1 --variants v4_both
    uv run python scripts/experiment_prompts.py --quick        # 5 per class

COST. 60 x 4 = 240 generations at ~3.8s = roughly 15-20 minutes.
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

from regrag import config
from regrag.generation import llm

# The rule and the original set come from the first experiment, IMPORTED so they
# cannot drift. Running a script from scripts/ puts scripts/ on sys.path.
from experiment_intent import LABELS, SETS as SET1, decide_b, grid

MODEL = "qwen3:1.7b"
W = 94
LINE = "=" * W
THIN = "-" * W


# ===========================================================================
# SET 2 — 60 FRESH QUESTIONS. Same construction rules as set 1, no overlap.
# ===========================================================================
# INCOMPLETE  = the corpus holds two or more DIFFERENT answers for a facet the
#               question did not give. Only MODEL_RISK_MANAGEMENT (US 2 / UK 1)
#               and STRESS_TESTING (US 6 / GLOBAL 2) qualify.
# OUT_OF_CORPUS = an empty (subject x jurisdiction) cell, a Basel volume not
#               held, or the recorded 12 CFR 217 known_gap. Absence by
#               construction — no judgement call.
# ANSWERABLE  = names a document the registry holds. NOTE: this proves the
#               DOCUMENT is held, not that the specific provision survived
#               parsing. Unverified, and stated as such.

S2_OUT_OF_DOMAIN = [
    "What time does the sun set in Delhi today?",
    "How do I remove a coffee stain from a white shirt?",
    "Explain the offside rule in football.",
    "What is the difference between RAM and an SSD?",
    "Give me a recipe for chocolate brownies.",
    "How many days are there in a leap year?",
    "Who painted the Mona Lisa?",
    "What is a good beginner yoga routine?",
    "How do I change a car tyre?",
    "What is the tallest mountain in Africa?",
    "Write a haiku about the monsoon.",
    "What does HTTP status code 404 mean?",
    "How long should I boil an egg?",
    "What is the currency of Brazil?",
    "Suggest a name for a golden retriever puppy.",
]

S2_INCOMPLETE = [
    "What are the requirements for model change management?",
    "How should model risk appetite be set?",
    "What are the reporting requirements to senior management on model risk?",
    "What must be included in a model development record?",
    "How should conceptual soundness be assessed?",
    "What are the expectations for model calibration?",
    "Who owns a model and what are their responsibilities?",
    "What are the requirements for benchmarking a model?",
    "How should model tiering or materiality be determined?",
    "What are the requirements for outcomes analysis and back-testing?",
    "What severity of scenarios must a stress test cover?",
    "How frequently must stress tests be run?",
    "What governance is required over the stress testing process?",
    "What must be documented about stress testing assumptions?",
    "How should stress test results feed into capital planning?",
]

S2_OUT_OF_CORPUS = [
    # empty (subject x jurisdiction) cells
    "What does the PRA require on capital buffers for ring-fenced banks?",
    "What are the US risk-based capital ratio minimums for a bank holding company?",
    "What does ASC 326 require on credit loss measurement?",
    "How must UK banks apply IFRS 9 staging under PRA rules?",
    "What do US regulators require on data lineage and aggregation?",
    "What does the PRA expect on risk data aggregation capabilities?",
    "What does BCBS require of banks on model risk governance?",
    "What are the Bank of England's concurrent stress testing requirements?",
    # Basel volumes not held (corpus has CAP CRE LEX RBC SCO SRP32)
    "How is the leverage ratio exposure measure calculated?",
    "What are the Basel rules on the credit valuation adjustment capital charge?",
    "What is the standardised approach to operational risk capital?",
    "What are the Basel liquidity monitoring tools?",
    "What are the Basel disclosure requirements for remuneration?",
    # the recorded known_gap: 12 CFR 217 (Regulation Q)
    "What is the capital conservation buffer requirement for US banks under Regulation Q?",
    "What does 12 CFR 217.11 say about the maximum payout ratio?",
]

S2_ANSWERABLE = [
    "What does SR 26-2 say about model risk management expectations?",
    "Under SS1/23, what does Principle 2 require on model governance?",
    "What does IFRS 9 say about determining significant increases in credit risk?",
    "What does BCBS 239 require on the timeliness of risk data?",
    "What does SR 11-7 say about the role of internal audit in model risk?",
    "What does IFRS 9 say about the time value of money in measuring ECL?",
    "What does 12 CFR Part 252 require on capital plan submissions?",
    "What does Basel RBC say about minimum risk-based capital ratios?",
    "What does SS1/23 say about the model risk management framework?",
    "What does Basel SRP32 say about Pillar 2 risk types?",
    "What does SR 15-19 say about capital planning for large firms?",
    "What does BCBS d403 say about the definition of non-performing exposures?",
    "What do the 2026 Stress Test Scenarios describe as the severely adverse scenario?",
    "What does Basel CRE say about eligible financial collateral?",
    "What does Basel SCO say about the treatment of investments in banking entities?",
]

SET2 = {"OUT_OF_DOMAIN": S2_OUT_OF_DOMAIN, "INCOMPLETE": S2_INCOMPLETE,
        "OUT_OF_CORPUS": S2_OUT_OF_CORPUS, "ANSWERABLE": S2_ANSWERABLE}


# ===========================================================================
# THE FOUR PROMPTS. Only this text differs between variants.
# ===========================================================================
_HEAD = """You read a user's question and extract facts from it. You do not answer it,
and you do not decide whether it can be answered."""

_KEYS = """Reply with JSON only, exactly these keys:
  "banking"      true if the question is about banking regulation, else false
  "subject"      one of {subjects} or null if none clearly applies
  "jurisdiction" one of {jurisdictions} or null if the question does not say
  "document"     the name of a regulation or standard the question NAMES,
                 written exactly as the question writes it, or null
  "specific"     true if the question asks about a particular rule, principle
                 or provision; false if it is a broad open question"""

_V1_TAIL = """Do not guess a jurisdiction that is not in the question. If the question does
not say US or UK, jurisdiction is null. That null is useful information, and a
guess destroys it."""

_ISSUER = """JURISDICTION IS OFTEN NAMED BY THE ISSUER, NOT BY THE WORD "US" OR "UK".
Naming a regulator or a body IS naming its jurisdiction. This is reading, not
guessing:
  UK      PRA, FCA, Bank of England, ring-fenced bank, CRR as applied in the UK
  US      Federal Reserve, the Fed, OCC, FDIC, SEC, CFR, Regulation Q / Y / YY,
          SR letter, US GAAP, ASC, CECL, bank holding company, CCAR, DFAST
  GLOBAL  Basel, BCBS, the Basel Committee, IFRS, IASB, Pillar 1 / 2 / 3

If the question names none of these and does not say US or UK, jurisdiction is
null. That null is useful information, and a guess destroys it."""

_DOMAIN = """"banking" IS WIDE. It is true for any question about how banks are supervised
or must behave — capital, credit losses and provisioning, risk models and their
validation, stress testing, risk data, governance and reporting to a board or
supervisor. A question does not need the word "bank", "regulator" or a named
rule in it. If it would be asked by a risk manager, a model validator or a
supervisor at a bank, it is banking.

  "What must be included in a model development record?"   -> banking true
  "How should conceptual soundness be assessed?"           -> banking true
  "How do I change a car tyre?"                            -> banking false"""


def build_prompt(variant: str, subjects, jurisdictions) -> str:
    keys = _KEYS.format(subjects=subjects, jurisdictions=jurisdictions)
    parts = {
        "v1_baseline": [_HEAD, keys, _V1_TAIL],
        "v2_issuer":   [_HEAD, keys, _ISSUER],
        "v3_domain":   [_HEAD, _DOMAIN, keys, _V1_TAIL],
        "v4_both":     [_HEAD, _DOMAIN, keys, _ISSUER],
    }[variant]
    return "\n\n".join(parts)


VARIANTS = ("v1_baseline", "v2_issuer", "v3_domain", "v4_both")


# ===========================================================================
def run(variant: str, questions, g, subjects, jurisdictions, names) -> dict:
    system = build_prompt(variant, subjects, jurisdictions)
    print(f"\n{LINE}\n{variant}   {len(system):,} chars of prompt\n{LINE}")
    rows = []
    for i, (truth, q) in enumerate(questions, 1):
        t0 = time.perf_counter()
        try:
            out = llm.chat(system, q, model=MODEL, fmt="json",
                           max_tokens=200, think=False)
            raw = json.loads(out.text)
            got, why = decide_b(raw, g, names)
            sec = out.seconds
        except Exception as exc:                                   # noqa: BLE001
            raw, got, why = None, "UNPARSABLE", f"{type(exc).__name__}: {exc}"
            sec = time.perf_counter() - t0
        ok = got == truth
        rows.append({"truth": truth, "got": got, "question": q, "why": why,
                     "raw": raw, "ok": ok, "seconds": round(sec, 2)})
        print(f"  {'ok ' if ok else 'MISS'} [{i:>2}] {truth:<14} -> {got:<14} "
              f"{sec:>5.1f}s  {q[:42]}")
    return {"variant": variant, "model": MODEL, "prompt_chars": len(system),
            "rows": rows}


def table(results: list[dict]) -> None:
    print(f"\n{LINE}\nCOMPARISON — every column read against the one beside it\n{LINE}")
    hdr = "".join(f"{x[:9]:>12}" for x in LABELS)
    print(f"  {'variant':<14}{'total':>9}{hdr}{'med s':>8}{'chars':>8}")
    base = None
    for r in results:
        rows = r["rows"]
        per = collections.defaultdict(lambda: [0, 0])
        for x in rows:
            per[x["truth"]][1] += 1
            per[x["truth"]][0] += x["ok"]
        sec = [x["seconds"] for x in rows]
        tot = sum(x["ok"] for x in rows)
        cells = "".join(f"{per[k][0]}/{per[k][1]:<10}" for k in LABELS)
        mark = ""
        if r["variant"] == "v1_baseline":
            base = tot
        elif base is not None:
            mark = f"  ({tot - base:+d})"
        print(f"  {r['variant']:<14}{tot:>4}/{len(rows):<4}{cells}"
              f"{statistics.median(sec):>7.1f}{r['prompt_chars']:>8,}{mark}")

    print(f"\n{THIN}\n  THE TWO NUMBERS THAT DECIDE IT\n{THIN}")
    for r in results:
        rows = r["rows"]
        ooc = [x for x in rows if x["truth"] == "OUT_OF_CORPUS"]
        ans = [x for x in rows if x["truth"] == "ANSWERABLE"]
        print(f"  {r['variant']:<14} out-of-corpus caught "
              f"{sum(x['ok'] for x in ooc)}/{len(ooc)}"
              f"   answerable kept "
              f"{sum(x['got'] == 'ANSWERABLE' for x in ans)}/{len(ans)}")
    print("\n  A variant that lifts the first while dropping the second has NOT")
    print("  improved. It has moved toward the switch that failed in August.")

    print(f"\n{THIN}\n  WHERE EACH VARIANT WENT WRONG\n{THIN}")
    for r in results:
        c = collections.Counter((x["truth"], x["got"])
                                for x in r["rows"] if not x["ok"])
        print(f"\n  {r['variant']}")
        for (t, gt), n in c.most_common(5):
            print(f"    {t:<15} -> {gt:<15} {n}")
        if not c:
            print("    none")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="2", choices=["1", "2"],
                    help="2 = the fresh set (default, tune here). "
                         "1 = experiment_intent's set, never used to pick a "
                         "prompt, so its number is comparable to the 46/60.")
    ap.add_argument("--variants", default="", help="e.g. v1_baseline,v4_both")
    ap.add_argument("--quick", action="store_true", help="5 per class")
    ap.add_argument("--list", action="store_true")
    ns = ap.parse_args()

    g, subjects, jurisdictions, names = grid()
    src = SET2 if ns.set == "2" else SET1
    n = 5 if ns.quick else 15
    questions = [(lab, q) for lab in LABELS for q in src[lab][:n]]

    if ns.list:
        for lab in LABELS:
            print(f"\n{lab}")
            for i, q in enumerate(src[lab][:n], 1):
                print(f"  {i:>2}. {q}")
        print(f"\n{LINE}\nPROMPTS\n{LINE}")
        for v in VARIANTS:
            p = build_prompt(v, subjects, jurisdictions)
            print(f"\n----- {v}  ({len(p):,} chars) -----\n{p}")
        return 0

    want = [v.strip() for v in ns.variants.split(",") if v.strip()] or list(VARIANTS)
    for v in want:
        if v not in VARIANTS:
            sys.exit(f"unknown variant {v!r} — use {list(VARIANTS)}")

    try:
        tags = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10).json()
    except Exception as exc:                                       # noqa: BLE001
        sys.exit(f"Ollama unreachable: {exc}")
    if not any(m["name"].startswith(MODEL) for m in tags.get("models", [])):
        sys.exit(f"{MODEL} not pulled")

    print(f"{LINE}\nPROMPT VARIANTS — model {MODEL} FIXED, decide_b FIXED, "
          f"set {ns.set}\n{LINE}")
    print(f"  {len(questions)} questions x {len(want)} variant(s) = "
          f"{len(questions) * len(want)} generations")
    if ns.set == "2":
        print("  Selecting on set 2. The winner's set-2 score is OPTIMISTIC by")
        print("  construction. Re-run it with --set 1 for a comparable number.")

    out = pathlib.Path("evaluation") / (
        f"prompts_set{ns.set}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    out.parent.mkdir(exist_ok=True)

    results = []
    for v in want:
        try:
            results.append(run(v, questions, g, subjects, jurisdictions, names))
        except KeyboardInterrupt:
            print("\n  interrupted — keeping finished variants")
            break
        out.write_text(json.dumps({"set": ns.set, "results": results}, indent=1),
                       encoding="utf-8")
    if not results:
        sys.exit("nothing completed")

    table(results)
    print(f"\n  raw -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
