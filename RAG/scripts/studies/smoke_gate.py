"""STAGE 8.6 — THE LIVE HALF OF THE GATE. Needs Ollama; tests/test_gate.py does not.

WHAT THIS COVERS THAT THE UNIT TESTS CANNOT
    `tests/test_gate.py` hands `decide()` a dictionary I wrote by hand. It
    proves the RULE is right and proves nothing about whether a 1.7B model ever
    produces that dictionary. This is the seam between the two — the only place
    a real model meets the real rule.

    That seam is where the quiet failure lives. If the model returns
    "subject": "capital" instead of "CAPITAL_ADEQUACY", the grid lookup misses,
    `decide()` returns INCOMPLETE, and NOTHING COMPLAINS. The user is asked a
    clarifying question about a subject the corpus covers perfectly well. No
    exception, no log line, just a worse system. The vocabulary check below is
    the reason this script exists.

FIVE CHECKS
    1 SCHEMA      all five keys present, and of the right TYPE. A model that
                  returns a list where a string belongs is a bug the rule will
                  swallow rather than raise on.
    2 VOCABULARY  every non-null subject / jurisdiction is a value the registry
                  actually uses. THE CHECK THAT MATTERS MOST — see above.
    3 STOP        a question the gate should stop comes back with answer=None
                  and a message, and NEVER reaches retrieval.
    4 PASS        a question that should pass reaches the pipeline and produces
                  a cited answer.
    5 TRANSPARENT the pass-through path returns the SAME answer text as calling
                  `answer()` directly. The gate must add nothing and remove
                  nothing on the path where it is supposed to do nothing.

WHAT IT DELIBERATELY DOES NOT DO
    It is not an accuracy measurement. Six questions cannot tell you whether the
    gate classifies well — `scripts/experiment_prompts.py` did that on 120
    questions across two independent sets, and those are the numbers to quote.
    This asks a smaller question: does the wiring hold?

    A FAILURE HERE IS A PLUMBING BUG. A wrong DECISION here is not necessarily
    one — with 12-13 of 15 out-of-corpus questions declined, one miss in six is
    inside the measured rate. Read a wrong decision as information, not as a
    regression, and go to the experiment for the real number.

    uv run python scripts/smoke_gate.py
    uv run python scripts/smoke_gate.py --show      # print each extraction
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from regrag import config
from regrag.gate import classify, respond
from regrag.gate.intent import corpus_grid
from regrag.generation import llm

W = 88
fails: list[str] = []


def check(label: str, passed: bool, detail: str = "") -> None:
    if not passed:
        fails.append(label)
    print(f"  [{'PASS' if passed else 'FAIL'}] {label:<50}{detail}")


# question, what the gate SHOULD say. Two of each stopping kind, two that pass.
CASES = [
    ("How do I cook basmati rice?", "OUT_OF_DOMAIN"),
    ("Who won the last football World Cup?", "OUT_OF_DOMAIN"),
    ("What does the PRA require on capital buffers for UK banks?", "OUT_OF_CORPUS"),
    ("How does CECL differ from the incurred loss model under US GAAP?", "OUT_OF_CORPUS"),
    ("What are the model validation requirements?", "INCOMPLETE"),
    ("What does SR 11-7 say about the use of vendor models?", "ANSWERABLE"),
]

REQUIRED = {"banking": bool, "subject": (str, type(None)),
            "jurisdiction": (str, type(None)), "document": (str, type(None)),
            "specific": (bool, type(None))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print every extraction")
    ns = ap.parse_args()

    print("=" * W)
    print(f"GATE SMOKE   {config.GATE_MODEL} / {config.GATE_PROMPT}   "
          f"enabled={config.GATE_ENABLED}")
    print("=" * W)

    if not config.GATE_ENABLED:
        sys.exit("REGRAG_GATE=0 — the gate is off, so there is nothing to smoke test.")
    problem = llm.why_unavailable()
    if problem:
        sys.exit(f"  Ollama: {problem}")

    _g, subjects, jurisdictions, _n = corpus_grid()
    print(f"  vocabulary: {len(subjects)} subject(s), "
          f"{len(jurisdictions)} jurisdiction(s)\n")

    # ---- 1 + 2: schema and vocabulary --------------------------------
    print("1-2. SCHEMA AND VOCABULARY — the seam between model and rule")
    print("-" * W)
    right = 0
    for q, want in CASES:
        it = classify(q)
        x = it.extraction

        missing = [k for k in REQUIRED if k not in x]
        bad_type = [k for k, t in REQUIRED.items()
                    if k in x and not isinstance(x[k], t)]
        check(f"schema  {q[:40]!r}", not missing and not bad_type,
              f"missing {missing} bad-type {bad_type}" if (missing or bad_type)
              else f"{it.seconds:.1f}s")

        off = []
        if x.get("subject") and x["subject"] not in subjects:
            off.append(f"subject={x['subject']!r}")
        if x.get("jurisdiction") and x["jurisdiction"] not in jurisdictions:
            off.append(f"jurisdiction={x['jurisdiction']!r}")
        check("  values are from the registry vocabulary", not off,
              "; ".join(off) if off else "ok")

        right += it.decision == want
        if it.decision != want:
            print(f"       decision {it.decision} (expected {want}) — "
                  f"{it.reason}")
        if ns.show:
            print(f"       {json.dumps(x)}")

    print(f"\n  decisions matching expectation: {right}/{len(CASES)}")
    print("  NOT an accuracy measure — see this file's docstring. The measured")
    print("  numbers are 53/60 and 55/60 from scripts/experiment_prompts.py.")

    # ---- 3: a stopped question never reaches retrieval ----------------
    print("\n" + "=" * W)
    print("3. STOPPED — no retrieval, no generation, one message")
    print("=" * W)
    t0 = time.perf_counter()
    r = respond("How do I cook basmati rice?")
    stop_s = time.perf_counter() - t0
    check("gate stopped it", r.stopped, r.intent.decision if r.intent else "?")
    check("  no Answer object was produced", r.answer is None)
    check("  a message was returned", bool(r.text.strip()), f"{len(r.text)} chars")
    check("  it was cheap — nothing downstream ran", stop_s < 15,
          f"{stop_s:.1f}s vs a 15-125s pipeline")

    # ---- 4 + 5: pass-through, and TRANSPARENT ------------------------
    print("\n" + "=" * W)
    print("4-5. PASSED THROUGH — reaches the pipeline, and changes nothing")
    print("=" * W)
    q = "What does SR 11-7 say about the use of vendor models?"
    r = respond(q)
    check("gate passed it", not r.stopped,
          r.intent.decision if r.intent else "gate off")
    if r.stopped:
        print("  cannot check transparency — the gate stopped a question it "
              "should have passed. Read the reason above before reading on.")
    else:
        a = r.answer
        check("  an answer was produced", not a.refused, a.refusal_reason or "")
        check("  claims were produced", bool(a.claims), f"{len(a.claims)} claim(s)")
        check("  every source label resolves", not a.invalid,
              f"{len(a.invalid)} invalid")
        check("  markers and SOURCES are present",
              bool(a.sources) and "[1]" in a.text, f"{len(a.sources)} source(s)")

        # THE TRANSPARENCY CHECK. Same question, no gate, same answer.
        # Temperature 0 and seed 0 make this a real equality test rather than a
        # similarity one - if it ever differs, either the gate is touching the
        # question or generation is not reproducible, and BOTH are findings.
        from regrag.generation.answer import answer as bare
        direct = bare(q)
        check("  the gate changed the answer in NO way",
              direct.text == a.text and direct.sources == a.sources,
              "identical" if direct.text == a.text else "DIFFERENT — investigate")

    print("\n" + "=" * W)
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED:")
        for f in fails:
            print(f"  - {f}")
        print("\n  A VOCABULARY failure is the quiet one: the rule keeps working")
        print("  and silently asks for clarification it does not need. Fix the")
        print("  prompt, then re-run BOTH question sets — the 53/60 and 55/60")
        print("  no longer describe the code once the prompt changes.")
    else:
        print("ALL CHECKS PASSED — the gate is wired correctly.")
        print("  Wiring only. Accuracy lives in scripts/experiment_prompts.py.")
    print("=" * W)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
