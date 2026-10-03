"""WHY DID THE TRANSPARENCY CHECK FAIL? Two candidates, separated in one run.

THE OBSERVATION (2026-09-02, scripts/smoke_gate.py)
    `respond(q).answer.text` != `answer(q).text` for the same question.

TWO CANDIDATE CAUSES, AND THEY ARE NOT THE SAME SIZE
    A. THE GATE IS NOT TRANSPARENT. Something on the respond() path changes
       what reaches the model. Bad, but local and fixable.
    B. GENERATION IS NOT REPRODUCIBLE. Two identical calls to answer() give two
       different answers. That is far worse: temperature 0 and seed 0 are the
       reason every recorded number in this project is comparable to every
       other one, and Stage 9 cannot exist without it. `stage7_check.py`
       asserts determinism in its preflight — but on a BARE prompt with no
       retrieval, which is a weaker test than a full answer.

    NOTE WHICH WAY THE EVIDENCE ALREADY LEANS. `respond()` passes the question
    STRING to answer() unchanged — same object, no edit, no added filter. For A
    to be true, something would have to change through a path that does not
    touch the question. So B is the more likely cause. That is a prior, not a
    finding, and this script is here to replace it with a measurement.

THE DESIGN — three arms, each differing from the last by ONE thing
    1 BASELINE      answer(q) three times, nothing in between.
                    Differences here prove B and clear the gate entirely.
    2 GATE CALL     classify(junk) then answer(q).
                    A difference ONLY here means the gate's model call disturbs
                    the next generation - Ollama state, not the question.
    3 FULL PATH     respond(q).
                    The failing case, reproduced.

    Arm 1 is the one that matters. If it already varies, arms 2 and 3 tell you
    nothing new and the gate is innocent.

READ THE DIFF, NOT JUST THE VERDICT
    It prints the first point where two answers part company. "One extra
    sentence" and "a different citation" are different problems: the first is
    sampling noise, the second means retrieval or label resolution moved.

    uv run python scripts/diagnose_transparency.py
"""
from __future__ import annotations

import sys

from regrag import config
from regrag.gate import classify, respond
from regrag.generation import llm
from regrag.generation.answer import answer

Q = "When is a financial asset credit-impaired?"
W = 88


def first_difference(a: str, b: str) -> str:
    if a == b:
        return "identical"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return (f"diverge at char {i}\n"
                    f"      A ...{a[max(0, i-40):i+60]!r}\n"
                    f"      B ...{b[max(0, i-40):i+60]!r}")
    return f"one is a prefix of the other ({len(a)} vs {len(b)} chars)"


def main() -> int:
    problem = llm.why_unavailable()
    if problem:
        sys.exit(problem)

    print("=" * W)
    print(f"TRANSPARENCY DIAGNOSIS   {config.GEN_MODEL}  "
          f"temp {config.TEMPERATURE}  seed {config.GEN_SEED}")
    print(f"  {Q!r}")
    print("=" * W)

    # --- ARM 1: is answer() reproducible AT ALL? ----------------------
    print("\n1. BASELINE — answer() three times, nothing in between")
    print("-" * W)
    runs = []
    for i in range(3):
        a = answer(Q)
        runs.append(a)
        print(f"   run {i+1}   {len(a.text):>5} chars  {len(a.claims)} claim(s)  "
              f"{a.seconds:>5.0f}s  sources={len(a.sources)}")

    same12 = runs[0].text == runs[1].text
    same13 = runs[0].text == runs[2].text
    reproducible = same12 and same13
    print(f"\n   run1 == run2 : {same12}")
    print(f"   run1 == run3 : {same13}")
    if not reproducible:
        print(f"\n   {first_difference(runs[0].text, runs[1].text)}")
        print("\n   *** GENERATION IS NOT REPRODUCIBLE. THE GATE IS INNOCENT. ***")
        print("   This is the bigger finding. Every comparison in this project")
        print("   assumes two identical calls give the same answer — the four")
        print("   prompt variants, the six intent arms, the refusal experiment.")
        print("   None of those are wrong yet, but the assumption under them")
        print("   needs re-checking before Stage 9 rests on it.")
        print("\n   ALSO CHECK, because they are cheap and would explain it:")
        print("     * do the RETRIEVALS match? printed below")
        print("     * is `seed` reaching Ollama? llm.chat sends it in options")
        print("     * did `think` stay off? a reasoning block is not deterministic")
        r1, r2 = runs[0].retrieval, runs[1].retrieval
        if r1 and r2:
            ids1 = [p.parent_id for p in r1.passages]
            ids2 = [p.parent_id for p in r2.passages]
            print(f"\n   retrieval identical: {ids1 == ids2}"
                  f"   ({len(ids1)} vs {len(ids2)} passages)")
            if ids1 != ids2:
                print("   RETRIEVAL ITSELF VARIES — look there first, not at "
                      "the model.")
        return 1

    print("\n   answer() IS reproducible. The gate is not cleared — continue.")
    base = runs[0]

    # --- ARM 2: does a gate CALL disturb the next generation? ---------
    print("\n2. GATE CALL FIRST — classify(junk), then answer()")
    print("-" * W)
    it = classify("How do I cook basmati rice?")
    print(f"   classify -> {it.decision} in {it.seconds:.1f}s")
    after = answer(Q)
    same = after.text == base.text
    print(f"   answer after a gate call == baseline : {same}")
    if not same:
        print(f"\n   {first_difference(base.text, after.text)}")
        print("\n   *** THE GATE'S MODEL CALL DISTURBS THE NEXT GENERATION. ***")
        print("   The question is untouched, so this is Ollama state — most")
        print("   likely the differing num_predict/context between the gate")
        print("   call and the answer call. NOT PROVEN. To isolate: repeat")
        print("   arm 2 with GATE_MAX_TOKENS set to the answer call's value.")
        return 1

    # --- ARM 3: the failing case ---------------------------------------
    print("\n3. FULL PATH — respond()")
    print("-" * W)
    r = respond(Q)
    if r.stopped:
        print(f"   gate STOPPED it: {r.intent.decision} — {r.intent.reason}")
        print("   That is the real failure: the gate is blocking an answerable")
        print("   question, and the smoke test reported it as a transparency")
        print("   failure because it could not get an answer to compare.")
        return 1
    same = r.answer.text == base.text
    print(f"   respond().answer == baseline : {same}")
    if not same:
        print(f"\n   {first_difference(base.text, r.answer.text)}")
        print("\n   Arms 1 and 2 were clean, so the difference is on the")
        print("   respond() path itself. Read respond() — it is 15 lines.")
        return 1

    print("\n" + "=" * W)
    print("ALL THREE ARMS MATCH — the failure did not reproduce.")
    print("  An intermittent difference is still a real finding: it means the")
    print("  pipeline is only USUALLY deterministic, which is not the same as")
    print("  deterministic. Run this again before dismissing it.")
    print("=" * W)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
