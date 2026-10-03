"""Print the TEXT retrieved for each true-positive candidate, so a human can read it.

WHY THIS EXISTS
    The refusal experiment measures how often each arm wrongly refuses a question
    the corpus CAN answer. That number is meaningless if the "can answer" list is
    a guess. scripts/calibrate_gate.py says so about its own REAL list:

        "The REAL questions below are my guesses at answerable questions, not
         verified ones. A low score on one of them might mean the corpus is
         silent, not that retrieval failed."

    This is the fix. It prints what came back so you can read it and decide.

THE ONE RULE
    NEVER verify by looking at the citation. The citation is metadata; it is
    composed from registry fields and will look correct whether or not the text
    answers anything. READ THE TEXT. If the text does not answer the question,
    the candidate is not a true positive, however good the citation looks.

    Three outcomes, and the middle one is common:
      ANSWERS      the text states the answer          -> verified=True
      MENTIONS     the text is on-topic but does not   -> NOT a true positive
                   state the answer                       (this is the trap)
      UNRELATED    retrieval missed                    -> NOT a true positive

    A MENTIONS is not a failure of the experiment. It is a finding about the
    corpus, and it belongs in the notes.

    uv run python scripts/verify_positives.py          # all candidates
    uv run python scripts/verify_positives.py 3        # just candidate 3
"""
from __future__ import annotations

import sys
import textwrap

from regrag.evaluation.refusal_set import TRUE_POSITIVE_CANDIDATES
from regrag.retrieval.search import retrieve

CHARS = 900          # how much of each passage to show
PASSAGES = 2         # how many passages to show per question


def show(i: int, cand) -> None:
    print(f"\n{'=' * 100}")
    print(f"[{i}] {cand.text}")
    print(f"     expected in: {cand.expect_in}     verified: {cand.verified}")
    print("=" * 100)

    # rerank=True so the cross-encoder scores are computed and visible here too.
    # They are NOT used to judge the candidate -- a human reading the text does
    # that. They are printed so the score distribution can be eyeballed early.
    r = retrieve(cand.text, rerank=True)

    if r.refused:
        print(f"  !! RETRIEVAL REFUSED: {r.refused}")
        return
    if not r.passages:
        print("  !! NO PASSAGES")
        return

    best = max((h.rerank for p in r.passages for h in p.children
                if h.rerank is not None), default=None)
    print(f"  plan: {r.plan.mode}, {len(r.plan.facets)} facet(s) -> "
          f"{len(r.passages)} passage(s), {r.total_chars:,} chars"
          + (f"   best cross-encoder score {best:+.2f}" if best is not None else ""))

    for p in r.passages[:PASSAGES]:
        kid = max(p.children, key=lambda h: (h.rerank if h.rerank is not None else h.rrf))
        sc = f"  ce{kid.rerank:+.2f}" if kid.rerank is not None else ""
        print(f"\n  --- [{p.short_name}] {p.heading[:70]!r}{sc}")
        body = " ".join((p.text or "").split())
        print(textwrap.fill(body[:CHARS], width=96,
                            initial_indent="      ", subsequent_indent="      "))
        if len(body) > CHARS:
            print(f"      ... (+{len(body) - CHARS:,} more chars)")

    print("\n  READ THE TEXT ABOVE.  ANSWERS / MENTIONS / UNRELATED ?")


def main() -> int:
    picks = [int(a) for a in sys.argv[1:]] or list(range(len(TRUE_POSITIVE_CANDIDATES)))
    for i in picks:
        show(i, TRUE_POSITIVE_CANDIDATES[i])
    print(f"\n{'=' * 100}")
    print("Mark each ANSWERS candidate verified=True in evaluation/refusal_set.py.")
    print("Anything else: replace the question or drop it. Do not keep a maybe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
