"""Did the facet margin break the three-way fan-out? Same query, margin on and off.

WHY THIS MATTERS MORE THAN THE OTHER FILTERS
    "what is model validation?" is the AXIS 1 case this project was built to
    handle: subjects OVERLAP, so the answer legitimately lives in SR 11-7 (US
    guidance), SS1/23 (UK) and Basel CRE36 section 8 (capital eligibility) at
    once. integration_check asserts at least three document families come back.
    That is a designed capability, not a tuning knob.

    The facet margin drops a slice whose best chunk is far below the winner. On
    a question where one family simply says MORE about the topic, that is
    exactly what happens - and coverage is lost for a reason that looks like
    noise but is really density. Recorded 2026-08-23: SR 11-7 has ~101
    "validat*" in 21pp against Basel CRE's ~39 in 323pp.

ONE VARIABLE. Same question, same index, margin on then off.

    uv run python scripts/diagnose_facet_margin.py
"""
from __future__ import annotations

from regrag import config
from regrag.retrieval.search import retrieve

QUERIES = ["what is model validation?",
           "how should a bank validate its models?",
           "what are the core elements of model validation?"]


def show(tag: str, q: str, **kw) -> set[str]:
    r = retrieve(q, **kw)
    fams = {p.short_name for p in r.passages}
    print(f"  {tag:<12}{len(r.passages)} psg  {r.total_chars:>7,}c  "
          f"{len(fams)} famil(y/ies): {', '.join(sorted(fams))}")
    for n in r.notes:
        if n.startswith(("facet margin", "relevance floor")):
            print(f"               ! {n}")
    return fams


def main() -> int:
    for q in QUERIES:
        print("=" * 96)
        print(q)
        print("=" * 96)
        # FOUR COMBINATIONS, ONE VARIABLE AT A TIME. Toggling both at once is
        # the before/after-across-two-edits mistake this project keeps
        # relearning; it proves nothing about which filter did what.
        keep_m, keep_f = config.FACET_DROP_MARGIN, config.RERANK_DROP_BELOW
        both = show("both ON", q)

        config.FACET_DROP_MARGIN = None
        no_margin = show("margin OFF", q)
        config.FACET_DROP_MARGIN = keep_m

        config.RERANK_DROP_BELOW = None
        no_floor = show("floor OFF", q)
        config.RERANK_DROP_BELOW = keep_f

        config.FACET_DROP_MARGIN = config.RERANK_DROP_BELOW = None
        neither = show("both OFF", q)
        config.FACET_DROP_MARGIN, config.RERANK_DROP_BELOW = keep_m, keep_f

        # No reranking at all - the Stage 5 behaviour the check was written for.
        pre = show("pre-Stage-7", q, rerank=False)

        print(f"               -> families: both {len(both)}, no-margin "
              f"{len(no_margin)}, no-floor {len(no_floor)}, neither "
              f"{len(neither)}, rerank-off {len(pre)}   (check wants >= 3)")
        if len(pre) >= 3 > len(both):
            culprit = ("the FLOOR" if len(no_floor) >= 3
                       else "the MARGIN" if len(no_margin) >= 3
                       else "RERANKING ITSELF (ordering), not either filter")
            print(f"               -> the check passed before Stage 7 and fails now: "
                  f"{culprit}")
        print()
    print("If the margin costs a family on an Axis-1 question, the margin is wrong")
    print("for inferred fan-outs on overlapping SUBJECTS - not the check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
