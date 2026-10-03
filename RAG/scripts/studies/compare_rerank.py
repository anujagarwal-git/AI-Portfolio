"""Did the cross-encoder actually change anything? Before and after, side by side.

The Stage 6 question is not "does it run" but "does it demote a bad candidate
that RRF ranked highly". This prints, per question, the top hits WITHOUT
reranking and WITH it, so a demotion is visible rather than assumed.

First run downloads ~90MB from HuggingFace.

    uv run python scripts/compare_rerank.py
"""
from __future__ import annotations

import time

from regrag import config
from regrag.retrieval.search import retrieve

QUESTIONS = [
    "what did SR 11-7 say about spreadsheets?",
    "what is model validation?",
    "what does CRE36.122 require?",
    "how often must a bank review its model inventory?",
    "what are the minimum requirements for the IRB approach?",
]
W = 100


def cites(r):
    return [(h.cite, h.arms) for pas in r.passages for h in pas.children]


def main() -> int:
    print(f"model {config.RERANK_MODEL} | candidates/facet {config.RERANK_CANDIDATES} "
          f"| quota {config.FACET_QUOTA_DEFAULT}\n")
    retrieve("warm up the models", rerank=True)          # pay the one-off loads

    rows = []
    for q in QUESTIONS:
        t = time.perf_counter(); off = retrieve(q, rerank=False); t_off = time.perf_counter() - t
        t = time.perf_counter(); on = retrieve(q, rerank=True);  t_on = time.perf_counter() - t

        print("=" * W); print(q); print("=" * W)
        a, b = cites(off), cites(on)
        print(f"  {'RRF ONLY':<48}{'RERANKED'}")
        for i in range(max(len(a), len(b))):
            left = f"{a[i][0][:38]:<38} {a[i][1][:9]}" if i < len(a) else ""
            right = f"{b[i][0][:38]:<38} {b[i][1]}" if i < len(b) else ""
            mark = " " if (i < len(a) and i < len(b) and a[i][0] == b[i][0]) else ">"
            print(f" {mark}{left:<48}{right}")

        moved = sum(1 for i in range(min(len(a), len(b))) if a[i][0] != b[i][0])
        gone = {c for c, _ in a} - {c for c, _ in b}
        print(f"\n  {moved} position(s) changed | {len(gone)} candidate(s) dropped out entirely")
        if gone:
            print(f"  demoted: {sorted(gone)[:4]}")
        print(f"  rerank cost {on.timings.get('rerank', 0) * 1000:.0f}ms "
              f"| total {t_on * 1000:.0f}ms vs {t_off * 1000:.0f}ms without")
        rows.append((q, moved, len(gone), on.timings.get("rerank", 0) * 1000, t_on * 1000))

    print("\n" + "=" * W); print("SUMMARY"); print("=" * W)
    print(f"  {'question':<52}{'moved':>7}{'dropped':>9}{'rerank':>9}{'total':>8}")
    for q, mv, gn, rr, tt in rows:
        print(f"  {q[:50]:<52}{mv:>7}{gn:>9}{rr:>8.0f}m{tt:>7.0f}m")
    print("\n  DONE WHEN: the reranked list is visibly better on at least 2 of 5.")
    print("  'Better' means a candidate you can see is off-topic got demoted.")
    print("  Read the pairs above and judge — a count of moves is not quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
