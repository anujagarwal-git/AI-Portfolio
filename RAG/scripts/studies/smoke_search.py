"""First run of search.py against the live index. Reads, does not decide.

    uv run python scripts/smoke_search.py

Needs Qdrant up (docker compose up -d). The FIRST query builds the BM25 index
over all 9,935 chunks — a few seconds, once per process. Read the timings from
the second query onward.
"""
from __future__ import annotations

from regrag import config
from regrag.retrieval.search import retrieve

CASES = [
    ("the bug fix",        "what did SR 11-7 say about spreadsheets?"),
    ("3-facet fan-out",    "what is model validation?"),
    ("version fan-out",    "did the treatment of spreadsheets change under the new guidance?"),
    ("BM25 should win",    "what does CRE36.122 require?"),
    ("no matching doc",    "what are the UK rules on expected credit loss?"),
    ("refused",            "?????"),
]


def main() -> int:
    print(f"PARENT_MAX_CHARS {config.PARENT_MAX_CHARS:,} | "
          f"CONTEXT_MAX_CHARS {config.CONTEXT_MAX_CHARS:,} | "
          f"quota {config.FACET_QUOTA_DEFAULT} | dense {config.DENSE_TOP_K} | "
          f"bm25 {config.LEXICAL_TOP_K} | rrf_k {config.RRF_K}")

    rows = []
    for label, q in CASES:
        print(f"\n{'=' * 96}\n{label.upper()}\n{'=' * 96}")
        r = retrieve(q)
        print(r)
        rows.append((label, r))

    print(f"\n{'=' * 96}\nSUMMARY  (ignore row 1 — it built the BM25 index)\n{'=' * 96}")
    print(f"  {'case':<18}{'mode':<20}{'psg':>4}{'chars':>8}{'embed':>8}{'search':>9}{'total':>8}")
    for label, r in rows:
        t = r.timings
        print(f"  {label:<18}{(r.plan.mode if not r.refused else 'REFUSED'):<20}"
              f"{len(r.passages):>4}{r.total_chars:>8,}"
              f"{t.get('embed', 0) * 1000:>7.0f}m{t.get('search', 0) * 1000:>8.0f}m"
              f"{t.get('total', 0) * 1000:>7.0f}m")

    print("\n  WHAT TO CHECK")
    print("  1. does SR 11-7 actually appear for case 1 — the bug this rule exists for")
    print("  2. CRE36.122: look for dense#-/bm25#1 — the lexical arm earning its place")
    print("  3. the UK/ECL facet should be SKIPPED, not answered weakly")
    print("  4. 'search' from case 2 onward decides whether Stage 6 is affordable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
