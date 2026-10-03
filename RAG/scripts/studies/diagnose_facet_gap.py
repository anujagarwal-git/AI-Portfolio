"""Where do the ~305ms per facet go? The three obvious steps sum to 40ms.

Measures _run_facet as a WHOLE, then each piece in the same conditions the real
code uses — including constructing a fresh VectorStore per call, which the
previous test did not do because it reused one warm instance.

    uv run python scripts/diagnose_facet_gap.py
"""
from __future__ import annotations

import time

from regrag import config
from regrag.index.embedder import embed_query
from regrag.index.vector_store import VectorStore
from regrag.retrieval.planner import plan
from regrag.retrieval.search import _corpus, _run_facet, _matches, tokenize

W = 78


def clock(fn, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts) * 1000, sum(ts) / len(ts) * 1000


def main() -> int:
    payloads, tokens, bm25 = _corpus("parentdoc")
    p = plan("what is model validation?")
    facet = p.facets[0]
    q = "what is model validation?"
    vec = embed_query(q)

    warm = VectorStore(strategy="parentdoc")
    warm.search(vec, limit=1, where=facet.where)

    print(f"facet {facet.label} | corpus {len(payloads):,} chunks\n")
    print(f"  {'step':<44}{'best':>10}{'avg':>10}")
    print("  " + "-" * (W - 2))

    rows = [
        ("WHOLE _run_facet (what the code does)",
         lambda: _run_facet(facet, vec, payloads, tokens, bm25, warm)),
        ("VectorStore(...) construction only",
         lambda: VectorStore(strategy="parentdoc")),
        ("fresh VectorStore + .search()",
         lambda: VectorStore(strategy="parentdoc").search(vec, limit=config.DENSE_TOP_K, where=facet.where)),
        ("warm store .search() only",
         lambda: warm.search(vec, limit=config.DENSE_TOP_K, where=facet.where)),
        ("filter pass over all payloads",
         lambda: [i for i, pl in enumerate(payloads) if _matches(pl, facet.where)]),
        ("bm25.get_scores()",
         lambda: bm25.get_scores(tokenize(q))),
    ]
    measured = {}
    for label, fn in rows:
        lo, avg = clock(fn)
        measured[label] = avg
        print(f"  {label:<44}{lo:>9.0f}m{avg:>9.0f}m")

    # the sort, timed on a real eligible list
    elig = [i for i, pl in enumerate(payloads) if _matches(pl, facet.where)]
    sc = bm25.get_scores(tokenize(q))
    lo, avg = clock(lambda: sorted(elig, key=lambda i: -sc[i]))
    measured["sort of eligible by score"] = avg
    print(f"  {'sort of eligible by score':<44}{lo:>9.0f}m{avg:>9.0f}m   ({len(elig):,} items)")

    whole = measured["WHOLE _run_facet (what the code does)"]
    parts = (measured["fresh VectorStore + .search()"]
             + measured["filter pass over all payloads"]
             + measured["bm25.get_scores()"]
             + measured["sort of eligible by score"])
    print("\n  " + "-" * (W - 2))
    print(f"  {'whole':<44}{whole:>19.0f}m")
    print(f"  {'sum of the parts':<44}{parts:>19.0f}m")
    print(f"  {'unexplained':<44}{whole - parts:>19.0f}m")

    con = measured["VectorStore(...) construction only"]
    warm_s = measured["warm store .search() only"]
    fresh_s = measured["fresh VectorStore + .search()"]
    print(f"\n  client cost = fresh search - warm search = {fresh_s - warm_s:.0f}ms per facet")
    print(f"  construction alone = {con:.0f}ms")
    if fresh_s - warm_s > 50:
        print(f"  -> building a VectorStore per facet is expensive. Passing ONE store in")
        print(f"     would save about {(fresh_s - warm_s) * len(p.facets):.0f}ms on a "
              f"{len(p.facets)}-facet question.")
    else:
        print("  -> the per-facet client is NOT the cost. The gap is elsewhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
