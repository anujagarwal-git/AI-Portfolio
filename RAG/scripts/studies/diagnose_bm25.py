"""Test two hypotheses about search.py BEFORE changing anything.

H1  The lexical arm cannot find "CRE36.122" because the chunk TEXT says
    "36.122" and the "CRE" lives only in the `locator` metadata, which is not
    indexed. Prediction: the query token `cre36.122` appears in ZERO chunks,
    and adding the locator to the indexed text moves the right chunk up.

H2  Per-facet search time is dominated by bm25.get_scores(), which rescores all
    9,935 chunks with the SAME query once per facet. Prediction: get_scores
    accounts for most of the ~345ms, and the Qdrant call is small.

Either can fail. The script prints what it finds, not what was expected.

    uv run python scripts/diagnose_bm25.py
"""
from __future__ import annotations

import time
from collections import Counter

from rank_bm25 import BM25Okapi

from regrag import config
from regrag.index.embedder import embed_query
from regrag.index.vector_store import VectorStore
from regrag.retrieval.planner import plan
from regrag.retrieval.search import _corpus, tokenize

TARGET = "CRE36.122"
QUESTION = "what does CRE36.122 require?"
W = 92


def rank_of(scores, chunk_id, payloads, top=5):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    pos = next((r for r, i in enumerate(order) if payloads[i]["chunk_id"] == chunk_id), None)
    head = [(payloads[i].get("locator") or payloads[i].get("parent_heading") or "?",
             round(scores[i], 3)) for i in order[:top]]
    return pos, head


def main() -> int:
    payloads, tokens, bm25 = _corpus("parentdoc")
    print(f"corpus: {len(payloads):,} chunks\n")

    # ================= H1 =================
    print("=" * W); print("H1  does the locator live outside the indexed text?"); print("=" * W)

    exact = [p for p in payloads if (p.get("locator") or "") == TARGET]
    if not exact:
        near = [p for p in payloads if TARGET.lower() in (p.get("locator") or "").lower()]
        print(f"  NO chunk carries locator == {TARGET!r}. H1 cannot be tested as stated.")
        print(f"  nearest locators containing it: {[p.get('locator') for p in near][:5]}")
        looks = [p for p in payloads if "36.122" in (p.get("text") or "")][:3]
        for p in looks:
            print(f"    text starts: {(p.get('text') or '')[:90]!r}  locator={p.get('locator')!r}")
        return 1

    tgt = exact[0]
    print(f"  chunk_id  {tgt['chunk_id']}")
    print(f"  locator   {tgt.get('locator')!r}   ({tgt.get('locator_kind')})")
    print(f"  text      {(tgt.get('text') or '')[:110]!r}")

    qtok = set(tokenize(QUESTION))
    ttok = set(tokenize(tgt.get("text") or ""))
    shared = sorted(qtok & ttok)
    print(f"\n  query tokens   {sorted(qtok)}")
    print(f"  shared with the target chunk's TEXT: {shared}")

    df = Counter()
    for t in tokens:
        for w in set(t):
            df[w] += 1
    for w in ("cre36.122", "cre36", "36.122", "122"):
        print(f"    df[{w!r}] = {df.get(w, 0):,} chunks contain it")

    # simulate the fix
    print(f"\n  --- rank of {TARGET} for {QUESTION!r} ---")
    q = tokenize(QUESTION)
    before, head_b = rank_of(bm25.get_scores(q), tgt["chunk_id"], payloads)
    print(f"  BEFORE (text only)          rank {before}   top5 {head_b}")

    t0 = time.perf_counter()
    tokens2 = [tokenize(f"{p.get('locator') or ''} {p.get('text') or ''}") for p in payloads]
    bm25b = BM25Okapi(tokens2)
    build = time.perf_counter() - t0
    after, head_a = rank_of(bm25b.get_scores(q), tgt["chunk_id"], payloads)
    print(f"  AFTER  (locator + text)     rank {after}   top5 {head_a}")
    print(f"  (rebuild cost {build:.1f}s, one-off per process)")

    if before is None or after is None:
        print("\n  VERDICT: inconclusive — the chunk did not rank at all in one of the runs.")
    elif after < before:
        print(f"\n  VERDICT: H1 SUPPORTED. rank {before} -> {after}, "
              f"and it now enters the top {config.LEXICAL_TOP_K} candidate list "
              f"{'YES' if after < config.LEXICAL_TOP_K else 'NO'}.")
    else:
        print(f"\n  VERDICT: H1 NOT SUPPORTED. rank did not improve ({before} -> {after}). "
              f"My explanation was wrong; the cause is elsewhere.")

    # ================= H2 =================
    print("\n" + "=" * W); print("H2  is bm25.get_scores() what makes a facet cost ~345ms?"); print("=" * W)

    p = plan("what is model validation?")
    facet = p.facets[0]
    store = VectorStore(strategy="parentdoc")
    vec = embed_query("what is model validation?")
    store.search(vec, limit=config.DENSE_TOP_K, where=facet.where)   # warm

    def clock(fn, n=3):
        best = []
        for _ in range(n):
            t = time.perf_counter(); fn(); best.append(time.perf_counter() - t)
        return min(best) * 1000, sum(best) / len(best) * 1000

    d_min, d_avg = clock(lambda: store.search(vec, limit=config.DENSE_TOP_K, where=facet.where))
    b_min, b_avg = clock(lambda: bm25.get_scores(tokenize(facet.query)))
    f_min, f_avg = clock(lambda: [i for i, pl in enumerate(payloads)
                                  if all(pl.get(k) in (v if isinstance(v, list) else [v])
                                         for k, v in facet.where.items())])

    print(f"  {'step':<34}{'best':>10}{'avg of 3':>12}")
    print(f"  {'dense: Qdrant search':<34}{d_min:>9.0f}ms{d_avg:>11.0f}ms")
    print(f"  {'lexical: bm25.get_scores()':<34}{b_min:>9.0f}ms{b_avg:>11.0f}ms")
    print(f"  {'lexical: python filter pass':<34}{f_min:>9.0f}ms{f_avg:>11.0f}ms")
    total = d_avg + b_avg + f_avg
    print(f"  {'sum':<34}{'':>10}{total:>11.0f}ms   (measured facet cost was ~345ms)")

    if b_avg > d_avg + f_avg:
        saved = b_avg * (len(p.facets) - 1)
        print(f"\n  VERDICT: H2 SUPPORTED. get_scores is the largest part. Computing it ONCE "
              f"instead of per facet would save ~{saved:.0f}ms on a {len(p.facets)}-facet question.")
    else:
        print(f"\n  VERDICT: H2 NOT SUPPORTED. get_scores is not the dominant cost; "
              f"the saving would be small. Look at the other two rows instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
