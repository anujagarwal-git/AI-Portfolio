"""A question did not return the chunk it obviously should. WHERE did it fall out?

Four things can go wrong, and they need four different fixes. This tells them apart
instead of guessing:

  1. THE CONTENT IS NOT THERE       -> a parsing or chunking problem
  2. NEITHER ARM RANKS IT           -> a retrieval problem (wording, spelling, IDF)
  3. AN ARM RANKS IT, THE FACET DOES NOT SEE IT  -> the facet filter or the quota
  4. IT REACHES THE CANDIDATES AND LOSES         -> a ranking problem (RRF or rerank)

Also checks SPELLING. The tokenizer does no stemming and no British/American
normalisation, so "specialized" and "specialised" are different tokens to BM25.
Dense retrieval may bridge that gap; the lexical arm cannot.

    uv run python scripts/diagnose_missing.py
    uv run python scripts/diagnose_missing.py --needle "counterparty credit risk"
"""
from __future__ import annotations

import sys

from regrag import config
from regrag.index.embedder import embed_query
from regrag.index.vector_store import VectorStore
from regrag.retrieval.planner import plan
from regrag.retrieval.search import _corpus, _matches, tokenize

NEEDLE = "specialis"          # matched case-insensitively; also tries the -z- form
QUESTIONS = [
    "which asset classes fall under specialised lending?",
    "which asset classes fall under specialized lending?",     # American spelling
    "what is the supervisory slotting approach for specialised lending?",
]
DEEP = 200
W = 100


def variants(n: str) -> tuple[str, ...]:
    return tuple({n.lower(), n.lower().replace("s", "z", 1) if "specialis" in n.lower() else n.lower(),
                  n.lower().replace("specialis", "specializ"), n.lower().replace("specializ", "specialis")})


def main() -> int:
    needle = sys.argv[sys.argv.index("--needle") + 1] if "--needle" in sys.argv else NEEDLE
    forms = variants(needle)
    payloads, tokens, bm25 = _corpus("parentdoc")
    store = VectorStore(strategy="parentdoc")
    hit = lambda t: any(f in (t or "").lower() for f in forms)

    # ---- 1. does the content exist at all? -----------------------------
    print("=" * W); print(f"1. IS IT IN THE INDEX?   forms tried: {forms}"); print("=" * W)
    in_head = [p for p in payloads if hit(p.get("parent_heading"))]
    in_text = [p for p in payloads if hit(p.get("text"))]
    idx = {p["chunk_id"]: i for i, p in enumerate(payloads)}
    targets = {p["chunk_id"]: p for p in in_head + in_text}
    print(f"  chunks whose PARENT HEADING matches : {len(in_head):,}")
    print(f"  chunks whose TEXT matches           : {len(in_text):,}")
    print(f"  distinct target chunks              : {len(targets):,}")
    heads = sorted({(p.get('short_name'), p.get('parent_heading')) for p in in_head})
    for sn, h in heads[:10]:
        n = sum(1 for p in in_head if p.get("parent_heading") == h)
        print(f"    [{str(sn)[:14]:<14}] {str(h)[:62]!r}  ({n} children)")
    if not targets:
        print("\n  -> CAUSE 1: the content is not in the index. Parsing/chunking, not retrieval.")
        return 0
    print(f"\n  df of the exact token: ", end="")
    for f in forms:
        print(f"{f!r}={sum(1 for t in tokens if f in t):,}  ", end="")
    print()

    # ---- 2-4. per question ---------------------------------------------
    for q in QUESTIONS:
        print("\n" + "=" * W); print(q); print("=" * W)
        p = plan(q)
        vec = embed_query(q)

        sc = bm25.get_scores(tokenize(q))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        bpos = {payloads[i]["chunk_id"]: r for r, i in enumerate(order[:DEEP])}
        dpos = {h.payload["chunk_id"]: r
                for r, h in enumerate(store.search(vec, limit=DEEP))}

        best_b = min((bpos[c] for c in targets if c in bpos), default=None)
        best_d = min((dpos[c] for c in targets if c in dpos), default=None)
        print(f"  plan: [{p.mode}] {len(p.facets)} facet(s)")
        print(f"  best target rank, UNFILTERED, top {DEEP}:  "
              f"bm25 {best_b if best_b is not None else 'not in top ' + str(DEEP)}   "
              f"dense {best_d if best_d is not None else 'not in top ' + str(DEEP)}")

        reached = False
        for f in p.facets:
            elig = [i for i, pl in enumerate(payloads) if _matches(pl, f.where)]
            elig.sort(key=lambda i: -sc[i])
            lex20 = [payloads[i]["chunk_id"] for i in elig[:config.LEXICAL_TOP_K] if sc[i] > 0]
            den20 = [h.payload["chunk_id"]
                     for h in store.search(vec, limit=config.DENSE_TOP_K, where=f.where)]
            got = [c for c in set(lex20) | set(den20) if c in targets]
            n_elig = sum(1 for i in elig if payloads[i]["chunk_id"] in targets)
            print(f"    facet {f.label:<22} eligible targets {n_elig:>4} | "
                  f"in the 20 candidates: {len(got)}")
            reached = reached or bool(got)

        print()
        if best_b is None and best_d is None:
            print("  -> CAUSE 2: neither arm ranks it in the top "
                  f"{DEEP} of the WHOLE corpus. A retrieval problem — wording, spelling or IDF.")
        elif not reached:
            print("  -> CAUSE 3: an arm ranks it corpus-wide, but no FACET sees it. The facet")
            print("     filter excludes it, or it loses inside the facet before the top 20.")
        else:
            print("  -> CAUSE 4: it REACHES the candidates and loses the ranking. RRF or rerank.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
