"""Open the box: what each arm found, what RRF did to it, and what reaches the model.

Prints, for one question:
  1. the DENSE top 20 per facet
  2. the BM25 top 20 per facet
  3. the RRF table — both ranks side by side, the fused score, and the quota line
  4. the passages actually delivered, with text, so relevance can be JUDGED not assumed

`*` marks a chunk whose text or parent heading contains the needle — the thing we
believe the question is about. It is a probe, not a label: it says "this is on the
topic", not "this is the answer".

    uv run python scripts/show_fusion.py
    uv run python scripts/show_fusion.py --q "what is model validation?" --needle validation
"""
from __future__ import annotations

import sys

from regrag import config
from regrag.index.embedder import embed_query
from regrag.index.vector_store import VectorStore
from regrag.retrieval.planner import plan
from regrag.retrieval.search import _corpus, _matches, _rrf, retrieve, tokenize

Q = "which asset classes fall under specialised lending?"
NEEDLE = "specialis"
W = 118


def arg(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> int:
    q, needle = arg("--q", Q), arg("--needle", NEEDLE).lower()
    payloads, tokens, bm25 = _corpus("parentdoc")
    store = VectorStore(strategy="parentdoc")
    vec = embed_query(q)
    p = plan(q)

    is_t = lambda pl: needle in (pl.get("text") or "").lower() \
        or needle in (pl.get("parent_heading") or "").lower()
    cite = lambda pl: (f"{pl.get('locator') or pl.get('parent_heading') or '?'}")[:46]
    n_targets = sum(1 for pl in payloads if is_t(pl))

    print(f"{q!r}\n  plan [{p.mode}] {len(p.facets)} facet(s) | "
          f"needle {needle!r} matches {n_targets:,} of {len(payloads):,} chunks\n")

    sc = bm25.get_scores(tokenize(q))

    for f in p.facets:
        print("=" * W); print(f"FACET {f.label}   filter {f.where}"); print("=" * W)

        dense_pts = store.search(vec, limit=config.DENSE_TOP_K, where=f.where)
        dense_ids = [x.payload["chunk_id"] for x in dense_pts]
        by_id = {x.payload["chunk_id"]: x.payload for x in dense_pts}
        dscore = {x.payload["chunk_id"]: x.score for x in dense_pts}

        elig = [i for i, pl in enumerate(payloads) if _matches(pl, f.where)]
        elig.sort(key=lambda i: -sc[i])
        lex_ids, bscore = [], {}
        for i in elig[:config.LEXICAL_TOP_K]:
            if sc[i] <= 0:
                break
            cid = payloads[i]["chunk_id"]
            lex_ids.append(cid); bscore[cid] = sc[i]
            by_id.setdefault(cid, payloads[i])

        n_el = sum(1 for i in elig if is_t(payloads[i]))
        print(f"  {n_el} of {len(elig):,} eligible chunks are on-topic\n")

        print(f"  {'DENSE top 20':<58}{'BM25 top 20'}")
        for i in range(max(len(dense_ids), len(lex_ids))):
            L = R = ""
            if i < len(dense_ids):
                c = dense_ids[i]
                L = f"{i+1:>2}{'*' if is_t(by_id[c]) else ' '} {cite(by_id[c]):<44}{dscore[c]:.3f}"
            if i < len(lex_ids):
                c = lex_ids[i]
                R = f"{i+1:>2}{'*' if is_t(by_id[c]) else ' '} {cite(by_id[c]):<44}{bscore[c]:.1f}"
            print(f"  {L:<58}{R}")

        fused = _rrf(dense_ids, lex_ids, config.RRF_K)
        dr = {c: r for r, c in enumerate(dense_ids)}
        lr = {c: r for r, c in enumerate(lex_ids)}
        order = sorted(fused, key=lambda c: -fused[c])

        print(f"\n  RRF  (k={config.RRF_K}) — sum of 1/(k+rank) over the arms that found it")
        print(f"  {'':4}{'dense':>7}{'bm25':>7}{'RRF':>9}   chunk")
        for r, c in enumerate(order):
            d = dr.get(c); l = lr.get(c)
            cut = " <-- quota" if r == f.quota - 1 else ""
            both = "  BOTH" if d is not None and l is not None else ""
            print(f"  {r+1:>2}{'*' if is_t(by_id[c]) else ' '} "
                  f"{(d+1) if d is not None else '-':>6} {(l+1) if l is not None else '-':>6} "
                  f"{fused[c]:>9.5f}   {cite(by_id[c])}{both}{cut}")
            if r >= 24:
                print(f"     ... {len(order)-25} more"); break
        print()

    print("=" * W); print("WHAT ACTUALLY REACHES THE MODEL"); print("=" * W)
    r = retrieve(q, rerank=False)
    print(f"  {len(r.passages)} passage(s), {r.total_chars:,} chars\n")
    for i, pas in enumerate(r.passages, 1):
        flag = "*" if needle in (pas.heading or "").lower() or needle in pas.text.lower() else " "
        print(f"  {i}.{flag}[{pas.short_name}] {pas.heading[:58]!r}  {len(pas.text):,}c")
        for h in pas.children:
            print(f"      child {h.cite[:44]:<46}{h.arms}")
            print(f"        {((h.payload.get('text') or '')[:104])!r}")
    for n in r.notes:
        print(f"  ! {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
