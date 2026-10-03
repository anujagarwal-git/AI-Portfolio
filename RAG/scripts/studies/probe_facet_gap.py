"""Inside ONE question, how far ahead is the facet that actually has the material?

An ABSOLUTE score floor died on 2026-08-23: noise peaked at 0.820, a real question
sat at 0.774, the ranges overlap. But that compared scores ACROSS questions.
This asks a narrower question, which may have a different answer:

    within a single query, does the facet that owns the topic stand clearly
    apart from the two that do not?

Every question here fans out to US / UK / GLOBAL, so two of the three facets are
searching a corpus that cannot answer. If the winner is consistently far ahead
when one jurisdiction owns the topic, and the facets bunch together when the
topic is shared or absent, a RELATIVE gate becomes possible.

Reports THREE instruments side by side so they can be compared on one run:
    dense (bi-encoder cosine) · BM25 · cross-encoder

`owner` is MY hand annotation of which jurisdiction should have the material. It
is a reading aid, not data. If a threshold is ever FITTED to these labels, it is
fitted to my judgement rather than to truth — use this to see whether a gap
EXISTS and how wide, not to pick a number to three decimals.

    uv run python scripts/probe_facet_gap.py
"""
from __future__ import annotations

from regrag import config
from regrag.index.embedder import embed_query
from regrag.index.vector_store import VectorStore
from regrag.retrieval import reranker
from regrag.retrieval.planner import plan
from regrag.retrieval.search import _corpus, _matches, _rrf, tokenize

# Chosen to avoid every subject term, so they route to `fallback` — and to give a
# mix: one jurisdiction owns it / several share it / nobody has it.
CASES = [
    ("which asset classes fall under specialised lending?",            "GLOBAL"),
    ("what haircuts apply to collateralised transactions?",            "GLOBAL"),
    ("how are guarantees and credit derivatives recognised?",          "GLOBAL"),
    ("what counts as eligible financial collateral?",                  "GLOBAL"),
    ("what is the definition of default?",                             "GLOBAL"),
    ("what is the treatment of purchased receivables?",                "GLOBAL"),
    ("what must be disclosed about the countercyclical buffer?",       "GLOBAL"),
    ("what are the reporting timelines for the annual submission?",    "US"),
    ("how should a firm document its assumptions?",                    "SHARED"),
    ("who must approve the firm's remuneration policy?",               "NONE"),
]
TOPN = 10
W = 104


def main() -> int:
    payloads, tokens, bm25 = _corpus("parentdoc")
    store = VectorStore(strategy="parentdoc")
    rows = []

    for q, owner in CASES:
        p = plan(q)
        flag = "" if p.mode == "fallback" else f"   <-- NOT fallback ({p.mode}) — swap this one"
        print("=" * W); print(f"{q}\n  owner: {owner}{flag}"); print("=" * W)
        vec = embed_query(q)
        sc = bm25.get_scores(tokenize(q))

        per, flat = {}, []
        for f in p.facets:
            dense = store.search(vec, limit=config.DENSE_TOP_K, where=f.where)
            d_ids = [h.payload["chunk_id"] for h in dense]
            by = {h.payload["chunk_id"]: h.payload for h in dense}
            d_sc = [h.score for h in dense]

            elig = [i for i, pl in enumerate(payloads) if _matches(pl, f.where)]
            elig.sort(key=lambda i: -sc[i])
            l_ids, b_sc = [], []
            for i in elig[:config.LEXICAL_TOP_K]:
                if sc[i] <= 0:
                    break
                l_ids.append(payloads[i]["chunk_id"]); b_sc.append(sc[i])
                by.setdefault(payloads[i]["chunk_id"], payloads[i])

            fused = _rrf(d_ids, l_ids, config.RRF_K)
            top = sorted(fused, key=lambda c: -fused[c])[:TOPN]
            per[f.label] = dict(elig=len(elig),
                                d_best=max(d_sc, default=0.0),
                                d_top5=sum(d_sc[:5]) / max(1, len(d_sc[:5])),
                                b_best=max(b_sc, default=0.0),
                                cands=[by[c] for c in top])
            flat += [(f.label, pl) for pl in per[f.label]["cands"]]

        scores = reranker.score(q, [reranker.pair_text(pl, metadata=True) for _, pl in flat])
        for (lab, _), s in zip(flat, scores):
            per[lab].setdefault("ce", []).append(s)

        print(f"  {'facet':<20}{'eligible':>10}{'dense best':>12}{'dense top5':>12}"
              f"{'bm25 best':>11}{'ce best':>10}")
        for lab, d in per.items():
            ce = max(d.get("ce", [0.0]))
            d["ce_best"] = ce
            print(f"  {lab:<20}{d['elig']:>10,}{d['d_best']:>12.3f}{d['d_top5']:>12.3f}"
                  f"{d['b_best']:>11.1f}{ce:>10.1f}")

        rank = sorted(per.items(), key=lambda kv: -kv[1]["d_best"])
        win, second = rank[0], rank[1]
        ce_rank = sorted(per.items(), key=lambda kv: -kv[1]["ce_best"])
        gap_d = win[1]["d_best"] - second[1]["d_best"]
        gap_c = ce_rank[0][1]["ce_best"] - ce_rank[1][1]["ce_best"]
        print(f"\n  dense winner {win[0]:<18} gap to 2nd {gap_d:+.3f}")
        print(f"  ce    winner {ce_rank[0][0]:<18} gap to 2nd {gap_c:+.1f}   "
              f"(best ce overall {max(d['ce_best'] for d in per.values()):+.1f})\n")
        rows.append((q, owner, p.mode, win[0], gap_d, ce_rank[0][0], gap_c,
                     max(d["ce_best"] for d in per.values()),
                     min(d["ce_best"] for d in per.values())))

    print("=" * W); print("SUMMARY"); print("=" * W)
    print(f"  {'question':<44}{'owner':>8}{'dense win':>11}{'gap':>8}"
          f"{'ce win':>9}{'gap':>7}{'ce hi':>7}{'ce lo':>7}")
    for q, owner, mode, dw, gd, cw, gc, hi, lo in rows:
        print(f"  {q[:42]:<44}{owner:>8}{dw.split('+')[0]:>11}{gd:>8.3f}"
              f"{cw.split('+')[0]:>9}{gc:>7.1f}{hi:>7.1f}{lo:>7.1f}")

    print("\n  HOW TO READ IT")
    print("  - rows marked GLOBAL/US: did the right facet win, and by how much?")
    print("  - row marked SHARED: the facets SHOULD bunch. A small gap here is correct.")
    print("  - row marked NONE: all three should be low. If `ce hi` is high, the")
    print("    cross-encoder is confident about something that does not exist.")
    print("  - a usable gate needs the gaps on GLOBAL/US rows to sit clearly ABOVE the")
    print("    gaps on the SHARED row. If they overlap, no relative threshold works either.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
