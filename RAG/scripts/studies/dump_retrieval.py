"""Every chunk retrieved for the golden set, with all four scores. CSV + console.

WHY THIS IS A RE-RUN AND NOT A READ
    run_golden.py saved the passage TEXT but not the per-child hit scores, so
    they cannot be read back out of golden_run_*.json. Retrieval is cheap and
    carries no sampling, so re-running it reproduces what the graded run saw -
    but "should reproduce" is not "did", so this CROSS-CHECKS the parent_ids
    against the saved run and reports any drift instead of assuming there is
    none. If drift appears, the scores below belong to a different retrieval
    than the one that was graded and must not be read alongside those numbers.

WHAT THE FOUR SCORES MEAN, since they are not interchangeable
    dense_rank    position in the vector search for this facet (0 = best).
                  None means the dense arm never returned this chunk.
    lexical_rank  position in BM25 for this facet. None means BM25 missed it.
                  A chunk found by only ONE arm is a chunk the other disagreed
                  about - that disagreement is what RRF is built to exploit.
    rrf           reciprocal rank fusion of the two arms. THIS IS WHAT ORDERS
                  DELIVERY: the cross-encoder's scores are used for the floor
                  and the facet margin, NOT for ordering (2026-08-29 - resorting
                  by cross-encoder collapsed multi-document coverage).
    rerank        cross-encoder score. Negative is normal. The floor drops a
                  chunk below RERANK_DROP_BELOW (0.0), keeping each facet's best.
    gap_to_best   distance below the best rerank in THIS question. The only
                  relative score, and the shape the facet margin uses.

ONE PARENT CAN HOLD SEVERAL MATCHED CHILDREN. The parent is delivered once;
every child that matched inside it is listed, because the child that WON the
parent its place is not necessarily the paragraph the answer used - that was
the whole citation-aligner finding of 2026-09-03.

    uv run python scripts/dump_retrieval.py
    uv run python scripts/dump_retrieval.py --id gs-02
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
import sys

W = 100
LINE = "=" * W
GOLDEN = pathlib.Path("evaluation") / "golden_set_v2.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="", help="one question id, e.g. gs-02")
    ap.add_argument("--out", default="evaluation/golden_retrieval.csv")
    ap.add_argument("--chars", type=int, default=160, help="text preview width")
    ns = ap.parse_args()

    from regrag.retrieval.search import is_bin, retrieve

    rows = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    if ns.id:
        rows = [r for r in rows if r["id"] == ns.id] or sys.exit(f"no {ns.id!r}")

    # What the graded run actually delivered, for the drift check.
    saved: dict[str, list[str]] = {}
    runs = sorted(glob.glob("evaluation/golden_run_*.json"))
    if runs:
        prev = json.load(open(runs[-1], encoding="utf-8"))
        saved = {r["id"]: r.get("locators", []) for r in prev["results"]}
        print(f"  cross-checking against {runs[-1]}")

    out, drift = [], []
    for item in rows:
        r = retrieve(item["question"])
        best = max((h.rerank for p in r.passages for h in p.children
                    if h.rerank is not None), default=None)
        print(f"\n{LINE}\n{item['id']}  [{item['scenario']}]  {item['question']}")
        print(f"  plan {r.plan.mode}  facets {len(r.plan.facets)}  "
              f"-> {len(r.passages)} passage(s), {r.total_chars:,} chars")
        for n in r.notes:
            print(f"  ! {n}")
        print(f"  {'S':<3}{'doc':<16}{'locator':<12}{'chars':>7}{'dense':>7}"
              f"{'bm25':>6}{'rrf':>9}{'rerank':>8}{'gap':>7}  heading")
        for i, p in enumerate(r.passages, 1):
            for j, h in enumerate(p.children):
                d = "-" if h.dense_rank is None else h.dense_rank + 1
                l = "-" if h.lexical_rank is None else h.lexical_rank + 1
                rr = "-" if h.rerank is None else f"{h.rerank:+.2f}"
                gp = ("-" if (best is None or h.rerank is None)
                      else f"{best - h.rerank:.2f}")
                tag = f"S{i}" if j == 0 else " ↳"
                print(f"  {tag:<3}{p.short_name[:15]:<16}"
                      f"{str(h.payload.get('locator') or '')[:11]:<12}"
                      f"{len(p.text) if j == 0 else 0:>7}{str(d):>7}{str(l):>6}"
                      f"{h.rrf:>9.5f}{rr:>8}{gp:>7}  {p.heading[:30]}")
                out.append({
                    "id": item["id"], "scenario": item["scenario"],
                    "question": item["question"], "plan_mode": r.plan.mode,
                    "label": f"S{i}", "child_of_parent": j,
                    "short_name": p.short_name, "heading": p.heading,
                    "parent_id": p.parent_id,
                    "locator": h.payload.get("locator") or "",
                    "facet": h.facet,
                    "passage_chars": len(p.text), "full_chars": p.full_chars,
                    "windowed": p.windowed, "is_bin": is_bin(p.heading),
                    "dense_rank": h.dense_rank, "lexical_rank": h.lexical_rank,
                    "arms": ("both" if h.dense_rank is not None and h.lexical_rank is not None
                             else "dense_only" if h.lexical_rank is None else "bm25_only"),
                    "rrf": round(h.rrf, 6), "rerank": h.rerank,
                    "gap_to_best": (None if (best is None or h.rerank is None)
                                    else round(best - h.rerank, 3)),
                    "n_children": len(p.children),
                    "text": " ".join(p.text.split())[:ns.chars],
                })
        got = [p.parent_id for p in r.passages]
        if item["id"] in saved and saved[item["id"]] != got:
            drift.append(item["id"])

    o = pathlib.Path(ns.out)
    with o.open("w", encoding="utf-8", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        wtr.writeheader(); wtr.writerows(out)
    print(f"\n{LINE}\n  {len(out)} chunk row(s) across {len(rows)} question(s) -> {o}")
    if drift:
        print(f"  *** DRIFT: {drift} delivered DIFFERENT parents than the graded run.")
        print("  These scores are NOT the ones behind those recall numbers. Do not")
        print("  read them together until you know why retrieval changed.")
    elif saved:
        print("  no drift: every question delivered the same parents as the graded run,")
        print("  so these scores are the ones behind those recall numbers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
