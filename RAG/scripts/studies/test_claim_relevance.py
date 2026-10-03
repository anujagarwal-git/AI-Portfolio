"""Does the cross-encoder separate an ON-TOPIC claim from an OFF-TOPIC one?

THE CASE THIS EXISTS FOR
    "What capabilities must a bank have for risk data aggregation?" produced,
    among four claims:

      "A bank must establish and maintain policies and procedures governing its
       liquidity stress testing practices..."   [12 CFR Part 252]  100% overlap

    Correctly cited. Perfectly grounded. Nothing to do with the question. Every
    check we have passes it, because overlap compares the claim to its PASSAGE
    and nothing compares it to the QUESTION.

THE IDEA
    Score (question, claim) with the same cross-encoder that scores
    (question, chunk). A claim is a short passage; judging whether a passage
    answers a query is the model's only job.

WHY THIS MIGHT WORK WHERE THE REFUSAL GATE DID NOT
    The gate needed an ABSOLUTE boundary holding across different questions, and
    died: hard negatives reached +6.99, true positives fell to +3.38. This needs
    only a RELATIVE comparison - is this claim far below the best claim in the
    SAME answer? That is the comparison that worked for facets at a margin of
    6.0. Relative inside one question: works. Absolute across questions: does not.

NO GENERATION. Reads claims already produced and only scores them.

    uv run python scripts/test_claim_relevance.py
"""
from __future__ import annotations

import glob
import json
import pathlib

from regrag.retrieval import reranker

W = 96


def load() -> list[tuple[str, list[str]]]:
    """(question, [claim, ...]) from the newest saved generation run."""
    files = sorted(glob.glob("evaluation/stage7_*.json")) + \
            sorted(glob.glob("evaluation/json_run_*.json"))
    if not files:
        raise SystemExit("no saved run found in evaluation/")
    f = sorted(files, key=lambda p: pathlib.Path(p).stat().st_mtime)[-1]
    print(f"source: {f}\n")
    out = []
    for r in json.load(open(f, encoding="utf-8"))["rows"]:
        q = r.get("q") or r.get("question")
        if "claims" in r and r["claims"] and isinstance(r["claims"][0], dict) \
                and "text" in r["claims"][0]:
            claims = [c["text"] for c in r["claims"]]
        elif r.get("raw"):
            try:
                claims = [c.get("text", "") for c in json.loads(r["raw"]).get("claims", [])]
            except Exception:
                continue
        else:
            continue
        if q and claims:
            out.append((q, [c for c in claims if c.strip()]))
    return out


def main() -> int:
    if not reranker.available():
        raise SystemExit("cross-encoder unavailable")
    data = load()
    all_margins, flagged = [], []

    for q, claims in data:
        scores = reranker.score(q, claims)          # ONE batched pass
        ranked = sorted(zip(scores, claims), key=lambda x: -x[0])
        top = ranked[0][0]
        print("=" * W)
        print(q)
        print("=" * W)
        for sc, cl in ranked:
            gap = top - sc
            all_margins.append(gap)
            mark = ""
            if gap > 6.0:
                mark = "   <-- 6.0+ below the best claim"
                flagged.append((q, cl, sc, gap))
            print(f"  ce{sc:+6.2f}  gap {gap:5.2f}  {cl[:74]}{mark}")
        print()

    print("=" * W)
    print("MARGIN DISTRIBUTION (0.00 = the best claim in its own answer)")
    print("=" * W)
    nz = sorted(m for m in all_margins if m > 0)
    if nz:
        for p, lab in ((0, "min"), (len(nz) // 4, "p25"), (len(nz) // 2, "median"),
                       (3 * len(nz) // 4, "p75"), (len(nz) - 1, "max")):
            print(f"  {lab:<8}{nz[p]:6.2f}")
    print(f"\n  claims scored {len(all_margins)}, non-zero gaps {len(nz)}")
    print(f"\n  AT A 6.0 MARGIN, {len(flagged)} CLAIM(S) WOULD BE DROPPED:")
    for q, cl, sc, gap in flagged:
        print(f"    {q[:56]}")
        print(f"      ce{sc:+.2f} ({gap:.1f} below)  {cl[:70]}")
    if not flagged:
        print("    none")
    print("\n  READ THEM. The test is whether the liquidity-stress-testing claim on the")
    print("  risk-data-aggregation question is among them, and whether anything")
    print("  legitimate is too. Do NOT pick a margin before reading the spread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
