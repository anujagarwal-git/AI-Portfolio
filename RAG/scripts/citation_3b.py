"""LAYER 3b - IS IT THE RIGHT PARAGRAPH? A GRADING WORKSHEET, NOT A METRIC.

WHY THIS SCRIPT SCORES NOTHING
    `align.py` chooses the paragraph. Anything that asked align.py whether
    align.py chose correctly could only ever agree with itself - the same shape
    as the `n_chars > 200` proxy rejected on 2026-08-23. So this script does
    exactly one thing: it puts the CLAIM and the PARAGRAPHS THE MODEL READ side
    by side, and Anuj decides. The verdict comes from outside the aligner or it
    is not a verdict.

    It therefore prints NO rate, NO percentage and NO pass/fail. It writes a
    worksheet with an empty `hand_verdict` on every claim.

WHAT THE THREE VERDICTS MEAN - fill these in by hand
    RIGHT       the sentence came from the paragraph that was cited
    WRONG       it came from a DIFFERENT paragraph in the same window
    UNGRADABLE  it draws on more than one paragraph, so there is no single
                right answer. NOT a failure. That question should LEAVE the
                set - a question that cannot be graded cleanly must not be
                graded generously.

    Report the result as "n of m claims correctly located, hand-graded", with
    the misses quoted. Never as a percentage of the corpus: fifteen questions
    chosen for gradability are not a sample of anything.

TWO PASSES
    --run     generate answers for the 3b set and write the worksheet
    (then edit the worksheet by hand, filling in hand_verdict)
    --report  read the filled worksheet back and print the tally

THE ALIGNER'S OWN VERDICT IS SHOWN, DELIBERATELY, AND LAST
    MATCH / FLOOR / TIE / NOLOC tells you what the aligner thought it was
    doing, which is what makes a WRONG informative - a wrong MATCH is a scoring
    bug, a wrong FLOOR means it never chose at all and shipped the retrieved
    child's locator. Read the claim BEFORE you read that column.

    uv run python scripts/citation_3b.py --run
    uv run python scripts/citation_3b.py --report
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
SET = EVAL / "citation_3b_set.json"

VERDICTS = ("RIGHT", "WRONG", "UNGRADABLE")


def _units(passage):
    """Paragraph units from the aligner itself, so this worksheet cannot show a
    different set of paragraphs than the one align.py chose among."""
    try:
        from regrag.generation import align as _al
        return _al.paragraph_units(passage)
    except Exception:                                         # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
def run(limit: int, only: str) -> int:
    from regrag.generation.answer import answer

    spec = json.loads(SET.read_text(encoding="utf-8"))
    qs = spec["questions"]
    if only:
        qs = [q for q in qs if q["id"] in {s.strip() for s in only.split(",")}]
    if limit:
        qs = qs[:limit]
    if not qs:
        print("no questions selected")
        return 1

    print(f"3b worksheet — {len(qs)} question(s). Generation only, no judge.\n")
    rows = []
    for i, q in enumerate(qs, 1):
        t0 = time.perf_counter()
        try:
            a = answer(q["question"])
            err = ""
        except Exception as exc:                              # noqa: BLE001
            traceback.print_exc()
            a, err = None, f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0

        if a is None:
            rows.append({**q, "error": err, "claims": [], "seconds": round(dt, 1)})
            print(f"  {i:2d}/{len(qs)}  {q['id']}  ERROR after {dt:.1f}s")
            continue

        passages = list(a.retrieval.passages) if a.retrieval else []
        # THE WINDOWS, IN FULL. The whole point of the worksheet is that the
        # grader reads what the model read. Truncating here would make the
        # grade an opinion about an excerpt.
        windows = []
        for idx, pas in enumerate(passages, 1):
            windows.append({
                "label": f"S{idx}",
                "short_name": pas.short_name,
                "paragraphs": [{"locator": u.locator, "text": u.text}
                               for u in _units(pas)],
            })

        claims = [{
            "text": c.text,
            "label": c.label,
            "cited_locator": c.locator,
            "aligner_verdict": c.align_verdict,
            "overlap": c.overlap,
            # *** FILL THIS IN BY HAND: RIGHT | WRONG | UNGRADABLE ***
            "hand_verdict": "",
            # If WRONG, write the locator it SHOULD have cited. That turns a
            # tally into a diagnosis - a cluster of misses one paragraph above
            # the cited one is a different bug from scattered misses.
            "should_have_been": "",
            "note": "",
        } for c in a.claims]

        rows.append({**q, "error": "", "seconds": round(dt, 1),
                     "plan_actual": a.retrieval.plan.mode if a.retrieval else "",
                     "config_hash": a.config_hash,
                     "refused": a.refused,
                     "claims": claims, "windows": windows})
        print(f"  {i:2d}/{len(qs)}  {q['id']}  {len(claims)} claim(s), "
              f"{len(windows)} window(s), {dt:.0f}s")

    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out = EVAL / f"citation_3b_worksheet_{stamp}.json"
    out.write_text(json.dumps({
        "built": stamp,
        "set": SET.name,
        "generator_config_hash": next((r.get("config_hash") for r in rows
                                       if r.get("config_hash")), ""),
        "instructions": [
            "For each claim: read `text`, then read the paragraphs of the "
            "window named by `label` under `windows`.",
            "Decide which paragraph that sentence actually came from.",
            "Write RIGHT, WRONG or UNGRADABLE into `hand_verdict`.",
            "If WRONG, put the correct locator into `should_have_been`.",
            "UNGRADABLE means the claim spans paragraphs - drop that question "
            "from the set rather than grading it.",
            "Read the claim BEFORE `aligner_verdict`. Knowing what the aligner "
            "thought biases the grade.",
        ],
        "rows": rows,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    n_claims = sum(len(r["claims"]) for r in rows)
    print(f"\nsaved  {out.relative_to(ROOT)}")
    print(f"{n_claims} claim(s) to grade. Fill in `hand_verdict`, then:")
    print(f"  uv run python scripts/citation_3b.py --report "
          f"--worksheet {out.name}")
    return 0


# ---------------------------------------------------------------------------
def report(name: str) -> int:
    path = (EVAL / name if name
            else max(EVAL.glob("citation_3b_worksheet_*.json"), default=None,
                     key=lambda p: p.stat().st_mtime))
    if path is None or not path.exists():
        print("no worksheet found")
        return 1
    blob = json.loads(path.read_text(encoding="utf-8"))
    rows = blob["rows"]

    graded, ungraded, bad = [], 0, []
    for r in rows:
        for c in r["claims"]:
            v = (c.get("hand_verdict") or "").strip().upper()
            if not v:
                ungraded += 1
            elif v not in VERDICTS:
                bad.append((r["id"], v))
            else:
                graded.append((r, c, v))

    print(f"worksheet    {path.name}")
    print(f"generator    {blob.get('generator_config_hash', '?')}\n")
    if bad:
        print(f"UNRECOGNISED VERDICTS: {bad}")
        print(f"  allowed: {', '.join(VERDICTS)}\n")
    if ungraded:
        print(f"{ungraded} claim(s) still blank — grade them or they are "
              f"silently excluded.\n")
    if not graded:
        return 1

    tally = Counter(v for _r, _c, v in graded)
    right, wrong = tally["RIGHT"], tally["WRONG"]
    denom = right + wrong

    print("LAYER 3b — HAND-GRADED SAMPLE. NOT A RATE.")
    print("-" * 62)
    print(f"  RIGHT       {right}")
    print(f"  WRONG       {wrong}")
    print(f"  UNGRADABLE  {tally['UNGRADABLE']}   (excluded — these questions "
          f"should leave the set)")
    print(f"\n  {right} of {denom} claims correctly located, hand-graded on "
          f"{len({r['id'] for r, _c, _v in graded})} question(s).")
    print("  Quote it that way. It is not a percentage of the corpus.")

    # WHAT THE ALIGNER THOUGHT IT WAS DOING WHEN IT WAS WRONG. A wrong MATCH is
    # a scoring bug; a wrong FLOOR means it never chose and shipped the
    # retrieved child's locator. Different bugs, different fixes.
    if wrong:
        print("\nMISSES, BY WHAT THE ALIGNER THOUGHT")
        print("-" * 62)
        by_verdict = Counter(c["aligner_verdict"] for _r, c, v in graded
                             if v == "WRONG")
        for k, n in by_verdict.most_common():
            print(f"  {k or '(none)':<10} {n}")
        print()
        for r, c, v in graded:
            if v != "WRONG":
                continue
            print(f"  {r['id']}  [{c['label']}] cited {c['cited_locator']!r}"
                  f"  should be {c['should_have_been'] or '?'!r}"
                  f"  ({c['aligner_verdict']}, overlap {c['overlap']})")
            print(f"        {c['text'][:110]}")
            if c.get("note"):
                print(f"        note: {c['note']}")

    # PER PLAN MODE, because the set was built one-per-mode on purpose.
    modes = Counter()
    mode_right = Counter()
    for r, _c, v in graded:
        if v == "UNGRADABLE":
            continue
        m = r.get("plan_actual") or r.get("plan_expected") or "?"
        modes[m] += 1
        mode_right[m] += (v == "RIGHT")
    if modes:
        print("\nBY PLAN MODE")
        print("-" * 62)
        for m in sorted(modes):
            print(f"  {m:<24}{mode_right[m]:>3} of {modes[m]:<3}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true",
                    help="generate answers and write a blank worksheet")
    ap.add_argument("--report", action="store_true",
                    help="read a filled worksheet and tally it")
    ap.add_argument("--worksheet", default="", help="worksheet file name")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated ids, e.g. cb-01,cb-03")
    ns = ap.parse_args()

    if ns.run == ns.report:
        ap.error("choose exactly one of --run or --report")
    return run(ns.limit, ns.only) if ns.run else report(ns.worksheet)


if __name__ == "__main__":
    sys.exit(main())
