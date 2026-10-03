"""CITATION ACCURACY, LAYERS 1-3a. Mechanical, no model, no Qdrant, no judge.

WHAT THIS MEASURES, AND THE ONE THING IT DOES NOT
    1   CITED     the claim carries a source label at all
    2   VALID     that label exists in the table the answer itself was given
    3a  IN WINDOW the cited paragraph is one the model actually READ

    Every one is a string check against evidence the run saved. None of them
    needs a judge, so the number is reproducible from the JSON alone.

    3b - IS IT THE RIGHT PARAGRAPH OF THE ONES IT READ - IS NOT HERE, and the
    reason is a rigor point, not an omission. `align.py` CHOOSES the paragraph.
    A metric that asked align.py whether align.py chose correctly could only
    ever agree with itself - the same shape as the `n_chars > 200` proxy that
    was rejected on 2026-08-23: a check that can only move the flattering
    direction is not a check. 3b needs ground truth from outside the aligner,
    which is a HAND-GRADED SAMPLE (see evaluation/citation_3b_set.json) and is
    reported as a sample, never as a rate.

WHY 3a IS WORTH MEASURING EVEN THOUGH IT IS WEAKER THAN 3b
    It is the falsifiable half. A citation pointing at a paragraph outside the
    delivered window is wrong with certainty and needs nobody's judgement -
    the model cited text that was never in front of it. That is exactly the
    failure class `align.py` was built after (an answer citing CRE53.56 whose
    claims came from 53.50, 53.51 and 53.55), and the guard that keeps a future
    change to the aligner from silently pointing outside the window.

READS A SAVED RUN. It never generates. So it can be re-run against any
`evaluation/golden_run_*.json` produced after 2026-09-10 - older runs saved
only `n_claims` and are reported as UNMEASURABLE rather than as zero.

    uv run python scripts/citation_accuracy.py
    uv run python scripts/citation_accuracy.py --run evaluation/golden_run_X.json
    uv run python scripts/citation_accuracy.py --misses      # print every fail
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"

RX_LABEL = re.compile(r"^S\d+$")


# ---------------------------------------------------------------------------
# THE THREE CHECKS. One claim in, three booleans out - or None for "cannot say".
#
# None IS NOT False. A claim whose passage recorded no paragraph locators
# (a window with no markers) cannot be judged on 3a, and scoring it as a miss
# would blame the citation for a parsing gap. Unmeasurable is counted, printed,
# and kept out of the denominator.
# ---------------------------------------------------------------------------
def grade_claim(claim: dict, labels: list[str],
                unit_locators: list[list[str]]) -> dict:
    label = str(claim.get("label") or "").strip()
    locator = str(claim.get("locator") or "").strip()

    cited = bool(label)
    valid = bool(cited and RX_LABEL.match(label) and label in labels)

    in_window: bool | None = None
    if valid:
        # "S3" -> passages[2]. The label table is built by enumerate(...,1) in
        # answer.py, so the index is the label's own number minus one. Read
        # from the label rather than from position, because a claim list is not
        # in passage order.
        idx = int(label[1:]) - 1
        units = unit_locators[idx] if 0 <= idx < len(unit_locators) else []
        known = {u for u in units if u}
        if not known:
            in_window = None          # nothing to check against - see above
        elif not locator:
            in_window = False         # cited a passage, named no paragraph
        else:
            in_window = locator in known

    return {"cited": cited, "valid": valid, "in_window": in_window,
            "label": label, "locator": locator, "text": claim.get("text", "")}


def grade_row(row: dict) -> list[dict] | None:
    """None means this row carries no citation evidence at all."""
    if "claims" not in row:
        return None
    labels = row.get("labels") or []
    units = row.get("unit_locators") or []
    return [grade_claim(c, labels, units) for c in row["claims"]]


# ---------------------------------------------------------------------------
def rate(hits: int, total: int) -> str:
    return f"{hits}/{total}" + (f"  {hits / total:5.1%}" if total else "     -")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--misses", action="store_true",
                    help="print every claim that failed a check")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    path = (pathlib.Path(args.run) if args.run
            else max(EVAL.rglob("golden_run_*.json"), default=None,
                     key=lambda p: p.stat().st_mtime))
    if path is None or not path.exists():
        print("no golden run found under evaluation/ (searched subfolders too)")
        return 1

    blob = json.loads(path.read_text(encoding="utf-8"))
    rows = blob["results"] if isinstance(blob, dict) else blob
    print(f"run          {path.name}")
    print(f"generator    {blob.get('generator_config_hash', '?')}")
    print(f"questions    {len(rows)}\n")

    # -- THE UNMEASURABLE CHECK COMES FIRST, because a metric computed on the
    #    wrong half of a file is worse than no metric. -----------------------
    graded: dict[str, list[dict]] = {}
    no_evidence = []
    for r in rows:
        g = grade_row(r)
        if g is None:
            no_evidence.append(r.get("id", "?"))
        else:
            graded[r.get("id", "?")] = g

    if no_evidence:
        print(f"NO CITATION EVIDENCE in {len(no_evidence)} of {len(rows)} rows.")
        print("  This run predates the 2026-09-10 field, which saves each")
        print("  claim's label and locator plus the delivered paragraph")
        print("  locators. Layers 1-3a CANNOT be computed from it - the claims")
        print("  are not in the file. Re-run scripts/run_golden.py.")
        if not graded:
            return 2
        print(f"  Grading the {len(graded)} rows that do carry it.\n")

    all_claims = [c for cs in graded.values() for c in cs]
    if not all_claims:
        print("no claims to grade (every answer refused or empty)")
        return 2

    n = len(all_claims)
    cited = sum(c["cited"] for c in all_claims)
    valid = sum(c["valid"] for c in all_claims)
    checkable = [c for c in all_claims if c["in_window"] is not None]
    inwin = sum(bool(c["in_window"]) for c in checkable)

    print("CITATION ACCURACY - layers 1 to 3a")
    print("-" * 60)
    print(f"  1  cited      {rate(cited, n)}        of all claims")
    print(f"  2  valid      {rate(valid, cited)}        of CITED claims")
    print(f"  3a in window  {rate(inwin, len(checkable))}        of VALID claims"
          f" with known paragraph locators")
    unmeasurable = sum(1 for c in all_claims
                       if c["valid"] and c["in_window"] is None)
    if unmeasurable:
        print(f"     ({unmeasurable} valid claim(s) had no paragraph markers in "
              f"their window - not counted either way)")

    # -- BY PLAN MODE. A single number hides which shape of question fails. ---
    by_mode: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        g = graded.get(r.get("id", "?"))
        if not g:
            continue
        m = r.get("plan_actual") or "?"
        for c in g:
            by_mode[m]["n"] += 1
            by_mode[m]["cited"] += c["cited"]
            by_mode[m]["valid"] += c["valid"]
            if c["in_window"] is not None:
                by_mode[m]["checkable"] += 1
                by_mode[m]["inwin"] += bool(c["in_window"])

    print("\nBY PLAN MODE")
    print("-" * 60)
    print(f"  {'mode':<22}{'claims':>7}{'cited':>8}{'valid':>8}{'in window':>12}")
    for m in sorted(by_mode):
        k = by_mode[m]
        w = f"{k['inwin']}/{k['checkable']}" if k["checkable"] else "-"
        print(f"  {m:<22}{k['n']:>7}{k['cited']:>8}{k['valid']:>8}{w:>12}")

    # -- THE MISSES. A rate with no examples cannot be acted on. -------------
    if args.misses:
        print("\nMISSES")
        print("-" * 60)
        for qid, cs in graded.items():
            for c in cs:
                why = ("no label" if not c["cited"]
                       else "label not in table" if not c["valid"]
                       else "locator outside the window"
                       if c["in_window"] is False else "")
                if why:
                    print(f"  {qid}  [{c['label'] or '-'}] "
                          f"{c['locator'] or '-'}  {why}")
                    print(f"        {c['text'][:110]}")

    if args.save:
        out = EVAL / (f"citation_accuracy_"
                      f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
        out.write_text(json.dumps({
            "run": path.name,
            "generator_config_hash": blob.get("generator_config_hash", ""),
            "layers": "1+2+3a",
            "n_claims": n, "cited": cited, "valid": valid,
            "in_window": inwin, "checkable": len(checkable),
            "unmeasurable_3a": unmeasurable,
            "rows_without_evidence": no_evidence,
            "by_mode": {m: dict(k) for m, k in by_mode.items()},
            "note": "3b (is it the RIGHT paragraph) is not here - align.py "
                    "chooses the paragraph, so it cannot grade its own choice. "
                    "See the hand-graded 3b sample.",
        }, indent=2), encoding="utf-8")
        print(f"\nsaved  {out.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
