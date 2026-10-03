"""LABEL IN EXCEL, NOT IN A TERMINAL. Export -> you fill one column -> import.

WHY THIS EXISTS
    Regulatory text carries page-break markers, footnote brackets and stray
    whitespace that a terminal renders badly. Judging relevance from text you
    cannot read comfortably produces labels that describe the rendering rather
    than the passage — and the labels ARE this study.

THE SHEET
    One row per delivered passage. Every retrieval score is there, and so is
    the mentor's first pass, so you can scan rather than start from nothing.

    YOU FILL ONE COLUMN: `your_label`, yellow, with a dropdown.
        r  relevant    it helps answer THIS question
        i  irrelevant  it does not, whatever else is true of it
        u  unsure      excluded from the analysis, and that is fine

    Leave a row blank and it stays unlabelled. Nothing else in the sheet is
    read back on import — edit anything else and it is simply ignored.

TWO THINGS DELIBERATELY PLACED
    `mentor_label` sits NEXT TO your column, not hidden. Anchoring is real: you
    will agree more often than you would have alone. That is the trade for
    speed, and it is why the agreement rate is reported but never used as
    evidence on its own — a 95% agreement partly measures the anchor.
    IF YOU WANT A CLEAN SECOND OPINION, hide column M before you start.

    THE SCORES SIT TO THE RIGHT OF THE TEXT. Reading "ce+6.8" before the
    passage is how a label becomes a description of the score being tested.
    Read the passage, decide, then look right if you are curious.

    uv run python scripts/label_sheet.py --export
    ... label in Excel, save ...
    uv run python scripts/label_sheet.py --import
    uv run python scripts/study_relevance.py --analyse
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

JSON = pathlib.Path("evaluation") / "relevance_study.json"
XLSX = pathlib.Path("evaluation") / "relevance_labels.xlsx"

FONT = "Arial"
COLS = [
    ("row", 6), ("qid", 5), ("mode", 12), ("plan_mode", 19), ("doc", 16),
    ("heading", 34), ("question", 46), ("passage", 100),
    ("your_label", 11), ("mentor_label", 13),
    ("rerank", 9), ("gap_to_best", 11), ("rrf", 9), ("dense_rank", 11),
    ("bm25_rank", 10), ("chars", 8), ("is_bin", 8), ("windowed", 9),
    ("off_target", 10), ("facet", 22), ("locator", 14),
]
SHORT = {"relevant": "r", "irrelevant": "i", "unsure": "u"}
LONG = {v: k for k, v in SHORT.items()}

# openpyxl REFUSES control characters outright (IllegalCharacterError), and
# text carved out of regulatory PDFs is full of them — form feeds at page
# breaks, vertical tabs inside tables, stray \x00 from the parser. They are
# invisible in a terminal, which is why this only surfaces at write time.
# Strip them here rather than upstream: the chunk text in Qdrant is what the
# model actually reads, so cleaning it for a spreadsheet's sake would make the
# sheet show something the pipeline never saw.
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(v):
    return _CTRL.sub(" ", v) if isinstance(v, str) else v


def export() -> int:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    d = json.loads(JSON.read_text(encoding="utf-8"))
    rows = d["rows"]

    wb = Workbook()
    ws = wb.active
    ws.title = "labels"

    head = Font(name=FONT, bold=True, color="FFFFFF")
    body = Font(name=FONT, size=10)
    grey = PatternFill("solid", fgColor="404040")
    yellow = PatternFill("solid", fgColor="FFFF00")

    for i, (name, width) in enumerate(COLS, 1):
        c = ws.cell(row=1, column=i, value=name)
        c.font, c.fill = head, grey
        c.alignment = Alignment(vertical="center")
        ws.column_dimensions[get_column_letter(i)].width = width

    for n, r in enumerate(rows, start=2):
        vals = [
            n - 2, r["qid"], r["want_mode"], r.get("plan_mode"),
            r["short_name"], r["heading"], r["question"],
            # Excel's hard cap is 32,767 characters per cell. A parent can be
            # larger, so it is trimmed HERE with a visible marker rather than
            # being silently refused by Excel on save.
            (r["text"][:30000] + "\n[... trimmed for Excel ...]"
             if len(r["text"]) > 30000 else r["text"]),
            SHORT.get(r.get("verdict") or "", ""),
            SHORT.get(r.get("verdict_claude") or "", ""),
            r.get("rerank"), r.get("gap_to_best"), r.get("rrf"),
            r.get("dense_rank"), r.get("lexical_rank"), r["chars"],
            r["is_bin"], r["windowed"], r["off_target"], r["facet"],
            r.get("locator"),
        ]
        for i, v in enumerate(vals, 1):
            c = ws.cell(row=n, column=i, value=clean(v))
            c.font = body
            if i in (6, 7, 8):                      # heading, question, passage
                c.alignment = Alignment(wrap_text=True, vertical="top")
            else:
                c.alignment = Alignment(vertical="top")
        ws.cell(row=n, column=9).fill = yellow      # your_label
        ws.row_dimensions[n].height = 110

    dv = DataValidation(type="list", formula1='"r,i,u"', allow_blank=True,
                        showDropDown=False)
    dv.error = "Use r (relevant), i (irrelevant) or u (unsure)."
    ws.add_data_validation(dv)
    dv.add(f"I2:I{len(rows) + 1}")

    ws.freeze_panes = "I2"        # question and passage stay visible while you scroll
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLS))}{len(rows) + 1}"

    # --- legend -----------------------------------------------------------
    lg = wb.create_sheet("read me")
    done = sum(1 for r in rows if r.get("verdict_claude"))
    lines = [
        ("HOW TO USE THIS SHEET", True),
        ("", False),
        ("Fill column I (your_label, yellow) with r, i or u. Nothing else is read back.", False),
        ("   r = relevant     this passage helps answer THAT ROW'S question", False),
        ("   i = irrelevant   it does not, whatever else is true of it", False),
        ("   u = unsure       excluded from the analysis, and that is fine", False),
        ("Blank = not yet labelled. Filter column I on (Blanks) to find them.", False),
        ("", False),
        ("THE COMPARISON THAT MATTERS", True),
        ("Column J is the mentor's first pass. It is NEXT TO yours on purpose, for", False),
        ("speed — but seeing it will pull your judgement toward it. That anchoring is", False),
        ("real, so the agreement rate is reported and never treated as evidence on its", False),
        ("own. FOR A CLEAN SECOND OPINION, HIDE COLUMN J BEFORE YOU START.", False),
        ("", False),
        ("WHY THE SCORES ARE TO THE RIGHT OF THE PASSAGE", True),
        ("Reading 'ce+6.8' before the text is how a label becomes a description of the", False),
        ("score being tested. Read the passage, decide, then look right if curious.", False),
        ("", False),
        ("WHEN YOU ARE DONE", True),
        ("   uv run python scripts/label_sheet.py --import", False),
        ("   uv run python scripts/study_relevance.py --analyse", False),
        ("Import is additive: it only fills rows you labelled and leaves the rest.", False),
        ("", False),
        ("COUNTS AT EXPORT — static, not live", True),
        (f"   passages            {len(rows)}", False),
        (f"   mentor labelled     {done}", False),
        (f"   yours labelled      {sum(1 for r in rows if r.get('verdict'))}", False),
        ("", False),
        ("The 15 multi-facet questions (mode <> 'named') are where irrelevance is", False),
        ("expected — filter column C to them first if you are short of time.", False),
    ]
    lg.column_dimensions["A"].width = 92
    for i, (text, bold) in enumerate(lines, 1):
        c = lg.cell(row=i, column=1, value=text)
        c.font = Font(name=FONT, bold=bold, size=11)

    XLSX.parent.mkdir(exist_ok=True)
    wb.save(XLSX)
    print(f"  {len(rows)} rows -> {XLSX}")
    print(f"  mentor pre-labelled {done}; fill column I, save, then --import")
    return 0


def do_import() -> int:
    from openpyxl import load_workbook

    if not XLSX.exists():
        sys.exit(f"{XLSX} not found — run --export first")
    d = json.loads(JSON.read_text(encoding="utf-8"))
    rows = d["rows"]
    ws = load_workbook(XLSX, data_only=True)["labels"]

    header = [c.value for c in ws[1]]
    i_row, i_lab = header.index("row") + 1, header.index("your_label") + 1

    changed = bad = 0
    for r in ws.iter_rows(min_row=2):
        idx, raw = r[i_row - 1].value, r[i_lab - 1].value
        if idx is None or raw in (None, ""):
            continue
        key = str(raw).strip().lower()[:1]
        if key not in LONG:
            bad += 1
            continue
        # `row` is the index into the JSON, written at export. Trust it rather
        # than sheet position: a sorted or filtered sheet would otherwise write
        # every label onto the wrong passage, silently.
        if rows[int(idx)].get("verdict") != LONG[key]:
            rows[int(idx)]["verdict"] = LONG[key]
            changed += 1

    JSON.write_text(json.dumps(d, indent=1), encoding="utf-8")
    done = sum(1 for r in rows if r.get("verdict"))
    print(f"  {changed} label(s) updated, {done}/{len(rows)} now labelled by you")
    if bad:
        print(f"  {bad} cell(s) ignored — not r / i / u")
    both = [r for r in rows if r.get("verdict") and r.get("verdict_claude")]
    if both:
        same = sum(1 for r in both if r["verdict"] == r["verdict_claude"])
        print(f"  agreement with the mentor pass: {same}/{len(both)} "
              f"({same / len(both):.0%})")
    print("  next: uv run python scripts/study_relevance.py --analyse")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export", action="store_true")
    g.add_argument("--import", dest="imp", action="store_true")
    ns = ap.parse_args()
    return export() if ns.export else do_import()


if __name__ == "__main__":
    raise SystemExit(main())
