"""Probe: can Docling extract the ERBA table completely?

THE PROBLEM. Docling's current settings LOSE CELLS. On CRE page 190, table 2
("ERBA risk weights for long-term ratings"), the AAA row extracts as:

    'AAA' | '15%' | '20%' | '15%'          4 cells

but the PDF says:

    AAA     15%     20%     15%     70%    5 values

The non-senior 5-year risk weight — 70% — is simply absent. Not blank in the
source: `pdftotext -layout` renders it correctly. Docling dropped it.

Across the corpus, 21% of table data rows have fewer cells than the widest row
in their own table.

WHY THIS MATTERS MORE THAN A MISSING PARAGRAPH. Absent text makes the system
say "I don't know". A table row missing one value makes it answer with the
wrong number and a clean citation. That is the failure this whole project is
built to prevent.

WHAT THIS PROBES. `mode` already defaults to ACCURATE, so that is not the
cause. The prime suspect is `do_cell_matching`, whose own docstring says:

    True:  Matches predictions back to PDF cells. CAN BREAK TABLE OUTPUT if
           PDF cells are merged across table columns.
    False: Let the table structure model define the text cells, ignore PDF cells.

Four configurations are tried on ONE PAGE, so this takes a minute rather than
re-parsing 323 pages.

Run:  uv run python scripts/probe_table_extraction.py
"""

from __future__ import annotations

from regrag import config

PDF = config.PROJECT_ROOT / "data" / "CreditRisk" / "BASEL_CRE.pdf"
PAGE = 190
EXPECTED = {  # read off the PDF by hand — the ground truth for this probe
    "AAA": ["15%", "20%", "15%", "70%"],
    "AA+": ["15%", "30%", "15%", "90%"],
}


def build(variant: str):
    """Return a DocumentConverter for one configuration, or None if the
    backend is unavailable in this Docling build."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True

    if variant == "current (V1, cell_matching=True)":
        pass  # defaults: TableFormer V1, ACCURATE, do_cell_matching=True

    elif variant == "V1, cell_matching=False":
        opts.table_structure_options.do_cell_matching = False

    elif variant == "V2":
        from docling.datamodel.pipeline_options import TableStructureV2Options
        opts.table_structure_options = TableStructureV2Options()

    elif variant == "V2, cell_matching=False":
        from docling.datamodel.pipeline_options import TableStructureV2Options
        opts.table_structure_options = TableStructureV2Options(do_cell_matching=False)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def erba_rows(doc) -> dict[str, dict]:
    """Pull the AAA / AA+ rows out of whichever table holds them.

    NO DE-DUPLICATION. An earlier version collapsed consecutive repeats to undo
    column spans, and destroyed the evidence: AAA legitimately holds 15% TWICE
    (senior 1-year and non-senior 1-year), so collapsing turned a correct
    ['15%','20%','15%','70%'] into ['15%','70%'] and made a working extractor
    look broken. Raw grid and raw cells are reported instead — judge from the
    data, not from a transform of it.
    """
    out: dict[str, dict] = {}
    for tb in doc.tables:
        data = tb.model_dump() if hasattr(tb, "model_dump") else {}
        d = data.get("data") or {}
        grid = d.get("grid") or []
        cells = d.get("table_cells") or []
        for r_i, row in enumerate(grid):
            texts = [(c.get("text") or "").strip() for c in row]
            if texts and texts[0] in EXPECTED and texts[0] not in out:
                raw = [
                    (c.get("text") or "").strip()
                    for c in cells
                    if c.get("start_row_offset_idx") == r_i
                ]
                out[texts[0]] = {"grid": texts, "cells": raw}
    return out


def main() -> None:
    variants = [
        "current (V1, cell_matching=True)",
        "V1, cell_matching=False",
        "V2",
        "V2, cell_matching=False",
    ]

    print(f"\nProbing {PDF.name} page {PAGE} — ERBA risk weights, Table 2")
    print("Ground truth read from the PDF by hand:")
    for k, v in EXPECTED.items():
        print(f"    {k}: {v}")
    print()

    for variant in variants:
        try:
            conv = build(variant)
            result = conv.convert(PDF, page_range=(PAGE, PAGE))
            rows = erba_rows(result.document)
        except Exception as e:
            print(f"  {variant:34}  UNAVAILABLE — {type(e).__name__}: {str(e)[:70]}")
            continue

        if not rows:
            print(f"\n  {variant}\n      no ERBA rows found")
            continue

        # A variant is COMPLETE if every expected value appears in the row's
        # raw cells. Order and span-repeats are ignored — what matters is that
        # no value was LOST.
        complete = all(
            all(v in rows.get(label, {}).get("cells", []) for v in expected)
            for label, expected in EXPECTED.items()
        )
        print(f"\n  {variant}   ->  {'COMPLETE' if complete else 'LOSSY'}")
        for label in EXPECTED:
            got = rows.get(label)
            if not got:
                print(f"      {label}: row not found")
                continue
            missing = [v for v in EXPECTED[label] if v not in got["cells"]]
            print(f"      {label} cells: {got['cells']}")
            print(f"      {label} grid : {got['grid']}")
            if missing:
                print(f"      {label} MISSING: {missing}")

    print("\nIf any variant reports COMPLETE, set it in parser.py and re-warm the cache.")
    print("If all are LOSSY, the parent table text must come from a pdftotext -layout")
    print("region extraction instead, and Docling's structure is used only for row keys.")


if __name__ == "__main__":
    main()
