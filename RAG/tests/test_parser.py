"""Smoke test for regrag.ingestion.parser — verifies the module works INDEPENDENTLY
of the notebook it was born in. Runs the real Docling parse on SR 11-7 and
checks the Stage 1 'done when' criteria against the quality report.

Run:  pytest tests/test_parser.py -s
  or: python tests/test_parser.py   (standalone, prints the report)
"""

from pathlib import Path

from regrag.ingestion.parser import parse_pdf, parse_quality_report

# Path is HARD-CODED, deliberately, even though registry.py could supply it.
# This is a UNIT test: it must prove parser.py works on its own. If it read the
# path from the registry, a broken registry would fail this test too and the
# error would point at the wrong module.
# Reading the path from the registry belongs in an INTEGRATION test — the
# separate step that checks registry -> parser -> chunker actually compose.
PDF = (
    Path(__file__).resolve().parents[1]
    / "data" / "MRM" / "SR-11-7_ModelRiskMgmt_2011.pdf"
)


def _report():
    doc = parse_pdf(PDF)
    return doc, parse_quality_report(doc)


def test_parse_quality():
    doc, r = _report()

    assert r["conversion_ok"], "conversion produced an empty document"
    assert r["n_text_items"] > 50, f"suspiciously few blocks: {r['n_text_items']}"
    assert r["n_body"] > 0, "no BODY content — everything landed in furniture"
    assert r["n_section_headers"] >= 7, "expected the 7 roman-numeral SR 11-7 headings"
    assert r["n_footnotes"] > 0, "footnotes disappeared (merged into body?)"
    assert r["prov_coverage"] > 0.95, "some blocks lost page provenance"

    # Interior reading-order tripwire: print the block sequence around the
    # first BODY table so the order can be eyeballed (check 8 stays manual).
    _print_interior_context(doc)


def _print_interior_context(doc, window: int = 3):
    texts = doc.texts
    for i, t in enumerate(texts):
        lbl = t.label.name if hasattr(t.label, "name") else str(t.label)
        if lbl == "TABLE" or (getattr(doc, "tables", None) and lbl == "TABLE"):
            break
    else:
        i = len(texts) // 2  # no inline table label; sample mid-document
    print("\n--- interior reading-order sample ---")
    for t in texts[max(0, i - window): i + window]:
        lbl = t.label.name if hasattr(t.label, "name") else str(t.label)
        page = t.prov[0].page_no if getattr(t, "prov", None) else "?"
        print(f"[p{page}][{lbl}] {t.text[:90]}")


if __name__ == "__main__":
    doc, r = _report()
    import json
    print(json.dumps(r, indent=2, default=str))
    _print_interior_context(doc)
    print("\nOK — parser smoke run complete.")
