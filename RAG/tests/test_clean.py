"""Tests for regrag.ingestion.clean.

Cleaning is the only destructive step in the pipeline, so most of these tests
assert what must SURVIVE rather than what must be removed. Over-cleaning is the
dangerous direction: text that has lost something still reads fluently, still
embeds, still retrieves, and nothing reports it.

Fixtures are copied from the real corpus — the CRE subscript line, the SR 11-7
mangled marker, the SR 15-19 intact marker — because the sections.py experience
showed that fixtures invented from a mental model only test whether the code
matches that model, not whether the model is right.

Run:  pytest tests/test_clean.py -q
  or: python tests/test_clean.py
"""

from __future__ import annotations

import os

from regrag import config
from regrag.ingestion.clean import CleanedDoc, clean_document, clean_report
from regrag.registry import load

SKIP_SLOW = os.getenv("REGRAG_SKIP_SLOW") == "1"


class _Prov:
    def __init__(self, page_no):
        self.page_no = page_no


class _Label:
    def __init__(self, name):
        self.name = name


class _Item:
    def __init__(self, text, label="TEXT", page=1, orig=None):
        self.text = text
        self.orig = orig            # Docling supplies both; `orig` keeps markers
        self.label = _Label(label)
        self.prov = [_Prov(page)]


class _Doc:
    def __init__(self, items):
        self.texts = items


class _Ref:
    def __init__(self, cref):
        self.cref = cref


class _Table:
    def __init__(self, data, page=1):
        self._data = data
        self.prov = [_Prov(page)]

    def model_dump(self):
        return {"data": self._data}


class _Group:
    def __init__(self, refs):
        self.children = [_Ref(r) for r in refs]


class _Body:
    def __init__(self, refs):
        self.children = [_Ref(r) for r in refs]


class _StructuredDoc:
    """Mirrors DoclingDocument closely enough to exercise _iter_document:
    texts, tables, groups and a body whose children reference all three."""

    def __init__(self, texts, tables=(), groups=(), body_refs=()):
        self.texts = list(texts)
        self.tables = list(tables)
        self.groups = list(groups)
        self.body = _Body(list(body_refs))


def _cell(text, row=0, col=0):
    return {"text": text, "start_row_offset_idx": row, "start_col_offset_idx": col}


def _table_data(rows, num_cols):
    """rows: list of lists of cell text -> Docling's grid + table_cells shape."""
    grid = [[_cell(t, r, c) for c, t in enumerate(row)] for r, row in enumerate(rows)]
    cells = [c for row in grid for c in row if c["text"]]
    return {"grid": grid, "table_cells": cells,
            "num_rows": len(rows), "num_cols": num_cols}


def _row(doc_id="bcbs-cre-consolidated"):
    return load().by_id(doc_id)


# =============================================================================
# 1. WHAT MUST SURVIVE — the important half
# =============================================================================


def test_math_subscripts_survive_as_content():
    """THE test this module exists for.

    CRE: 'Time period parameters: M{{i}}, E{{i}}, S{{i}} and T{{i}}'
    Deleting {{...}} yields 'M, E, S and T' — still reads like a formula,
    no longer means anything, and nothing anywhere reports it.
    """
    doc = _Doc([_Item("Time period parameters: M{{i}}, E{{i}}, S{{i}} and T{{i}}")])
    out = clean_document(doc, _row()).texts[0].text

    assert out == "Time period parameters: Mi, Ei, Si and Ti"
    for token in ("Mi", "Ei", "Si", "Ti"):
        assert token in out, f"subscript {token} was destroyed"


def test_enumeration_markers_are_recovered_from_orig():
    """Docling's `text` STRIPS the list marker; `orig` keeps it.

    Real pair from 12 CFR 225.8. Reading `text` discarded these across the
    corpus — 913 in CRE, 678 in Part 252. In the CFR the enumeration IS the
    structure (148 paragraphs under one heading), and it is also how the
    provision gets cited: "12 CFR 225.8(b)(1)", not "page 3".
    """
    doc = _Doc([_Item(
        text="Applicability. Except as provided in paragraph (c) of this section",
        orig="(1) Applicability. Except as provided in paragraph (c) of this section",
    )])
    out = clean_document(doc, _row("cfr-12-225-8"))

    assert out.texts[0].text.startswith("(1) "), "enumeration marker lost"
    assert out.report["markers_recovered"] == 1


def test_orig_is_ignored_when_it_matches_text():
    """Most items have orig == text. Those must not be counted as recoveries,
    or the metric stops meaning anything."""
    same = "A bank must maintain adequate capital."
    doc = _Doc([_Item(text=same, orig=same)])
    out = clean_document(doc, _row())

    assert out.texts[0].text == same
    assert out.report["markers_recovered"] == 0


def test_missing_orig_falls_back_to_text():
    """Not every source supplies `orig`. Absence must be harmless."""
    doc = _Doc([_Item(text="plain text with no orig field")])
    assert clean_document(doc, _row()).texts[0].text == "plain text with no orig field"


def test_glossary_link_is_unwrapped_not_deleted():
    """{{IRB}} is a glossary reference. Same syntax as the subscript above, so
    one rule has to serve both: unwrap, never delete."""
    doc = _Doc([_Item("banks using the {{IRB}} approach must")])
    assert clean_document(doc, _row()).texts[0].text == "banks using the IRB approach must"


def test_paragraph_numbers_and_cross_references_are_untouched():
    """'36.122' is a citation locator and 'LEX30' is a cross-reference. Both
    look like noise to a naive cleaner and both are load-bearing."""
    src = "36.122 Banks must validate as specified in LEX30 and CRE36.53."
    doc = _Doc([_Item(src)])
    assert clean_document(doc, _row()).texts[0].text == src


def test_ordinary_prose_is_byte_identical():
    """Most text must not be touched at all. If a clean pass rewrites ordinary
    sentences, some rule is too broad."""
    src = ("A bank should ensure that it has sufficient capital to meet the "
           "Pillar 1 requirements, including under adverse conditions.")
    doc = _Doc([_Item(src)])
    assert clean_document(doc, _row()).texts[0].text == src


def test_footnotes_are_kept_and_tagged():
    """Decision: keep footnotes, tag them. Regulatory footnotes carry real
    content — SR 15-19's cites 84 Fed. Reg. 59032; SR 11-7's define terms."""
    doc = _Doc([
        _Item("body paragraph", label="TEXT"),
        _Item("See 84 Fed. Reg. 59032 (November 1, 2019).", label="FOOTNOTE"),
    ])
    out = clean_document(doc, _row("sr-15-19-2015r2021")).texts

    assert len(out) == 2, "footnotes must not be dropped"
    assert out[0].is_footnote is False
    assert out[1].is_footnote is True
    assert "84 Fed. Reg. 59032" in out[1].text


# =============================================================================
# 2. WHAT MUST BE REMOVED
# =============================================================================


def test_page_furniture_dropped_by_label_not_pattern():
    """Basel footers are '2/20', SR letters use 'Page 7'. Docling already
    classifies both, so this uses its structural judgement rather than a
    pattern that could match a real numeric reference in body text."""
    doc = _Doc([
        _Item("real content"),
        _Item("2/20", label="PAGE_FOOTER"),
        _Item("Page 7", label="PAGE_FOOTER"),
        _Item("Basel Committee on Banking Supervision", label="PAGE_HEADER"),
    ])
    res = clean_document(doc, _row())

    assert [i.text for i in res.texts] == ["real content"]
    assert res.report["dropped_page_furniture"] == 3


def test_intact_footnote_marker_removed():
    """SR 15-19's real form, 63 occurrences: digit glued to the word, then
    '[Footnote'."""
    doc = _Doc([_Item("subject to the Board's tailoring framework,2[Footnote")])
    out = clean_document(doc, _row("sr-15-19-2015r2021"))

    assert out.texts[0].text == "subject to the Board's tailoring framework,"
    assert out.report["footnote_markers_clean"] == 1


def test_mangled_footnote_marker_removed():
    """SR 11-7's real form. The font encoding transposes the characters:
    '.1[Footnote' arrives as '[Fo tn1oe'."""
    doc = _Doc([_Item("decision making.[Fo tn1oe They routinely use models")])
    out = clean_document(doc, _row("sr-11-7-2011"))

    assert "[Fo" not in out.texts[0].text
    assert "They routinely use models" in out.texts[0].text
    assert out.report["footnote_markers_mangled"] == 1


def test_unknown_marker_corruption_is_reported_not_guessed_at():
    """A form neither pattern handles must SURFACE, not be absorbed by
    loosening a regex until it disappears. A new corruption is a prompt to go
    and look at the PDF."""
    doc = _Doc([_Item("some text [Fo0tn0te weird corruption here")])
    rep = clean_report(doc, _row())

    assert rep["n_unknown_marker_residue"] >= 1
    assert "[Fo" in rep["unknown_marker_residue"][0]


def test_empty_items_dropped():
    doc = _Doc([_Item("content"), _Item("   "), _Item("")])
    assert clean_document(doc, _row()).report["dropped_empty"] == 2


# =============================================================================
# 3. PROSE-LAYOUT TABLES — content Docling files under `tables`
# =============================================================================


def test_ifrs9_style_hanging_indent_is_recovered():
    """IFRS 9's paragraph numbers sit in a left margin, so Docling reads the
    layout as a two-column table. That text lives ONLY in `tables` — 16,224
    characters of IFRS 9, including the ECL provisions, were absent from the
    index entirely, which is why the document scored 0% citable.

    Flattening puts the number back at the start of the line, so the existing
    locator extractor recovers '5.5.12' with no special handling.
    """
    data = _table_data([
        ["5.5.12", "If the contractual cash flows on a financial asset have been "
                   "renegotiated or modified and the financial asset was not "
                   "derecognised, an entity shall assess whether there has been a "
                   "significant increase in credit risk."],
        ["", "(a) the risk of a default occurring at the reporting date based on "
             "the modified contractual terms of the financial asset."],
    ], num_cols=2)
    doc = _StructuredDoc(texts=[], tables=[_Table(data)], body_refs=["#/tables/0"])
    out = clean_document(doc, _row("ifrs9-extract")).texts

    assert len(out) == 2, [i.text[:40] for i in out]
    assert out[0].text.startswith("5.5.12 If the contractual")
    assert "significant increase in credit risk" in out[0].text


def test_genuine_grid_is_left_alone():
    """Docling loses roughly 40% of values in multi-column numeric tables, so
    those are deferred to a pdfplumber pass rather than indexed with
    known-wrong numbers. Only prose layouts are recovered here."""
    data = _table_data([
        ["Rating", "1 year", "5 years", "1 year", "5 years"],
        ["AAA", "15%", "20%", "15%", "70%"],
        ["AA+", "15%", "30%", "15%", "90%"],
    ], num_cols=5)
    doc = _StructuredDoc(texts=[], tables=[_Table(data)], body_refs=["#/tables/0"])

    assert clean_document(doc, _row("bcbs-cre-consolidated")).texts == []


def test_contents_listing_is_not_treated_as_content():
    """A table of contents is furniture. Dotted leaders identify it."""
    data = _table_data([
        ["Introduction .................................................", "4"],
        ["Definition ...................................................", "6"],
        ["Objectives ..................................................."
         " and a long trailing description to clear the prose threshold "
         "so only the dotted leaders can disqualify it.", "8"],
    ], num_cols=2)
    doc = _StructuredDoc(texts=[], tables=[_Table(data)], body_refs=["#/tables/0"])

    assert clean_document(doc, _row("bcbs-239-2013")).texts == []


def test_text_nested_inside_a_group_is_not_dropped():
    """body.children is NOT flat — it contains '#/groups/N' entries whose own
    children hold the text. A flat walk silently dropped more than half of
    IFRS 9 while appearing to work. The traversal must recurse."""
    texts = [_Item("Top-level paragraph, directly under body."),
             _Item("Nested paragraph, reachable only through the group.")]
    doc = _StructuredDoc(
        texts=texts,
        groups=[_Group(["#/texts/1"])],
        body_refs=["#/texts/0", "#/groups/0"],
    )
    out = clean_document(doc, _row()).texts

    assert len(out) == 2, "grouped text was dropped"
    assert "Nested paragraph" in out[1].text


def test_reading_order_is_preserved_across_texts_and_tables():
    """A recovered table row must land where it sits in the document, so it
    falls inside the right chapter and inherits that chapter's date."""
    data = _table_data([
        ["5.5.12", "A recovered provision, written long enough to clear the prose "
                   "threshold that separates a hanging-indent layout from a genuine "
                   "numeric grid, with plenty of characters to spare."],
    ], num_cols=2)
    doc = _StructuredDoc(
        texts=[_Item("Before the table."), _Item("After the table.")],
        tables=[_Table(data)],
        body_refs=["#/texts/0", "#/tables/0", "#/texts/1"],
    )
    out = [i.text[:20] for i in clean_document(doc, _row("ifrs9-extract")).texts]

    assert out[0].startswith("Before")
    assert out[1].startswith("5.5.12")
    assert out[2].startswith("After")


def test_falls_back_to_texts_when_there_is_no_body():
    """Any backend without a body — and every stub in these tests — must still
    work."""
    doc = _Doc([_Item("Plain paragraph with no body structure at all.")])
    assert len(clean_document(doc, _row()).texts) == 1


# =============================================================================
# 4. THE CONTRACT WITH THE REST OF THE PIPELINE
# =============================================================================


def test_cleaned_doc_works_with_find_sections():
    """clean -> sections -> chunk. CleanedDoc exposes .texts, so find_sections
    consumes it unchanged and never learns that cleaning happened."""
    from regrag.ingestion.sections import find_sections

    doc = _Doc([
        _Item("LEX10 Definitions and application", label="SECTION_HEADER"),
        _Item("Version effective as of 01 Jan 2023", label="SECTION_HEADER"),
        _Item("2/20", label="PAGE_FOOTER"),
        _Item("10.1 body text"),
    ])
    row = _row("bcbs-lex-consolidated")
    cleaned = clean_document(doc, row)
    secs = find_sections(cleaned, row)

    assert [s.code for s in secs] == ["LEX10"]
    assert str(secs[0].effective_from) == "2023-01-01"


def test_provenance_survives_cleaning():
    """A chunk cannot cite a page it has lost. page_no must survive the
    conversion into CleanItem."""
    doc = _Doc([_Item("content", page=7)])
    item = clean_document(doc, _row()).texts[0]

    assert item.page == 7
    assert item.prov[0].page_no == 7


def test_original_document_is_not_mutated():
    src = "text with {{IRB}} and 2[Footnote"
    item = _Item(src)
    doc = _Doc([item])
    clean_document(doc, _row())

    assert item.text == src, "clean_document must not mutate its input"


def test_orig_idx_traces_back_to_the_raw_parse():
    """Every chunk must be traceable to the exact text block it came from in
    the unmodified parse — otherwise a suspect answer cannot be audited."""
    doc = _Doc([
        _Item("first"),
        _Item("2/20", label="PAGE_FOOTER"),
        _Item("third"),
    ])
    out = clean_document(doc, _row()).texts

    assert [i.orig_idx for i in out] == [0, 2]


# =============================================================================
# 4. REAL DOCUMENTS
# =============================================================================


def test_real_documents_lose_little_text():
    """A small single-digit percentage is artifact removal. A large drop means
    a rule is eating content — the failure mode that reads perfectly fine."""
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return

    from regrag.ingestion.parser import parse_pdf

    reg = load()
    for doc_id in ("sr-11-7-2011", "bcbs-lex-consolidated"):
        row = reg.by_id(doc_id)
        rep = clean_report(parse_pdf(config.PROJECT_ROOT / row.file), row)
        print(f"\n  {rep['short_name']}: -{rep['chars_removed_pct']}% chars, "
              f"{rep['items_in']}->{rep['items_out']} items, "
              f"{rep['n_footnotes']} footnotes, "
              f"{rep['n_unknown_marker_residue']} unknown residue")

        assert rep["chars_removed_pct"] < 8.0, (
            f"{rep['short_name']}: cleaning removed "
            f"{rep['chars_removed_pct']}% of characters — a rule is eating content"
        )
        assert rep["items_out"] > 0


# =============================================================================
if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
