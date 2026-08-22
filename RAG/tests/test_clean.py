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
    def __init__(self, text, label="TEXT", page=1):
        self.text = text
        self.label = _Label(label)
        self.prov = [_Prov(page)]


class _Doc:
    def __init__(self, items):
        self.texts = items


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
# 3. THE CONTRACT WITH THE REST OF THE PIPELINE
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
