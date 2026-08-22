"""Tests for regrag.ingestion.sections.

TWO TIERS, same reasoning as test_integration.py:

  SYNTHETIC (fast) — a fake document made of plain text items. No Docling, no
      PDF, runs instantly. This is where the FAILURE cases live, because you
      cannot easily produce a corrupt PDF on demand but you can trivially build
      a chapter with a missing stamp.

  REAL (slow) — the actual LEX PDF through Docling, asserting the four chapters
      and the exact dates read out of the file by hand beforehand. Set
      REGRAG_SKIP_SLOW=1 to skip.

The synthetic tier matters more than it looks. Every guard in sections.py
exists because its absence produces a SILENT wrong answer — a Basel file
indexed with no chapter dates, or a wrong date attached to a chapter. None of
those crash. Tests are the only thing that can observe them.

Run:  pytest tests/test_sections.py -q
  or: python tests/test_sections.py
"""

from __future__ import annotations

import os
from datetime import date

from regrag import config
from regrag.ingestion.sections import Section, SectionError, find_sections, sections_report
from regrag.registry import load

SKIP_SLOW = os.getenv("REGRAG_SKIP_SLOW") == "1"


# =============================================================================
# a minimal stand-in for a DoclingDocument
# =============================================================================


class _Prov:
    def __init__(self, page_no: int):
        self.page_no = page_no


class _Item:
    def __init__(self, text: str, page: int = 1):
        self.text = text
        self.prov = [_Prov(page)]


class _Doc:
    def __init__(self, lines: list[str], page_every: int = 10):
        self.texts = [_Item(t, page=1 + i // page_every) for i, t in enumerate(lines)]


def _lex_like() -> _Doc:
    """Mirrors the REAL Docling output for BASEL_LEX.pdf, item for item.

    An earlier version of this fixture was invented from pdftotext output —
    chapter code alone on a line, date on the line after the phrase. Every
    synthetic test passed against it, and the module was still broken, because
    the fixture encoded the SAME wrong assumption as the code. Only the real
    PDF exposed it.

    Two things Docling actually does, and does INCONSISTENTLY within one file:
      - LEX10 arrives merged with its title; LEX20/30/40 arrive alone
      - the date sits INSIDE the stamp item, not after it
      - each chapter's change-note repeats on both sides of the stamp
    """
    return _Doc(
        ["Basel Committee on Banking Supervision", "LEX", "Large exposures",
         "Contents", "Definitions and application  4"]
        + ["LEX10 Definitions and application",                  # MERGED with title
           "Cross references to LEX30 updated.",
           "Version effective as of 01 Jan 2023",                # date INLINE
           "Cross references to LEX30 updated.", "2/20"]
        + [f"10.{i} body text mentioning LEX30 in passing" for i in range(1, 12)]
        + ["LEX20", "Requirements",                              # code ALONE
           "First version in the format of the consolidated framework.",
           "Version effective as of 15 Dec 2019",
           "First version in the format of the consolidated framework.", "7/20"]
        + [f"20.{i} body text" for i in range(1, 8)]
        + ["LEX30", "Exposure measurement",
           "Reflects changes in market risk requirements published January 2017.",
           "Version effective as of 01 Jan 2023",
           "Reflects changes in market risk requirements published January 2017.", "9/20"]
        + [f"30.{i} body text" for i in range(1, 20)]
        + ["  LEX40  ", "Large exposure rules for G-SIBs",       # indented
           "First version in the format of the consolidated framework.",
           "Version effective as of 15 Dec 2019", "19/20"]
        + [f"40.{i} body text" for i in range(1, 6)]
    )


def _pdftotext_like() -> _Doc:
    """The OTHER layout — stamp phrase and date as separate items.

    Kept deliberately. Docling's grouping is not guaranteed stable across
    versions, and the same BIS content renders this way through other
    extractors. Both shapes must resolve to the same dates.
    """
    return _Doc(
        ["LEX", "Large exposures"]
        + ["LEX10", "Definitions and application", "Version effective as of", "01 Jan 2023"]
        + ["10.1 body"] * 5
        + ["LEX20", "Requirements", "Version effective as of", "15 Dec 2019"]
        + ["20.1 body"] * 5
        + ["LEX30", "Exposure measurement", "Version effective as of", "01 Jan 2023"]
        + ["30.1 body"] * 5
        + ["LEX40", "G-SIB rules", "Version effective as of", "15 Dec 2019"]
        + ["40.1 body"] * 5
    )


def _row(doc_id: str):
    return load().by_id(doc_id)


def _expect(mutate_doc, doc_id: str, *, contains: str) -> None:
    try:
        find_sections(mutate_doc, _row(doc_id))
    except SectionError as e:
        assert contains.lower() in str(e).lower(), f"wrong error: {e}"
        return
    raise AssertionError(f"expected SectionError mentioning {contains!r}")


# =============================================================================
# 1. HAPPY PATH — chaptered and non-chaptered
# =============================================================================


def test_lex_like_document_yields_four_chapters():
    secs = find_sections(_lex_like(), _row("bcbs-lex-consolidated"))

    assert [s.code for s in secs] == ["LEX10", "LEX20", "LEX30", "LEX40"]
    assert [s.effective_from for s in secs] == [
        date(2023, 1, 1), date(2019, 12, 15), date(2023, 1, 1), date(2019, 12, 15)
    ]
    assert all(s.kind == "chapter" for s in secs)


def test_indented_chapter_code_still_matches():
    """LEX40 is indented in the real PDF and LEX10 is not. An intolerant
    pattern would silently drop one chapter in four."""
    secs = find_sections(_lex_like(), _row("bcbs-lex-consolidated"))
    assert "LEX40" in [s.code for s in secs]


def test_chapter_code_merged_with_its_title_still_matches():
    """The bug the real PDF caught. Docling emits 'LEX10 Definitions and
    application' as ONE item while LEX20 arrives alone. Requiring a whole-item
    match dropped LEX10 with no error at all."""
    secs = find_sections(_lex_like(), _row("bcbs-lex-consolidated"))
    assert secs[0].code == "LEX10", "merged code+title was not matched"
    assert secs[0].effective_from == date(2023, 1, 1)


def test_both_extractor_layouts_give_the_same_dates():
    """Date inline with the phrase (Docling) vs date as the next item
    (pdftotext). Docling's grouping is not guaranteed stable across versions,
    so both shapes must resolve identically."""
    row = _row("bcbs-lex-consolidated")
    a = [s.effective_from for s in find_sections(_lex_like(), row)]
    b = [s.effective_from for s in find_sections(_pdftotext_like(), row)]
    assert a == b == [
        date(2023, 1, 1), date(2019, 12, 15), date(2023, 1, 1), date(2019, 12, 15)
    ]


def test_repeated_change_note_between_phrase_and_date_is_skipped():
    """Every BIS chapter repeats its change-note around the stamp. Treating the
    first non-empty item after the phrase as 'must be the date' raised on LEX20
    against the real file."""
    doc = _Doc(
        ["LEX10", "Definitions",
         "Version effective as of",
         "First version in the format of the consolidated framework.",
         "01 Jan 2023"]
        + ["body"] * 5
    )
    secs = find_sections(doc, _row("bcbs-lex-consolidated"))
    assert secs[0].effective_from == date(2023, 1, 1)


def test_non_basel_document_returns_one_whole_document_section():
    doc = _Doc(["SUPERVISORY GUIDANCE ON MODEL RISK MANAGEMENT", "I. INTRODUCTION",
                "Use of models within the banking industry continues to grow."])
    secs = find_sections(doc, _row("sr-26-2-2026"))

    assert len(secs) == 1
    assert secs[0].kind == "whole_document"
    assert secs[0].code is None
    assert secs[0].effective_from is None, (
        "the registry row's own date applies; this must not invent one"
    )


def test_sections_partition_the_document():
    """No gaps, no overlaps. Items before the first chapter are cover/contents
    and are legitimately outside any section."""
    doc = _lex_like()
    secs = find_sections(doc, _row("bcbs-lex-consolidated"))

    first_chapter_idx = next(
        i for i, t in enumerate(doc.texts) if t.text.strip().startswith("LEX10")
    )
    assert secs[0].start_idx == first_chapter_idx  # cover + contents excluded
    for a, b in zip(secs, secs[1:]):
        assert a.end_idx == b.start_idx
    assert secs[-1].end_idx == len(doc.texts)


# =============================================================================
# 2. THE GUARDS — each one prevents a SILENT wrong answer
# =============================================================================


def test_basel_document_with_no_chapters_raises():
    """THE most important test in this file.

    A Basel parse that finds no chapters would otherwise return one
    whole-document section with effective_from=None — byte-identical to a
    correct SR 26-2 result. CRE's 323 pages would index with no chapter dates
    and nothing would report a problem.
    """
    doc = _Doc(["Large exposures", "some body text", "more body text"])
    _expect(doc, "bcbs-lex-consolidated", contains="found none")


def test_chapter_without_a_version_stamp_raises():
    """A missing stamp must not become None. None already means 'living text /
    no fixed edition' in the schema — letting a parse failure share that value
    turns a bug into a metadata fact nobody questions later."""
    doc = _Doc(["LEX10", "Definitions and application"] + ["body"] * 30)
    _expect(doc, "bcbs-lex-consolidated", contains="no 'version effective as of'")


def test_unparseable_date_raises():
    doc = _Doc(["LEX10", "Version effective as of", "sometime in 2023"] + ["body"] * 5)
    _expect(doc, "bcbs-lex-consolidated", contains="no dd mon yyyy date")


def test_cross_reference_with_paragraph_number_does_not_open_a_section():
    """'LEX30.1' and 'LEX301' must not start a section, but 'LEX30 Exposure
    measurement' must. The negative lookahead is what separates them."""
    doc = _Doc(
        ["LEX10 Definitions", "Version effective as of 01 Jan 2023"]
        + ["LEX30.1 is a cross-reference, not a chapter start",
           "LEX301 would be a different chapter entirely"]
        + ["body"] * 5
    )
    secs = find_sections(doc, _row("bcbs-lex-consolidated"))
    assert [s.code for s in secs] == ["LEX10"]


def test_single_chapter_file_with_code_at_end_of_title():
    """SRP32's real Docling layout: 'Supervisory review process SRP32' — the
    standard's name first, the chapter code LAST. A start anchor finds nothing.
    Unanchored search is safe for single-chapter files because there are no
    sibling chapters to confuse a stray mention with."""
    doc = _Doc(
        ["Basel Committee on Banking Supervision", "SRP",
         "Supervisory review process SRP32", "Credit risk",
         "Version effective as of 01 Jan 2023",
         "Cross references updated to take account of the revised standards."]
        + ["32.1 A bank should ensure it has sufficient capital"] * 8
    )
    secs = find_sections(doc, _row("bcbs-srp32-consolidated"))
    assert [s.code for s in secs] == ["SRP32"]
    assert secs[0].effective_from == date(2023, 1, 1)


def test_cross_reference_does_not_open_a_section():
    """SRP32 cites CRE36.53, and Basel chapters cross-reference each other
    constantly. Matching is anchored to the row's OWN volume, so a mention of
    another standard cannot open a spurious section."""
    doc = _Doc(
        ["SRP32", "Credit risk", "Version effective as of", "01 Jan 2023"]
        + ["see CRE36.53 for the treatment", "CRE20", "LEX10 is not relevant here"]
        + ["body"] * 5
    )
    secs = find_sections(doc, _row("bcbs-srp32-consolidated"))
    assert [s.code for s in secs] == ["SRP32"], "a cross-reference opened a section"


def test_stamp_window_does_not_reach_into_the_next_chapter():
    """A window wide enough to cross a chapter boundary would attach the NEXT
    chapter's date — a wrong date, which is worse than a loud failure."""
    gap = config.SECTION_STAMP_WINDOW + 5
    doc = _Doc(
        ["LEX10", "Definitions"] + ["body"] * gap
        + ["LEX20", "Requirements", "Version effective as of", "15 Dec 2019"]
        + ["body"] * 3
    )
    _expect(doc, "bcbs-lex-consolidated", contains="no 'version effective as of'")


# =============================================================================
# 3. THE CONTRACT WITH THE CHUNKER
# =============================================================================


def test_payload_overlay_never_nulls_a_known_date():
    """The chunker merges this ON TOP of Document.payload(). Emitting
    effective_from=None for a non-chaptered document would overwrite the
    registry's real date with the 'living text' signal."""
    whole = Section("whole_document", None, None, 0, 10, 1, 3)
    assert "effective_from" not in whole.payload_overlay()
    assert whole.payload_overlay()["section_kind"] == "whole_document"

    chapter = Section("chapter", "CRE36", date(2023, 1, 1), 0, 10, 5, 9)
    ov = chapter.payload_overlay()
    assert ov["effective_from"] == "2023-01-01"
    assert ov["chapter"] == "CRE36"


def test_chapter_date_overrides_null_document_date():
    """The whole point: Basel rows carry effective_from=None, and the chapter
    supplies it. Verified against the real registry row."""
    row = _row("bcbs-lex-consolidated")
    assert row.effective_from is None, "Basel rows must be chapter-level"

    payload = row.payload()
    assert payload["effective_from"] is None

    secs = find_sections(_lex_like(), row)
    payload.update(secs[0].payload_overlay())
    assert payload["effective_from"] == "2023-01-01"
    assert payload["chapter"] == "LEX10"


# =============================================================================
# 4. REAL PDF — the numbers were read out of the file by hand first
# =============================================================================


def test_real_lex_pdf():
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return

    from regrag.ingestion.parser import parse_pdf

    row = _row("bcbs-lex-consolidated")
    doc = parse_pdf(config.PROJECT_ROOT / row.file)
    rep = sections_report(doc, row)

    print("\n ", rep)
    assert rep["n_sections"] == 4, f"expected 4 LEX chapters, got {rep['n_sections']}"
    assert rep["codes"] == ["LEX10", "LEX20", "LEX30", "LEX40"]
    assert rep["effective_dates"] == [
        "2023-01-01", "2019-12-15", "2023-01-01", "2019-12-15"
    ], "dates must match what pdftotext showed before this module existed"
    assert rep["distinct_dates"] == ["2019-12-15", "2023-01-01"]


def test_real_non_chaptered_pdf():
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return

    from regrag.ingestion.parser import parse_pdf

    row = _row("sr-26-2-2026")
    doc = parse_pdf(config.PROJECT_ROOT / row.file)
    rep = sections_report(doc, row)

    assert rep["kind"] == "whole_document"
    assert rep["n_sections"] == 1
    assert rep["effective_dates"] == [None]


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
