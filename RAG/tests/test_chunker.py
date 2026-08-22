"""Tests for regrag.ingestion.chunker — parent boundary detection.

The fixtures are copied from real cached parses, not invented. The sections.py
experience made the reason plain: a fixture built from the same mental model as
the code only tests that they agree, never that the model is right.

The centrepiece is the (i) ambiguity. In 12 CFR 225.8 the single-letter markers
appear in this real order:

    a b i c d e i i i v i f i g h i i i i i i i j i i i k

a-h, j and k appear once each — genuine top-level paragraphs. "i" appears 16
times and "v" once: roman numerals nested three levels down, textually
identical to letters. A naive rule would have created 16 spurious parents in
one 12-page regulation.

Run:  pytest tests/test_chunker.py -q
  or: python tests/test_chunker.py
"""

from __future__ import annotations

import os

from regrag import config
from regrag.ingestion.chunker import (
    ParentSpan, chunk_document, chunk_report, find_parents, parents_report,
)
from regrag.registry import load

SKIP_SLOW = os.getenv("REGRAG_SKIP_SLOW") == "1"


class _Item:
    def __init__(self, text, label="LIST_ITEM", page=1, is_footnote=False):
        self.text = text
        self.label = label
        self.page = page
        self.is_footnote = is_footnote


class _Section:
    """Stands in for a sections.py Section."""

    def __init__(self, code, start, end, eff="2023-01-01"):
        self.code, self.start_idx, self.end_idx, self._eff = code, start, end, eff

    def payload_overlay(self):
        d = {"chapter": self.code, "section_kind": "chapter" if self.code else "whole_document"}
        if self._eff:
            d["effective_from"] = self._eff
        return d


class _Cleaned:
    def __init__(self, items):
        self.texts = items


def _items(pairs):
    return [_Item(t, l) for t, l in pairs]


def _row(doc_id):
    return load().by_id(doc_id)


def _cfr_225_8_like():
    """The real 225.8 shape: § header, then (a)..(k) interleaved with roman
    numerals at depth three."""
    return _items([
        ("This content is from the eCFR and is authoritative but unofficial.", "TEXT"),
        ("§ 225.8 Capital planning and stress capital buffer requirement.", "SECTION_HEADER"),
        ("(a) Purpose. This section establishes capital planning requirements.", "LIST_ITEM"),
        ("(b) Scope and reservation of authority -", "LIST_ITEM"),
        ("(1) Applicability. Except as provided in paragraph (c) of this section", "LIST_ITEM"),
        ("(i) Any top-tier bank holding company domiciled in the United States", "LIST_ITEM"),
        ("(ii) Any other bank holding company domiciled in the United States", "LIST_ITEM"),
        ("(iii) Any U.S. intermediate holding company subject to this section", "LIST_ITEM"),
        ("(iv) Any nonbank financial company supervised by the Board", "LIST_ITEM"),
        ("(v) Any covered savings and loan holding company", "LIST_ITEM"),
        ("(2) Average total consolidated assets. For purposes of this section", "LIST_ITEM"),
        ("(c) Transition. A company that becomes subject to this section", "LIST_ITEM"),
        ("(d) Definitions. For purposes of this section, the following apply", "LIST_ITEM"),
    ])


# =============================================================================
# 1. THE (i) AMBIGUITY — the reason this module is not three lines long
# =============================================================================


def test_roman_numerals_do_not_open_parents():
    """(i) through (v) here are nested under (b)(1), not top-level paragraphs.
    Accepting them would have produced 5 spurious parents in 13 items."""
    spans = find_parents(_cfr_225_8_like(), 0, 13, _row("cfr-12-225-8"))
    markers = [s.marker for s in spans if s.marker]

    assert markers == ["(a)", "(b)", "(c)", "(d)"], f"got {markers}"
    for bad in ("(i)", "(ii)", "(iii)", "(iv)", "(v)"):
        assert bad not in markers, f"{bad} opened a parent — roman numeral treated as a letter"


def test_letter_i_is_accepted_when_it_follows_h():
    """A genuine top-level (i) must still work — it is only rejected when out
    of sequence."""
    items = _items([(f"({c}) Paragraph {c} text here.", "LIST_ITEM") for c in "abcdefghij"])
    spans = find_parents(items, 0, len(items), _row("cfr-12-225-8"))

    assert [s.marker for s in spans] == [f"({c})" for c in "abcdefghij"]


def test_digit_markers_do_not_open_parents():
    """(1)/(2) are one level below (a)/(b) — they are children, not parents."""
    items = _items([
        ("(a) Purpose. This section establishes requirements.", "LIST_ITEM"),
        ("(1) A bank holding company that is a U.S. bank holding company;", "LIST_ITEM"),
        ("(2) A U.S. intermediate holding company; or", "LIST_ITEM"),
        ("(b) Scope of this section.", "LIST_ITEM"),
    ])
    spans = find_parents(items, 0, len(items), _row("cfr-12-225-8"))

    assert [s.marker for s in spans] == ["(a)", "(b)"]
    assert spans[0].n_items == 3, "the (1)/(2) items must fall INSIDE parent (a)"


def test_section_sign_restarts_the_enumeration():
    """Every § restarts at (a). Without a reset, the second section's (a) would
    be rejected as out of sequence and its text would fold into (b)."""
    items = _items([
        ("§ 252.13 Risk committee requirement.", "SECTION_HEADER"),
        ("(a) First provision of 252.13.", "LIST_ITEM"),
        ("(b) Second provision of 252.13.", "LIST_ITEM"),
        ("§ 252.14 Liquidity requirement.", "SECTION_HEADER"),
        ("(a) First provision of 252.14.", "LIST_ITEM"),
        ("(b) Second provision of 252.14.", "LIST_ITEM"),
    ])
    spans = find_parents(items, 0, len(items), _row("cfr-12-252"))

    assert [s.marker for s in spans] == [None, "(a)", "(b)", None, "(a)", "(b)"]
    assert [s.label for s in spans].count("heading") == 2


# =============================================================================
# 2. THE RULE IS CHOSEN FROM THE REGISTRY
# =============================================================================


def test_basel_uses_headings_and_ignores_enumeration():
    """CRE has 913 "(N)" sub-paragraph markers. Under the enumeration rule they
    would shatter every chapter; Basel writes real headings, so it uses those."""
    items = _items([
        ("Standardised approach: individual exposures", "SECTION_HEADER"),
        ("(1) The standardised approach assigns standardised risk weights.", "LIST_ITEM"),
        ("(2) To determine the risk weights in the standardised approach", "LIST_ITEM"),
        ("Definition of exposure value", "SECTION_HEADER"),
        ("(1) Equity investments in funds are addressed in CRE60.", "LIST_ITEM"),
    ])
    spans = find_parents(items, 0, len(items), _row("bcbs-cre-consolidated"))

    assert len(spans) == 2, "Basel must split on headings only"
    assert all(s.label == "heading" for s in spans)
    assert all(s.marker is None for s in spans)


def test_same_items_split_differently_per_document_class():
    """The point of the whole design: identical text, different rule."""
    items = _items([
        ("(a) First provision.", "LIST_ITEM"),
        ("(b) Second provision.", "LIST_ITEM"),
        ("(c) Third provision.", "LIST_ITEM"),
    ])
    cfr = find_parents(items, 0, 3, _row("cfr-12-225-8"))
    basel = find_parents(items, 0, 3, _row("bcbs-cre-consolidated"))

    assert len(cfr) == 3, "CFR splits on enumeration"
    assert len(basel) == 1, "Basel has no heading here, so it stays one span"


def test_sr_letters_use_headings():
    items = _items([
        ("III. OVERVIEW OF MODEL RISK MANAGEMENT", "SECTION_HEADER"),
        ("For the purposes of this guidance, the term model refers to", "TEXT"),
        ("IV. MODEL DEVELOPMENT, IMPLEMENTATION, AND USE", "SECTION_HEADER"),
        ("Model development is not a straightforward or routine exercise.", "TEXT"),
    ])
    spans = find_parents(items, 0, len(items), _row("sr-26-2-2026"))
    assert len(spans) == 2 and all(s.label == "heading" for s in spans)


# =============================================================================
# 3. COVERAGE — nothing may be lost
# =============================================================================


def test_text_before_the_first_boundary_is_kept():
    """The eCFR preamble sits above the first § and would otherwise vanish."""
    items = _cfr_225_8_like()
    spans = find_parents(items, 0, len(items), _row("cfr-12-225-8"))

    assert spans[0].start_idx == 0
    assert spans[0].label == "chapter_start"
    assert "authoritative but unofficial" in spans[0].heading


def test_parents_partition_the_span_exactly():
    items = _cfr_225_8_like()
    spans = find_parents(items, 0, len(items), _row("cfr-12-225-8"))

    assert spans[0].start_idx == 0
    assert spans[-1].end_idx == len(items)
    for a, b in zip(spans, spans[1:]):
        assert a.end_idx == b.start_idx, "gap or overlap between parents"


def test_parents_never_straddle_a_chapter():
    """find_parents is called per chapter, so a parent cannot span two chapters
    with different effective dates — which is what sections.py established."""
    items = _items([(f"({c}) text", "LIST_ITEM") for c in "abcd"])
    first = find_parents(items, 0, 2, _row("cfr-12-225-8"))
    second = find_parents(items, 2, 4, _row("cfr-12-225-8"))

    assert first[-1].end_idx == 2 and second[0].start_idx == 2


# =============================================================================
# 4. CHILDREN — what gets embedded
# =============================================================================


def test_noise_is_dropped_not_merged():
    """'FAQ' (76x), 'FAQ1' (48x), 'Footnotes' (31x) are real corpus items that
    carry no meaning alone.

    An earlier version MERGED them into the following paragraph, which polluted
    a real chunk with a meaningless prefix. Dropping costs nothing — there is
    nothing in 'FAQ' to lose — and it removed 36% of all merges corpus-wide.
    """
    items = _items([
        ("Standardised approach", "SECTION_HEADER"),
        ("FAQ", "TEXT"),
        ("FAQ1", "LIST_ITEM"),
        ("Should banks assess climate-related financial risks as part of the due "
         "diligence analyses with respect to counterparty creditworthiness?", "TEXT"),
    ])
    chunks, _ = chunk_document(_Cleaned(items), [_Section("CRE20", 0, 4)],
                               _row("bcbs-cre-consolidated"))

    assert len(chunks) == 1
    assert not chunks[0].text.startswith("FAQ"), "noise was merged instead of dropped"
    assert chunks[0].text.startswith("Should banks assess")


def test_page_furniture_that_escaped_clean_is_dropped_here():
    """'Page 3' reached a chunk in SR 11-7 because Docling labelled it TEXT, so
    clean.py's label-based drop never saw it. It then merged into the following
    sentence: 'Page 3 from those calculations. Finally, the quality of model
    outputs...'"""
    items = _items([
        ("Heading", "SECTION_HEADER"),
        ("Page 3", "TEXT"),
        ("The quality of model outputs depends on the quality of input data and "
         "assumptions, and errors will lead to inaccurate results.", "TEXT"),
    ])
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 3, eff=None)],
                               _row("sr-11-7-2011"))

    assert len(chunks) == 1
    assert not chunks[0].text.startswith("Page 3")


def test_contents_page_numbers_are_dropped():
    """SR 11-7 produced '1 2 3 5 9 16 21 This guidance describes...' — a run of
    bare table-of-contents numbers glued onto the opening paragraph."""
    items = _items([("Heading", "SECTION_HEADER")]
                   + [(n, "TEXT") for n in ("1", "2", "3", "5", "9", "16", "21")]
                   + [("This guidance describes the key aspects of effective model "
                       "risk management and supervisory expectations.", "TEXT")])
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 9, eff=None)],
                               _row("sr-11-7-2011"))

    assert len(chunks) == 1
    assert chunks[0].text.startswith("This guidance describes")


def test_split_sentence_is_repaired_when_both_signals_agree():
    """Ends without terminal punctuation AND the next item starts lowercase —
    a genuine split across a column or page break. 3.2% of adjacent pairs."""
    items = _items([
        ("Heading", "SECTION_HEADER"),
        ("This guidance is expected to be most relevant to banking organizations "
         "with over $30", "TEXT"),
        ("billion in total assets, measured on a consolidated basis over the four "
         "most recent quarters.", "TEXT"),
    ])
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 3, eff=None)],
                               _row("sr-26-2-2026"))

    assert len(chunks) == 1, "the split sentence was not rejoined"
    assert "over $30 billion in total assets" in chunks[0].text, chunks[0].text


def test_complete_sentences_are_never_merged():
    """Signal 1 blocks it. Two complete short sentences are two good retrieval
    keys — gluing them together loses precision for no gain."""
    items = _items([
        ("Heading", "SECTION_HEADER"),
        ("A bank must maintain adequate capital.", "TEXT"),
        ("Supervisors will review the assessment annually.", "TEXT"),
    ])
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 3, eff=None)],
                               _row("sr-26-2-2026"))

    assert len(chunks) == 2


def test_open_ended_item_is_not_merged_when_next_starts_uppercase():
    """The 23% ambiguous case, and the one that caused the damage. Headings,
    list items and formulas legitimately take no full stop — merging on signal 1
    alone fused unrelated content."""
    items = _items([
        ("Heading", "SECTION_HEADER"),
        ("External Credit Risk Assessment Approach (ECRA)", "TEXT"),
        ("Banks must apply the risk weights set out in the table below.", "TEXT"),
    ])
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 3, eff=None)],
                               _row("bcbs-cre-consolidated"))

    assert len(chunks) == 2, "an open-ended heading absorbed the following sentence"


def test_sentence_is_repaired_across_an_interleaved_footnote():
    """The real SR 26-2 case. The sentence continues on the next page, but a
    footnote sits between the two halves in reading order:

        [15] body      '...banking organizations with over $30'
        [16] footnote  '1 See 12 CFR Part 4, Subpart F...'
        [18] body      'billion in total assets...'

    Read in raw order the footnote blocks the repair twice — it does not start
    lowercase, and it ends with a full stop. The result was a chunk beginning
    mid-sentence with the $30 BILLION THRESHOLD CUT IN HALF.

    Body and footnotes are separate streams, so the body sentence rejoins.
    """
    items = [
        _Item("Heading", "SECTION_HEADER"),
        _Item("This guidance is expected to be most relevant to banking "
              "organizations with over $30", "TEXT", page=2),
        _Item("1 See 12 CFR Part 4, Subpart F, Appendix A (OCC); 12 CFR Part 262, "
              "Appendix A (Board).", "FOOTNOTE", page=2, is_footnote=True),
        _Item("billion in total assets, measured on a consolidated basis.", "TEXT", page=3),
    ]
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 4, eff=None)],
                               _row("sr-26-2-2026"))

    body = [c for c in chunks if not c.is_footnote]
    assert len(body) == 1, f"body sentence not rejoined: {[c.text[:40] for c in body]}"
    assert "over $30 billion in total assets" in body[0].text, body[0].text
    assert not any(c.text[:1].islower() for c in chunks), "a chunk still starts mid-sentence"


def test_footnote_and_body_are_never_merged():
    """Structurally impossible now — they are separate streams. Previously 81
    cases fused a footnote onto a paragraph because one of them was short."""
    items = [
        _Item("Heading", "SECTION_HEADER"),
        _Item("1.5. The policy comes into effect on Friday 17 May 2024.", "TEXT"),
        _Item("1 While the scope of this SS includes banks, building societies and "
              "designated investment firms, the term 'banks' is used throughout.",
              "FOOTNOTE", is_footnote=True),
    ]
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 3, eff=None)],
                               _row("pra-ss1-23-2023"))

    assert len(chunks) == 2
    assert sum(c.is_footnote for c in chunks) == 1
    assert not any("17 May 2024" in c.text and "scope of this SS" in c.text for c in chunks)


def test_children_stay_in_reading_order():
    """Splitting into streams must not reorder the parent's children."""
    items = [
        _Item("Heading", "SECTION_HEADER"),
        _Item("First body paragraph of the provision, complete and self-contained.", "TEXT"),
        _Item("1 A footnote attached to the first paragraph.", "FOOTNOTE", is_footnote=True),
        _Item("Second body paragraph, also complete and self-contained.", "TEXT"),
    ]
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 4, eff=None)],
                               _row("sr-26-2-2026"))

    assert [c.payload["orig_idx"] for c in chunks] == [1, 2, 3]


def test_unrelated_footnotes_are_not_fused():
    """'1 Basel Committee, Enhancements...' + '2 MIS in this context refers...'
    were fused under the size rule. Both end with a full stop, so signal 1 now
    keeps them apart."""
    items = _items([
        ("Footnotes", "SECTION_HEADER"),
        ("1 Basel Committee, Enhancements to the Basel II framework (July 2009) "
         "at www.bis.org/publ/bcbs158.pdf.", "LIST_ITEM"),
        ("2 MIS in this context refers to risk management information.", "LIST_ITEM"),
    ])
    chunks, _ = chunk_document(_Cleaned(items), [_Section("CRE20", 0, 3)],
                               _row("bcbs-cre-consolidated"))

    assert len(chunks) == 2, "two unrelated footnotes were fused"


def test_normal_children_are_left_alone():
    """Merging only rescues fragments. Parentdoc depends on a SMALL, PRECISE
    retrieval key, so ordinary paragraphs must not be glued together."""
    para = "A" * 300
    items = _items([("Heading", "SECTION_HEADER"), (para, "TEXT"), (para, "TEXT")])
    chunks, _ = chunk_document(_Cleaned(items), [_Section("CRE20", 0, 3)],
                               _row("bcbs-cre-consolidated"))

    assert len(chunks) == 2, "normal-sized paragraphs were merged"


def test_oversized_child_is_split_at_sentence_ends():
    """6 children in the corpus exceed bge-small's limit and would be SILENTLY
    truncated. A truncated requirement is not a weaker requirement — it is a
    different statement."""
    long_text = " ".join(f"Sentence number {i} of the requirement text." for i in range(90))
    items = _items([("Heading", "SECTION_HEADER"), (long_text, "TEXT")])
    chunks, _ = chunk_document(_Cleaned(items), [_Section("CRE20", 0, 2)],
                               _row("bcbs-cre-consolidated"))

    assert len(chunks) > 1, "oversized child was not split"
    assert all(len(c.text) <= config.CHILD_MAX_CHARS for c in chunks)
    assert all(c.text.rstrip().endswith(".") for c in chunks), "split mid-sentence"


def test_basel_locator_matches_bis_citation_style():
    """BIS cites 'CRE36.122'. The chapter gives 'CRE36', the paragraph text
    opens '36.122' — so the prefix comes from the chapter, the number from the
    text."""
    items = _items([
        ("Section 8: validation of internal estimates", "SECTION_HEADER"),
        ("36.122 Banks must have a robust system in place to validate the accuracy "
         "and consistency of rating systems and processes.", "LIST_ITEM"),
    ])
    chunks, _ = chunk_document(_Cleaned(items), [_Section("CRE36", 0, 2)],
                               _row("bcbs-cre-consolidated"))

    assert chunks[0].locator == "CRE36.122"
    assert chunks[0].payload["locator"] == "CRE36.122"


def test_locator_works_without_a_chapter_code():
    """Every publisher numbers provisions its own way, and each form IS the
    citation. Requiring a Basel-style chapter code meant every other document
    scored 0% citable — 51% of BCBS 239, 43% of PAP, all of IFRS 9 — even with
    the paragraph number sitting at the start of the text.
    """
    cases = [
        ("pra-ss1-23-2023", "1.5 The policy comes into effect on Friday 17 May 2024.", "1.5"),
        ("bcbs-239-2013",
         "21. Risk data aggregation capabilities and risk reporting practices are "
         "considered separately in this paper.", "21"),
        ("ifrs9-extract",
         "5.5.1 An entity shall recognise a loss allowance for expected credit "
         "losses on a financial asset measured at amortised cost.", "5.5.1"),
        ("bcbs-pap-d403",
         "3. The definitions promote harmonisation in the measurement and "
         "application of asset quality measures.", "3"),
    ]
    for doc_id, text, expected in cases:
        chunks, _ = chunk_document(
            _Cleaned(_items([("Heading", "SECTION_HEADER"), (text, "TEXT")])),
            [_Section(None, 0, 2, eff=None)], _row(doc_id))
        assert chunks[0].locator == expected, \
            f"{doc_id}: got {chunks[0].locator!r}, expected {expected!r}"


def test_basel_still_prefixes_with_its_chapter():
    """The new patterns must not break BIS's own form: chapter CRE36 plus
    paragraph 36.122 is cited 'CRE36.122', not '36.122'."""
    items = _items([
        ("Section 8: validation of internal estimates", "SECTION_HEADER"),
        ("36.122 Banks must have a robust system in place to validate.", "LIST_ITEM"),
    ])
    chunks, _ = chunk_document(_Cleaned(items), [_Section("CRE36", 0, 2)],
                               _row("bcbs-cre-consolidated"))
    assert chunks[0].locator == "CRE36.122"


def test_unnumbered_document_falls_back_to_its_section_heading():
    """SR 26-2, SR 11-7 and d450 carry NO paragraph numbers — 0 of 41 and 0 of
    117 body items. Their finest citable unit is the section, which is exactly
    how they are cited in practice: 'SR 26-2, III. Overview of Model Risk'."""
    items = _items([
        ("III. OVERVIEW OF MODEL RISK AND MODEL RISK MANAGEMENT", "SECTION_HEADER"),
        ("Models are simplified representations of real-world relationships among "
         "observed characteristics, values, and events.", "TEXT"),
    ])
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 2, eff=None)],
                               _row("sr-26-2-2026"))

    assert chunks[0].locator == "III. OVERVIEW OF MODEL RISK AND MODEL RISK MANAGEMENT"


def test_cfr_locator_combines_parent_and_child_markers():
    """'(b)(1)' — the form an examiner cites. Parent supplies (b), child (1)."""
    items = _items([
        ("(a) Purpose. This section establishes capital planning and prior notice "
         "and approval requirements for capital distributions.", "LIST_ITEM"),
        ("(b) Scope and reservation of authority -", "LIST_ITEM"),
        ("(1) Applicability. Except as provided in paragraph (c) of this section, "
         "this section applies to any top-tier bank holding company.", "LIST_ITEM"),
    ])
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 3, eff=None)],
                               _row("cfr-12-225-8"))
    locs = [c.locator for c in chunks]

    assert "(b)(1)" in locs, locs
    assert "(a)" in locs, locs


def test_enumeration_heading_line_is_not_embedded_as_its_own_child():
    """'(b) Scope and reservation of authority -' is a HEADING that happens to
    carry the label LIST_ITEM. For Basel the equivalent SECTION_HEADER is never
    embedded; the CFR must behave the same, or the 39-char stub merges into the
    provision below and drags its locator from (b)(1) down to (b).

    The text is not lost — the PARENT record still holds it, and the parent is
    what gets delivered.
    """
    items = _items([
        ("(a) Purpose. This section establishes capital planning requirements for "
         "certain bank holding companies operating in the United States.", "LIST_ITEM"),
        ("(b) Scope and reservation of authority -", "LIST_ITEM"),
        ("(1) Applicability. Except as provided in paragraph (c) of this section, "
         "this section applies to any top-tier bank holding company.", "LIST_ITEM"),
    ])
    chunks, parents = chunk_document(_Cleaned(items), [_Section(None, 0, 3, eff=None)],
                                     _row("cfr-12-225-8"))

    assert not any(c.text.startswith("(b) Scope") for c in chunks), \
        "the marker line was embedded as a child"
    assert any("(b) Scope" in p.text for p in parents), \
        "the marker line vanished — it must survive in the parent"


def test_chunk_carries_registry_and_chapter_metadata():
    """Every chunk must know what document it is, whether that document is in
    force, and WHEN its chapter took effect — the chapter date overriding the
    null document-level date."""
    items = _items([("Heading", "SECTION_HEADER"), ("36.1 Body text of the provision.", "LIST_ITEM")])
    row = _row("bcbs-cre-consolidated")
    assert row.effective_from is None, "Basel rows are chapter-level"

    chunks, _ = chunk_document(_Cleaned(items), [_Section("CRE36", 0, 2)], row)
    p = chunks[0].payload

    assert p["doc_id"] == "bcbs-cre-consolidated"
    assert p["status"] == "in_force"
    assert p["authority_rank"] == 2
    assert p["is_requirement_source"] is True
    assert p["chapter"] == "CRE36"
    assert p["effective_from"] == "2023-01-01", "chapter date did not override the null doc date"
    assert "notes" not in p and "file" not in p


def test_parent_text_is_stored_once_not_per_child():
    """A 4,000-char parent with 20 children would be duplicated 20 times if the
    text lived in the payload. Chunks carry a parent_id; parents are separate."""
    items = _items([("Heading", "SECTION_HEADER")]
                   + [(f"{i}.1 " + "B" * 200, "LIST_ITEM") for i in range(6)])
    chunks, parents = chunk_document(_Cleaned(items), [_Section("CRE20", 0, 7)],
                                     _row("bcbs-cre-consolidated"))

    assert len(parents) == 1 and len(chunks) == 6
    assert all(c.parent_id == parents[0].parent_id for c in chunks)
    assert all("parent_text" not in c.payload for c in chunks)
    assert parents[0].n_children == 6


def test_footnote_flag_survives_into_the_chunk():
    items = [_Item("Heading", "SECTION_HEADER"),
             _Item("See 84 Fed. Reg. 59032 (November 1, 2019) for more information on the "
                   "tailoring framework and its application.", "FOOTNOTE", is_footnote=True)]
    chunks, _ = chunk_document(_Cleaned(items), [_Section(None, 0, 2, eff=None)],
                               _row("sr-15-19-2015r2021"))

    assert chunks[0].is_footnote is True
    assert chunks[0].payload["is_footnote"] is True


def test_chunk_ids_are_unique_and_traceable():
    items = _items([("Heading", "SECTION_HEADER")]
                   + [(f"{i}.1 " + "C" * 200, "LIST_ITEM") for i in range(4)])
    chunks, _ = chunk_document(_Cleaned(items), [_Section("CRE20", 0, 5)],
                               _row("bcbs-cre-consolidated"))

    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert all(c.chunk_id.startswith(c.parent_id) for c in chunks)
    assert all(c.payload["orig_idx"] is not None for c in chunks)


# =============================================================================
# 5. REAL DOCUMENTS
# =============================================================================


def test_real_cfr_225_8_gains_real_structure():
    """Before: 3 headings for 209 items, one parent holding 34,245 chars.
    After: the enumeration should give a usable number of bounded provisions."""
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return

    from regrag.ingestion.cache import parse_cached
    from regrag.ingestion.clean import clean_document
    from regrag.ingestion.sections import find_sections

    row = _row("cfr-12-225-8")
    doc = parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
    cleaned = clean_document(doc, row)
    rep = parents_report(cleaned, find_sections(cleaned, row), row)

    print(f"\n  {rep}")
    assert rep["enumerated"] is True
    assert rep["n_parents"] >= 8, f"expected the (a)-(k) provisions, got {rep['n_parents']}"
    assert rep["max"] < 34_245, "the single 34k parent should be gone"


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
