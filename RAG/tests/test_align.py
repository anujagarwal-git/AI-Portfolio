"""THE ALIGNER, TESTED WITHOUT A MODEL OR AN INDEX.

Every case here is built from a hand-written passage, so a failure means the
alignment logic changed - never that Ollama was slow or Qdrant was empty.

WHAT IS ACTUALLY BEING PROTECTED
    Not "does it pick a good paragraph" - that was measured on 35 real claims
    and belongs in evaluation/, not in a unit test. What is protected here is
    the set of GUARANTEES a wrong answer would violate quietly:
      * it never invents a locator the passage does not announce,
      * it never returns an empty one,
      * it defers to the shipped citation instead of degrading it,
      * a heading is never mistaken for a chapter prefix.
    Each of those was a real defect on 2026-09-02, found by reading output.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from regrag.generation import align


@dataclass
class FakeHit:
    payload: dict


@dataclass
class FakePassage:
    text: str
    children: list = field(default_factory=list)
    heading: str = ""


def passage(text: str, *locators: str) -> FakePassage:
    return FakePassage(text, [FakeHit({"locator": loc, "text": ""})
                              for loc in locators])


CRE53 = passage(
    "53.50 The bank must have a rigorous programme of backtesting in place, "
    "comparing realised outcomes against model forecasts before the model is "
    "put into production.\n\n"
    "53.51 An initial validation must be carried out by a party independent of "
    "the developers, and the results documented for the supervisor.\n\n"
    "53.55 Stress testing of the portfolio must cover concentration risk and "
    "illiquidity under stressed market conditions.\n\n"
    "53.56 The bank must review its model inputs at least annually.",
    "CRE53.56")


# =========================================================== paragraph_units
def test_a_paragraph_takes_the_locator_it_announces():
    units = align.paragraph_units(CRE53)
    assert [u.locator for u in units] == [
        "CRE53.50", "CRE53.51", "CRE53.55", "CRE53.56"]


def test_the_chapter_prefix_comes_from_a_matched_child_not_the_body():
    # The body prints "53.50"; a citation must read "CRE53.50". Nothing in the
    # passage text carries the chapter code.
    assert align.paragraph_units(CRE53)[0].locator.startswith("CRE")


def test_a_heading_in_the_locator_field_is_NOT_a_prefix():
    # 2026-09-02: a chunk with no paragraph number carries its HEADING in the
    # locator field, and taking the leading word produced `Investments30.22` -
    # a reference that looks real and cannot be looked up. The guarantee is
    # that no locator is ever FABRICATED, not that the passage ends up
    # unlabelled: with no usable prefix the units fall back to the children,
    # which is what shipped anyway.
    assert not align.RX_PREFIX.fullmatch("Investments in own shares (treasury)")
    assert align.RX_PREFIX.fullmatch("CAP30.22").group(1) == "CAP"

    p = passage("30.22 Investments in own shares must be deducted from CET1 "
                "capital in the calculation of regulatory adjustments.",
                "Investments in own shares (treasury stock)")
    assert all(not u.locator.startswith("Investments3")
               for u in align.paragraph_units(p))


def test_short_fragments_and_headings_are_not_units():
    p = passage("Section header\n\n"
                "30.1 A bank must at all times meet the minimum capital "
                "requirements set out in this chapter of the framework.",
                "CAP30.1")
    assert len(align.paragraph_units(p)) == 1


def test_a_passage_with_no_markers_falls_back_to_its_matched_children():
    # SR letters, BCBS papers and FAQ bins number nothing. Without this the
    # aligner would have no units at all and could never move a citation.
    p = FakePassage("Model risk management should be commensurate with the "
                    "bank's exposure to model risk and the complexity of use.",
                    [FakeHit({"locator": "Footnotes", "text": "some text"})])
    units = align.paragraph_units(p)
    assert len(units) == 1 and units[0].locator == "Footnotes"


# =========================================================== align
def test_it_recovers_the_paragraph_the_claim_came_from():
    # THE WHOLE POINT. Retrieval matched CRE53.56; the claim is CRE53.51's.
    a = align.align(
        "An initial validation must be carried out by a party independent of "
        "the developers and documented for the supervisor.",
        align.paragraph_units(CRE53), current="CRE53.56")
    assert a.verdict == "MATCH" and a.locator == "CRE53.51"


def test_below_the_floor_it_KEEPS_the_shipped_locator():
    # 2026-09-02: the fallback used to return a RANGE and replaced a correct
    # CAP30.1 with CAP30.1-CAP30.2. Deferring beats degrading.
    a = align.align("Basel III was agreed by the Committee in 2010.",
                    align.paragraph_units(CRE53), current="CRE53.56")
    assert a.verdict == "FLOOR" and a.locator == "CRE53.56"


def test_it_never_returns_an_empty_locator():
    # "(no locator)" shipped where "Footnotes" had. Useless-but-stable wins.
    p = passage("The bank must maintain a rigorous programme of backtesting "
                "comparing realised outcomes against model forecasts.", "")
    a = align.align("The bank must maintain a rigorous backtesting programme "
                    "comparing realised outcomes against forecasts.",
                    align.paragraph_units(p), current="Model risk chapter")
    assert a.locator == "Model risk chapter"
    assert a.verdict in {"NOLOC", "FLOOR", "TIE"}


def test_two_indistinguishable_paragraphs_are_not_chosen_between():
    p = passage(
        "30.1 The bank must hold capital against credit risk exposures at all "
        "times under the standardised approach.\n\n"
        "30.2 The bank must hold capital against credit risk exposures at all "
        "times under the standardised approach.",
        "CAP30.9")
    a = align.align("A bank must hold capital against credit risk exposures "
                    "under the standardised approach.",
                    align.paragraph_units(p), current="CAP30.9")
    assert a.verdict == "TIE" and a.locator == "CAP30.9"


def test_no_units_at_all_defers_rather_than_raising():
    a = align.align("anything", [], current="CAP30.1")
    assert a.verdict == "EMPTY" and a.locator == "CAP30.1"


def test_only_MATCH_counts_as_moved():
    assert align.Alignment("MATCH", "x", 0.9, 0.1).moved
    for v in ("FLOOR", "TIE", "NOLOC", "EMPTY"):
        assert not align.Alignment(v, "x", 0.9, 0.1).moved


# =========================================================== scoring
def test_rare_words_outweigh_boilerplate():
    # "the bank must" is in every paragraph; "backtesting" is in one. Without
    # the idf weighting the common phrasing would drown the signal.
    units = align.paragraph_units(CRE53)
    idf = align.build_idf([u.text for u in units])
    assert idf["backtesting"] > idf["model"]


def test_similarity_is_zero_for_a_claim_with_no_content_words():
    idf = align.build_idf(["some text here about capital"])
    assert align.similarity("of the and it", "some text here", idf) == 0.0


def test_the_floor_is_a_knob_not_a_constant():
    # Written against the score the aligner ACTUALLY produces rather than a
    # guessed number: the first draft asserted FLOOR at 0.99 on a claim that
    # scores 1.00, and the test was wrong, not the code.
    claim = "An initial validation must be carried out by an independent party."
    units = align.paragraph_units(CRE53)
    loose = align.align(claim, units, current="CRE53.56", floor=0.0)
    assert loose.verdict == "MATCH"
    tight = align.align(claim, units, current="CRE53.56",
                        floor=loose.best + 0.01)
    assert tight.verdict == "FLOOR" and tight.locator == "CRE53.56"
