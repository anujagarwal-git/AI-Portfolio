"""THE TWO ZERO-COST PRECISION FIXES FROM THE 2026-09-03 RELEVANCE STUDY.

Both were built because the study measured junk that costs nothing to remove -
no model call, no threshold, and no relevant material lost. That is what makes
them different from every score-tuning experiment in this project, all of which
traded precision against recall along one flat curve.

WHAT IS PROTECTED HERE

1. AN ALIAS OWNED BY TWO DOCUMENTS IDENTIFIES NEITHER.
   `_aliases_for` shortens "SR 15-18" to "sr 15" and "SR 15-19" to "sr 15" as
   well. So "What does SR 15-18 expect...?" matched BOTH, `_documents_in`
   returned two doc_ids, planner rule 1 fired, and retrieval searched a
   document the user never asked for. Measured: 8 passages across q27/q28,
   17,133 characters, NOT ONE judged usable.

   The regression that matters most is the OTHER direction - over-correcting
   by dropping the shortening itself would make "sr 11" stop resolving to
   SR 11-7, which is a legitimate handle no other document answers to. Both
   directions are asserted.

2. A PASSAGE WHOLLY INSIDE ANOTHER FROM THE SAME DOCUMENT IS WASTE.
   BCBS 239 Principle 3 arrived as a 268-char chunk AND inside the 2,300-char
   section; Principle 5 as 610 inside 1,970. Different parent_ids, so pass 1's
   de-duplication cannot see them.

   THE SAME-DOCUMENT RESTRICTION IS THE WHOLE DESIGN, not an optimisation.
   SR 15-18 and SR 15-19 are near-identical twins by construction. On "how
   does SR 15-18 differ from SR 15-19?" both copies ARE the answer, and a
   cross-document collapse would delete half the evidence for the comparison
   the user asked for. That case is asserted explicitly.

WHAT WAS MEASURED AND REJECTED, so nobody rebuilds it
   A same-document SIMILARITY threshold. On full text the only pairs above
   0.90 were two true duplicates (0.941, 0.921); the nearest pair below was
   0.807 - the 12 CFR SPV look-through provisions, one for a covered company
   and one for a covered FOREIGN entity. Near-identical wording, different
   rules. A cut in that band has 0.05 of headroom and three data points behind
   it, and would have saved roughly 0.16 seconds of prompt reading per
   question. Deleting a regulatory provision to save 0.16s is a bad trade.
   Containment needs no threshold and cannot lose information, so only
   containment was built.
"""
from __future__ import annotations

import pytest

from regrag.registry import load as load_registry
from regrag.retrieval import planner
from regrag.retrieval.search import Passage, _contained_in


# ---------------------------------------------------------------------------
# FIX 1 — ambiguous aliases
# ---------------------------------------------------------------------------
def test_naming_sr_15_18_finds_only_sr_15_18():
    found = planner._documents_in(
        "what does sr 15-18 expect of capital planning at large firms?")
    assert [sn for sn, _a in found.values()] == ["SR 15-18"]


def test_naming_sr_15_19_finds_only_sr_15_19():
    found = planner._documents_in(
        "what does sr 15-19 say about capital planning at smaller firms?")
    assert [sn for sn, _a in found.values()] == ["SR 15-19"]


def test_the_ambiguous_alias_itself_now_matches_nothing():
    """'sr 15' is a handle for two documents, so it identifies neither.

    Asking for it should find NO document rather than guessing one - a guess
    here is what produced the 17,133 characters of off-target junk."""
    assert planner._documents_in("what does sr 15 say about capital planning?") == {}


def test_the_shortening_still_works_where_it_is_UNAMBIGUOUS():
    """The over-correction guard. Only ambiguity is disqualifying, not
    shortening - no other document answers to 'sr 11'."""
    found = planner._documents_in("what did sr 11 say about spreadsheets?")
    assert [sn for sn, _a in found.values()] == ["SR 11-7"]


def test_no_document_became_unnameable():
    """Dropping ambiguous aliases must not strip a document of every handle;
    a document nobody can name can never be filtered to."""
    have = {sn for _did, sn, _alias, _pat in planner._doc_index()}
    want = {row.short_name for row in load_registry().indexable()}
    assert want - have == set()


def test_every_alias_in_the_index_belongs_to_exactly_one_document():
    owners: dict[str, set[str]] = {}
    for doc_id, _sn, alias, _pat in planner._doc_index():
        owners.setdefault(alias, set()).add(doc_id)
    assert [a for a, v in owners.items() if len(v) > 1] == []


# ---------------------------------------------------------------------------
# FIX 2 — containment
# ---------------------------------------------------------------------------
def psg(pid: str, short_name: str, text: str) -> Passage:
    return Passage(parent_id=pid, short_name=short_name, heading="Principle 5",
                   text=text, windowed=False, full_chars=len(text), children=[])


PRINCIPLE = ("Timeliness - A bank should be able to generate aggregate and "
             "up-to-date risk data in a timely manner.")
SECTION = PRINCIPLE + (" 44. A bank's risk data aggregation capabilities should "
                       "ensure that it is able to produce aggregate risk "
                       "information on a timely basis.")


def test_a_passage_wholly_inside_another_is_contained():
    assert _contained_in(psg("p1", "BCBS 239", PRINCIPLE),
                         psg("p2", "BCBS 239", SECTION))


def test_the_container_is_not_contained_in_the_contained():
    assert not _contained_in(psg("p2", "BCBS 239", SECTION),
                             psg("p1", "BCBS 239", PRINCIPLE))


def test_containment_across_DIFFERENT_documents_is_not_collapsed():
    """The case the restriction exists for. SR 15-18 and SR 15-19 are twins;
    on a comparison question both copies are the answer."""
    assert not _contained_in(psg("p1", "SR 15-18", PRINCIPLE),
                             psg("p2", "SR 15-19", SECTION))


def test_a_passage_is_never_contained_in_itself():
    a = psg("p1", "BCBS 239", PRINCIPLE)
    assert not _contained_in(a, a)


def test_two_passages_sharing_the_same_parent_id_do_not_match():
    assert not _contained_in(psg("p1", "BCBS 239", PRINCIPLE),
                             psg("p1", "BCBS 239", SECTION))


def test_whitespace_differences_do_not_hide_containment():
    """The chunk and the parent are stored separately, so line wrapping can
    differ. Without normalising, a real containment would be missed."""
    wrapped = PRINCIPLE.replace(" ", "\n  ", 3)
    assert _contained_in(psg("p1", "BCBS 239", wrapped),
                         psg("p2", "BCBS 239", SECTION))


def test_an_empty_passage_is_not_contained_in_everything():
    """An empty string is inside every string. Without the guard, a passage
    with no text would be silently swallowed by any other."""
    assert not _contained_in(psg("p1", "BCBS 239", ""),
                             psg("p2", "BCBS 239", SECTION))


def test_a_passage_that_merely_OVERLAPS_is_kept():
    """Shares an opening, then diverges - the 12 CFR '(a) Risk committee -'
    shape. Its body carries content the other lacks, so it must survive."""
    other = PRINCIPLE[:60] + " but then says something entirely different."
    assert not _contained_in(psg("p1", "BCBS 239", other),
                             psg("p2", "BCBS 239", SECTION))


# ---------------------------------------------------------------------------
# THE GATE MUST NOT ASK ABOUT A QUESTION THAT NAMES ITS DOCUMENTS (2026-09-08)
# ---------------------------------------------------------------------------
# `decide(x)` reads the LLM's extraction, never the question, so the
# "names a document -> ANSWERABLE" rule silently depended on a small model
# filling x["document"]. It did not, and the gate asked the user to choose from
# all 19 documents about a question naming two of them. These run without
# Ollama BECAUSE the short-circuit returns before any model call - if that
# stops being true, they will fail by hanging, which is the correct alarm.
from regrag.gate import intent as gate_intent


def test_a_question_naming_two_documents_is_answerable_without_a_model():
    i = gate_intent.classify(
        "How do SR 11-7 and SS1/23 each frame the components of model validation?")
    assert i.decision == "ANSWERABLE"
    assert "SR 11-7" in i.reason and "SS1/23" in i.reason


def test_a_question_naming_one_document_is_answerable_without_a_model():
    assert gate_intent.classify(
        "What does BCBS 239 require on the accuracy of risk data?"
    ).decision == "ANSWERABLE"


def test_documents_named_reads_the_question_not_an_extraction():
    got = gate_intent.documents_named(
        "compare sr 11-7 with ss1/23 on validation")
    assert sorted(got) == ["SR 11-7", "SS1/23"]


def test_a_document_we_do_not_hold_is_not_short_circuited():
    """Solvency II must still reach the model path and its OUT_OF_CORPUS
    branch. If the short-circuit ever claims it, the gate would answer from a
    corpus that does not contain it."""
    assert gate_intent.documents_named("what does Solvency II require?") == []
