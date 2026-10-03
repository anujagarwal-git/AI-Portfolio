"""Ten cases, one per branch plus the traps that branch on the same words.

These test what the planner DECIDES, not what retrieval FINDS. There are no
expected answers and no ground-truth passages here, so nothing can leak into
retrieval tuning.
"""

import pytest

from regrag import config
from regrag.registry import load
from regrag.retrieval.planner import HISTORIC, plan


def _doc_id(fragment: str) -> str:
    for row in load().indexable():
        if fragment.lower() in row.short_name.lower():
            return row.doc_id
    raise AssertionError(f"no registry row matching {fragment!r}")


def _statuses(p):
    return {s for f in p.facets for s in f.where.get("status", [])}


# ---------------------------------------------------------------- the branches
def test_1_subject_only_fans_out_over_jurisdiction():
    p = plan("what is model validation?")
    assert p.mode == "jurisdiction_fanout"
    assert len(p.facets) == 3
    assert {f.where["jurisdiction"] for f in p.facets} == {"US", "UK", "GLOBAL"}
    assert HISTORIC[0] not in _statuses(p)


def test_2_one_jurisdiction_is_a_filter_not_a_fanout():
    p = plan("what does the PRA expect for model tiering?")
    assert p.mode == "single_jurisdiction"
    assert len(p.facets) == 1
    assert p.facets[0].where["jurisdiction"] == "UK"


def test_3_two_jurisdictions_fan_out_even_though_differ_is_a_change_word():
    # "differ" is a weak version word. The thing being compared is NAMED, so the
    # jurisdiction split must win over the version split.
    p = plan("how does the UK differ from the US on model validation?")
    assert p.mode == "jurisdiction_fanout"
    assert {f.where["jurisdiction"] for f in p.facets} == {"UK", "US"}
    assert HISTORIC[0] not in _statuses(p)


def test_4_naming_a_superseded_document_reaches_it():
    # The bug this rule exists for: status defaulted to current, which EXCLUDES
    # SR 11-7, and the answer came from SR 26-2 saying the opposite.
    p = plan("what did SR 11-7 say about spreadsheets?")
    assert p.mode == "single_document"
    assert len(p.facets) == 1
    assert p.facets[0].where == {"doc_id": _doc_id("SR 11-7")}
    assert "status" not in p.facets[0].where


@pytest.mark.parametrize("q", [
    "what did SR 11 say about spreadsheets?",
    "what did sr11-7 say about spreadsheets?",
    "what did SR 11-7 say about spreadsheets?",
])
def test_5_loose_document_aliases_all_reach_the_same_row(q):
    p = plan(q)
    assert p.mode == "single_document"
    assert p.facets[0].where["doc_id"] == _doc_id("SR 11-7")


def test_6_two_documents_fan_out_by_document():
    p = plan("how does SR 26-2 differ from SR 11-7?")
    assert p.mode == "document_fanout"
    assert {f.where["doc_id"] for f in p.facets} == {_doc_id("SR 26-2"), _doc_id("SR 11-7")}


WEAK_Q = "did the treatment of spreadsheets change under the new guidance?"


def test_7_weak_change_wording_is_OFF_BY_DEFAULT():
    """CHANGED 2026-09-02 AT ANUJ'S INSTRUCTION — this is the DEFAULT now.

    Weak version wording is a change word ("change", "differ") plus any
    document word ("guidance", "framework", "standard"). It fired on questions
    that were not about versions at all: "Under CRE, in which cases does the
    treatment DIFFER..." became a version fan-out and buried the named
    document. Rule 4 (one named document) should have won and did not.

    So the weak path is now off unless REGRAG_WEAK_VERSION is set. Strong
    wording - "previous", "superseded", "what changed" - is unaffected and is
    covered by test 8.
    """
    assert not config.WEAK_VERSION_TERMS_ENABLED, "the default flipped back"
    assert plan(WEAK_Q).mode != "version_fanout"


def test_7b_weak_change_wording_still_works_when_switched_on(monkeypatch):
    # The path is disabled, NOT deleted. If it is ever re-enabled it must still
    # behave - an experiment switch that silently rots is worse than no switch.
    monkeypatch.setattr(config, "WEAK_VERSION_TERMS_ENABLED", True)
    p = plan(WEAK_Q)
    assert p.mode == "version_fanout"
    assert len(p.facets) == 2
    assert HISTORIC[0] in _statuses(p)


@pytest.mark.parametrize("q", [
    "what was the previous definition of a model?",   # strong word
    "what changed in the model definition?",          # past-tense phrase
])
def test_8_strong_version_wording_fires_on_its_own(q):
    p = plan(q)
    assert p.mode == "version_fanout"
    assert HISTORIC[0] in _statuses(p)


def test_9_traps_must_not_reach_superseded_documents():
    # (a) a change word with no document word is about CURRENT requirements
    a = plan("what changes are required to the model inventory?")
    # (b) recency words mean "the current one", not "what changed"
    b = plan("what is the latest guidance on model risk?")
    for p in (a, b):
        assert p.mode != "version_fanout"
        assert HISTORIC[0] not in _statuses(p), p


def test_10_degenerate_input_widens_rather_than_narrowing():
    # Recognising nothing means knowing LESS, so the search must widen. The
    # mode still records the miss even though the behaviour is a fan-out.
    for q in ("", "hello", "tell me about banks", "?????", "modelvalidation"):
        p = plan(q)
        assert p.mode == "fallback", q
        assert len(p.facets) == 3
        assert {f.where["jurisdiction"] for f in p.facets} == {"US", "UK", "GLOBAL"}


# ------------------------------------------------------------- the invariants
@pytest.mark.parametrize("q", [
    "", "hello", "what is model validation?", "what did SR 11-7 say?",
    "how does SR 26-2 differ from SR 11-7?", "what is the latest guidance?",
    "how does the UK differ from the US?", "thus the bank must validate",
    "did the treatment of spreadsheets change under the new guidance?",
])
def test_invariants_hold_for_every_question(q):
    p = plan(q)
    assert p.facets, "never zero facets"
    assert len(p.facets) <= 3, "at most one axis fans out"
    assert all(f.why for f in p.facets), "a facet must explain itself"
    assert all(k in config.PAYLOAD_INDEX_FIELDS
               for f in p.facets for k in f.where), "filtering on an unindexed field"
    assert str(plan(q)) == str(p), "same question, same plan"
