"""WHAT THE CUTS DISCARDED, TESTED WITHOUT AN INDEX.

Every case here is built from hand-made Hits and Passages, so a failure means
the accounting logic changed - never that Qdrant was empty.

WHAT IS ACTUALLY BEING PROTECTED
    Not "are the cuts right" - that is the study's question, and no unit test
    can answer it. What is protected here is the ACCOUNTING, because every
    number the study prints is divided by it:

      * a drop whose parent arrived anyway COST NOTHING and must not be
        counted as a loss. This is the single number most able to inflate the
        study: the quota discards ranks 5-19 of every facet, and many of those
        chunks share a parent with a chunk that survived. Counting them would
        turn a small loss into a catastrophic-looking one.
      * a drop whose parent never arrived IS a loss,
      * `rank_in_facet` must carry the TRUE rank, not the index inside the
        discarded slice - the quota records `hits[5:]`, and if the ranks were
        taken from that slice every quota casualty would report rank 0 and the
        "ranks 5-7 are the marginal ones" filter would select the wrong rows,
      * `Retrieval` must still build from four positional arguments, because
        the existing suite constructs it that way.

THE FALSIFYING CONDITIONS, FIXED HERE BEFORE THE RUN
    If `loss` can return "absent" for a parent that was delivered whole, the
    study overstates loss and every conclusion about the cuts is wrong in the
    direction that flatters the finding. If it can return "none" for a parent
    that never arrived, it understates it. Both directions are tested.

    `windowed_out` is a HEURISTIC and is tested as one: it is checked only for
    self-consistency (text present -> none, text absent -> windowed_out), never
    for correctness against a real Docling parent. Do not read a passing test
    here as evidence that the windowed_out count in the study is exact.
"""
from __future__ import annotations

import pytest

from regrag.retrieval.search import (Dropped, Hit, Passage, Retrieval,
                                     _record, _resolve_loss)


def hit(chunk_id: str, parent_id: str, text: str = "some chunk text",
        rerank: float | None = 1.0) -> Hit:
    return Hit(chunk_id=chunk_id,
               payload={"parent_id": parent_id, "text": text,
                        "short_name": "Basel CRE", "parent_heading": "Section 3",
                        "locator": "36.122"},
               facet="GLOBAL", dense_rank=0, lexical_rank=1, rrf=0.03,
               rerank=rerank)


def passage(parent_id: str, text: str, windowed: bool = False) -> Passage:
    return Passage(parent_id=parent_id, short_name="Basel CRE",
                   heading="Section 3", text=text, windowed=windowed,
                   full_chars=len(text), children=[])


def drop(**kw) -> Dropped:
    base = dict(cut="quota", facet="GLOBAL", parent_id="p1",
                short_name="Basel CRE", heading="Section 3",
                text="some text", rerank=0.4, rrf=0.01, dense_rank=7,
                lexical_rank=None, rank_in_facet=6, locator="36.122", chars=9)
    base.update(kw)
    return Dropped(**base)


# ---------------------------------------------------------------------------
# THE ACCOUNTING - did the drop cost the prompt anything?
# ---------------------------------------------------------------------------
def test_a_drop_whose_parent_arrived_whole_cost_nothing():
    """The number most able to inflate this study. See the module docstring."""
    d = [drop(text="the chunk that lost its slot")]
    _resolve_loss(d, [passage("p1", "a whole parent, delivered untrimmed")])
    assert d[0].loss == "none"


def test_a_drop_whose_parent_never_arrived_is_a_loss():
    d = [drop(parent_id="p_missing", text="text nobody read")]
    _resolve_loss(d, [passage("p1", "a different parent entirely")])
    assert d[0].loss == "absent"


def test_nothing_delivered_at_all_makes_every_drop_a_loss():
    d = [drop(), drop(parent_id="p2")]
    _resolve_loss(d, [])
    assert [x.loss for x in d] == ["absent", "absent"]


def test_a_context_budget_drop_is_always_absent():
    """A budget drop discards the WHOLE passage, so its parent is by
    definition not in the delivered list. Asserted rather than assumed: the
    parent-lookup branch would agree for the wrong reason if a budget-dropped
    passage were ever left in the delivered list by a later edit."""
    d = [drop(cut="context_budget", text="a whole passage that did not fit",
              rank_in_facet=None, dense_rank=None)]
    _resolve_loss(d, [passage("p1", "a whole passage that did not fit")])
    assert d[0].loss == "absent"


def test_a_trimmed_parent_still_holding_the_text_cost_nothing():
    d = [drop(text="the internal model must be", locator="53.50")]
    _resolve_loss(d, [passage(
        "p1", "53.50 the internal model must be validated annually",
        windowed=True)])
    assert d[0].loss == "none"


def test_a_trimmed_parent_that_lost_the_text_is_windowed_out():
    d = [drop(text="a paragraph the window cut away", locator="53.99")]
    _resolve_loss(d, [passage("p1", "53.50 something else entirely",
                              windowed=True)])
    assert d[0].loss == "windowed_out"


def test_whitespace_differences_do_not_invent_a_loss():
    """The chunk payload and the parent text are stored separately, so line
    wrapping can differ between them. Normalising is what stops that from
    reporting a loss that did not happen."""
    d = [drop(text="the internal   model\n\n  must be", locator="53.50")]
    _resolve_loss(d, [passage("p1", "the internal model must be validated",
                              windowed=True)])
    assert d[0].loss == "none"


def test_an_empty_chunk_text_in_a_trimmed_parent_is_not_called_none():
    """An empty string is inside every string. Without the guard, a chunk with
    no text would be silently scored as delivered."""
    d = [drop(text="", chars=0)]
    _resolve_loss(d, [passage("p1", "some delivered window", windowed=True)])
    assert d[0].loss == "windowed_out"


# ---------------------------------------------------------------------------
# THE RANK - the quota records a SLICE, and the slice does not start at zero
# ---------------------------------------------------------------------------
def test_rank_in_facet_is_the_true_rank_not_the_slice_index():
    sink: list = []
    hits = [hit(f"c{i}", f"p{i}") for i in range(8)]
    _record(sink, "quota", hits[5:], list(range(5, 8)))
    assert [d.rank_in_facet for d in sink] == [5, 6, 7]


def test_record_without_ranks_falls_back_to_position():
    sink: list = []
    _record(sink, "facet_margin", [hit("c0", "p0"), hit("c1", "p1")])
    assert [d.rank_in_facet for d in sink] == [0, 1]


def test_record_carries_the_payload_fields_the_study_reads():
    sink: list = []
    _record(sink, "quota", [hit("c0", "p9", text="a chunk")], [5])
    d = sink[0]
    assert (d.parent_id, d.short_name, d.text, d.chars) == ("p9", "Basel CRE",
                                                            "a chunk", 7)
    assert d.cut == "quota"
    assert d.loss == "?"          # unresolved until _resolve_loss runs


def test_a_sparse_payload_does_not_raise():
    """Bin chunks and unnumbered parents have sparse payloads. A study that
    crashes on question 31 of 45 collects nothing."""
    sink: list = []
    _record(sink, "relevance_floor", [Hit(chunk_id="c0", payload={}, facet="UK")])
    assert sink[0].parent_id == ""
    assert sink[0].chars == 0


# ---------------------------------------------------------------------------
# THE REGRESSION GUARD - the existing suite builds Retrieval positionally
# ---------------------------------------------------------------------------
def test_retrieval_still_builds_without_dropped():
    r = Retrieval("q", None, [], ["a note"])
    assert r.dropped == []


def test_two_retrievals_do_not_share_a_dropped_list():
    """A mutable default would make every Retrieval in a process share one
    list, and the study's per-question counts would accumulate silently."""
    a, b = Retrieval("q1", None, []), Retrieval("q2", None, [])
    a.dropped.append("x")
    assert b.dropped == []
