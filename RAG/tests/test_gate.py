"""Stage 8.6 — the intent gate. NO OLLAMA NEEDED.

`decide()` is pure: an extraction dict in, a decision out, everything else read
from registry.yaml. That is the half of the gate carrying the corpus knowledge,
so it is the half worth testing hard. The model call is tested by
scripts/experiment_prompts.py, which measures it on 120 questions — a unit test
cannot tell you whether a 1.7B model reads a sentence correctly.

Every expected value below is derived from the GRID, not typed from memory, so
adding a document to registry.yaml cannot make a test silently wrong.
"""
from __future__ import annotations

import pytest

from regrag import config
from regrag.gate import intent as gate


@pytest.fixture(scope="module")
def grid():
    g, subjects, jurisdictions, names = gate.corpus_grid()
    return g, subjects, jurisdictions, names


def _one_jurisdiction_subject(g, subjects):
    """A subject the corpus covers for exactly ONE jurisdiction."""
    for su in subjects:
        have = [j for (s, j) in g if s == su and g[(s, j)]]
        if len(have) == 1:
            return su, have[0]
    pytest.skip("no single-jurisdiction subject in this registry")


def _multi_jurisdiction_subject(g, subjects):
    for su in subjects:
        have = [j for (s, j) in g if s == su and g[(s, j)]]
        if len(have) > 1:
            return su, sorted(have)
    pytest.skip("no multi-jurisdiction subject in this registry")


def _empty_cell(g, subjects, jurisdictions):
    for su in subjects:
        for ju in jurisdictions:
            if not g.get((su, ju)):
                return su, ju
    pytest.skip("no empty cell in this registry")


# --- the four decisions ----------------------------------------------------
def test_not_banking_is_refused():
    d, why = gate.decide({"banking": False})
    assert d == "OUT_OF_DOMAIN", why


def test_named_document_we_hold_passes(grid):
    _g, _s, _j, names = grid
    d, why = gate.decide({"banking": True, "document": next(iter(names.values()))})
    assert d == "ANSWERABLE", why


def test_named_document_we_do_not_hold_is_declined():
    d, why = gate.decide({"banking": True,
                          "document": "Solvency II Delegated Regulation"})
    assert d == "OUT_OF_CORPUS", why


def test_a_family_name_alone_is_not_a_named_document(grid):
    """"Basel" is six documents, so it identifies none of them.

    THE BUG THIS GUARDS (2026-09-02): a substring match made "Basel" hit
    "Basel CAP", so a question about an ABSENT Basel volume came back
    ANSWERABLE. Finding "Jane Smith" is not finding John Smith's file.
    """
    _g, _s, _j, names = grid
    kind, _hit = gate._names_hit("basel", names)
    assert kind == "FAMILY", "a shared word must not resolve to one document"


def test_a_family_name_falls_through_rather_than_declining(grid):
    """The other half of the same fix, and the easier one to get wrong.

    "What does Basel say about capital?" names no volume, but the corpus DOES
    cover capital globally. Declining here would trade one bug for a worse one:
    refusing a question we can answer.
    """
    g, s, _j, _n = grid
    su = next(su for su in s if len({j for (a, j) in g if a == su and g[(a, j)]}) == 1)
    d, why = gate.decide({"banking": True, "document": "Basel",
                          "subject": su, "jurisdiction": None})
    assert d == "ANSWERABLE", why


def test_a_distinctive_token_still_resolves(grid):
    _g, _s, _j, names = grid
    target = next((n for n in names.values() if n.lower().startswith("basel ")), None)
    if target is None:
        pytest.skip("no multi-word Basel volume in this registry")
    kind, hit = gate._names_hit(target.lower(), names)
    assert (kind, hit) == ("EXACT", target)


def test_empty_grid_cell_ASKS_rather_than_declining(grid):
    """CHANGED 2026-09-02 AT ANUJ'S INSTRUCTION. This used to expect
    OUT_OF_CORPUS, and that was the wrong call.

    An empty (subject x jurisdiction) cell has TWO causes and the gate cannot
    tell them apart: the corpus really lacks it, or THE SUBJECT WAS MIS-READ.
    The second happened eight times in hand testing - a Basel credit-risk
    question was classified into a subject that only exists for US and UK, and
    a confident OUT_OF_CORPUS refused a question the corpus can answer.

    Declining costs an answerable question. Asking costs one turn. So an empty
    cell now ASKS, and names the documents rather than the jurisdictions -
    the document field has never been wrong, the subject field has.
    """
    g, s, j, _n = grid
    su, ju = _empty_cell(g, s, j)
    d, why = gate.decide({"banking": True, "subject": su, "jurisdiction": ju})
    assert d == "INCOMPLETE", why
    assert "mis-read" in why, "the message must not assert the corpus lacks it"


def test_populated_grid_cell_passes(grid):
    g, _s, _j, _n = grid
    (su, ju) = next(k for k, v in g.items() if v)
    d, why = gate.decide({"banking": True, "subject": su, "jurisdiction": ju})
    assert d == "ANSWERABLE", why


# --- the interesting branch: no jurisdiction given -------------------------
def test_missing_jurisdiction_with_one_option_passes_rather_than_asking(grid):
    """Asking a question with only one possible answer is a WASTED TURN.

    The clarifying turn is the gate's main new cost to a user. It must only be
    spent where the corpus genuinely holds two different answers.
    """
    g, s, _j, _n = grid
    su, _only = _one_jurisdiction_subject(g, s)
    d, why = gate.decide({"banking": True, "subject": su, "jurisdiction": None})
    assert d == "ANSWERABLE", why


def test_missing_jurisdiction_with_several_options_asks(grid):
    g, s, _j, _n = grid
    su, have = _multi_jurisdiction_subject(g, s)
    d, why = gate.decide({"banking": True, "subject": su, "jurisdiction": None})
    assert d == "INCOMPLETE", why
    for j in have:
        assert j in why, f"the reason should name the real options: {why}"


def test_no_subject_asks_rather_than_guessing():
    d, why = gate.decide({"banking": True, "subject": None, "jurisdiction": None})
    assert d == "INCOMPLETE", why


# --- the messages a person actually reads ----------------------------------
def test_every_stopping_decision_has_exactly_one_fixed_message():
    """Anuj's rule: one voice per outcome, not one per internal cause."""
    assert set(gate.MESSAGES) == {"OUT_OF_DOMAIN", "OUT_OF_CORPUS"}
    for text in gate.MESSAGES.values():
        assert text and text[-1] == "." and len(text) < 400


def test_clarifying_question_names_the_real_options(grid):
    g, s, _j, _n = grid
    su, have = _multi_jurisdiction_subject(g, s)
    q = gate.clarifying_question({"subject": su}, "")
    for j in have:
        assert j in q, f"must offer what the corpus holds, not a generic ask: {q}"


def test_clarifying_question_falls_back_when_subject_is_unknown():
    # It must ACTUALLY ASK. A clarifying turn phrased as a statement leaves the
    # user unsure whether a reply is wanted - and this turn is the gate's whole
    # cost to them, so it has to earn it.
    q = gate.clarifying_question({"subject": None}, "")
    assert q and "?" in q


def test_the_clarifying_question_names_documents_not_only_jurisdictions(grid):
    """The menu must be answerable by someone who does not know our vocabulary.

    "UK or US?" is unanswerable when the SUBJECT behind it was mis-read - that
    is how a Basel question got a UK/US menu. Document names always resolve.
    """
    g, s, _j, _n = grid
    su, _only = _one_jurisdiction_subject(g, s)
    for x in ({"subject": None}, {"subject": su}):
        assert "\n  " in gate.clarifying_question(x, ""), x


# --- the properties that keep this safe to ship ----------------------------
def test_decide_is_deterministic(grid):
    g, _s, _j, _n = grid
    x = {"banking": True, "subject": next(iter(g))[0], "jurisdiction": None}
    assert [gate.decide(x) for _ in range(5)].count(gate.decide(x)) == 5


def test_decide_never_raises_on_a_junk_extraction():
    """The model can return anything. `decide` must survive all of it."""
    for x in ({}, {"banking": "yes"}, {"banking": True, "subject": "NOPE"},
              {"banking": True, "subject": None, "jurisdiction": "MARS"},
              {"banking": True, "document": ""},
              {"banking": True, "document": None, "subject": 7}):
        d, _why = gate.decide(x)
        assert d in gate.DECISIONS


def test_grid_uses_indexable_documents_only():
    """An unverified row is refused by the indexer, so it is NOT coverage.

    Using registry.documents here would let the gate promise a document the
    retriever cannot reach — a false ANSWERABLE, which is the worst outcome
    this gate has.
    """
    from regrag import registry
    reg = registry.load_cached()
    g, _s, _j, names = gate.corpus_grid()
    assert len(names) == len(reg.indexable())
    for d in reg.blocked():
        assert d.short_name.lower() not in names


def test_config_switch_exists_and_defaults_on():
    assert isinstance(config.GATE_ENABLED, bool)
    assert config.GATE_MODEL and config.GATE_PROMPT


def test_prompt_file_loads_and_carries_the_vocabularies():
    text, sha = gate._prompt(config.GATE_PROMPT)
    _g, subjects, jurisdictions, _n = gate.corpus_grid()
    assert len(sha) == 16
    for v in list(subjects) + list(jurisdictions):
        assert v in text, f"{v} missing — the prompt lost the controlled vocab"
    assert "{subjects}" not in text and "{jurisdictions}" not in text
