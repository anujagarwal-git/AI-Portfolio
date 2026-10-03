"""Unit tests for Stage 7 — generation.

Split the same way as test_search.py: most of the logic here is pure functions
over text and needs no Qdrant and no Ollama. The tests that do are collected
under `live` / `llm_live` and skip themselves, so `pytest` runs clean on a
machine with nothing started.

WHAT IS DELIBERATELY NOT TESTED
    That the system refuses an unanswerable question. It does not. Measured
    2026-08-26 and recorded in MENTOR_PROGRESS.md: a score gate is dead and a
    prompt rule refused 10 of 10 ANSWERABLE questions. Writing a test that
    asserts refusal would encode a capability the project does not have.
"""
from __future__ import annotations

import re

import pytest

from regrag import config
from regrag.generation import prompts, render
from regrag.generation.answer import Claim, answer, config_hash, overlap_of


# =========================================================== prompts.py
def test_the_current_prompt_loads_and_is_identified():
    p = prompts.load()
    assert p.version == prompts.CURRENT
    assert len(p.sha) == 16
    # An S-LABEL, not the bracket form. The prose prompts write "[S1]"; the
    # JSON prompt writes {"source": "S1"}. Two spellings of one fact, and
    # asserting the punctuation made the CURRENT prompt fail its own test.
    assert re.search(r"\bS\d+\b", p.text)


def test_every_prompt_on_disk_still_loads():
    # Dead prompts are kept as evidence of a measured rejection. If one stops
    # loading, the evidence is quietly gone.
    for v in ("answer_v0_norule", "answer_v1_rule", "answer_v2", "answer_v3_json"):
        assert prompts.load(v).text


def test_a_missing_prompt_names_what_is_available():
    with pytest.raises(FileNotFoundError) as e:
        prompts.load("answer_v99")
    assert "answer_v3_json" in str(e.value)


def test_the_sha_changes_with_the_CONTENT_not_just_the_name():
    # The whole reason prompts.py exists: a file can be edited without its
    # version name changing, and config_hash would then lie.
    a, b = prompts.load("answer_v0_norule"), prompts.load("answer_v2")
    assert a.version != b.version and a.sha != b.sha


# =========================================================== config_hash
def test_config_hash_carries_everything_that_changes_an_answer():
    h = config_hash(prompts.load())
    for part in (config.GEN_MODEL, prompts.CURRENT, str(config.TEMPERATURE),
                 str(config.RERANK_DROP_BELOW), str(config.FACET_DROP_MARGIN)):
        assert part in h


def test_two_prompts_produce_two_hashes():
    assert config_hash(prompts.load("answer_v2")) != config_hash(prompts.load())


# =========================================================== overlap screen
def test_overlap_finds_words_that_are_present():
    assert overlap_of("banks must validate models",
                      "Banks must validate models regularly.") == 1.0


def test_overlap_is_low_for_unrelated_text():
    assert overlap_of("liquidity coverage ratio requirements",
                      "A model is a quantitative method.") < 0.45


def test_overlap_ignores_word_order_and_says_so():
    # Documented weakness, pinned so nobody later reads it as meaning.
    a = overlap_of("banks validate models", "models validate banks")
    b = overlap_of("models validate banks", "banks validate models")
    assert a == b == 1.0


def test_overlap_of_an_empty_claim_is_zero_not_a_crash():
    assert overlap_of("", "anything") == 0.0
    assert overlap_of("of the a to", "anything") == 0.0


# =========================================================== rendering
def _labels():
    return {"S1": "SR 26-2, II. PURPOSE AND SCOPE (in_force; eff. 2026-04-17)",
            "S2": "SS1/23, Principle 1.2 Model inventory (in_force; eff. 2024-05-17)"}


def test_the_marker_sits_INSIDE_the_sentence():
    # "...boundaries [1]." not "...boundaries. [1]" - a marker after the stop
    # floats between two sentences and the reader cannot tell which it serves.
    body, _ = render.format_answer([{"text": "A model has boundaries.", "source": "S2"}],
                                   _labels())
    assert body.endswith("boundaries [1].")
    assert ". [1]" not in body


def test_markers_are_numbered_by_first_appearance_and_deduplicated():
    body, sources = render.format_answer(
        [{"text": "One.", "source": "S2"},
         {"text": "Two.", "source": "S1"},
         {"text": "Three.", "source": "S2"}], _labels())
    assert "One [1]." in body and "Two [2]." in body and "Three [1]." in body
    assert len(sources) == 2 and sources[0].startswith("[1] SS1/23")


def test_two_claims_on_ONE_passage_can_get_TWO_numbers():
    # Since the aligner, two claims reading the same parent can cite two
    # different paragraphs. Numbering keys on the CITATION, not the label -
    # keying on the label would collapse them and put the wrong paragraph under
    # half the markers.
    body, sources = render.format_answer(
        [{"text": "One.", "source": "S1", "citation": "Basel CRE, CRE53.50 (in_force)"},
         {"text": "Two.", "source": "S1", "citation": "Basel CRE, CRE53.55 (in_force)"},
         {"text": "Three.", "source": "S1", "citation": "Basel CRE, CRE53.50 (in_force)"}],
        _labels())
    assert "One [1]." in body and "Two [2]." in body and "Three [1]." in body
    assert len(sources) == 2 and sources[1].endswith("CRE53.55 (in_force)")


def test_without_a_citation_the_label_still_resolves():
    # Alignment off, or a caller that predates it: behaviour must be identical
    # to what shipped before.
    body, sources = render.format_answer(
        [{"text": "One.", "source": "S2"}], _labels())
    assert "One [1]." in body and sources[0].startswith("[1] SS1/23")


def test_a_claim_with_no_usable_source_is_MARKED_not_dropped():
    # An unattributed sentence hidden inside clean prose is the exact failure
    # this design exists to prevent.
    body, sources = render.format_answer(
        [{"text": "Something unsupported.", "source": "S9"}], _labels())
    assert "[unattributed]" in body
    assert sources == []


def test_a_sentence_without_a_full_stop_still_gets_one():
    body, _ = render.format_answer([{"text": "No stop here", "source": "S1"}], _labels())
    assert body == "No stop here [1]."


# =========================================================== Claim
def test_an_invalid_label_makes_the_claim_invalid():
    assert not Claim("x", "S9", "", 0.0).valid
    assert Claim("x", "S1", "SR 26-2", 0.9).valid


def test_weak_only_applies_to_a_VALID_claim():
    # An invalid claim is a bug, not a weak one - do not let it be reported as
    # merely weak.
    assert not Claim("x", "S9", "", 0.0).weak
    assert Claim("x", "S1", "SR 26-2", 0.1).weak


# =========================================================== live: retrieval
@pytest.fixture(scope="module")
def live():
    from regrag.index.vector_store import VectorStore
    try:
        VectorStore(strategy="parentdoc").stats()
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"Qdrant not reachable: {type(exc).__name__}")


@pytest.fixture(scope="module")
def llm_live(live):
    from regrag.generation import llm
    problem = llm.why_unavailable()
    if problem:
        pytest.skip(problem)


def test_a_content_free_question_is_refused_without_calling_the_model(live):
    a = answer("?????")
    assert a.refused and not a.claims and a.text == ""
    assert a.config_hash                       # stamped even on a refusal


def test_naming_a_document_is_never_emptied_by_the_floor(live):
    # The identifier query that broke on 2026-08-29: the cross-encoder demotes
    # exact-identifier matches, every chunk fell below the floor, and a question
    # the corpus answers came back empty. A named target must always survive.
    from regrag.retrieval.search import retrieve
    r = retrieve("what did SR 11-7 say about spreadsheets?")
    assert r.passages and not r.refused
    assert {p.short_name for p in r.passages} == {"SR 11-7"}


# =========================================================== live: generation
def test_an_answerable_question_is_answered_and_cited(llm_live):
    a = answer("When is a financial asset credit-impaired?")
    assert not a.refused, a.refusal_reason
    assert a.claims, "no claims produced"
    assert not a.invalid, [c.label for c in a.invalid]
    assert a.sources and "[1]" in a.text


def test_every_marker_in_the_text_has_a_source_line(llm_live):
    import re
    a = answer("What does a firm's model inventory have to record?")
    used = {int(n) for n in re.findall(r"\[(\d+)\]", a.text)}
    assert used and max(used) <= len(a.sources)


def test_the_same_question_twice_gives_the_same_answer(llm_live):
    # temperature 0 AND a fixed seed. Without this every measurement in
    # evaluation/ is the generator's noise rather than a result.
    q = "Are spreadsheets treated as models?"
    assert answer(q).text == answer(q).text


def test_retrieval_can_be_supplied_so_two_prompts_see_one_evidence_set(llm_live):
    # Holding retrieval fixed is what makes a prompt comparison a comparison.
    from regrag.retrieval.search import retrieve
    r = retrieve("What is the definition of a model?")
    a = answer(r.question, retrieval=r)
    b = answer(r.question, prompt_version="answer_v3_json", retrieval=r)
    assert a.retrieval is r and b.retrieval is r
    assert a.config_hash == b.config_hash


def test_a_broken_model_response_becomes_a_refusal_not_a_crash(monkeypatch, live):
    from regrag.generation import answer as mod

    class Fake:
        text = "{not json at all"
    monkeypatch.setattr(mod.llm, "chat", lambda *a, **k: Fake())
    a = answer("What is the definition of a model?")
    assert a.refused and "unusable" in a.refusal_reason
