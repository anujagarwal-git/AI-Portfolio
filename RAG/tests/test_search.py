"""Unit tests for search.py.

Most of these need NO Qdrant and NO models — the parts of retrieval that carry
real logic are pure functions over text. The live ones are collected under
`live` and skip themselves when Qdrant is not reachable, so `pytest` still runs
clean on a machine with nothing started.
"""

import pytest

from regrag import config
from regrag.retrieval.reranker import pair_text
from regrag.retrieval.search import _matches, _rrf, _window, is_bin, tokenize


# ============================================================ tokenizer
def test_identifier_is_kept_whole_and_split():
    t = tokenize("what does CRE36.122 require?")
    assert "cre36.122" in t          # the citation form, df=0 before the fix
    assert "cre36" in t
    assert "122" in t


def test_word_boundaries_hold():
    # "us" inside "thus"/"must" would tag nearly every question as US-scoped
    assert tokenize("modelvalidation") == ["modelvalidation"]
    assert "model" not in tokenize("modelvalidation")


@pytest.mark.parametrize("raw,expect", [
    ("12 CFR 252.54", "252.54"),
    ("SS1/23", "ss1/23"),
    ("SR 11-7", "11-7"),
])
def test_regulatory_identifiers_survive_tokenizing(raw, expect):
    assert expect in tokenize(raw)


def test_the_same_tokenizer_runs_at_build_and_query_time():
    # two tokenizers is the classic way to get an index that matches nothing
    assert tokenize("CRE36.122") == tokenize("cre36.122")


# ============================================================ RRF
def test_rrf_rewards_agreement_over_one_arms_first_place():
    # b is 2nd in both arms; a is 1st in one and absent from the other
    scores = _rrf(dense=["a", "b"], lexical=["c", "b"], k=config.RRF_K)
    assert scores["b"] > scores["a"], "a chunk BOTH arms found must beat a single arm's favourite"


def test_rrf_k_flattens_the_gap_between_nearby_ranks():
    s = _rrf(["x"] + [f"f{i}" for i in range(9)], [], k=60)
    assert (s["x"] - s["f8"]) < s["x"] * 0.2   # rank 1 vs rank 10 differ by <20%


def test_rrf_never_invents_a_chunk():
    s = _rrf(["a"], ["b"], k=60)
    assert set(s) == {"a", "b"}


# ============================================================ windowing
def _para(n, size=100):
    return "\n\n".join(f"p{i} " + "x" * size for i in range(n))


def test_a_parent_under_the_cap_is_delivered_whole():
    text = _para(3)
    out, windowed = _window(text, "p1", cap=10_000)
    assert out == text and windowed is False


def test_an_oversized_parent_is_trimmed_and_flagged():
    text = _para(40, 200)
    out, windowed = _window(text, "p20 ", cap=2_000)
    assert windowed is True and len(out) < len(text)


def test_the_window_is_built_from_WHOLE_paragraphs():
    # a character cut lands mid-requirement, and half a requirement reads
    # exactly like a whole one
    text = _para(40, 200)
    originals = set(text.split("\n\n"))
    out, _ = _window(text, "p20 ", cap=2_000)
    for chunk in out.split("\n\n"):
        if chunk != "[...]":
            assert chunk in originals, "a paragraph was cut in half"


def test_the_window_is_centred_on_the_matched_child():
    text = _para(40, 200)
    out, _ = _window(text, "p20 ", cap=2_000)
    assert "p20 " in out


def test_elision_is_marked_at_a_trimmed_end():
    text = _para(40, 200)
    out, _ = _window(text, "p20 ", cap=2_000)
    assert "[...]" in out


def test_a_missing_child_does_not_crash():
    text = _para(10, 200)
    out, windowed = _window(text, "text that is not in the parent", cap=800)
    assert windowed is True and out          # KNOWN LIMIT: falls back to paragraph 0


# ============================================================ bins
@pytest.mark.parametrize("heading", ["Footnotes", "footnotes", "FOOTNOTES", "notes", "", "Tables"])
def test_a_bag_is_a_bin(heading):
    assert is_bin(heading) is True


@pytest.mark.parametrize("heading", [
    "Section 8: validation of internal estimates",
    "Principle 1.2 Model inventory",
    "III. OVERVIEW OF MODEL RISK MANAGEMENT",
])
def test_a_real_section_is_not_a_bin(heading):
    assert is_bin(heading) is False


def test_a_paragraph_masquerading_as_a_heading_is_a_bin():
    assert is_bin("x" * 120) is True


# ============================================================ filters
def test_a_scalar_filter_matches_exactly():
    assert _matches({"jurisdiction": "UK"}, {"jurisdiction": "UK"}) is True
    assert _matches({"jurisdiction": "US"}, {"jurisdiction": "UK"}) is False


def test_a_list_filter_matches_any_member():
    p = {"status": "partially_revised"}
    assert _matches(p, {"status": ["in_force", "partially_revised"]}) is True
    assert _matches(p, {"status": ["superseded"]}) is False


def test_every_key_must_match():
    p = {"jurisdiction": "UK", "status": "in_force"}
    assert _matches(p, {"jurisdiction": "UK", "status": "superseded"}) is False


# ============================================================ rerank input
def test_without_metadata_the_model_reads_only_the_text():
    p = {"text": "Banks must validate.", "locator": "CRE36.122", "parent_heading": "Section 8"}
    assert pair_text(p, metadata=False) == "Banks must validate."


def test_with_metadata_the_locator_and_heading_come_first():
    p = {"text": "Banks must validate.", "locator": "CRE36.122", "parent_heading": "Section 8"}
    out = pair_text(p, metadata=True)
    assert out.startswith("CRE36.122") and "Section 8" in out and out.endswith("Banks must validate.")


def test_a_locator_that_IS_the_heading_is_not_repeated():
    p = {"text": "body", "locator": "Model validation", "parent_heading": "Model validation"}
    assert pair_text(p, metadata=True).count("Model validation") == 1


def test_missing_metadata_degrades_to_the_text():
    assert pair_text({"text": "body"}, metadata=True) == "body"


# ============================================================ live
@pytest.fixture(scope="module")
def live():
    from regrag.index.vector_store import VectorStore
    try:
        VectorStore(strategy="parentdoc").stats()
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"Qdrant not reachable: {type(exc).__name__}")


@pytest.mark.parametrize("q", ["", "?????", "hi"])
def test_a_content_free_question_is_refused_before_embedding(live, q):
    from regrag.retrieval.search import retrieve
    r = retrieve(q)
    assert r.refused and not r.passages


def test_naming_a_superseded_document_reaches_it(live):
    # the bug: status defaulted to current, which EXCLUDES SR 11-7
    from regrag.retrieval.search import retrieve
    r = retrieve("what did SR 11-7 say about spreadsheets?")
    assert r.passages
    assert {p.short_name for p in r.passages} == {"SR 11-7"}


def test_the_context_budget_is_never_exceeded(live):
    from regrag.retrieval.search import retrieve
    for q in ("what is model validation?",
              "which asset classes fall under specialised lending?",
              "what does CRE36.122 require?"):
        assert retrieve(q).total_chars <= config.CONTEXT_MAX_CHARS


def test_a_parent_is_delivered_once_however_many_children_matched(live):
    from regrag.retrieval.search import retrieve
    r = retrieve("what is model validation?")
    ids = [p.parent_id for p in r.passages]
    assert len(ids) == len(set(ids))
