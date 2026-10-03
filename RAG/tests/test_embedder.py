"""Tests for regrag.index.embedder.

Nearly every test here guards the SAME defect: BGE is asymmetric, and getting
that wrong produces no error at all. Queries take an instruction prefix,
documents do not. Swap them and retrieval quietly degrades — plausible top-k,
wrong ordering, no symptom until an eval you cannot yet run.

These tests download and run the real model, so they are slow on first use.
Set REGRAG_SKIP_SLOW=1 to skip.

Run:  pytest tests/test_embedder.py -q -s
  or: python tests/test_embedder.py
"""

from __future__ import annotations

import math

import os

from regrag import config
from regrag.index.embedder import (
    embed_documents, embed_query, measure_throughput, sanity_check,
)

SKIP_SLOW = os.getenv("REGRAG_SKIP_SLOW") == "1"

PASSAGE = ("Banks must have a robust system in place to validate the accuracy "
           "and consistency of rating systems and processes.")
QUESTION = "What must banks do to validate their internal rating systems?"


def _skip() -> bool:
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return True
    return False


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# =============================================================================
# 1. THE ASYMMETRY
# =============================================================================


def test_query_and_document_paths_differ():
    """Same text, two functions, two different vectors — because the query
    path prepends the instruction. If these ever match, the prefix is not being
    applied and retrieval silently degrades."""
    if _skip():
        return
    assert embed_query(PASSAGE) != embed_documents([PASSAGE])[0]


def test_documents_get_no_instruction_prefix():
    """Embedding the prefix text itself must equal embedding it as a document —
    proving embed_documents adds nothing of its own."""
    if _skip():
        return
    manual = embed_documents([config.BGE_QUERY_INSTRUCTION + PASSAGE])[0]
    viaquery = embed_query(PASSAGE)
    assert manual == viaquery, "embed_query does not apply exactly the configured prefix"


def test_related_passage_beats_unrelated():
    """The end-to-end check that the asymmetry is wired the right way round.
    A question about validation must score higher against the validation
    passage than against a large-exposures passage."""
    if _skip():
        return
    r = sanity_check()
    print(f"\n  related={r['related_score']}  unrelated={r['unrelated_score']}  "
          f"margin={r['margin']}")
    assert r["query_prefix_applied"] is True
    assert r["related_score"] > r["unrelated_score"], "asymmetry is wired backwards"
    assert r["margin"] > 0.05, f"margin only {r['margin']} — suspiciously weak"


# =============================================================================
# 2. VECTOR PROPERTIES QDRANT DEPENDS ON
# =============================================================================


def test_vectors_have_the_configured_dimension():
    """config.EMBED_DIM sizes the Qdrant collection. A mismatch is rejected at
    insert time, but only after the whole corpus has been embedded."""
    if _skip():
        return
    assert len(embed_query(QUESTION)) == config.EMBED_DIM
    assert len(embed_documents([PASSAGE])[0]) == config.EMBED_DIM


def test_vectors_are_l2_normalised():
    """Normalised vectors make a dot product equal cosine similarity, so scores
    are comparable across queries. Both sides must be normalised or the
    comparison is meaningless."""
    if _skip():
        return
    for v in (embed_query(QUESTION), embed_documents([PASSAGE])[0]):
        assert abs(_dot(v, v) - 1.0) < 1e-4, "vector is not unit length"


def test_embedding_is_deterministic():
    """Two runs must produce identical vectors — the same reason temperature 0
    matters in generation. Without it, two eval runs are not comparable."""
    if _skip():
        return
    assert embed_documents([PASSAGE]) == embed_documents([PASSAGE])
    assert embed_query(QUESTION) == embed_query(QUESTION)


def test_batching_does_not_change_vectors():
    """Batch size is a performance knob. If it altered the output, throughput
    tuning would silently change retrieval results.

    COMPARED BY DIRECTION, NOT BY BITS — and the difference matters.
    This asserted exact float equality until 2026-08-29, when it went red at
    the 7th decimal. Floating-point addition is not associative and batching
    reshapes every matmul, so tiny differences are expected; the old assertion
    was passing on luck.

    But a failing assertion is not loosened on that reasoning alone. Measured
    first (scripts/diagnose_batching.py): the two batchings agree in direction
    to more than nine decimals. Retrieval compares vectors by COSINE, so a
    difference that small cannot move a single result.

    NOTE THE CONTRAST with test_two_runs_are_identical above, which keeps
    exact equality. Same input, same batch size, same code path -> the bits
    must match, and anything else is real non-determinism. Different batch
    size is a different code path, so direction is the honest invariant.
    """
    if _skip():
        return
    texts = [PASSAGE, QUESTION, "A short third passage about capital buffers."]
    a = embed_documents(texts, batch_size=1)
    b = embed_documents(texts, batch_size=3)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        dot = sum(p * q for p, q in zip(x, y))
        nx = math.sqrt(sum(p * p for p in x))
        ny = math.sqrt(sum(q * q for q in y))
        assert 1.0 - dot / (nx * ny) < 1e-9        # same direction
        assert max(abs(p - q) for p, q in zip(x, y)) < 1e-4


def test_empty_input_returns_empty():
    if _skip():
        return
    assert embed_documents([]) == []


# =============================================================================
# 3. REAL CHUNKS
# =============================================================================


def test_real_corpus_chunks_embed_and_rank_sensibly():
    """A question drawn from the corpus should rank its own source chunk near
    the top. Not a retrieval benchmark — a check that nothing is inverted."""
    if _skip():
        return

    from regrag.ingestion.cache import parse_cached
    from regrag.ingestion.chunker import chunk_document
    from regrag.ingestion.clean import clean_document
    from regrag.ingestion.sections import find_sections
    from regrag.registry import load

    reg = load()
    row = reg.by_id("bcbs-cre-consolidated")
    doc = parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
    cleaned = clean_document(doc, row)
    chunks, _ = chunk_document(cleaned, find_sections(cleaned, row), row)

    subset = chunks[:400]
    vecs = embed_documents([c.text for c in subset])
    qv = embed_query("How must a bank validate its internal rating estimates?")

    ranked = sorted(range(len(subset)), key=lambda i: -_dot(qv, vecs[i]))
    top = subset[ranked[0]]
    print(f"\n  top hit [{top.locator}] {top.text[:100]!r}")

    assert len(vecs) == len(subset)
    best = _dot(qv, vecs[ranked[0]])
    worst = _dot(qv, vecs[ranked[-1]])
    assert best > worst, "ranking is inverted"
    assert best > 0.3, f"best score only {best:.3f} — nothing matched meaningfully"


def test_throughput_is_measurable():
    """The number that decides whether the index can be rebuilt freely."""
    if _skip():
        return
    texts = [f"Paragraph {i} concerning capital requirements and risk weights." * 3
             for i in range(64)]
    tp = measure_throughput(texts, batch_size=32)
    print(f"\n  {tp}   -> 9,935 chunks in ~{tp.estimate(9935)/60:.1f} min")
    assert tp.per_second > 0


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
