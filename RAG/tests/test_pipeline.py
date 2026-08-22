"""End-to-end: registry -> cache -> clean -> sections -> chunker.

STEP 5 OF THE ARC. Every module passes its own tests. This is the one that asks
whether they still work when chained — and it is where the failures that matter
live, because they are the ones no unit test can see:

  - a payload field silently missing across 10,000 chunks
  - Basel chunks carrying effective_from=None because the chapter date never
    made it through the overlay
  - a locator format that works on a fixture and not on the real document
  - text lost between two stages, so a provision simply is not retrievable

Every assertion here is about the WHOLE corpus, not a sample. A field that is
right 99% of the time is a field that is wrong 100 times.

Reads from the parse cache, so it is seconds. Warm it first:
    python -m regrag.ingestion.cache --warm

Run:  pytest tests/test_pipeline.py -q -s
  or: python tests/test_pipeline.py
"""

from __future__ import annotations

import os

from regrag import config
from regrag.ingestion.cache import parse_cached
from regrag.ingestion.chunker import ENUMERATED_FRAMEWORKS, chunk_document
from regrag.ingestion.clean import clean_document
from regrag.ingestion.sections import find_sections
from regrag.registry import load

SKIP_SLOW = os.getenv("REGRAG_SKIP_SLOW") == "1"

_CACHE: dict = {}


def _corpus():
    """Run the full chain once for every indexable document, then reuse."""
    if _CACHE:
        return _CACHE
    reg = load()
    for row in reg.indexable():
        doc = parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
        cleaned = clean_document(doc, row)
        chapters = find_sections(cleaned, row)
        chunks, parents = chunk_document(cleaned, chapters, row)
        _CACHE[row.doc_id] = (row, cleaned, chapters, chunks, parents)
    return _CACHE


def _skip() -> bool:
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return True
    return False


# =============================================================================
# 1. THE CHAIN RUNS AT ALL
# =============================================================================


def test_every_indexable_document_produces_chunks():
    """A document that silently yields zero chunks is invisible to the system
    while looking perfectly healthy in the registry."""
    if _skip():
        return
    empty = [row.short_name for row, _, _, ck, _ in _corpus().values() if not ck]
    assert not empty, f"produced no chunks: {empty}"

    total = sum(len(ck) for _, _, _, ck, _ in _corpus().values())
    print(f"\n  {len(_corpus())} documents -> {total:,} chunks")
    assert total > 5_000, f"only {total} chunks — a stage is dropping content"


# =============================================================================
# 2. THE METADATA CONTRACT — what Qdrant will filter on
# =============================================================================


def test_every_chunk_carries_the_filterable_fields():
    """These are exactly the fields config.PAYLOAD_INDEX_FIELDS tells Qdrant to
    index. One missing on a subset means a filter silently drops those chunks —
    no error, just absent results."""
    if _skip():
        return
    for row, _, _, chunks, _ in _corpus().values():
        for c in chunks:
            missing = [f for f in config.PAYLOAD_INDEX_FIELDS if f not in c.payload]
            assert not missing, f"{row.short_name} chunk {c.chunk_id}: missing {missing}"


def test_payload_never_leaks_registry_internals():
    if _skip():
        return
    for row, _, _, chunks, _ in _corpus().values():
        for c in chunks:
            for leaked in ("notes", "file", "verified"):
                assert leaked not in c.payload, f"{row.short_name}: payload leaked {leaked!r}"


def test_basel_chunks_all_carry_a_chapter_level_effective_date():
    """THE reason sections.py exists. Basel registry rows have
    effective_from=None deliberately; the chapter supplies it. If the overlay
    fails, chunks look like living texts with no fixed edition — and no error
    is raised anywhere.
    """
    if _skip():
        return
    from regrag.domain import Framework

    checked = 0
    for row, _, _, chunks, _ in _corpus().values():
        if row.framework is not Framework.BASEL_FRAMEWORK:
            continue
        # SRP32 is the documented exception: a SINGLE-chapter file, so one
        # document-level date is legitimate there. The five multi-chapter
        # bundles are the ones that must be null.
        if row.volume != "SRP32":
            assert row.effective_from is None, (
                f"{row.short_name}: multi-chapter Basel rows must be chapter-level"
            )
        for c in chunks:
            assert c.payload["effective_from"], (
                f"{row.short_name} {c.chunk_id}: no chapter date reached the chunk"
            )
            assert c.payload["chapter"], f"{row.short_name} {c.chunk_id}: no chapter code"
            checked += 1
    print(f"\n  {checked:,} Basel chunks all carry a chapter date")
    assert checked > 3_000


def test_superseded_document_is_marked_on_every_chunk():
    """SR 11-7 must never be answerable as current. The guard has to hold on
    EVERY chunk, not on the document row."""
    if _skip():
        return
    _, _, _, chunks, _ = _corpus()["sr-11-7-2011"]
    assert chunks
    for c in chunks:
        assert c.payload["status"] == "superseded"
        assert c.payload["effective_to"] == "2026-04-17"


def test_reference_data_is_never_a_requirement_source():
    """The 2026 DFAST results contain the numeral 4.5 — the same number as the
    Basel CET1 minimum. Neither is US law."""
    if _skip():
        return
    for doc_id in ("fed-2026-scenarios", "fed-2026-dfast-results"):
        _, _, _, chunks, _ = _corpus()[doc_id]
        for c in chunks:
            assert c.payload["authority_rank"] == 0
            assert c.payload["is_requirement_source"] is False


# =============================================================================
# 3. NOTHING IS LOST BETWEEN STAGES
# =============================================================================


def test_chunks_retain_most_of_the_cleaned_text():
    """Noise dropping is deliberate; losing provisions is not. A large gap
    between cleaned text and chunk text means a stage is discarding content."""
    if _skip():
        return
    for row, cleaned, _, chunks, _ in _corpus().values():
        cleaned_chars = sum(len(i.text) for i in cleaned.texts)
        chunk_chars = sum(len(c.text) for c in chunks)
        kept = chunk_chars / cleaned_chars if cleaned_chars else 0
        assert kept > 0.60, (
            f"{row.short_name}: only {kept:.0%} of cleaned text reached chunks"
        )


def test_every_chunk_points_at_a_real_parent():
    if _skip():
        return
    for row, _, _, chunks, parents in _corpus().values():
        ids = {p.parent_id for p in parents}
        for c in chunks:
            assert c.parent_id in ids, f"{row.short_name}: dangling parent_id {c.parent_id}"


def test_chunk_ids_are_globally_unique():
    """They become Qdrant point ids. A collision silently overwrites a chunk."""
    if _skip():
        return
    seen: set[str] = set()
    for row, _, _, chunks, _ in _corpus().values():
        for c in chunks:
            assert c.chunk_id not in seen, f"duplicate chunk_id {c.chunk_id}"
            seen.add(c.chunk_id)


# =============================================================================
# 4. READY FOR THE EMBEDDER
# =============================================================================


def test_no_chunk_would_be_truncated_by_the_embedder():
    """bge-small drops the tail past ~512 tokens with no error. Silent
    truncation of a requirement is the failure mode with no symptom."""
    if _skip():
        return
    for row, _, _, chunks, _ in _corpus().values():
        for c in chunks:
            assert len(c.text) <= config.CHILD_MAX_CHARS, (
                f"{row.short_name} {c.chunk_id}: {len(c.text)} chars would be truncated"
            )


def test_no_empty_chunks():
    if _skip():
        return
    for row, _, _, chunks, _ in _corpus().values():
        for c in chunks:
            assert c.text.strip(), f"{row.short_name}: empty chunk {c.chunk_id}"


def test_locators_are_present_where_the_document_class_supports_them():
    """Basel and the CFR number their provisions, so most of their chunks should
    be citable. SR letters and the PRA use prose headings — low coverage there
    is expected, not a defect."""
    if _skip():
        return
    from regrag.domain import Framework

    # The bar catches a rule that BROKE, not a quality target. Anything above
    # a few percent means locator extraction is running; the actual coverage is
    # printed so it can be judged on its merits rather than by an invented
    # threshold. (Basel CAP sits near 20% because much of it is criteria lists
    # and FAQ text carrying no paragraph number.)
    print(f"\n  {'document':32} {'precise':>8} {'section':>8}")
    tot_p = tot_n = 0
    for row, _, _, chunks, _ in _corpus().values():
        precise = sum(1 for c in chunks if c.locator_kind == "paragraph")
        pct = 100 * precise / len(chunks)
        tot_p += precise
        tot_n += len(chunks)
        marker = ""
        if row.framework is Framework.BASEL_FRAMEWORK or row.framework in ENUMERATED_FRAMEWORKS:
            marker = "  <- numbered provisions"
            assert pct > 5, f"{row.short_name}: {pct:.0f}% precise — locator rule is not running"
        print(f"  {row.short_name:32} {pct:7.1f}% {100-pct:7.1f}%{marker}")

    # EVERY chunk has SOME locator, because the section heading is always
    # available as a fallback. That makes "has a locator" a useless metric —
    # it measures "has a parent", which is always true. What matters is how
    # many can be cited PRECISELY.
    assert all(c.locator for _, _, _, ck, _ in _corpus().values() for c in ck)
    print(f"\n  {tot_p:,} of {tot_n:,} chunks ({100*tot_p/tot_n:.1f}%) cite a specific provision;")
    print("  the rest cite their section, which is how unnumbered documents are cited anyway.")


def test_corpus_summary():
    """Not an assertion — the numbers you need to size Stage 4."""
    if _skip():
        return
    tot_c = tot_p = 0
    sizes: list[int] = []
    for _, _, _, chunks, parents in _corpus().values():
        tot_c += len(chunks)
        tot_p += len(parents)
        sizes += [len(c.text) for c in chunks]
    sizes.sort()
    print(f"\n  chunks {tot_c:,}   parents {tot_p:,}   "
          f"median {sizes[len(sizes)//2]}   p90 {sizes[int(0.9*len(sizes))]}   max {sizes[-1]}")
    print(f"  vectors: {tot_c:,} x {config.EMBED_DIM} dims x 4 bytes "
          f"= {tot_c * config.EMBED_DIM * 4 / 1e6:.1f} MB")


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
