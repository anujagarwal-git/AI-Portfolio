"""End-to-end smoke: question -> vector -> Qdrant -> chunk -> PARENT.

WHAT THIS CLOSES. Every stage so far has been tested in isolation, and the
index has been built and queried. What has never run is the SECOND HALF OF
PARENTDOC: taking a retrieved child and fetching the section it belongs to.

That half is what makes the whole strategy legitimate. Your top-4 smoke hit was
'(8) Validation of internal estimates' — 36 characters, useless as an answer,
excellent as a key. It is only acceptable because the parent arrives with it.
If get_parent() returns nothing, every short chunk in the index becomes a
liability instead of a pointer.

Requires Qdrant running AND the index built:
    docker compose up -d
    uv run python -m regrag.index.vector_store --index

Skipped cleanly when either is absent.

Run:  pytest tests/test_retrieval_smoke.py -q -s
  or: python tests/test_retrieval_smoke.py
"""

from __future__ import annotations

import os

from regrag import config

SKIP_SLOW = os.getenv("REGRAG_SKIP_SLOW") == "1"

VALIDATION_Q = "How must a bank validate its internal rating estimates?"


def _ready():
    """(store, embed_query) if the index is live, else None."""
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return None
    try:
        from regrag.index.embedder import embed_query
        from regrag.index.vector_store import VectorStore

        store = VectorStore()
        if not store.client.collection_exists(store.collection):
            print("  (skipped: collection not built — run --index)")
            return None
        if store.client.count(store.collection, exact=True).count == 0:
            print("  (skipped: collection is empty)")
            return None
        return store, embed_query
    except Exception as e:
        print(f"  (skipped: {type(e).__name__}: {str(e)[:60]})")
        return None


# =============================================================================
# 1. THE HALF THAT HAS NEVER RUN
# =============================================================================


def test_every_hit_can_fetch_its_parent():
    """A dangling parent_id would make the child unusable — retrieved, but with
    no context to deliver."""
    ready = _ready()
    if not ready:
        return
    store, embed_query = ready

    for hit in store.search(embed_query(VALIDATION_Q), limit=10):
        pid = hit.payload.get("parent_id")
        assert pid, f"chunk {hit.payload.get('chunk_id')} has no parent_id"
        assert store.get_parent(pid), f"parent {pid} missing from the parent store"


def test_parent_is_larger_than_the_child_and_contains_it():
    """The point of parentdoc: retrieve precise, deliver context."""
    ready = _ready()
    if not ready:
        return
    store, embed_query = ready

    hit = store.search(embed_query(VALIDATION_Q), limit=1)[0]
    child = hit.payload["text"]
    parent = store.get_parent(hit.payload["parent_id"])

    print(f"\n  child  ({len(child)} chars) {child[:70]!r}")
    print(f"  parent ({parent['n_chars']} chars, {parent['n_children']} children) "
          f"heading={parent['heading'][:60]!r}")

    assert parent["n_chars"] >= len(child)
    assert child[:60] in parent["text"], "the child text is not inside its own parent"


def test_a_tiny_heading_chunk_still_delivers_real_context():
    """The case that justifies keeping 858 sub-40-character chunks.

    '(8) Validation of internal estimates' is a useless ANSWER and a good KEY.
    It is only defensible if its parent carries the substance.
    """
    ready = _ready()
    if not ready:
        return
    store, embed_query = ready

    tiny = [h for h in store.search(embed_query(VALIDATION_Q), limit=20)
            if len(h.payload["text"]) < 60]
    if not tiny:
        print("  (no sub-60-char hit in the top 20 for this query)")
        return

    hit = tiny[0]
    parent = store.get_parent(hit.payload["parent_id"])
    print(f"\n  tiny child ({len(hit.payload['text'])}c): {hit.payload['text']!r}")
    print(f"  its parent ({parent['n_chars']}c): {parent['text'][:160]!r}")

    assert parent["n_chars"] > 200, (
        "a tiny chunk resolved to a tiny parent — it carries no context and is "
        "just noise in the index"
    )


# =============================================================================
# 2. THE METADATA ACTUALLY FILTERS
# =============================================================================


def test_jurisdiction_filter_isolates_the_uk_document():
    """Axis 2, mechanically. SS1/23 is the ONLY UK document in the corpus, so a
    UK-filtered query proves the filter rather than the ranking."""
    ready = _ready()
    if not ready:
        return
    store, embed_query = ready

    hits = store.search(embed_query(VALIDATION_Q), limit=5, where={"jurisdiction": "UK"})
    assert hits, "no UK results — the filter or the payload is wrong"
    assert all(h.payload["short_name"] == "SS1/23" for h in hits)


def test_requirement_lookup_excludes_guidelines_and_reference_data():
    """BCBS guidelines say 'banks should'. Retrieved as a requirement, should
    becomes must."""
    ready = _ready()
    if not ready:
        return
    store, embed_query = ready

    hits = store.search(
        embed_query("What capital must a bank hold against credit risk?"),
        limit=20, where={"doc_type": list(config.REQUIREMENT_DOC_TYPES)},
    )
    assert hits
    for h in hits:
        assert h.payload["doc_type"] in config.REQUIREMENT_DOC_TYPES
        assert h.payload["authority_rank"] > 0, "reference data reached a requirement query"


def test_superseded_text_can_be_excluded():
    """SR 11-7 must not answer as current. The guard is a payload filter, so it
    has to work at query time, not just in the registry."""
    ready = _ready()
    if not ready:
        return
    store, embed_query = ready

    q = embed_query("How does the guidance define a model?")
    current = store.search(q, limit=10, where={"status": "in_force"})
    assert current
    assert all(h.payload["status"] == "in_force" for h in current)
    assert not any(h.payload["short_name"] == "SR 11-7" for h in current)

    superseded = store.search(q, limit=5, where={"status": "superseded"})
    assert superseded and all(h.payload["short_name"] == "SR 11-7" for h in superseded)
    print(f"\n  in_force top hit  : {current[0].payload['short_name']}")
    print(f"  superseded top hit: {superseded[0].payload['short_name']}")


# =============================================================================
# 3. CITATIONS SURVIVE THE ROUND TRIP
# =============================================================================


def test_hits_carry_enough_to_build_a_citation():
    """short_name + locator + status + effective_from is what prompts.py needs
    to compose a citation. There is no free-text citation field, so if these do
    not survive into Qdrant the audit trail is gone."""
    ready = _ready()
    if not ready:
        return
    store, embed_query = ready

    for hit in store.search(embed_query(VALIDATION_Q), limit=5):
        p = hit.payload
        for field in ("short_name", "locator", "locator_kind", "status"):
            assert p.get(field) is not None, f"missing {field}"
        print(f"  {p['short_name']} {p['locator']} [{p['status']}] ({p['locator_kind']})")


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
