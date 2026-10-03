"""Tests for regrag.index.vector_store.

Two tiers. The ID-derivation tests need nothing running. The rest need Qdrant
on localhost:6333 (`docker compose up -d`) and are skipped cleanly if it is
not there, so the suite still passes on a machine without Docker.

They use a THROWAWAY collection, never the real one — a test that wipes your
index is worse than no test.

Run:  pytest tests/test_vector_store.py -q -s
  or: python tests/test_vector_store.py
"""

from __future__ import annotations

import os

from regrag import config
from regrag.index.vector_store import VectorStore, point_id

SKIP_SLOW = os.getenv("REGRAG_SKIP_SLOW") == "1"
TEST_STRATEGY = "recursive"        # a real strategy name, but not the one in use


class _Chunk:
    """Minimal stand-in for chunker.Chunk."""

    def __init__(self, chunk_id, text, payload):
        self.chunk_id = chunk_id
        self.text = text
        self.payload = payload


def _chunk(i, doc_id="test-doc", **payload):
    base = {
        "doc_id": doc_id, "short_name": "Test Doc", "subject": "CAPITAL_ADEQUACY",
        "framework": "BASEL_FRAMEWORK", "jurisdiction": "GLOBAL", "issuer": "BCBS",
        "doc_type": "standard", "status": "in_force", "applicability": "ALL",
        "authority_rank": 2, "locator": f"CRE36.{i}",
    }
    base.update(payload)
    return _Chunk(f"{doc_id}::CRE36::0000::{i:03d}", f"Provision number {i}.", base)


def _vec(seed: int) -> list[float]:
    """A deterministic unit vector — no model needed."""
    v = [((seed * 7 + j * 13) % 100) / 100 for j in range(config.EMBED_DIM)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def _store():
    """A store on the throwaway collection, or None if Qdrant is unreachable."""
    if SKIP_SLOW:
        return None
    try:
        s = VectorStore(strategy=TEST_STRATEGY)
        s.client.get_collections()
        return s
    except Exception as e:
        print(f"  (skipped: Qdrant unreachable — {type(e).__name__})")
        return None


# =============================================================================
# 1. POINT IDS — no Qdrant needed, and the most consequential detail
# =============================================================================


def test_point_id_is_deterministic():
    """Qdrant ids must be a UUID or an integer, so string chunk_ids get hashed.
    That hash MUST be stable: re-indexing the same chunk has to overwrite the
    same point. With a random id, a second run silently doubles the corpus and
    every search returns near-duplicate neighbours."""
    cid = "bcbs-cre-consolidated::CRE36::0007::002"
    assert point_id(cid) == point_id(cid)


def test_different_chunks_get_different_ids():
    ids = {point_id(f"doc::CRE36::0000::{i:03d}") for i in range(500)}
    assert len(ids) == 500, "id collision"


def test_point_id_is_a_valid_uuid():
    import uuid as _uuid
    _uuid.UUID(point_id("any::chunk::id"))


# =============================================================================
# 2. COLLECTION SETUP
# =============================================================================


def test_collection_name_comes_from_config():
    """Never hard-coded — the strategy is part of the name, because a vector
    means something different under a different chunking strategy."""
    assert VectorStore(strategy="parentdoc").collection == config.collection_name("parentdoc")
    assert "parentdoc" in config.collection_name("parentdoc")


def test_unknown_strategy_is_rejected():
    try:
        VectorStore(strategy="made-up")
    except ValueError:
        return
    raise AssertionError("an unknown strategy must not silently create a collection")


def test_ensure_collection_creates_payload_indexes():
    """Without payload indexes Qdrant filters by scanning every point, so a
    UK-scoped search reads all 9,935 vectors to return one document."""
    s = _store()
    if s is None:
        return
    s.ensure_collection(recreate=True)
    info = s.client.get_collection(s.collection)
    indexed = set((info.payload_schema or {}).keys())
    missing = [f for f in config.PAYLOAD_INDEX_FIELDS if f not in indexed]
    print(f"\n  indexed payload fields: {sorted(indexed)}")
    assert not missing, f"not indexed: {missing}"


# =============================================================================
# 3. WRITE, SEARCH, FILTER
# =============================================================================


def test_upsert_is_idempotent():
    """THE reason ids are derived. Indexing the same chunks twice must leave
    the same number of points, not double them."""
    s = _store()
    if s is None:
        return
    s.ensure_collection(recreate=True)
    chunks = [_chunk(i) for i in range(20)]
    vecs = [_vec(i) for i in range(20)]

    s.upsert(chunks, vecs)
    first = s.client.count(s.collection, exact=True).count
    s.upsert(chunks, vecs)
    second = s.client.count(s.collection, exact=True).count

    assert first == second == 20, f"{first} then {second} — points duplicated"


def test_mismatched_lengths_raise():
    s = _store()
    if s is None:
        return
    try:
        s.upsert([_chunk(0)], [_vec(0), _vec(1)])
    except ValueError:
        return
    raise AssertionError("chunk/vector length mismatch must raise, not silently truncate")


def test_filters_are_applied_inside_qdrant():
    """The limit must apply AFTER the filter. Asking for 5 UK chunks should
    return 5 UK chunks — not 5 chunks of which some happen to be UK."""
    s = _store()
    if s is None:
        return
    s.ensure_collection(recreate=True)
    chunks, vecs = [], []
    for i in range(30):
        j = "UK" if i % 5 == 0 else "US"
        chunks.append(_chunk(i, jurisdiction=j))
        vecs.append(_vec(i))
    s.upsert(chunks, vecs)

    hits = s.search(_vec(3), limit=5, where={"jurisdiction": "UK"})
    assert len(hits) == 5, f"got {len(hits)} — limit applied before the filter?"
    assert all(h.payload["jurisdiction"] == "UK" for h in hits)


def test_filter_accepts_a_list_of_values():
    """requirement_lookup needs doc_type IN (standard, rule)."""
    s = _store()
    if s is None:
        return
    s.ensure_collection(recreate=True)
    chunks, vecs = [], []
    for i, dt in enumerate(["standard", "rule", "guideline", "reference_data"] * 5):
        chunks.append(_chunk(i, doc_type=dt))
        vecs.append(_vec(i))
    s.upsert(chunks, vecs)

    hits = s.search(_vec(1), limit=20, where={"doc_type": ["standard", "rule"]})
    kinds = {h.payload["doc_type"] for h in hits}
    assert kinds <= {"standard", "rule"}, f"guidelines leaked in: {kinds}"


def test_payload_carries_text_and_chunk_id():
    """Retrieval must be able to return the text and trace it back."""
    s = _store()
    if s is None:
        return
    s.ensure_collection(recreate=True)
    s.upsert([_chunk(1)], [_vec(1)])
    hit = s.search(_vec(1), limit=1)[0]

    assert hit.payload["text"].startswith("Provision number 1")
    assert hit.payload["chunk_id"].endswith("::001")
    assert hit.payload["locator"] == "CRE36.1"


# =============================================================================
# 4. delete_by_doc — what makes an update possible instead of a rebuild
# =============================================================================


def test_delete_by_doc_removes_only_that_document():
    """A regulator reissues a document. Without this, re-indexing leaves BOTH
    versions and retrieval returns both — version bleed created by the
    plumbing rather than by the corpus."""
    s = _store()
    if s is None:
        return
    s.ensure_collection(recreate=True)
    chunks = [_chunk(i, doc_id="doc-a") for i in range(10)]
    chunks += [_chunk(i, doc_id="doc-b") for i in range(10)]
    s.upsert(chunks, [_vec(i) for i in range(20)])
    assert s.client.count(s.collection, exact=True).count == 20

    s.delete_by_doc("doc-a")
    remaining = s.client.count(s.collection, exact=True).count
    assert remaining == 10, f"{remaining} left — wrong points removed"

    hits = s.search(_vec(0), limit=20)
    assert all(h.payload["doc_id"] == "doc-b" for h in hits)


def test_reindexing_a_document_after_delete_does_not_duplicate():
    s = _store()
    if s is None:
        return
    s.ensure_collection(recreate=True)
    chunks = [_chunk(i, doc_id="doc-a") for i in range(10)]
    vecs = [_vec(i) for i in range(10)]
    s.upsert(chunks, vecs)
    s.delete_by_doc("doc-a")
    s.upsert(chunks, vecs)

    assert s.client.count(s.collection, exact=True).count == 10


# =============================================================================
# 5. PARENT STORE
# =============================================================================


class _Parent:
    def __init__(self, pid):
        self.parent_id = pid
        self.doc_id = "test-doc"
        self.chapter = "CRE36"
        self.heading = "Section 8: validation of internal estimates"
        self.text = "The full section text delivered as context."
        self.n_chars = 43
        self.n_children = 3


def test_parents_round_trip():
    """Parents are fetched by id after retrieval — never embedded, never
    searched — so they live on disk rather than in a vector database."""
    s = _store()
    if s is None:
        return
    n = s.save_parents([_Parent(f"p{i}") for i in range(5)])
    assert n == 5

    s._parents = None                       # force a re-read from disk
    got = s.get_parent("p3")
    assert got and got["chapter"] == "CRE36"
    assert "delivered as context" in got["text"]
    assert s.get_parent("does-not-exist") is None


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
