"""Vector store — Qdrant for chunks, a local file for parents.

WHY PARENTS ARE NOT IN QDRANT

Parents are never embedded and never searched. They are fetched BY ID once a
child has been retrieved. Putting them in a vector database would mean
inventing a vector for something that has no reason to have one, and paying
Qdrant's storage and indexing cost for data that only ever needs a dict lookup.
So chunks go to Qdrant; parents go to a gzipped JSON file keyed by parent_id.
1,660 parents is about 3 MB.

WHY ONE COLLECTION PER CHUNKING STRATEGY, NOT PER FRAMEWORK

Framework, jurisdiction and status are payload FILTERS. Making them collections
would mean adding a jurisdiction required new infrastructure rather than a new
filter value — which breaks "a corpus change is a DATA change". Chunking
strategy is different: it changes what a vector MEANS, so those genuinely
cannot share an index.

WHY POINT IDS ARE DERIVED, NOT RANDOM

Qdrant point ids must be an unsigned integer or a UUID, and chunk_ids are
strings like `bcbs-cre-consolidated::CRE36::0007::002`. Hashing them with
uuid5 makes the id DETERMINISTIC: re-indexing the same chunk overwrites the
same point instead of creating a duplicate. With a random id, a second run
would silently double the corpus and every search would return near-identical
neighbours.

WHY delete_by_doc MATTERS

Without a way to remove the old points, the corpus accumulates two versions of the same text and
retrieval returns both — which is precisely the version-bleed failure the whole
registry design exists to prevent. `delete_by_doc` makes an update possible
instead of a rebuild.
"""

from __future__ import annotations

import gzip
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from regrag import config

# Fixed namespace so a chunk_id always maps to the same point id, across
# machines and runs. Changing this orphans every existing point.
_NAMESPACE = uuid.UUID("6f9d1a3e-6f1e-4b0a-9d3c-1f1c8a2b7e40")

INDEX_DIR = config.DATA_DIR / "index"

# ONE CLIENT PER URL, SHARED BY EVERY VectorStore IN THE PROCESS.

_CLIENTS: dict[str, object] = {}


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


@dataclass
class StoreStats:
    collection: str
    points: int
    parents: int
    indexed_fields: list[str]

    def __str__(self) -> str:
        return (f"<{self.collection}: {self.points:,} points, "
                f"{self.parents:,} parents, {len(self.indexed_fields)} payload indexes>")


class VectorStore:
    """Chunks in Qdrant, parents on disk, one strategy per collection."""

    def __init__(self, strategy: str = "parentdoc", url: str | None = None):
        # VALIDATE FIRST, then connect. An unknown strategy is a caller error
        # and should fail on the spot — not after importing a client and
        # opening a connection, which would report it as an infrastructure
        # problem and send you looking in the wrong place.
        self.strategy = strategy
        self.collection = config.collection_name(strategy)
        self._parents_path = INDEX_DIR / f"{self.collection}_parents.json.gz"
        self._parents: dict | None = None
        self._url = url or config.QDRANT_URL
        self._client = None

    @property
    def client(self):
        """Connect lazily, so constructing a store costs nothing.

        Lets the id-derivation and naming logic be tested on a machine with no
        Qdrant running — which is most machines, most of the time.
        """
        if self._client is None:
            if self._url not in _CLIENTS:
                from qdrant_client import QdrantClient

                _CLIENTS[self._url] = QdrantClient(url=self._url)
            self._client = _CLIENTS[self._url]
        return self._client

    # ---- collection lifecycle -------------------------------------------

    def ensure_collection(self, *, recreate: bool = False) -> None:
        """Create the collection and its payload indexes if absent.

        Payload indexes are not optional at this size. Without them Qdrant
        filters by scanning every point, so a jurisdiction-scoped search reads
        all 9,935 vectors to return the one UK document.
        """
        from qdrant_client.models import Distance, VectorParams

        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=config.EMBED_DIM,
                    distance=Distance.COSINE,   # vectors are L2-normalised in embedder.py
                ),
            )
        for field in config.PAYLOAD_INDEX_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema="integer" if field == "authority_rank" else "keyword",
                )
            except Exception:
                pass  # already indexed

    # ---- writing ---------------------------------------------------------

    def upsert(self, chunks, vectors, *, batch_size: int = 256) -> int:
        """Insert or overwrite chunks. Idempotent, because ids are derived."""
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

        written = 0
        for i in range(0, len(chunks), batch_size):
            batch = [
                PointStruct(
                    id=point_id(c.chunk_id),
                    vector=v,
                    payload={**c.payload, "chunk_id": c.chunk_id, "text": c.text},
                )
                for c, v in zip(chunks[i:i + batch_size], vectors[i:i + batch_size])
            ]
            self.client.upsert(collection_name=self.collection, points=batch, wait=True)
            written += len(batch)
        return written

    def save_parents(self, parents) -> int:
        """Write the parent store. Replaces it wholesale — parents are derived
        from chunking, so a partial update would leave a stale mixture."""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            p.parent_id: {
                "doc_id": p.doc_id, "chapter": p.chapter,
                "heading": p.heading, "text": p.text,
                "n_chars": p.n_chars, "n_children": p.n_children,
            }
            for p in parents
        }
        tmp = self._parents_path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(self._parents_path)   # atomic: an interrupted write leaves the old file
        self._parents = payload
        return len(payload)

    def delete_by_doc(self, doc_id: str) -> None:
        """Remove every point for one document.

        Re-indexing a reissued document without this leaves BOTH versions in
        the collection, and retrieval returns both — version bleed introduced
        by the plumbing rather than by the corpus.
        """
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
            ),
            wait=True,
        )

    # ---- reading ---------------------------------------------------------

    def search(self, vector, *, limit: int = 10, where: dict | None = None):
        """Nearest children, optionally filtered.

        `where` maps payload fields to a value or list of values, e.g.
            {"jurisdiction": "UK"}
            {"status": "in_force", "doc_type": ["standard", "rule"]}
        Filtering happens INSIDE Qdrant, so the limit applies after the filter
        — asking for 10 UK chunks returns 10 UK chunks, not 10 chunks of which
        some happen to be UK.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        qfilter = None
        if where:
            must = []
            for key, val in where.items():
                match = MatchAny(any=list(val)) if isinstance(val, (list, tuple, set)) \
                    else MatchValue(value=val)
                must.append(FieldCondition(key=key, match=match))
            qfilter = Filter(must=must)

        return self.client.query_points(
            collection_name=self.collection,
            query=list(vector),
            query_filter=qfilter,
            limit=limit,
            with_payload=True,
        ).points

    def get_parent(self, parent_id: str) -> dict | None:
        if self._parents is None:
            if not self._parents_path.exists():
                self._parents = {}
            else:
                with gzip.open(self._parents_path, "rt", encoding="utf-8") as fh:
                    self._parents = json.load(fh)
        return self._parents.get(parent_id)

    def stats(self) -> StoreStats:
        n = self.client.count(self.collection, exact=True).count \
            if self.client.collection_exists(self.collection) else 0
        self.get_parent("")   # force the parent store to load
        return StoreStats(
            collection=self.collection,
            points=n,
            parents=len(self._parents or {}),
            indexed_fields=list(config.PAYLOAD_INDEX_FIELDS),
        )


# =============================================================================
# CLI — `python -m regrag.index.vector_store [--index] [--recreate]`
# =============================================================================

if __name__ == "__main__":
    import sys
    import time

    from regrag.index.embedder import embed_documents, embed_query
    from regrag.ingestion.cache import parse_cached
    from regrag.ingestion.chunker import chunk_document
    from regrag.ingestion.clean import clean_document
    from regrag.ingestion.sections import find_sections
    from regrag.registry import load

    args = set(sys.argv[1:])
    store = VectorStore()

    if "--index" in args:
        store.ensure_collection(recreate="--recreate" in args)
        reg = load()
        all_chunks, all_parents = [], []
        for row in reg.indexable():
            doc = parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
            cleaned = clean_document(doc, row)
            ck, pa = chunk_document(cleaned, find_sections(cleaned, row), row)
            all_chunks += ck
            all_parents += pa
            print(f"  {row.short_name:32} {len(ck):>5} chunks")

        print(f"\nembedding {len(all_chunks):,} chunks ...")
        t0 = time.perf_counter()
        vectors = embed_documents([c.text for c in all_chunks], progress=True)
        print(f"  {time.perf_counter()-t0:.0f}s")

        print("writing to Qdrant ...")
        n = store.upsert(all_chunks, vectors)
        m = store.save_parents(all_parents)
        print(f"  {n:,} points, {m:,} parents")

    print(f"\n{store.stats()}")

    if store.stats().points:
        print("\nsmoke query — 'How must a bank validate its internal rating estimates?'")
        qv = embed_query("How must a bank validate its internal rating estimates?")
        for hit in store.search(qv, limit=5):
            p = hit.payload
            print(f"  {hit.score:.3f}  {p.get('short_name'):22} {str(p.get('locator'))[:24]:24} "
                  f"{p.get('text','')[:60]!r}")

        print("\nsame query, filtered to UK only")
        for hit in store.search(qv, limit=3, where={"jurisdiction": "UK"}):
            p = hit.payload
            print(f"  {hit.score:.3f}  {p.get('short_name'):22} {p.get('text','')[:60]!r}")
