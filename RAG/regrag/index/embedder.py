"""Embedding — text to vectors with bge-small, on CPU.

THE ONE THING THIS MODULE EXISTS TO GET RIGHT: BGE IS ASYMMETRIC.

A query and a document are NOT embedded the same way. BGE was trained with an
instruction prefix on the query side only:

    query     "Represent this sentence for searching relevant passages: " + text
    document  text, with no prefix at all

Put the prefix on documents, or leave it off queries, and nothing breaks. No
exception, no warning, no empty result set. Retrieval simply gets worse — the
query and document vectors sit in slightly different regions of the space, and
the top-k comes back plausible but wrong. It is the single most common silent
defect in a BGE pipeline, which is why `embed_query` and `embed_documents` are
separate functions rather than one function with a flag: it is hard to call the
wrong one by accident when they have different names.

NORMALISATION. Vectors are L2-normalised, so a dot product IS cosine
similarity. Qdrant's COSINE distance then costs nothing extra, and scores are
comparable across queries. Both sides must be normalised or the comparison is
meaningless — so it happens here, once, rather than being left to callers.

DETERMINISM. The model runs in eval mode with no sampling, so the same text
always gives the same vector. That matters for the same reason temperature 0
matters in generation: without it, two evaluation runs are not comparable and a
metric change cannot be attributed to a code change.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from regrag import config

_MODEL = None


def _model():
    """Load once per process. The first call downloads ~130 MB and takes a
    while; every call after it is instant."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        _MODEL.eval()
    return _MODEL


def embed_documents(texts: list[str], *, batch_size: int = 32,
                    progress: bool = False) -> list[list[float]]:
    """Embed CHUNK text. No instruction prefix — this is the document side."""
    if not texts:
        return []
    vecs = _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=progress,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vecs]


def embed_query(text: str) -> list[float]:
    """Embed a QUESTION. Instruction prefix applied — this is the query side.

    Never call embed_documents() on a question, or embed_query() on a chunk.
    """
    vec = _model().encode(
        [config.BGE_QUERY_INSTRUCTION + text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]
    return vec.tolist()


# =============================================================================
# THROUGHPUT — measure before committing to a full run
# =============================================================================


@dataclass
class Throughput:
    n: int
    seconds: float

    @property
    def per_second(self) -> float:
        return self.n / self.seconds if self.seconds else 0.0

    def estimate(self, total: int) -> float:
        """Seconds to embed `total` chunks at this rate."""
        return total / self.per_second if self.per_second else float("inf")

    def __str__(self) -> str:
        return f"{self.n} chunks in {self.seconds:.1f}s = {self.per_second:.1f}/s"


def measure_throughput(texts: list[str], *, batch_size: int = 32) -> Throughput:
    """Time a real subset.

    The first batch also pays for model loading, so warm up first — otherwise
    the rate is wrong by however long the load took, and every estimate built
    on it is wrong too.
    """
    _model()
    embed_documents(texts[: min(8, len(texts))], batch_size=batch_size)  # warm

    t0 = time.perf_counter()
    embed_documents(texts, batch_size=batch_size, progress=False)
    return Throughput(n=len(texts), seconds=time.perf_counter() - t0)


def sanity_check() -> dict:
    """Prove the asymmetry is wired correctly before indexing anything.

    A related question and passage should score clearly higher than an
    unrelated pair. If they do not, the prefix is on the wrong side — and that
    failure would otherwise show up only as mediocre retrieval much later.
    """
    passage = (
        "Banks must have a robust system in place to validate the accuracy and "
        "consistency of rating systems and processes."
    )
    unrelated = (
        "Large exposures regulation limits the maximum loss a bank could face "
        "in the event of a sudden counterparty failure."
    )
    question = "What must banks do to validate their internal rating systems?"

    dv = embed_documents([passage, unrelated])
    qv = embed_query(question)
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))

    related = dot(qv, dv[0])
    other = dot(qv, dv[1])

    # Same text through both paths must NOT be identical — that would mean the
    # instruction prefix is not being applied at all.
    prefix_applied = embed_query(passage) != dv[0]

    return {
        "related_score": round(related, 4),
        "unrelated_score": round(other, 4),
        "margin": round(related - other, 4),
        "query_prefix_applied": prefix_applied,
        "vector_dim": len(qv),
        "normalised": abs(dot(qv, qv) - 1.0) < 1e-4,
    }


# =============================================================================
# CLI — `python -m regrag.index.embedder`
#
# Measures throughput on a real subset and estimates the full run, WITHOUT
# committing to it. That number decides how freely the corpus can be re-indexed.
# =============================================================================

if __name__ == "__main__":
    import sys

    from regrag.ingestion.cache import parse_cached
    from regrag.ingestion.chunker import chunk_document
    from regrag.ingestion.clean import clean_document
    from regrag.ingestion.sections import find_sections
    from regrag.registry import load

    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    print("\n1. SANITY CHECK — is the query/document asymmetry wired correctly?\n")
    for k, v in sanity_check().items():
        print(f"   {k:24} {v}")
    print("\n   related_score must clearly exceed unrelated_score, and")
    print("   query_prefix_applied must be True. If not, STOP.\n")

    print("2. COLLECTING CHUNKS from the cached corpus ...")
    reg = load()
    texts: list[str] = []
    for row in reg.indexable():
        doc = parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
        cleaned = clean_document(doc, row)
        chunks, _ = chunk_document(cleaned, find_sections(cleaned, row), row)
        texts += [c.text for c in chunks]
    print(f"   {len(texts):,} chunks\n")

    sample = texts[:n_sample]
    print(f"3. THROUGHPUT on {len(sample)} chunks ...")
    tp = measure_throughput(sample)
    full = tp.estimate(len(texts))
    print(f"   {tp}")
    print(f"   full corpus ({len(texts):,} chunks): ~{full/60:.1f} minutes")
    print(f"   vectors: {len(texts) * config.EMBED_DIM * 4 / 1e6:.1f} MB\n")
    print("   Under ~10 min means the index can be rebuilt freely — tables,")
    print("   chunker changes and parameter sweeps all become cheap to redo.")
