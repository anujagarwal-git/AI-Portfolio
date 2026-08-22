"""Tests for regrag.ingestion.cache.

A cache that can serve a STALE entry is worse than no cache, because the run
looks completely normal. So most of these tests are about MISSING correctly:
a changed PDF must miss, and a different Docling version must miss.

The round-trip test is the one that matters most and the one I could not verify
while writing it — rebuilding a DoclingDocument from `export_to_dict()` depends
on docling-core's current API. If that call is wrong, this test fails loudly
here rather than the pipeline silently receiving a half-built object.

Run:  pytest tests/test_cache.py -q
  or: python tests/test_cache.py
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path

from regrag import config
from regrag.ingestion import cache
from regrag.registry import load

SKIP_SLOW = os.getenv("REGRAG_SKIP_SLOW") == "1"

# Smallest indexable PDF, so the one real parse in this file stays cheap.
SMALL_DOC = "sr-26-2-2026"


# =============================================================================
# 1. THE KEY — what must MISS
# =============================================================================


def test_key_includes_source_hash_and_docling_version():
    a = cache.cache_path("x", "aaaa1111", "2.113.0")
    b = cache.cache_path("x", "bbbb2222", "2.113.0")
    c = cache.cache_path("x", "aaaa1111", "2.200.0")

    assert a != b, "a different PDF must map to a different cache file"
    assert a != c, "a different Docling version must map to a different cache file"
    assert "aaaa1111" in a.name and "docling-2-113-0" in a.name


def test_hash_is_content_not_mtime():
    """mtime already lied in this corpus: four PDFs share a timestamp within 65
    seconds because they were bulk-copied. Only content can key a parse."""
    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = Path(tmp) / "a.pdf", Path(tmp) / "b.pdf"
        p1.write_bytes(b"%PDF-1.7 same bytes")
        p2.write_bytes(b"%PDF-1.7 same bytes")
        os.utime(p2, (0, 0))  # wildly different mtime, identical content

        assert cache.source_hash(p1) == cache.source_hash(p2)

        p2.write_bytes(b"%PDF-1.7 different bytes")
        assert cache.source_hash(p1) != cache.source_hash(p2)


def test_edited_pdf_misses_the_cache():
    """Re-download a document and the old parse must NOT be served."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "doc.pdf"
        p.write_bytes(b"%PDF-1.7 v1")
        before = cache.cache_path("d", cache.source_hash(p))

        p.write_bytes(b"%PDF-1.7 v2 with a new chapter")
        after = cache.cache_path("d", cache.source_hash(p))

        assert before != after


# =============================================================================
# 2. THE ENVELOPE
# =============================================================================


def test_cache_format_mismatch_raises_rather_than_guessing():
    """An envelope written by older code must not be reinterpreted."""
    with tempfile.TemporaryDirectory() as tmp:
        cp = Path(tmp) / "x.json.gz"
        with gzip.open(cp, "wt", encoding="utf-8") as fh:
            json.dump({"cache_format": 0, "document": {}}, fh)

        try:
            cache._load(cp)
        except cache.CacheError as e:
            assert "cache_format" in str(e)
            return
        raise AssertionError("stale cache_format must raise")


def test_envelope_records_provenance():
    """A cached parse must say where it came from and what produced it —
    otherwise a suspect chunk cannot be traced back to a parse."""
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return

    row = load().by_id(SMALL_DOC)
    cache.parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
    cp = cache.cache_path(row.doc_id, cache.source_hash(config.PROJECT_ROOT / row.file))

    with gzip.open(cp, "rt", encoding="utf-8") as fh:
        env = json.load(fh)

    for key in ("cache_format", "doc_id", "source", "source_sha256_16",
                "docling_version", "parsed_at", "document"):
        assert key in env, f"envelope missing {key}"
    assert env["doc_id"] == SMALL_DOC


# =============================================================================
# 3. ROUND TRIP — the test I could not verify while writing it
# =============================================================================


def test_cached_parse_matches_a_fresh_parse():
    """THE test. A cached DoclingDocument must be indistinguishable from a
    fresh one where the pipeline is concerned: same text items, same labels,
    same page provenance. If docling-core's deserialisation API differs from
    what cache._load assumes, this is where it surfaces."""
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return

    from regrag.ingestion.parser import parse_pdf

    row = load().by_id(SMALL_DOC)
    path = config.PROJECT_ROOT / row.file

    fresh = parse_pdf(path)
    cache.parse_cached(path, row.doc_id, refresh=True, verbose=False)
    restored = cache.parse_cached(path, row.doc_id, verbose=False)

    assert len(restored.texts) == len(fresh.texts), "item count changed through the cache"
    for a, b in zip(fresh.texts, restored.texts):
        assert a.text == b.text
        assert a.label == b.label
        pa = a.prov[0].page_no if getattr(a, "prov", None) else None
        pb = b.prov[0].page_no if getattr(b, "prov", None) else None
        assert pa == pb, "page provenance lost — a chunk could not cite its page"


def test_cached_parse_still_works_downstream():
    """clean -> sections must behave identically on a cached document."""
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return

    from regrag.ingestion.clean import clean_document
    from regrag.ingestion.sections import find_sections

    row = load().by_id(SMALL_DOC)
    doc = cache.parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
    secs = find_sections(clean_document(doc, row), row)

    assert len(secs) == 1 and secs[0].kind == "whole_document"


def test_second_call_does_not_reparse():
    """The whole point: the second call must be a disk read."""
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return

    import time

    row = load().by_id(SMALL_DOC)
    path = config.PROJECT_ROOT / row.file

    cache.parse_cached(path, row.doc_id, refresh=True, verbose=False)  # warm
    t0 = time.perf_counter()
    cache.parse_cached(path, row.doc_id, verbose=False)
    hit = time.perf_counter() - t0

    print(f"\n  cache hit took {hit:.2f}s")
    assert hit < 5.0, f"cache hit took {hit:.1f}s — that is not a disk read"


# =============================================================================
# 4. HOUSEKEEPING
# =============================================================================


def test_purge_is_dry_run_by_default():
    """Deleting cache entries is cheap to redo but annoying to do by accident."""
    rows = load().indexable()
    before = sorted(cache.CACHE_DIR.glob("*.json.gz")) if cache.CACHE_DIR.exists() else []
    cache.purge_stale(rows, dry_run=True)
    after = sorted(cache.CACHE_DIR.glob("*.json.gz")) if cache.CACHE_DIR.exists() else []

    assert before == after, "dry_run deleted files"


def test_cache_status_covers_every_indexable_document():
    rows = load().indexable()
    st = cache.cache_status(rows)

    assert len(st) == len(rows)
    assert {s["doc_id"] for s in st} == {r.doc_id for r in rows}
    for s in st:
        assert s["state"] in {"cached", "not cached", "SOURCE MISSING"}


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
