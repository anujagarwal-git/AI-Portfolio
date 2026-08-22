"""Parse cache — parse each PDF once, reuse it everywhere.

WHY THIS EXISTS.

Nothing cached parses. `save_processed()` has been in parser.py since July and
is called by no one, so every test run, every CLI run and every experiment
re-parsed from PDF. That is ~1,100 pages through Docling on CPU, with CRE alone
at 323. It is why `pytest -q` takes two minutes for a handful of documents.

Stage 3 is where that stops being tolerable. Chunking is the stage you iterate
on — try a parent cap, look, adjust, look again. Without a cache each loop pays
for CRE again. With one, the second run is a disk read.

TWO THINGS IN THE CACHE KEY, and both matter:

  source hash      Content, not mtime. mtime lied once already — four files in
                   this corpus share a timestamp within 65 seconds because they
                   were bulk-copied, so mtime says nothing about the bytes.

  docling version  A Docling upgrade CHANGES PARSE OUTPUT. We already saw its
                   grouping differ from pdftotext in ways that broke chapter
                   detection ('LEX10 Definitions' as one item vs two). A cache
                   entry written by a different version is not the same parse,
                   and silently reusing it would make a version upgrade
                   invisible — the worst kind of change.

A cache that can serve a stale entry is worse than no cache, because the run
looks normal. Both keys are in the filename, so a mismatch MISSES rather than
being detected later.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from regrag import config

CACHE_DIR = config.PROCESSED_DIR
CACHE_FORMAT = 1  # bump if the envelope below changes shape


class CacheError(Exception):
    pass


def docling_version() -> str:
    try:
        return version("docling")
    except PackageNotFoundError:  # pragma: no cover
        return "unknown"


def source_hash(path: Path) -> str:
    """SHA-256 of the file bytes, first 16 hex chars.

    Streamed rather than read whole: BASEL_PAP was 14 MB, and an image-only
    variant of it was larger still.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def cache_path(doc_id: str, sha: str, dv: str | None = None) -> Path:
    dv = (dv or docling_version()).replace(".", "-")
    return CACHE_DIR / f"{doc_id}__{sha}__docling-{dv}.json.gz"


# =============================================================================
# PUBLIC API
# =============================================================================


def parse_cached(path: Path | str, doc_id: str, *, refresh: bool = False, verbose: bool = True):
    """Return a DoclingDocument, from cache when the key matches.

    `doc_id` only names the cache file; correctness rests entirely on the
    source hash and the Docling version.
    """
    from regrag.ingestion.parser import parse_pdf  # local: keeps import graph flat

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    sha = source_hash(path)
    cp = cache_path(doc_id, sha)

    if cp.exists() and not refresh:
        if verbose:
            print(f"  cache HIT  {doc_id}  ({cp.name})")
        return _load(cp)

    if verbose:
        print(f"  cache MISS {doc_id}  parsing {path.name} ...")
    doc = parse_pdf(path)
    _store(cp, doc, doc_id=doc_id, sha=sha, source=str(path))
    return doc


def cache_status(rows) -> list[dict]:
    """One row per registry document: is it cached, and is the entry current?"""
    out = []
    for row in rows:
        p = config.PROJECT_ROOT / row.file
        if not p.exists():
            out.append({"doc_id": row.doc_id, "short_name": row.short_name,
                        "state": "SOURCE MISSING", "size_mb": None})
            continue
        sha = source_hash(p)
        cp = cache_path(row.doc_id, sha)
        stale = sorted(CACHE_DIR.glob(f"{row.doc_id}__*.json.gz")) if CACHE_DIR.exists() else []
        stale = [s for s in stale if s.name != cp.name]
        out.append({
            "doc_id": row.doc_id,
            "short_name": row.short_name,
            "state": "cached" if cp.exists() else "not cached",
            "size_mb": round(cp.stat().st_size / 1e6, 1) if cp.exists() else None,
            "stale_entries": [s.name for s in stale],
        })
    return out


def purge_stale(rows, *, dry_run: bool = True) -> list[str]:
    """Remove cache entries whose key no longer matches any current source.

    Stale entries are harmless — a mismatched key simply misses — but they
    accumulate a copy per PDF revision and per Docling upgrade.
    """
    keep = set()
    for row in rows:
        p = config.PROJECT_ROOT / row.file
        if p.exists():
            keep.add(cache_path(row.doc_id, source_hash(p)).name)
    removed = []
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.json.gz"):
            if f.name not in keep:
                removed.append(f.name)
                if not dry_run:
                    f.unlink()
    return removed


# =============================================================================
# INTERNALS
# =============================================================================


def _store(cp: Path, doc, *, doc_id: str, sha: str, source: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    envelope = {
        "cache_format": CACHE_FORMAT,
        "doc_id": doc_id,
        "source": source,
        "source_sha256_16": sha,
        "docling_version": docling_version(),
        "parsed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "document": doc.export_to_dict(),
    }
    tmp = cp.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(envelope, fh)
    tmp.replace(cp)  # atomic: an interrupted write never leaves a usable cache file


def _load(cp: Path):
    with gzip.open(cp, "rt", encoding="utf-8") as fh:
        envelope = json.load(fh)

    if envelope.get("cache_format") != CACHE_FORMAT:
        raise CacheError(f"{cp.name}: cache_format {envelope.get('cache_format')} "
                         f"!= {CACHE_FORMAT}; delete data/processed and re-warm")

    # NOTE: docling-core's DoclingDocument is a pydantic model, so model_validate
    # is the documented way back. If a Docling upgrade changes this, the round-trip
    # test fails loudly here rather than returning a half-built object.
    from docling_core.types.doc import DoclingDocument

    try:
        return DoclingDocument.model_validate(envelope["document"])
    except Exception as e:
        raise CacheError(
            f"{cp.name}: could not rebuild a DoclingDocument from the cached dict "
            f"({type(e).__name__}: {e}). Check the docling-core API for the current "
            "way to deserialise, then bump CACHE_FORMAT."
        ) from None


# =============================================================================
# CLI — `python -m regrag.ingestion.cache [--warm] [--status] [--purge]`
# =============================================================================

if __name__ == "__main__":
    import sys

    from regrag.registry import load

    args = set(sys.argv[1:])
    reg = load()
    rows = reg.indexable()

    if not args or "--status" in args:
        print(f"\nParse cache — {CACHE_DIR}\n")
        print(f"  docling {docling_version()}\n")
        print(f"  {'doc_id':28} {'state':12} {'size':>8}")
        print("  " + "-" * 52)
        total = 0.0
        for s in cache_status(rows):
            size = f"{s['size_mb']} MB" if s["size_mb"] else "-"
            total += s["size_mb"] or 0
            print(f"  {s['doc_id']:28} {s['state']:12} {size:>8}")
            for st in s.get("stale_entries", []):
                print(f"      stale: {st}")
        n = sum(1 for s in cache_status(rows) if s["state"] == "cached")
        print(f"\n  {n}/{len(rows)} cached, {total:.1f} MB total")
        if not args:
            print("\n  --warm   parse everything not yet cached (slow, once)")
            print("  --purge  delete entries whose key no longer matches")

    if "--warm" in args:
        print("\nWarming cache. CRE is 323 pages — this is the slow one.\n")
        for row in sorted(rows, key=lambda r: (config.PROJECT_ROOT / r.file).stat().st_size):
            parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, refresh="--refresh" in args)
        print("\nDone. Subsequent runs read from disk.")

    if "--purge" in args:
        stale = purge_stale(rows, dry_run=False)
        print(f"\nremoved {len(stale)} stale entries")
        for s in stale:
            print(f"  {s}")
