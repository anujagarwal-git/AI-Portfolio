"""Integration: registry.py + parser.py, working together.

The unit tests prove each module works ALONE. This one proves they COMPOSE —
that the registry can hand the parser a document and the parser can open it.
That is step 5 of the build arc, and it is where most self-taught projects go
soft: two modules that each pass their own tests but have never spoken.

TWO TIERS, on purpose:

  PREFLIGHT (fast, all 19 documents)
      No Docling. Checks every indexable document is a structurally intact PDF.
      This tier exists because of a real incident: BASEL_PAP.pdf was once a
      half-finished download — a valid %PDF header, no %%EOF trailer, growing
      1 MiB every 4 seconds. Docling would have failed on it deep inside a
      batch run, hours in. Catching it in two seconds is worth a test.

  PARSE (slow, ONE document)
      A real Docling conversion driven entirely off the registry. Uses the
      smallest load-bearing file, SR 26-2 (12 pages), so the test stays usable.
      Set REGRAG_SKIP_SLOW=1 to skip it.

Run:  pytest tests/test_integration.py -q
  or: python tests/test_integration.py
"""

from __future__ import annotations

import os
from pathlib import Path

from regrag import config
from regrag.ingestion.parser import parse_pdf, parse_quality_report
from regrag.registry import load

SKIP_SLOW = os.getenv("REGRAG_SKIP_SLOW") == "1"

# Smallest indexable document that carries both axis 2 and axis 3.
PARSE_TARGET = "sr-26-2-2026"


# =============================================================================
# PREFLIGHT — fast, whole corpus, no Docling
# =============================================================================


def test_every_indexable_document_is_an_intact_pdf():
    """Structural integrity of all 19 files, in about two seconds.

    Deliberately NOT a parse. The point is to fail fast and name the file,
    before a long batch run dies somewhere in the middle.
    """
    reg = load()
    problems = []

    for d in reg.indexable():
        path = config.PROJECT_ROOT / d.file
        if not path.exists():
            problems.append(f"{d.short_name}: file missing — {d.file}")
            continue

        size = path.stat().st_size
        if size < 10_000:
            problems.append(f"{d.short_name}: suspiciously small ({size} bytes)")

        head = path.read_bytes()[:5]
        if head != b"%PDF-":
            problems.append(f"{d.short_name}: not a PDF (starts with {head!r})")
            continue

        # A truncated download has a valid header and NO trailer. This is the
        # exact signature the half-downloaded BASEL_PAP.pdf showed.
        tail = path.read_bytes()[-2048:]
        if b"%%EOF" not in tail:
            problems.append(
                f"{d.short_name}: no %%EOF trailer — file is truncated or still "
                f"downloading ({size:,} bytes)"
            )

    assert not problems, "preflight failed:\n  " + "\n  ".join(problems)


def test_gate_is_what_ingestion_iterates():
    """Ingestion must loop over indexable(), never documents.

    Iterating .documents would silently parse rows the registry refused, which
    is precisely the failure the gate exists to prevent — and it would look
    like a completely normal run.
    """
    reg = load()
    assert set(d.doc_id for d in reg.indexable()) <= set(d.doc_id for d in reg.documents)
    for d in reg.indexable():
        assert d.verified, f"{d.doc_id} reached indexable() without being verified"


def test_registry_paths_are_relative_to_project_root():
    """`file` must resolve from PROJECT_ROOT, not from the caller's cwd.

    Otherwise the same registry works in a notebook and fails in a script,
    which is a miserable class of bug to chase.
    """
    for d in load().indexable():
        assert not Path(d.file).is_absolute(), f"{d.doc_id}: absolute path in registry"
        assert (config.PROJECT_ROOT / d.file).exists()


# =============================================================================
# PARSE — slow, one document, the full handoff
# =============================================================================


def test_registry_hands_parser_a_document():
    """The actual integration: registry -> path -> Docling -> quality report.

    Note what is NOT hard-coded here. The test never names a file path; it asks
    the registry for a document and uses whatever `file` that row carries. Move
    the PDF, update the registry, and this test still passes — which is the
    "corpus change is a DATA change" claim, demonstrated rather than asserted.
    """
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return

    reg = load()
    doc_row = reg.by_id(PARSE_TARGET)
    assert doc_row.verified, "parse target must be an indexable document"

    parsed = parse_pdf(config.PROJECT_ROOT / doc_row.file)
    report = parse_quality_report(parsed)

    assert report["conversion_ok"], f"{doc_row.short_name}: conversion produced nothing"
    assert report["n_body"] > 0, "everything landed in FURNITURE, nothing in BODY"
    assert report["prov_coverage"] > 0.95, (
        f"{report['prov_coverage']:.2%} of blocks have page provenance; "
        "without it a citation cannot name a page"
    )
    print(f"\n  parsed {doc_row.short_name}: {report['n_text_items']} blocks, "
          f"{report['n_section_headers']} headers, "
          f"prov {report['prov_coverage']:.1%}")


def test_parsed_document_and_registry_row_compose_into_chunk_metadata():
    """What a chunk will actually carry, assembled from both halves.

    The registry supplies WHAT the document is; the parser supplies WHAT IS
    INSIDE it. A chunk needs both. This asserts the two halves fit together
    before chunker.py exists to join them for real.
    """
    if SKIP_SLOW:
        print("  (skipped: REGRAG_SKIP_SLOW=1)")
        return

    reg = load()
    doc_row = reg.by_id(PARSE_TARGET)
    parsed = parse_pdf(config.PROJECT_ROOT / doc_row.file)

    first_body = next(
        t for t in parsed.texts if t.content_layer.name == "BODY" and t.text.strip()
    )
    page = first_body.prov[0].page_no

    payload = doc_row.payload()
    payload["page"] = page
    payload["text"] = first_body.text[:200]

    # the metadata half
    assert payload["doc_id"] == PARSE_TARGET
    assert payload["status"] == "in_force"
    assert payload["jurisdiction"] == "US"
    assert payload["is_requirement_source"] is False  # supervisory guidance
    assert "notes" not in payload and "file" not in payload

    # the citation the generator will emit for this chunk
    citation = doc_row.citation(f"p.{page}")
    assert citation.startswith("SR 26-2 p.")
    assert "in_force" in citation

    # NOT YET PRESENT — sections.py adds this, and until it does the five
    # multi-chapter Basel files cannot be indexed at all.
    assert payload["effective_from"] == "2026-04-17", (
        "SR 26-2 is single-version so the document-level date is correct here; "
        "for Basel this field is null and sections.py must supply it per chapter"
    )

    print(f"\n  chunk metadata: {citation}  page={page}  "
          f"fields={len(payload)}")


# =============================================================================
# standalone runner
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
