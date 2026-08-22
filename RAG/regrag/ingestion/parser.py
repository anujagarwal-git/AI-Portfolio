"""Parsing — owns turning a regulatory PDF into a structured DoclingDocument.

Thin wrapper over Docling. `parse_pdf` standardizes the conversion config
(digital PDF, no OCR, table structure on) and returns Docling's NATIVE
DoclingDocument — no anti-corruption layer here. Our own domain object
(the Chunk dataclass) is deferred to Stage 3, once the downstream contract
is known.

`parse_quality_report` automates checks 1-7 of the post-Docling QA tape
(check 8, the markdown spot-check, stays manual).
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def _build_converter() -> DocumentConverter:
    """One canonical converter config, so every doc is parsed identically."""
    opts = PdfPipelineOptions()
    opts.do_ocr = False              # SR 11-7 et al. are digital PDFs
    opts.do_table_structure = True   # keep tables as tables (Stage 3 needs this)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def parse_pdf(path: str | Path) -> "DoclingDocument":
    """Parse one PDF and return the native DoclingDocument.

    Raises FileNotFoundError if the path is missing (fail loud, not on
    a silently-empty document three stages later).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    result = _build_converter().convert(path)
    return result.document


def parse_quality_report(doc) -> dict:
    """Checks 1-7 of the QA tape as a dict. No assertions here — the caller
    (smoke test / notebook) decides what 'good' means; this just measures.
    """
    texts = doc.texts
    labels = Counter(t.label.name if hasattr(t.label, "name") else str(t.label)
                     for t in texts)
    layers = Counter(t.content_layer.name for t in texts)

    # provenance coverage: every text block should know its page
    with_page = sum(
        1 for t in texts if getattr(t, "prov", None) and len(t.prov) > 0
    )

    return {
        # 1. conversion ok
        "conversion_ok": doc is not None and len(texts) > 0,
        # 2. body volume
        "n_text_items": len(texts),
        "n_body": layers.get("BODY", 0),
        # 3. label distribution
        "label_counts": dict(labels),
        # 4. headers present
        "n_section_headers": labels.get("SECTION_HEADER", 0),
        # 5. BODY / FURNITURE split
        "layer_counts": dict(layers),
        # 6. tables
        "n_tables": len(getattr(doc, "tables", []) or []),
        # 7. footnotes + provenance
        "n_footnotes": labels.get("FOOTNOTE", 0),
        "prov_coverage": with_page / len(texts) if texts else 0.0,
    }


def save_processed(doc, name: str, out_dir: str | Path = "data/processed") -> dict:
    """Persist both exports with a parse-date stamp. Returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    md_path = out_dir / f"{name}__{stamp}.md"
    dict_path = out_dir / f"{name}__{stamp}.json"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
    dict_path.write_text(
        json.dumps(doc.export_to_dict(), ensure_ascii=False), encoding="utf-8"
    )
    return {"markdown": str(md_path), "dict": str(dict_path)}
