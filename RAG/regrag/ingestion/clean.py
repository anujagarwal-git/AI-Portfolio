"""Cleaning — removes extraction artifacts without removing content.

THIS IS THE ONLY DESTRUCTIVE STEP IN THE PIPELINE. Everything else adds
metadata; this one deletes text. And the dangerous direction is over-cleaning,
because over-cleaning is INVISIBLE: text that has lost something still reads
fluently, still embeds, still retrieves, and is silently wrong.

The concrete example that shaped this module. CRE contains:

    Time period parameters: M{{i}}, E{{i}}, S{{i}} and T{{i}}

Those are mathematical subscripts in the counterparty credit risk formulas.
CRE also contains {{IRB}}, which IS a glossary link. Same syntax, two meanings.
Strip {{...}} wholesale and the formula becomes "M, E, S and T" — still looks
like a formula, no longer means anything, and nothing anywhere reports it.
So the rule is UNWRAP, NEVER DELETE: {{X}} -> X. Right for the glossary link,
survivable for the subscript.

TWO KINDS OF CLEANING, and they carry very different risk:

  BY LABEL (safe)   Docling already classified page furniture as PAGE_FOOTER /
                    PAGE_HEADER. Dropping those uses its structural judgement,
                    not a guess about text.

  BY PATTERN (risky) Footnote markers arrive corrupted — SR 15-19 has 63 clean
                    '[Footnote' markers, while SR 11-7's font encoding mangles
                    the same thing into '[Fo tn1oe'. Patterns that tolerate
                    corruption also tolerate false positives, so every rule is
                    COUNTED and anything that looks like a marker but matches
                    no known pattern is REPORTED rather than silently left or
                    silently removed.

ORDER: clean -> sections -> chunk. Cleaning first means chapter detection is
not searching past page footers, and chunk boundaries are never influenced by
an artifact. CleanedDoc exposes `.texts`, so find_sections() consumes it
unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from regrag.domain import Document

# ---- what gets dropped whole, using Docling's own classification -------------
DROP_LABELS = {"PAGE_FOOTER", "PAGE_HEADER"}
FOOTNOTE_LABELS = {"FOOTNOTE"}

# ---- footnote markers -------------------------------------------------------
# Observed forms, and ONLY observed forms:
#   '...framework,2[Footnote'   SR 15-19, 63 occurrences, intact
#   '...making.[Fo tn1oe They'  SR 11-7, characters transposed by the font
_MARKER_CLEAN = re.compile(r"\d*\[Footnote\s*", re.IGNORECASE)
_MARKER_MANGLED = re.compile(r"\[F\s*o\s*t?\s*n\s*\d*\s*o?\s*e\s*", re.IGNORECASE)

# Anything starting '[Fo' that neither pattern removed is an UNKNOWN corruption.
# Reported, never guessed at — a new mangling should be looked at, not absorbed
# by loosening a regex until it disappears.
_MARKER_RESIDUE = re.compile(r"\[F\s*o", re.IGNORECASE)

# ---- glossary links and subscripts ------------------------------------------
_BRACES = re.compile(r"\{\{\s*([^{}]{1,40}?)\s*\}\}")

_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
              "ﬄ": "ffl", "’": "'", "“": '"', "”": '"',
              "–": "-", "—": "-", " ": " "}

_WS = re.compile(r"[ \t]{2,}")


@dataclass
class CleanItem:
    """One text block, after cleaning.

    parser.py deliberately returned Docling's native object and deferred a
    domain type "until the downstream contract is known". This is that type
    arriving — the seam where the pipeline stops depending on Docling's classes.

    `orig_idx` is kept so any chunk can be traced back to the exact text block
    it came from, in the unmodified parse.
    """

    text: str
    label: str
    page: int | None
    is_footnote: bool
    orig_idx: int

    @property
    def prov(self):  # keeps find_sections()/_page_span() working unchanged
        return [_Prov(self.page)] if self.page is not None else []


@dataclass
class _Prov:
    page_no: int


@dataclass
class CleanedDoc:
    """Exposes `.texts` so find_sections() consumes it with no changes."""

    texts: list[CleanItem]
    report: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.texts)


# =============================================================================
# PUBLIC API
# =============================================================================


def clean_document(doc, row: Document) -> CleanedDoc:
    """Return a cleaned copy of `doc`. The original is never mutated."""
    counts = {
        "dropped_page_furniture": 0,
        "dropped_empty": 0,
        "footnote_markers_clean": 0,
        "footnote_markers_mangled": 0,
        "braces_unwrapped": 0,
        "ligatures_normalised": 0,
        "whitespace_collapsed": 0,
    }
    residue: list[str] = []
    items: list[CleanItem] = []
    footnote_lengths: list[int] = []

    for idx, raw in enumerate(doc.texts):
        label = _label(raw)

        if label in DROP_LABELS:
            counts["dropped_page_furniture"] += 1
            continue

        text = (getattr(raw, "text", "") or "")
        cleaned = _clean_text(text, counts, residue)

        if not cleaned.strip():
            counts["dropped_empty"] += 1
            continue

        is_fn = label in FOOTNOTE_LABELS
        if is_fn:
            footnote_lengths.append(len(cleaned))

        items.append(
            CleanItem(
                text=cleaned,
                label=label,
                page=_page(raw),
                is_footnote=is_fn,
                orig_idx=idx,
            )
        )

    report = {
        "doc_id": row.doc_id,
        "short_name": row.short_name,
        "items_in": len(doc.texts),
        "items_out": len(items),
        "chars_in": sum(len(getattr(t, "text", "") or "") for t in doc.texts),
        "chars_out": sum(len(i.text) for i in items),
        "n_footnotes": len(footnote_lengths),
        "footnote_len_min": min(footnote_lengths) if footnote_lengths else None,
        "footnote_len_median": (
            sorted(footnote_lengths)[len(footnote_lengths) // 2] if footnote_lengths else None
        ),
        "footnote_len_max": max(footnote_lengths) if footnote_lengths else None,
        # Unknown '[Fo' corruptions. NOT an error — a prompt to go and look.
        "unknown_marker_residue": residue[:10],
        "n_unknown_marker_residue": len(residue),
        **counts,
    }
    report["chars_removed_pct"] = (
        round(100 * (1 - report["chars_out"] / report["chars_in"]), 2)
        if report["chars_in"] else 0.0
    )
    return CleanedDoc(texts=items, report=report)


def clean_report(doc, row: Document) -> dict:
    """Measurements only, same contract as parse_quality_report/sections_report."""
    return clean_document(doc, row).report


# =============================================================================
# INTERNALS
# =============================================================================


def _label(item) -> str:
    lbl = getattr(item, "label", None)
    return getattr(lbl, "name", None) or str(lbl or "UNKNOWN")


def _page(item) -> int | None:
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def _clean_text(text: str, counts: dict, residue: list[str]) -> str:
    original = text

    text, n = _MARKER_CLEAN.subn(" ", text)
    counts["footnote_markers_clean"] += n

    text, n = _MARKER_MANGLED.subn(" ", text)
    counts["footnote_markers_mangled"] += n

    for m in _MARKER_RESIDUE.finditer(text):
        residue.append(text[max(0, m.start() - 20): m.start() + 30])

    # UNWRAP, never delete — {{IRB}} is a glossary link, {{i}} is a subscript.
    text, n = _BRACES.subn(r"\1", text)
    counts["braces_unwrapped"] += n

    before = text
    for bad, good in _LIGATURES.items():
        text = text.replace(bad, good)
    if text != before:
        counts["ligatures_normalised"] += 1

    text = unicodedata.normalize("NFKC", text)

    before = text
    text = _WS.sub(" ", text).strip()
    if text != before:
        counts["whitespace_collapsed"] += 1

    return text if text != original or True else original


# =============================================================================
# CLI — `python -m regrag.ingestion.clean [doc_id ...]`
# Defaults to a small, fast, representative set rather than all 19 documents.
# =============================================================================

if __name__ == "__main__":
    import sys

    from regrag import config
    from regrag.ingestion.parser import parse_pdf
    from regrag.registry import load

    reg = load()
    # SR 11-7 = mangled markers; SR 15-19 = 63 intact markers; LEX = page
    # footers + Basel layout. Roughly 90 pages total.
    doc_ids = sys.argv[1:] or [
        "sr-11-7-2011", "sr-15-19-2015r2021", "bcbs-lex-consolidated"
    ]

    for doc_id in doc_ids:
        row = reg.by_id(doc_id)
        parsed = parse_pdf(config.PROJECT_ROOT / row.file)
        rep = clean_report(parsed, row)
        print(f"\n=== {rep['short_name']} ===")
        for k in ("items_in", "items_out", "chars_in", "chars_out",
                  "chars_removed_pct", "dropped_page_furniture", "dropped_empty",
                  "footnote_markers_clean", "footnote_markers_mangled",
                  "braces_unwrapped", "n_footnotes", "footnote_len_min",
                  "footnote_len_median", "footnote_len_max",
                  "n_unknown_marker_residue"):
            print(f"  {k:26} {rep[k]}")
        if rep["unknown_marker_residue"]:
            print("  UNKNOWN '[Fo' RESIDUE — go and look at these:")
            for s in rep["unknown_marker_residue"]:
                print(f"    {s!r}")
