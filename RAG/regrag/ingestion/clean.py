"""Cleaning — removes extraction artifacts without removing content.


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
                    silently removed."""

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

# =============================================================================
# PROSE-LAYOUT TABLES
#
# Docling reports some things as tables that are not grids at all — a hanging
# indent, a glossary, an FAQ block, a footnote list. IFRS 9 is the clearest
# case: its paragraph numbers sit in a left margin, so the layout model reads
#
#     r0 c0 '5.5.12'   r0 c1 'If the contractual cash flows on a financial...'
#     r1 c1 '(a)'      r1 c2 'the risk of a default occurring at the...'
#
# as a two-column table. That text lives ONLY in `tables` — it is NOT
# duplicated in `texts` — so 16,224 characters of IFRS 9 (14% of the document,
# including the ECL provisions) were absent from the index entirely, and every
# paragraph number with them, which is why IFRS 9 scored 0% citable.
#
# Corpus-wide this recovers 45,730 characters across 28 tables: IFRS 9's
# provisions, BCBS 239's glossary, CAP's FAQ blocks and footnote lists.
#
# GENUINE GRIDS ARE NOT TOUCHED. Docling's cell extraction is unreliable on
# numeric tables — roughly 40% of rows in multi-column tables lose a value —
# so those are deferred to a separate pdfplumber pass rather than indexed with
# known-wrong numbers.
# =============================================================================

PROSE_TABLE_MAX_COLS = 3
PROSE_TABLE_MIN_CELL = 120        # at least one cell of real prose

# Dotted leaders: 'Introduction .......... 4'. A table of contents is furniture.
_TOC_LEADER = re.compile(r"\.{5,}")


def _is_prose_table(data: dict) -> bool:
    cells = data.get("table_cells") or []
    if not cells or (data.get("num_cols") or 0) > PROSE_TABLE_MAX_COLS:
        return False
    if not any(len(c.get("text") or "") > PROSE_TABLE_MIN_CELL for c in cells):
        return False
    # a contents listing is furniture, not content
    dotted = sum(1 for c in cells if _TOC_LEADER.search(c.get("text") or ""))
    return dotted < max(2, len(cells) // 4)


def _prose_table_rows(data: dict) -> list[str]:
    """One line per row: cells joined left to right.

    For IFRS 9 this yields '5.5.12 If the contractual cash flows...', which
    puts the paragraph number exactly where the locator extractor expects it —
    so the citation scheme is recovered with no special handling.
    """
    grid = data.get("grid") or []
    out: list[str] = []
    for row in grid:
        seen, parts = None, []
        for c in row:
            t = (c.get("text") or "").strip()
            if t and t != seen:       # column spans repeat a cell across columns
                parts.append(t)
            seen = t or seen
        line = " ".join(parts).strip()
        if line:
            out.append(line)
    return out


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
        "markers_recovered": 0,
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

    for idx, raw in _iter_document(doc):
        label = _label(raw)

        if label in DROP_LABELS:
            counts["dropped_page_furniture"] += 1
            continue

        text = _source_text(raw, counts)
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
        "items_in": sum(1 for _ in _iter_document(doc)),
        "items_out": len(items),
        "chars_in": sum(
            len((getattr(t, "orig", "") or "").strip() or (getattr(t, "text", "") or ""))
            for _, t in _iter_document(doc)
        ),
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


class _TableLine:
    """One row of a prose-layout table, presented as if it were a text item.

    Keeps the rest of clean.py, sections.py and chunker.py unaware that tables
    exist — a recovered IFRS 9 paragraph flows through exactly like any other
    paragraph, and lands in the right chapter with the right effective date.
    """

    __slots__ = ("text", "orig", "label", "prov")

    def __init__(self, text: str, page):
        self.text = self.orig = text
        self.label = "TEXT"
        self.prov = [_Prov(page)] if page is not None else []


def _iter_document(doc):
    """Yield (index, item) in READING ORDER, including prose-layout tables.

    Docling records reading order in `body.children`, which interleaves texts
    and tables. Walking `doc.texts` alone — as this module used to — skips
    every table, and table text is NOT duplicated into `texts`.

    Falls back to `doc.texts` when there is no body (test stubs, and any
    backend that does not provide one).
    """
    body = getattr(doc, "body", None)
    children = getattr(body, "children", None) if body is not None else None
    texts = list(getattr(doc, "texts", []) or [])
    tables = list(getattr(doc, "tables", []) or [])
    groups = list(getattr(doc, "groups", []) or [])

    if not children:
        yield from enumerate(texts)
        return

    def ref_path(ref) -> str:
        return getattr(ref, "cref", None) or (
            ref.get("$ref", "") if isinstance(ref, dict) else str(ref)
        )

    counter = [0]
    seen_groups: set[int] = set()

    def walk(refs):
        """Depth-first, because body.children is NOT flat.

        It contains `#/groups/N` entries whose own children hold the text.
        Walking only the top level dropped every grouped item — more than half
        of IFRS 9 — while appearing to work. Recursion is not optional here.
        """
        for ref in refs:
            path = ref_path(ref)
            if "/texts/" in path:
                n = int(path.rsplit("/", 1)[-1])
                if n < len(texts):
                    yield counter[0], texts[n]
                    counter[0] += 1
            elif "/groups/" in path:
                n = int(path.rsplit("/", 1)[-1])
                if n < len(groups) and n not in seen_groups:
                    seen_groups.add(n)
                    yield from walk(getattr(groups[n], "children", []) or [])
            elif "/tables/" in path:
                n = int(path.rsplit("/", 1)[-1])
                if n >= len(tables):
                    continue
                tbl = tables[n]
                data = tbl.model_dump().get("data", {}) if hasattr(tbl, "model_dump") else {}
                if not _is_prose_table(data):
                    continue      # genuine grid — deferred to the pdfplumber pass
                prov = getattr(tbl, "prov", None)
                page = prov[0].page_no if prov else None
                for line in _prose_table_rows(data):
                    yield counter[0], _TableLine(line, page)
                    counter[0] += 1

    yield from walk(children)


def _source_text(item, counts: dict) -> str:
    """Prefer Docling's `orig` over `text`.

    Docling gives each item BOTH. `text` is its tidied version, and tidying
    STRIPS THE ENUMERATION MARKER:

        orig : "(1) Applicability. Except as provided in paragraph (c)..."
        text : "Applicability. Except as provided in paragraph (c)..."

    Reading `text` discarded those markers across the whole corpus — 913 in
    CRE, 678 in 12 CFR Part 252, 106 in PAP, 92 in SS1/23. Two costs:

      STRUCTURE   In the CFR the enumeration IS the structure. 12 CFR 225.8 has
                  148 paragraphs under a single heading, so (a)/(b)/(c) is the
                  only signal of where one provision ends and the next begins.

      CITATION    "12 CFR 225.8(b)(1)" is how an examiner cites. Without the
                  marker the best available citation is a page number, which
                  nobody can act on.

    Verified safe before switching: across 1,859 differing items in six
    documents, `orig` was NEVER shorter than `text` — it only ever restores a
    prefix. So this cannot lose content.
    """
    orig = (getattr(item, "orig", "") or "").strip()
    text = (getattr(item, "text", "") or "")
    if orig and orig != text.strip():
        counts["markers_recovered"] += 1
        return orig
    return text


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
