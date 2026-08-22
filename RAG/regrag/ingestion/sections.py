"""Sectioning — splits a parsed document into its chapters and finds each
chapter's version-effective date.

WHY THIS MODULE EXISTS.

Five of the six BASEL_FRAMEWORK files bundle several chapters, and EVERY
CHAPTER carries its own "Version effective as of" stamp. SCO is the extreme
case: its introduction chapter is effective 15 Dec 2019 while its cryptoasset
chapter is effective 01 Jan 2026. So `effective_from` is deliberately null on
those registry rows — a single document-level date would be false for most of
the file. This module supplies the missing per-chapter date, and until it
exists those five documents CANNOT BE INDEXED. That is ~510 of ~1,100 pages,
CRE included.

THE FAILURE MODE THIS MODULE IS DESIGNED AROUND.

If chapter detection silently fails on a Basel file, the natural result is one
whole-document section with effective_from=None — which is byte-for-byte
identical to a correct result for SR 26-2, a document that genuinely has no
chapters. A broken parse of CRE would look exactly like a healthy parse of an
SR letter, and 323 pages would be indexed with no chapter dates while nothing
reported a problem.

So this module does NOT guess from the text alone. It asks the registry what
to expect: a BASEL_FRAMEWORK document MUST yield chapters, and anything else
must not. Two situations that produce the same output are forced apart by
metadata rather than left to collapse into one value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from regrag import config
from regrag.domain import Document, Framework

STAMP_PHRASE = "version effective as of"

# BIS renders dates as "01 Jan 2023". Anchored so a stray date elsewhere in the
# body cannot be mistaken for a version stamp.
_DATE_RE = re.compile(r"^\s*(\d{2}\s+[A-Za-z]{3}\s+\d{4})\s*$")

# Same date, but embedded in the stamp item itself. Only ever applied to text
# that already begins with STAMP_PHRASE, so it cannot pick up a body date.
_DATE_INLINE_RE = re.compile(r"(\d{2}\s+[A-Za-z]{3}\s+\d{4})")

_DATE_FMT = "%d %b %Y"


class SectionError(Exception):
    """Raised when sectioning produces something the registry says is wrong."""


@dataclass(frozen=True)
class Section:
    """A span of text items sharing one version-effective date.

    `kind` is what stops a failed Basel parse impersonating a healthy SR-letter
    parse. Without it both are just "one section with no date".
    """

    kind: str                  # "chapter" | "whole_document"
    code: str | None           # e.g. "CRE36"; None for whole_document
    effective_from: date | None
    start_idx: int             # inclusive index into doc.texts
    end_idx: int               # exclusive
    page_from: int | None
    page_to: int | None

    @property
    def n_items(self) -> int:
        return self.end_idx - self.start_idx

    def payload_overlay(self) -> dict:
        """What the chunker merges ON TOP of Document.payload().

        Only non-null values are emitted. A chapter date must never overwrite a
        document-level date with None — that would turn a known date into the
        "living text" signal, which means something else entirely in the schema.
        """
        out: dict = {"chapter": self.code, "section_kind": self.kind}
        if self.effective_from is not None:
            out["effective_from"] = self.effective_from.isoformat()
        return out

    def __str__(self) -> str:
        d = self.effective_from.isoformat() if self.effective_from else "no date"
        return f"<{self.code or self.kind} {d} items {self.start_idx}:{self.end_idx}>"


# =============================================================================
# PUBLIC API
# =============================================================================


def find_sections(doc, row: Document) -> list[Section]:
    """Split `doc` into sections, using `row` to decide what to expect.

    Basel chapters are matched against the row's OWN volume code — CRE files
    look for `CRE\\d+`, LEX files for `LEX\\d+`. Deriving the prefix from the
    registry rather than hard-coding a list of Basel standards means a false
    positive is impossible (a stray "SRP30" reference inside CRE cannot open a
    section) and a new BIS standard needs no code change, only a registry row.
    """
    expects_chapters = row.framework is Framework.BASEL_FRAMEWORK

    if not expects_chapters:
        return [_whole_document(doc)]

    if not row.volume:
        raise SectionError(
            f"{row.doc_id}: framework is BASEL_FRAMEWORK but `volume` is null, so "
            "there is no chapter prefix to match. Every Basel row must carry its "
            "standard code (CAP/CRE/LEX/RBC/SCO/SRP32)."
        )

    starts = _chapter_starts(doc, row.volume)
    if not starts:
        raise SectionError(
            f"{row.doc_id} ({row.short_name}): expected chapter codes matching "
            f"{row.volume}<digits> and found NONE. Refusing to return a "
            "whole-document section, because that result is indistinguishable "
            "from a correct parse of a non-chaptered document — and would index "
            "the whole file with no chapter dates while looking healthy."
        )

    sections: list[Section] = []
    for i, (idx, code) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(doc.texts)
        eff = _stamp_after(doc, idx, end, code, row)
        p_from, p_to = _page_span(doc, idx, end)
        sections.append(
            Section(
                kind="chapter",
                code=code,
                effective_from=eff,
                start_idx=idx,
                end_idx=end,
                page_from=p_from,
                page_to=p_to,
            )
        )

    _check_coverage(doc, sections, row)
    return sections


def sections_report(doc, row: Document) -> dict:
    """Measurements, no assertions — same contract as parse_quality_report.
    The caller decides what "good" means.
    """
    secs = find_sections(doc, row)
    return {
        "doc_id": row.doc_id,
        "short_name": row.short_name,
        "kind": secs[0].kind,
        "n_sections": len(secs),
        "codes": [s.code for s in secs],
        "effective_dates": [
            s.effective_from.isoformat() if s.effective_from else None for s in secs
        ],
        "distinct_dates": sorted(
            {s.effective_from.isoformat() for s in secs if s.effective_from}
        ),
        "items_covered": sum(s.n_items for s in secs),
        "n_text_items": len(doc.texts),
        "page_spans": [(s.page_from, s.page_to) for s in secs],
    }


# =============================================================================
# INTERNALS
# =============================================================================


def _text(item) -> str:
    return (getattr(item, "text", "") or "").strip()


def _whole_document(doc) -> Section:
    p_from, p_to = _page_span(doc, 0, len(doc.texts))
    return Section(
        kind="whole_document",
        code=None,
        effective_from=None,   # the registry row's own date applies
        start_idx=0,
        end_idx=len(doc.texts),
        page_from=p_from,
        page_to=p_to,
    )


def _chapter_starts(doc, volume: str) -> list[tuple[int, str]]:
    """Indices of text items that are exactly a chapter code for this volume.

    Anchored and whitespace-tolerant: LEX40 is indented in the source while
    LEX10 is not, and an unanchored match would also fire on every in-text
    cross-reference to another chapter.
    """
    # `volume` sits at one of TWO levels of the BIS hierarchy, and both are
    # correct under the registry's own definition ("the citable address inside
    # the framework"):
    #
    #   volume="CRE"    a whole STANDARD -> its chapters are CRE20, CRE21, ...
    #   volume="SRP32"  a single CHAPTER lifted out of the SRP standard
    #
    # Deciding which by whether the volume already ends in digits keeps the
    # registry honest — neither row has to be distorted to suit the parser.
    # ANCHORED AT THE START, NOT THE WHOLE ITEM. Docling is inconsistent about
    # this even within one file: LEX20 arrives as its own item 'LEX20', while
    # LEX10 arrives merged with its title, 'LEX10 Definitions and application'.
    # Requiring a whole-item match silently dropped LEX10 — one chapter in four,
    # with no error, which is exactly the class of failure this module exists to
    # prevent.
    #
    # The negative lookahead keeps cross-references out: 'LEX30.1' and 'LEX301'
    # must not open a section, while 'LEX30 Exposure measurement' must. Body
    # references like '20.2 ... as specified in LEX30' are excluded by the
    # start anchor, since they begin with a paragraph number.
    # EXACTLY TWO DIGITS. Every BIS chapter number is two digits — CAP10,
    # CRE36, LEX40, SCO60, SRP32 — and relying on that is what separates a real
    # chapter start from text that merely begins with the code. With an open
    # \d+, an item reading 'LEX301 would be a different chapter' opens a
    # spurious section; the body then has no version stamp and the whole
    # document fails to resolve.
    if re.search(r"\d$", volume):
        # SINGLE-CHAPTER FILE (volume IS the chapter code, e.g. SRP32).
        # Search ANYWHERE in the item, and take only the first hit. Docling
        # renders this title block as 'Supervisory review process SRP32' —
        # standard name first, code last — so a start anchor finds nothing.
        # Unanchored is safe here in a way it would NOT be for a multi-chapter
        # standard: there are no sibling chapters for a stray mention to be
        # confused with, and only the first match opens a section.
        pattern = re.compile(rf"({re.escape(volume)})(?![\w.])")
        for i, item in enumerate(doc.texts):
            m = pattern.search(_text(item))
            if m:
                return [(i, m.group(1))]
        return []

    # MULTI-CHAPTER STANDARD (volume is the standard code, e.g. CRE).
    # Anchored at the start, exactly two digits — every BIS chapter number is
    # two digits, and that is what separates 'CRE36 Minimum requirements' from
    # a body line that merely mentions CRE36.
    pattern = re.compile(rf"^\s*({re.escape(volume)}\d{{2}})(?![\w.])")
    out: list[tuple[int, str]] = []
    for i, item in enumerate(doc.texts):
        m = pattern.match(_text(item))
        if m:
            out.append((i, m.group(1)))
    return out


def _stamp_after(doc, start: int, end: int, code: str, row: Document) -> date:
    """Find "Version effective as of" after a chapter start and parse the date.

    Raises rather than returning None. A missing stamp must not become None,
    because None already means "living text / no fixed edition" elsewhere in the
    schema. Letting the two share a value is how a parsing failure becomes a
    metadata fact nobody questions later.
    """
    window_end = min(end, start + config.SECTION_STAMP_WINDOW)

    for i in range(start, window_end):
        txt = _text(doc.texts[i])
        if not txt.lower().startswith(STAMP_PHRASE):
            continue

        # CASE 1 — Docling keeps the date in the SAME item:
        #     'Version effective as of 15 Dec 2019'
        m = _DATE_INLINE_RE.search(txt)
        if m:
            return _parse_date(m.group(1), code, row)

        # CASE 2 — the date is a following item, as pdftotext renders it.
        # Intervening non-date text is SKIPPED rather than treated as an error:
        # every BIS chapter repeats its change-note around the stamp, e.g.
        # 'First version in the format of the consolidated framework.'
        for j in range(i + 1, min(end, i + 5)):
            nxt = _text(doc.texts[j])
            if not nxt:
                continue
            m = _DATE_RE.match(nxt)
            if m:
                return _parse_date(m.group(1), code, row)

        raise SectionError(
            f"{row.doc_id} {code}: found '{STAMP_PHRASE}' in {txt!r} but no "
            "DD Mon YYYY date in that item or the four following it"
        )

    raise SectionError(
        f"{row.doc_id} {code}: no '{STAMP_PHRASE}' found within "
        f"{config.SECTION_STAMP_WINDOW} items of the chapter start. Either the "
        "window is too small or this chapter has no version stamp — check the "
        "PDF before widening the window, because a wrong date is worse than none."
    )


def dump_structure(doc, row: Document, context: int = 8) -> None:
    """Print the text items around every chapter code and every stamp phrase.

    Diagnostic, not part of the pipeline. Exists because the matcher was built
    against pdftotext output, and Docling groups text differently — when a
    chapter fails to resolve, the fix should come from LOOKING at what Docling
    actually produced, not from widening a pattern until it passes.
    """
    marks: set[int] = set()
    for i, item in enumerate(doc.texts):
        txt = _text(item)
        if row.volume and re.match(rf"^\s*{re.escape(row.volume)}\d*\s*$", txt):
            marks.add(i)
        if txt.lower().startswith(STAMP_PHRASE):
            marks.add(i)

    show: set[int] = set()
    for m in marks:
        show.update(range(max(0, m - 2), min(len(doc.texts), m + context)))

    print(f"\n=== {row.short_name} — {len(doc.texts)} text items, "
          f"{len(marks)} landmarks ===")
    last = -1
    for i in sorted(show):
        if i != last + 1:
            print("      ...")
        item = doc.texts[i]
        label = getattr(getattr(item, "label", None), "name", "?")
        flag = " <<<" if i in marks else ""
        print(f"  {i:4} [{label:14}] {_text(item)[:78]!r}{flag}")
        last = i


def _parse_date(raw: str, code: str, row: Document) -> date:
    try:
        return datetime.strptime(raw.strip(), _DATE_FMT).date()
    except ValueError as e:
        raise SectionError(f"{row.doc_id} {code}: cannot parse date {raw!r} ({e})") from None


def _page_span(doc, start: int, end: int) -> tuple[int | None, int | None]:
    pages = [
        item.prov[0].page_no
        for item in doc.texts[start:end]
        if getattr(item, "prov", None)
    ]
    return (min(pages), max(pages)) if pages else (None, None)


def _check_coverage(doc, sections: list[Section], row: Document) -> None:
    """Every text item after the first chapter must belong to exactly one section.

    Items BEFORE the first chapter are the cover page and table of contents —
    legitimately outside any chapter, and not indexed.
    """
    if not sections:
        return
    expected = len(doc.texts) - sections[0].start_idx
    covered = sum(s.n_items for s in sections)
    if covered != expected:
        raise SectionError(
            f"{row.doc_id}: sections cover {covered} text items but "
            f"{expected} follow the first chapter — sections must partition the "
            "document, with no gaps or overlaps"
        )
    for a, b in zip(sections, sections[1:]):
        if a.end_idx != b.start_idx:
            raise SectionError(
                f"{row.doc_id}: gap or overlap between {a.code} and {b.code}"
            )


# =============================================================================
# CLI — `python -m regrag.ingestion.sections <doc_id>`
#
# Prints the sectioning report, or on failure dumps the surrounding text items
# so the cause can be OBSERVED rather than guessed at.
# =============================================================================

if __name__ == "__main__":
    import sys

    from regrag.ingestion.parser import parse_pdf
    from regrag.registry import load

    reg = load()

    # DEFAULT TO THE BASEL DOCUMENTS ONLY, smallest first.
    #
    # Parsing all 19 means ~1,100 pages through Docling on CPU — tens of
    # minutes — and the 13 non-Basel documents return a whole-document section
    # regardless, so parsing them tells you nothing about chapter detection.
    # Smallest-first means a broken pattern surfaces on a 12-page file rather
    # than after CRE's 323 pages have already been converted.
    if sys.argv[1:]:
        doc_ids = sys.argv[1:]
    else:
        basel = [d for d in reg.indexable() if d.framework is Framework.BASEL_FRAMEWORK]
        order = {"SRP32": 0, "LEX": 1, "RBC": 2, "CAP": 3, "SCO": 4, "CRE": 5}
        basel.sort(key=lambda d: order.get(d.volume or "", 99))
        doc_ids = [d.doc_id for d in basel]
        print(f"Checking {len(doc_ids)} BASEL_FRAMEWORK documents, smallest first.")
        print("CRE is 323 pages and comes last — Ctrl+C once the others pass.\n")

    for doc_id in doc_ids:
        row = reg.by_id(doc_id)
        parsed = parse_pdf(config.PROJECT_ROOT / row.file)
        try:
            rep = sections_report(parsed, row)
            print(f"\nOK  {row.short_name}: {rep['n_sections']} section(s) "
                  f"{rep['codes']} dates={rep['effective_dates']}")
        except SectionError as e:
            print(f"\nFAIL {row.short_name}: {e}")
            dump_structure(parsed, row)
