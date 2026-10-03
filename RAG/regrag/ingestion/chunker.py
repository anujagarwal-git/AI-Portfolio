"""Chunking — parent/child spans.

PARENTDOC: the CHILD (a paragraph) is embedded and retrieved; the PARENT (the
enclosing provision) is delivered as context. Precise retrieval, coherent
context, and a citation that names the real unit.

This module currently implements PARENT BOUNDARY DETECTION. Child assembly and
the parent size policy follow.

WHY BOUNDARIES ARE PER DOCUMENT CLASS

One uniform rule — "split at Docling's SECTION_HEADER" — works for BIS, which
writes real headings, and fails for the CFR, which barely has any:

    12 CFR 225.8      209 items  ->  3 headers; 148 paragraphs under ONE of them
    12 CFR Part 252   2,793 items -> 191 parents, largest 33,036 chars

Those are the ONLY TWO BINDING INSTRUMENTS in the corpus (authority_rank 5), so
the documents that should win any conflict were the ones the parent unit
handled worst.

The CFR expresses structure as a legal enumeration — (a) / (1) / (i) / (A) —
not as headings. So the boundary rule is chosen from the REGISTRY, the same way
sections.py is told whether to expect chapters:

    BASEL_FRAMEWORK   SECTION_HEADER
    REG_Y / REG_YY    SECTION_HEADER *or* a top-level (a)/(b)/(c) marker
    everything else   SECTION_HEADER

THE (i) PROBLEM

In CFR hierarchy (a) -> (1) -> (i) -> (A), a bare "(i)" is either the ninth
top-level paragraph or the first roman numeral three levels down. Textually
identical. The observed sequence of single-letter markers in 12 CFR 225.8:

    a b i c d e i i i v i f i g h i i i i i i i j i i i k

Each of a-h, j, k appears ONCE — real top-level paragraphs. "i" appears 16
times and "v" once: roman numerals. A naive "single lowercase letter opens a
parent" rule would have created 16 spurious parents in one small document.

So markers are accepted only IN SEQUENCE: after (h) comes (i), never before.
Residual ambiguity is documented in _TopLevelLetters below.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

from regrag import config
from regrag.domain import Document, Framework

BOUNDARY_LABELS = {"SECTION_HEADER", "TITLE"}

# Enumeration-based classes. Everything else uses headings alone.
ENUMERATED_FRAMEWORKS = {Framework.REG_Y, Framework.REG_YY}

# "(a) Purpose. ..." — a single lowercase letter, then real text.
_TOP_LEVEL_MARKER = re.compile(r"^\(([a-z])\)\s+\S")

# "§ 252.13 Definitions." — a CFR section sign opens a provision even when
# Docling did not label it a header.
_SECTION_SIGN = re.compile(r"^\s*§+\s*\d")


@dataclass(frozen=True)
class ParentSpan:
    """A contiguous run of items delivered together as context."""

    start_idx: int          # inclusive, into the cleaned item list
    end_idx: int            # exclusive
    label: str              # what opened it: "heading" | "enumeration" | "chapter_start"
    marker: str | None      # e.g. "(b)" — None for headings
    heading: str            # first line, for display/debug
    n_chars: int

    @property
    def n_items(self) -> int:
        return self.end_idx - self.start_idx

    def __str__(self) -> str:
        return f"<parent {self.label} {self.marker or ''} {self.n_chars:,}c {self.heading[:44]!r}>"


class _TopLevelLetters:
    """Accepts (a),(b),(c)... only in sequence, so roman numerals are rejected.

    RESIDUAL AMBIGUITY, stated rather than hidden: immediately after (h) we
    expect (i), so the very next "(i)" is accepted as top-level even if it is
    really a roman numeral nested under (h). The cost is one parent boundary
    placed a few items early — a merged or slightly-misplaced span, never lost
    text. Resolving it properly needs indentation or a lookahead for (j), and
    is not worth the complexity until a golden question exposes it.
    """

    def __init__(self) -> None:
        self._expected = 0  # index into 'a'..'z'

    def accepts(self, letter: str) -> bool:
        if self._expected < len(string.ascii_lowercase) and letter == string.ascii_lowercase[self._expected]:
            self._expected += 1
            return True
        if letter == "a":
            self._expected = 1  # a new section restarts the enumeration
            return True
        return False

    def reset(self) -> None:
        self._expected = 0


# =============================================================================
# PUBLIC API
# =============================================================================


def find_parents(items, start: int, end: int, row: Document) -> list[ParentSpan]:
    """Split [start, end) into parent spans, using the rule for this document.

    `items` are CleanItems. `start`/`end` normally come from a chapter returned
    by sections.py, so a parent can never straddle two chapters — which is what
    keeps one chunk from carrying two different effective dates.
    """
    enumerated = row.framework in ENUMERATED_FRAMEWORKS
    letters = _TopLevelLetters()

    opens: list[tuple[int, str, str | None]] = []
    for i in range(start, end):
        text = (items[i].text or "").strip()
        if not text:
            continue

        if items[i].label in BOUNDARY_LABELS:
            opens.append((i, "heading", None))
            if enumerated:
                letters.reset()   # a new § restarts (a),(b),(c)
            continue

        if not enumerated:
            continue

        if _SECTION_SIGN.match(text):
            opens.append((i, "heading", None))
            letters.reset()
            continue

        m = _TOP_LEVEL_MARKER.match(text)
        if m and letters.accepts(m.group(1)):
            opens.append((i, "enumeration", f"({m.group(1)})"))

    # Text before the first boundary still belongs somewhere.
    if not opens or opens[0][0] != start:
        opens.insert(0, (start, "chapter_start", None))

    spans: list[ParentSpan] = []
    for k, (idx, kind, marker) in enumerate(opens):
        stop = opens[k + 1][0] if k + 1 < len(opens) else end
        n_chars = sum(len(items[j].text) for j in range(idx, stop))
        if n_chars == 0:
            continue
        spans.append(
            ParentSpan(
                start_idx=idx,
                end_idx=stop,
                label=kind,
                marker=marker,
                heading=(items[idx].text or "").strip()[:120],
                n_chars=n_chars,
            )
        )
    return spans


# =============================================================================
# CHILDREN — what actually gets embedded
# =============================================================================

CHILD_LABELS = {"TEXT", "LIST_ITEM", "PARAGRAPH", "FOOTNOTE", "CAPTION", "FORMULA"}

# PARAGRAPH NUMBERING, publisher by publisher.
#
# Every regulator numbers provisions its own way, and each form is how that
# document is actually cited. Ordered longest-first, because '5.5.1' must match
# before '5.5' does.
#
#   IFRS 9        5.5.1  5.4.9      three-part
#   Basel         36.122  20.2      two-part; combines with the chapter -> CRE36.122
#   SS1/23        1.5  2.13         two-part, no chapter
#   BCBS 239/PAP  21.  3.           plain integer with a trailing dot
_PARA_3 = re.compile(r"^(\d{1,3}\.\d{1,3}\.\d{1,3})\s")
_PARA_2 = re.compile(r"^(\d{1,3}\.\d{1,3})\.?\s")
_PARA_1 = re.compile(r"^(\d{1,3})\.\s+\S")

_CFR_MARKER = re.compile(r"^(\([a-zA-Z0-9ivx]{1,4}\))\s")

# "CRE36" -> "CRE", so the chapter prefix can be joined to a paragraph number.
_TRAILING_DIGITS = re.compile(r"\d+$")

_SENTENCE_END = re.compile(r"(?<=[.;:])\s+")

# A complete unit ends here. ';' and ':' count — legal drafting ends sub-items
# with them constantly.
_TERMINAL = (".", "?", "!", ";", ":")

# PURE NOISE — dropped, never merged into a neighbour.
#
# Merging noise into a real paragraph pollutes it; dropping it costs nothing,
# because none of these carry meaning on their own. 
_NOISE = re.compile(
    r"^(?:FAQ\s*\d*"          # 'FAQ' (76x), 'FAQ1' (48x)
    r"|Footnotes?"            # section stub (31x)
    r"|Page\s+\d+"            # 'Page 3' — label was TEXT, so clean.py missed it
    r"|\d{1,4}"               # bare numerals: contents pages, table cells
    r"|[·■•\-–—]"             # orphaned bullet glyphs
    r"|n/?a\.?"               # 'n/a'
    r"|\(?[a-zA-Z]\)?"        # a bare marker with no text after it
    r"|[ivxlc]{1,5}\.?"       # a bare roman numeral
    r")$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    """One CHILD: the unit that is embedded and retrieved.

    `text` is what goes to the embedder. `parent_id` is what gets DELIVERED —
    the chunk itself is a retrieval key, not the answer.
    """

    chunk_id: str
    doc_id: str
    parent_id: str
    text: str
    locator: str | None      # "CRE36.122" / "225.8(b)(1)" — for the citation
    locator_kind: str        # "paragraph" (precise) | "section" (coarse)
    is_footnote: bool
    page: int | None
    payload: dict

    def __str__(self) -> str:
        return f"<chunk {self.chunk_id} {self.locator or ''} {len(self.text)}c>"


@dataclass(frozen=True)
class ParentRecord:
    """The context delivered when one of its children is retrieved.

    Stored SEPARATELY from chunks rather than copied into each child's payload.
    A 4,000-char parent with 20 children would otherwise be duplicated 20 times,
    and the oversized ones far worse.
    """

    parent_id: str
    doc_id: str
    chapter: str | None
    heading: str
    text: str
    n_chars: int
    n_children: int


def _locator(text: str, chapter: str | None, parent_marker: str | None,
             enumerated: bool, heading: str | None = None) -> tuple[str | None, str]:
    """Returns (locator, kind).

    `kind` matters and is not decoration. "paragraph" means the chunk can be
    cited exactly — CRE36.122, 225.8(b)(1), IFRS 9 5.5.1 — and a validator can
    open the source at that provision. "section" means the best available
    reference is the enclosing heading, which narrows the reader to a few pages
    but does not name the requirement.

    Without this distinction the two are indistinguishable, and the coverage
    metric reports 100% for a corpus where a third of chunks can only be cited
    approximately. The generator should also treat them differently: a
    paragraph locator can be quoted as a pinpoint citation, a section locator
    should read "see SR 26-2, section III".
    """
    """The finest citable reference for this child, or None.

    Document.citation() prepends the short_name, so this returns only the part
    that identifies the provision WITHIN the document:

        Basel     "36.122" + chapter "CRE36"  ->  "CRE36.122"   (BIS's own form)
        CFR       parent "(b)" + child "(1)"  ->  "(b)(1)"
        SS1/23    "1.5"                       ->  "SS1/23 1.5"
        BCBS 239  "21."                       ->  "BCBS 239 21"
        IFRS 9    "5.5.1"                     ->  "IFRS 9 5.5.1"

    Falls back to the section heading, because for SR 26-2, SR 11-7 and d450 —
    which carry no paragraph numbers at all — the heading IS how they are cited
    ("SR 26-2, III. Overview of Model Risk").
    """
    if enumerated:
        m = _CFR_MARKER.match(text)
        child = m.group(1) if m else None
        if parent_marker and child and child != parent_marker:
            return f"{parent_marker}{child}", "paragraph"
        if child or parent_marker:
            return (child or parent_marker), "paragraph"
        return (heading or None), "section"

    for pattern in (_PARA_3, _PARA_2, _PARA_1):
        m = pattern.match(text)
        if not m:
            continue
        num = m.group(1)
        if chapter and _PARA_2.match(text) and not _PARA_3.match(text):
            # Basel: chapter "CRE36" + paragraph "36.122" -> "CRE36.122".
            # The chapter already carries its own number, so strip it first.
            prefix = _TRAILING_DIGITS.sub("", chapter)
            return f"{prefix}{num}", "paragraph"
        return num, "paragraph"

    # No number anywhere — cite the section, which is what a reader would do.
    return (heading or None), "section"


def _split_oversized(text: str, limit: int) -> list[str]:
    """Split at sentence ends, never mid-sentence.

    A truncated regulatory sentence is worse than two chunks: "A bank must
    maintain capital of at least" is not a weaker version of the requirement,
    it is a different statement.
    """
    if len(text) <= limit:
        return [text]
    out, buf = [], ""
    for piece in _SENTENCE_END.split(text):
        if buf and len(buf) + 1 + len(piece) > limit:
            out.append(buf.strip())
            buf = piece
        else:
            buf = f"{buf} {piece}".strip()
    if buf.strip():
        out.append(buf.strip())
    # A single sentence longer than the limit still has to be cut somewhere.
    final: list[str] = []
    for s in out:
        while len(s) > limit:
            final.append(s[:limit])
            s = s[limit:]
        if s:
            final.append(s)
    return final


def _assemble_children(items, span: ParentSpan) -> list[tuple[str, int, bool, int]]:
    """(text, first_item_idx, is_footnote, page) for each child of one parent.

    TWO OPERATIONS, AND ONLY TWO.

      DROP NOISE     items carrying no meaning alone (see _NOISE).

      REPAIR SPLITS  rejoin a sentence Docling broke across a column or page
                     boundary — and nothing else.


    TWO SIGNALS TO AGREE:

        the fragment does not end with terminal punctuation, AND
        the next item begins with a lowercase letter

    Measured over every adjacent pair in the corpus:

        72.7%   ends closed + next uppercase   clearly separate
        23.0%   ends OPEN   + next uppercase   AMBIGUOUS — headings, list items
                                               and formulas legitimately take no
                                               full stop. Merging these was the
                                               bug.
         3.2%   ends OPEN   + next lowercase   a real split. English does not
                                               start sentences lowercase.
         1.2%   ends closed + next lowercase   ambiguous

    KNOWN LIMIT: 194 fragments still begin mid-sentence, because their other
    half sits ABOVE them and this only merges forward. Repairing those needs
    the page-geometry signal — see MENTOR_PROGRESS.md.
    """
    raw = [
        i for i in range(span.start_idx, span.end_idx)
        if items[i].label in CHILD_LABELS and (items[i].text or "").strip()
    ]

    # THE ENUMERATION MARKER LINE IS A HEADING, NOT A CHILD.
    
    # For Basel a SECTION_HEADER opens a parent and is never itself embedded.
    # In the CFR the equivalent line — "(b) Scope and reservation of authority -"
    # — carries the label LIST_ITEM, so without this it becomes a child, merges
    # with the provision beneath it (being only 39 chars), and drags the
    # locator down from "(b)(1)" to "(b)". Same role, so same treatment.
    #
    # Dropped only when the parent has other children; the text is never lost,
    # because the PARENT record still contains it and is what gets delivered.
    if span.label == "enumeration" and len(raw) > 1 and raw[0] == span.start_idx:
        head = (items[raw[0]].text or "").strip()
        if len(head) < config.CHILD_MIN_CHARS:
            raw = raw[1:]

    if not raw:
        return []

    # 1. DROP NOISE
    kept = [i for i in raw if not _NOISE.match((items[i].text or "").strip())]
    if not kept:
        return []

    # 2. SPLIT INTO TWO INTERLEAVED STREAMS.
    #
    # A body sentence continues into the next BODY item — never into a footnote
    # that happens to sit between them on the page. 

    body = [i for i in kept if not getattr(items[i], "is_footnote", False)]
    notes = [i for i in kept if getattr(items[i], "is_footnote", False)]

    out: list[tuple[str, int, bool, int]] = []
    for stream, is_fn in ((body, False), (notes, True)):
        out += _repair_stream(items, stream, is_fn)

    # Restore reading order, so a parent's children stay in document sequence.
    out.sort(key=lambda t: t[1])
    return out


def _repair_stream(items, stream: list[int], is_fn: bool) -> list[tuple[str, int, bool, int]]:
    """Rejoin split sentences within ONE stream. Both signals must agree."""
    out: list[tuple[str, int, bool, int]] = []
    k = 0
    while k < len(stream):
        i = stream[k]
        text = (items[i].text or "").strip()
        page = getattr(items[i], "page", None)

        while k + 1 < len(stream):
            nxt = (items[stream[k + 1]].text or "").strip()
            if (
                text.rstrip().endswith(_TERMINAL)          # signal 1: ends closed -> stop
                or not nxt[:1].islower()                   # signal 2: next is a new unit -> stop
                or len(text) + len(nxt) > config.CHILD_MERGE_CEILING
            ):
                break
            text = f"{text} {nxt}"
            k += 1

        for piece in _split_oversized(text, config.CHILD_MAX_CHARS):
            out.append((piece, i, is_fn, page))
        k += 1
    return out


def chunk_document(cleaned, chapters, row: Document) -> tuple[list[Chunk], list[ParentRecord]]:
    """Turn one cleaned document into chunks plus the parents they point at."""
    items = cleaned.texts
    enumerated = row.framework in ENUMERATED_FRAMEWORKS
    chunks: list[Chunk] = []
    parents: list[ParentRecord] = []

    for ch in chapters:
        overlay = ch.payload_overlay()
        for p_no, span in enumerate(find_parents(items, ch.start_idx, ch.end_idx, row)):
            parent_id = f"{row.doc_id}::{ch.code or 'doc'}::{p_no:04d}"
            kids = _assemble_children(items, span)
            if not kids:
                continue

            parents.append(ParentRecord(
                parent_id=parent_id,
                doc_id=row.doc_id,
                chapter=ch.code,
                heading=span.heading,
                text="\n\n".join(items[i].text for i in range(span.start_idx, span.end_idx)
                                 if (items[i].text or "").strip()),
                n_chars=span.n_chars,
                n_children=len(kids),
            ))

            for c_no, (text, idx, is_fn, page) in enumerate(kids):
                payload = row.payload()
                payload.update(overlay)
                loc, loc_kind = _locator(text, ch.code, span.marker, enumerated, span.heading)
                payload.update({
                    "parent_id": parent_id,
                    "parent_heading": span.heading,
                    "locator": loc,
                    "locator_kind": loc_kind,
                    "is_footnote": is_fn,
                    "page": page,
                    "orig_idx": idx,
                })
                chunks.append(Chunk(
                    chunk_id=f"{parent_id}::{c_no:03d}",
                    doc_id=row.doc_id,
                    parent_id=parent_id,
                    text=text,
                    locator=loc,
                    locator_kind=loc_kind,
                    is_footnote=is_fn,
                    page=page,
                    payload=payload,
                ))
    return chunks, parents


def chunk_report(cleaned, chapters, row: Document) -> dict:
    chunks, parents = chunk_document(cleaned, chapters, row)
    sizes = sorted(len(c.text) for c in chunks)
    p = lambda q: sizes[min(len(sizes) - 1, int(q * len(sizes)))] if sizes else 0
    return {
        "doc_id": row.doc_id,
        "short_name": row.short_name,
        "n_parents": len(parents),
        "n_chunks": len(chunks),
        "chunks_per_parent": round(len(chunks) / len(parents), 1) if parents else 0,
        "child_median": p(0.50),
        "child_p90": p(0.90),
        "child_max": sizes[-1] if sizes else 0,
        "under_min": sum(1 for s in sizes if s < config.CHILD_MIN_CHARS),
        "over_max": sum(1 for s in sizes if s > config.CHILD_MAX_CHARS),
        "precise_locator": sum(1 for c in chunks if c.locator_kind == "paragraph"),
        "with_locator": sum(1 for c in chunks if c.locator),
        "footnote_chunks": sum(1 for c in chunks if c.is_footnote),
    }


def parents_report(cleaned, chapters, row: Document) -> dict:
    """Measurements only — same contract as the other stage reports."""
    items = cleaned.texts
    spans: list[ParentSpan] = []
    for ch in chapters:
        spans += find_parents(items, ch.start_idx, ch.end_idx, row)

    sizes = sorted(s.n_chars for s in spans)
    kinds: dict[str, int] = {}
    for s in spans:
        kinds[s.label] = kinds.get(s.label, 0) + 1

    def pct(q: float) -> int:
        return sizes[min(len(sizes) - 1, int(q * len(sizes)))] if sizes else 0

    return {
        "doc_id": row.doc_id,
        "short_name": row.short_name,
        "framework": row.framework.value,
        "enumerated": row.framework in ENUMERATED_FRAMEWORKS,
        "n_parents": len(spans),
        "opened_by": kinds,
        "median": sizes[len(sizes) // 2] if sizes else 0,
        "p90": pct(0.90),
        "max": sizes[-1] if sizes else 0,
        "over_8k": sum(1 for s in sizes if s > 8_000),
        "over_12k": sum(1 for s in sizes if s > 12_000),
    }


# =============================================================================
# CLI — `python -m regrag.ingestion.chunker [doc_id ...]`
# Defaults to the documents that exercise BOTH boundary rules.
# =============================================================================

if __name__ == "__main__":
    import sys

    from regrag import config
    from regrag.ingestion.cache import parse_cached
    from regrag.ingestion.clean import clean_document
    from regrag.ingestion.sections import find_sections
    from regrag.registry import load

    reg = load()
    doc_ids = sys.argv[1:] or [d.doc_id for d in reg.indexable()]

    print(f"\n{'document':30} {'rule':5} {'parents':>8} {'chunks':>7} {'c/p':>5} "
          f"{'med':>5} {'p90':>6} {'max':>6} {'loc%':>5} {'fn':>5}")
    print("-" * 92)
    tot_c = tot_p = tot_loc = 0
    for doc_id in doc_ids:
        row = reg.by_id(doc_id)
        doc = parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
        cleaned = clean_document(doc, row)
        chapters = find_sections(cleaned, row)
        r = chunk_report(cleaned, chapters, row)
        rule = "enum" if row.framework in ENUMERATED_FRAMEWORKS else "head"
        loc_pct = 100 * r["with_locator"] / r["n_chunks"] if r["n_chunks"] else 0
        tot_c += r["n_chunks"]; tot_p += r["n_parents"]; tot_loc += r["with_locator"]
        print(f"{r['short_name']:30} {rule:5} {r['n_parents']:>8,} {r['n_chunks']:>7,} "
              f"{r['chunks_per_parent']:>5} {r['child_median']:>5} {r['child_p90']:>6} "
              f"{r['child_max']:>6} {loc_pct:>4.0f}% {r['footnote_chunks']:>5}")

    print("-" * 92)
    print(f"{'TOTAL':30} {'':5} {tot_p:>8,} {tot_c:>7,} "
          f"{tot_c/tot_p if tot_p else 0:>5.1f} {'':5} {'':6} {'':6} "
          f"{100*tot_loc/tot_c if tot_c else 0:>4.0f}%")
    print(f"\n  {tot_c:,} chunks to embed with bge-small on CPU.")
