"""Measure the candidate parent/child units, to size the parentdoc chunker.

WHY: parentdoc retrieves a CHILD (paragraph) and delivers its PARENT (section).
That only works if parents are bounded. SR 11-7 — the one document the
prototype was validated on — is 21 pages of continuous prose. CRE is 323 pages
across 27 chapters, and if a single section there turns out to be fifteen
pages, "deliver the parent" means dumping fifteen pages into the context window.

So: measure before choosing. The parent cap should come from this distribution,
not from a number that felt about right.

THE HIERARCHY BEING MEASURED

    chapter   from sections.py        carries effective_from + chapter code
      parent  span between SECTION_HEADERs   what gets DELIVERED as context
        child paragraph / list item          what gets EMBEDDED and retrieved

Reads from the parse cache, so this is seconds rather than minutes.

Run:  uv run python scripts/measure_parent_units.py
      uv run python scripts/measure_parent_units.py bcbs-cre-consolidated
"""

from __future__ import annotations

import statistics as st
import sys

from regrag import config
from regrag.ingestion.cache import parse_cached
from regrag.ingestion.clean import clean_document
from regrag.ingestion.sections import find_sections
from regrag.registry import load

PARENT_BOUNDARY_LABELS = {"SECTION_HEADER", "TITLE"}
CHILD_LABELS = {"TEXT", "LIST_ITEM", "PARAGRAPH", "FOOTNOTE", "CAPTION", "FORMULA"}

# Candidate caps to report against, in characters.
CAPS = (2_000, 4_000, 6_000, 8_000, 12_000)


def parent_spans(items, start: int, end: int) -> list[tuple[int, int]]:
    """Split [start, end) at PARENT_BOUNDARY_LABELS. Text before the first
    boundary is its own span, so nothing is lost."""
    bounds = [i for i in range(start, end) if items[i].label in PARENT_BOUNDARY_LABELS]
    if not bounds or bounds[0] != start:
        bounds = [start] + bounds
    return [(b, bounds[k + 1] if k + 1 < len(bounds) else end) for k, b in enumerate(bounds)]


def dist(name: str, values: list[int]) -> None:
    if not values:
        print(f"  {name:22} (none)")
        return
    v = sorted(values)
    p = lambda q: v[min(len(v) - 1, int(q * len(v)))]
    print(f"  {name:22} n={len(v):<5} min={v[0]:<6} med={st.median(v):<7.0f} "
          f"p90={p(0.90):<7} max={v[-1]:<7}")


def main(doc_ids: list[str]) -> None:
    reg = load()
    rows = [reg.by_id(d) for d in doc_ids] if doc_ids else reg.indexable()

    all_parents: list[int] = []
    all_children: list[int] = []
    oversized: list[tuple[str, str, int]] = []

    for row in rows:
        doc = parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
        cleaned = clean_document(doc, row)
        items = cleaned.texts
        chapters = find_sections(cleaned, row)

        p_sizes: list[int] = []
        c_sizes: list[int] = []

        for ch in chapters:
            for s, e in parent_spans(items, ch.start_idx, ch.end_idx):
                size = sum(len(items[i].text) for i in range(s, e))
                if size == 0:
                    continue
                p_sizes.append(size)
                if size > CAPS[-1]:
                    head = items[s].text[:60].replace("\n", " ")
                    oversized.append((row.short_name, f"{ch.code or '-'} | {head}", size))
                for i in range(s, e):
                    if items[i].label in CHILD_LABELS and items[i].text.strip():
                        c_sizes.append(len(items[i].text))

        all_parents += p_sizes
        all_children += c_sizes

        print(f"\n{row.short_name}  ({len(chapters)} chapter(s), {len(items)} items)")
        dist("parent (section)", p_sizes)
        dist("child (paragraph)", c_sizes)

    print("\n" + "=" * 78)
    print("ACROSS ALL DOCUMENTS")
    dist("parent (section)", all_parents)
    dist("child (paragraph)", all_children)

    print("\n  how many parents exceed each candidate cap:")
    for cap in CAPS:
        over = [x for x in all_parents if x > cap]
        pct = 100 * len(over) / len(all_parents) if all_parents else 0
        print(f"    > {cap:>6,} chars : {len(over):>5} parents ({pct:4.1f}%)")

    if oversized:
        print(f"\n  {len(oversized)} parents over {CAPS[-1]:,} chars — these decide the cap:")
        for name, where, size in sorted(oversized, key=lambda x: -x[2])[:12]:
            print(f"    {size:>8,}  {name:<14} {where}")


if __name__ == "__main__":
    main(sys.argv[1:])
