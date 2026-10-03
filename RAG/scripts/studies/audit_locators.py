"""Split citation traceability per document, and split the UNTRACEABLE part
into (a) pipeline missed an available number and (b) the document offers none.

THE TEST FOR 'PIPELINE FAULT' — objective, and able to come back ZERO:
a chunk whose locator_kind is 'section' (coarse) but whose OWN TEXT begins with
a paragraph marker. The number was sitting in the text and was not picked up.
This does not rely on my reading of the extractor; it reads the raw text.

LIMITS, stated so the numbers are not over-read:
- a leading '(1)' is a citable sub-paragraph in the CFR but may be a plain list
  bullet in prose, so bracketed and dotted markers are reported SEPARATELY.
- 'no marker' does not prove the document is unnumbered at that point; it proves
  no marker survived into the chunk text. Doc-level marker density is printed
  alongside so the two can be told apart.
"""
from __future__ import annotations
import re, sys
from collections import Counter, defaultdict

from regrag import config
from regrag.ingestion.cache import parse_cached
from regrag.ingestion.chunker import chunk_document
from regrag.ingestion.clean import clean_document
from regrag.ingestion.sections import find_sections
from regrag.registry import load

DOTTED = re.compile(r"^\s*\(?\d{1,3}(?:\.\d{1,3}){1,3}\)?[\s.:)]")   # 36.122  20.2  99.16
BRACKET = re.compile(r"^\s*\((?:[a-zA-Z]|\d{1,2}|[ivxIVX]{1,4})\)\s")  # (a) (1) (iv)
BIN = {"footnotes", "footnote", "notes", "table", "tables", "", "?"}

def norm(s): return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()

def main() -> int:
    reg = load()
    rows = []
    coarse_bucket = Counter()
    samples = defaultdict(list)
    printed_keys = False

    for row in reg.indexable():
        doc = parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
        cleaned = clean_document(doc, row)
        chunks, parents = chunk_document(cleaned, find_sections(cleaned, row), row)
        by_id = {p.parent_id: p for p in parents}

        precise = coarse = other = 0
        dotted_anywhere = 0
        for c in chunks:
            global_keys = c.payload.keys()
            if not printed_keys:
                print("payload keys:", sorted(global_keys)); printed_keys = True
            if DOTTED.search(c.text): dotted_anywhere += 1
            kind = c.payload.get("locator_kind")
            if kind == "paragraph":
                precise += 1; continue
            if kind != "section":
                other += 1; continue
            coarse += 1
            par = by_id.get(c.parent_id)
            head = (par.heading if par else "") or ""
            nh = norm(head)
            if DOTTED.search(c.text):
                b = "A1 pipeline: dotted number in text, cited as section"
            elif BRACKET.search(c.text):
                b = "A2 pipeline?: bracketed marker in text, cited as section"
            elif (not nh) or nh in BIN or (par is None):
                b = "B  pipeline: parent is a bin/no heading -> nothing to cite"
            elif len(head) > 90:
                b = "B  pipeline: parent is a bin/no heading -> nothing to cite"
            else:
                b = "C  document: no marker in text; parent IS a named heading"
            coarse_bucket[b] += 1
            if len(samples[b]) < 6:
                samples[b].append(f"[{row.short_name}] {c.text[:58]!r} -> parent {head[:44]!r}")
        tot = precise + coarse + other
        rows.append((row.short_name, tot, precise, coarse, other, dotted_anywhere))

    W = 74
    print("\n" + "=" * W); print("PER DOCUMENT — how precisely a chunk can be cited"); print("=" * W)
    print(f"  {'document':<24}{'chunks':>7}{'paragraph':>11}{'section':>9}{'other':>7}   {'%precise':>8}")
    for sn, tot, p, c, o, _ in sorted(rows, key=lambda r: -r[1]):
        pct = 100 * p / tot if tot else 0
        print(f"  {sn[:24]:<24}{tot:>7,}{p:>11,}{c:>9,}{o:>7,}   {pct:>7.1f}%")
    T = sum(r[1] for r in rows); P = sum(r[2] for r in rows); C = sum(r[3] for r in rows); O = sum(r[4] for r in rows)
    print(f"  {'TOTAL':<24}{T:>7,}{P:>11,}{C:>9,}{O:>7,}   {100*P/T:>7.1f}%")

    print("\n" + "=" * W); print("THE COARSE CHUNKS — why is there no paragraph number?"); print("=" * W)
    for b, n in sorted(coarse_bucket.items()):
        print(f"  {b:<58}{n:>6,}  ({100*n/C:4.1f}% of coarse)")
    print("\n--- samples ---")
    for b in sorted(samples):
        print(f"\n  {b}")
        for s in samples[b]: print(f"    {s}")

    print("\n" + "=" * W); print("DOC-LEVEL: does the document use dotted numbering at all?"); print("=" * W)
    for sn, tot, p, c, o, d in sorted(rows, key=lambda r: -r[5]):
        print(f"  {sn[:24]:<24}{d:>6,} of {tot:>6,} chunks carry a dotted number  ({100*d/tot if tot else 0:4.1f}%)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
