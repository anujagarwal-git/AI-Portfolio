"""Audit the 858 sub-40-char chunks against the claim that justifies keeping them.

THE CLAIM (decision 2026-08-21, option 1): "a tiny child is a bad retrieval KEY
but never a bad answer, because it delivers its full parent."

THIS IS NOT A LINKING PROBLEM. Children are built inside their parent's loop and
inherit `parent_id` as a loop variable, so a child CANNOT point at the wrong
parent — the same guarantee a docstore key gives, established at chunk time.
An earlier note in MENTOR_PROGRESS said a chunk "resolved to the wrong section";
that was wrong, and Anuj caught it.

WHAT THE SMOKE TEST ACTUALLY FOUND. The string occurs TWICE in CRE36:

    Introduction
      36.1 This chapter presents the minimum requirements ...
      (8) Validation of internal estimates       <- occurrence 1: a contents line
    ...
    Section 8: validation of internal estimates  <- occurrence 2: the real heading
      36.122 Banks must have a robust system ...

Occurrence 1 is genuinely text inside the Introduction, so its parent link is
CORRECT. The retriever returned the right parent for the chunk it matched; the
chunk was simply the navigational copy rather than the substantive one.

So this is a CHUNK-SELECTION defect, not a linking defect: table-of-contents
furniture became a retrieval key. Same family as 'FAQ' and 'Page 3' — a stricter
version of the `_NOISE` filter that already exists, not a new mechanism.

Severity is bounded accordingly: the harm is a DUPLICATE KEY COMPETING with the
real one, not wrong context. In the smoke query the real chunk still won
(CRE36.122 at 0.846, ranked first). This script measures how often the duplicate
exists at all, so the fix can be justified by a number rather than one example.

THE OBJECTIVE TEST. A contents entry is a chunk whose text matches the HEADING OF
A DIFFERENT PARENT in the same document. No judgement calls: if 'Validation of
internal estimates' is also the heading of 'Section 8: validation of internal
estimates', the chunk duplicates a section that is ALREADY INDEXED with its own
children, so it adds nothing retrievable.

Run:  uv run python scripts/audit_tiny_chunks.py
      uv run python scripts/audit_tiny_chunks.py --show 25
"""

from __future__ import annotations

import re
import sys
from collections import Counter

from regrag import config
from regrag.ingestion.cache import parse_cached
from regrag.ingestion.chunker import chunk_document
from regrag.ingestion.clean import clean_document
from regrag.ingestion.sections import find_sections
from regrag.registry import load

TINY = 40


def norm(s: str) -> str:
    """Compare headings and chunk text on WORDS ONLY.

    '(8) Validation of internal estimates' and 'Section 8: validation of
    internal estimates' are the same pointer wearing different numbering, so
    leading markers, case and punctuation must not affect the match.
    """
    s = re.sub(r"^\s*(?:section|chapter|part|annex)?\s*[\(\[]?\d+[\)\].:]?\s*", "", s, flags=re.I)
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()
    # note: returns a normalised STRING, not a set — word order matters for a heading


def main(show: int = 12) -> int:
    reg = load()
    tiny_real, tiny_pointer, tiny_fragment = [], [], []
    coarse = precise = 0
    parent_of_tiny = Counter()

    for row in reg.indexable():
        doc = parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
        cleaned = clean_document(doc, row)
        chunks, parents = chunk_document(cleaned, find_sections(cleaned, row), row)

        by_id = {p.parent_id: p for p in parents}
        # every heading in THIS document, normalised, mapped to its parent_id
        headings = {}
        for p in parents:
            h = norm(p.heading or "")
            if h:
                headings.setdefault(h, p.parent_id)

        for c in chunks:
            kind = c.payload.get("locator_kind")
            if kind == "paragraph":
                precise += 1
            elif kind == "section":
                coarse += 1

            if len(c.text) >= TINY:
                continue

            par = by_id.get(c.parent_id)
            parent_of_tiny[(row.short_name, (par.heading or "?")[:40] if par else "MISSING")] += 1

            key = norm(c.text)
            target = headings.get(key)
            rec = (row.short_name, c.text, (par.heading if par else "?"), par.n_chars if par else 0)

            if target and target != c.parent_id:
                # points at a DIFFERENT section that is already indexed
                tiny_pointer.append(rec + (by_id[target].heading,))
            elif len(c.text.split()) <= 2 or not re.search(r"[a-z]{3}", c.text):
                # '15% other physical', '-2', '(3)' — no sentence, no pointer
                tiny_fragment.append(rec)
            else:
                tiny_real.append(rec)

    total = len(tiny_real) + len(tiny_pointer) + len(tiny_fragment)
    print(f"\n{'='*74}\nTINY CHUNKS (< {TINY} chars): {total:,}\n{'='*74}")
    for label, bucket, verdict in [
        ("REAL short text", tiny_real, "claim HOLDS — parent is its own section"),
        ("TOC DUPLICATES of a real heading", tiny_pointer, "link is CORRECT; the chunk is navigational furniture"),
        ("FRAGMENTS / table cells", tiny_fragment, "no sentence at all — root cause is tables"),
    ]:
        pct = 100 * len(bucket) / total if total else 0
        print(f"  {label:32} {len(bucket):>5}  ({pct:4.1f}%)   {verdict}")

    if tiny_pointer:
        print(f"\n--- TOC DUPLICATES: navigational line, its (correct) parent, the section it names ---")
        for sn, txt, got, n, meant in tiny_pointer[:show]:
            print(f"  [{sn}] {txt[:44]!r}")
            print(f"      sits in : {got[:52]!r} ({n} chars)  <- correct link")
            print(f"      names   : {meant[:52]!r}  <- already indexed separately")

    if tiny_fragment:
        print(f"\n--- FRAGMENTS (sample) ---")
        for sn, txt, got, n in tiny_fragment[:show]:
            print(f"  [{sn}] {txt[:40]!r}  -> {got[:44]!r}")

    print(f"\n{'='*74}\nLOCATOR PRECISION (all chunks)\n{'='*74}")
    tot = precise + coarse
    print(f"  paragraph (precise, citable)  {precise:>6,}  ({100*precise/tot:4.1f}%)")
    print(f"  section   (coarse)            {coarse:>6,}  ({100*coarse/tot:4.1f}%)")
    print("  A coarse hit cites as 'Basel CRE Introduction' — true, but not a locator")
    print("  a reader can verify. This is the number prompts.py has to live with.")

    print(f"\n{'='*74}\nWHERE TINY CHUNKS CLUSTER (top 12 doc / parent pairs)\n{'='*74}")
    for (sn, h), n in parent_of_tiny.most_common(12):
        print(f"  {n:>4}  [{sn}] {h!r}")
    return 0


if __name__ == "__main__":
    n = 12
    if "--show" in sys.argv:
        n = int(sys.argv[sys.argv.index("--show") + 1])
    raise SystemExit(main(n))
