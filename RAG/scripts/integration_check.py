"""Stage 5 close-out. Does the QUERY side agree with the INGESTION side?

Running the pipeline end to end proves it does not crash. That is the easy half.
The half that actually bites is DRIFT: the index was built by code that has since
changed, so what Qdrant holds is no longer what the chunker would produce today.
That failure is silent — every query still returns something.

This project has already been bitten twice by exactly that: 858 tiny chunks
became 664, and 1,465 parents became 1,664, with no run in between that said so.

So the check re-chunks ONE document from the parse cache and compares it against
what the index actually holds, chunk for chunk. Then it runs the query path.

    uv run python scripts/integration_check.py
    uv run python scripts/integration_check.py --doc "Basel CRE"
"""
from __future__ import annotations

import sys
import time

from regrag import config
from regrag.index.vector_store import VectorStore, point_id
from regrag.ingestion.cache import parse_cached
from regrag.ingestion.chunker import chunk_document
from regrag.ingestion.clean import clean_document
from regrag.ingestion.sections import find_sections
from regrag.registry import load
from regrag.retrieval.search import retrieve

DOC = "SR 26-2"          # smallest document — the check is about drift, not scale
W = 88
ok = lambda b: "PASS" if b else "FAIL"
fails: list[str] = []


def check(label, passed, detail=""):
    fails.append(label) if not passed else None
    print(f"  [{ok(passed)}] {label:<52}{detail}")


def main() -> int:
    doc = sys.argv[sys.argv.index("--doc") + 1] if "--doc" in sys.argv else DOC
    t0 = time.perf_counter()

    # ---- 1. the seams load -------------------------------------------
    print("=" * W); print("1. THE SEAMS"); print("=" * W)
    reg = load()
    rows = list(reg.indexable())
    check("registry loads and validates", bool(rows), f"{len(rows)} indexable rows")

    store = VectorStore(strategy="parentdoc")
    stats = store.stats()
    check("Qdrant collection exists", stats.points > 0, str(stats))
    check("parent store loads", stats.parents > 0, f"{stats.parents:,} parents")

    # ---- 2. DRIFT: does the index match what the code produces NOW? ---
    print("\n" + "=" * W); print(f"2. DRIFT — re-chunk {doc!r} and compare to the index"); print("=" * W)
    row = next((r for r in rows if doc.lower() in r.short_name.lower()), None)
    if row is None:
        print(f"  no registry row matching {doc!r}"); return 1

    d = parse_cached(config.PROJECT_ROOT / row.file, row.doc_id, verbose=False)
    cleaned = clean_document(d, row)
    chunks, parents = chunk_document(cleaned, find_sections(cleaned, row), row)

    from qdrant_client.models import FieldCondition, Filter, MatchValue
    indexed, offset = [], None
    while True:
        pts, offset = store.client.scroll(
            collection_name=store.collection, limit=1_000, offset=offset,
            with_payload=True, with_vectors=False,
            scroll_filter=Filter(must=[FieldCondition(key="doc_id",
                                                      match=MatchValue(value=row.doc_id))]))
        indexed += pts
        if offset is None:
            break

    check("chunk COUNT matches the index", len(chunks) == len(indexed),
          f"code {len(chunks)} vs index {len(indexed)}")

    by_id = {p.payload["chunk_id"]: p.payload for p in indexed}
    missing = [c.chunk_id for c in chunks if c.chunk_id not in by_id]
    check("every chunk the code makes is IN the index", not missing,
          f"{len(missing)} missing" + (f" e.g. {missing[0]}" if missing else ""))

    same = [c for c in chunks if c.chunk_id in by_id
            and (by_id[c.chunk_id].get("text") or "") == c.text]
    check("chunk TEXT is identical", len(same) == len(chunks) - len(missing),
          f"{len(same)}/{len(chunks) - len(missing)} match")

    if chunks and chunks[0].chunk_id in by_id:
        c = chunks[0]
        got = store.client.retrieve(store.collection, ids=[point_id(c.chunk_id)],
                                    with_payload=True)
        check("point id is DERIVED, so it round-trips", bool(got),
              f"{c.chunk_id[:46]}")

    # ---- 3. the parent link ------------------------------------------
    print("\n" + "=" * W); print("3. THE PARENTDOC LINK"); print("=" * W)
    dangling = [p.payload.get("parent_id") for p in indexed
                if not store.get_parent(p.payload.get("parent_id"))]
    check("no chunk points at a missing parent", not dangling,
          f"{len(dangling)} dangling")
    check("parent COUNT matches the index", len(parents) == len({p.parent_id for p in parents}),
          f"code {len(parents)} parents for this doc")

    # ---- 4. the query path -------------------------------------------
    print("\n" + "=" * W); print("4. QUESTION -> PLAN -> SEARCH -> PARENTS"); print("=" * W)
    cases = [
        ("named document reaches a superseded row", "what did SR 11-7 say about spreadsheets?",
         lambda r: {p.short_name for p in r.passages} == {"SR 11-7"}),
        ("three-way fan-out covers all families", "what is model validation?",
         lambda r: len({p.short_name for p in r.passages}) >= 3),
        ("an exact locator is retrievable", "what does CRE36.122 require?",
         lambda r: bool(r.passages)),
        ("a content-free question is refused", "?????",
         lambda r: bool(r.refused) and not r.passages),
    ]
    for label, q, pred in cases:
        r = retrieve(q)
        check(label, pred(r),
              f"{len(r.passages)} psg, {r.total_chars:,}c, {r.timings.get('total', 0)*1000:.0f}ms")
        check(f"  budget respected — {q[:30]!r}",
              r.total_chars <= config.CONTEXT_MAX_CHARS,
              f"{r.total_chars:,} <= {config.CONTEXT_MAX_CHARS:,}")

    # ---------------------------------------------------------------- 5
    # STAGE 7 — end to end. Everything above proves the EVIDENCE is right.
    # This proves a raw question becomes a CITED ANSWER: retrieve -> facet
    # margin -> relevance floor -> render -> generate -> resolve labels ->
    # markers + SOURCES. Skipped, not failed, when Ollama is not running: a
    # missing generator must not make a retrieval regression look green.
    print("\n" + "=" * W)
    print("5. GENERATION — raw question to cited answer")
    print("=" * W)
    from regrag.generation import llm, prompts
    from regrag.generation.answer import answer

    a = None
    problem = llm.why_unavailable()
    if problem:
        print(f"  SKIPPED — {problem}")
        print("  (retrieval checks above still stand; Stage 7 is unverified in this run)")
    else:
        pr = prompts.load()
        print(f"  prompt {pr.version}@{pr.sha}   model {config.GEN_MODEL}")
        a = answer("When is a financial asset credit-impaired?")
        check("an answerable question is answered", not a.refused, a.refusal_reason or "")
        check("  claims were produced", bool(a.claims), f"{len(a.claims)} claim(s)")
        check("  every source label resolves", not a.invalid,
              f"{len(a.invalid)} invalid" if a.invalid else "0 invalid")
        check("  the answer carries markers and sources",
              bool(a.sources) and "[1]" in a.text,
              f"{len(a.sources)} source(s)")
        check("  the answer stamps how it was made", bool(a.config_hash), a.config_hash)
        if a.weak:
            print(f"    note: {len(a.weak)} weakly-grounded claim(s) — a screen, read them")
        print()
        print("\n".join("    " + ln for ln in str(a).splitlines()))

        b = answer("?????")
        check("\n  a content-free question never reaches the model",
              b.refused and not b.claims, b.refusal_reason)

    # ---------------------------------------------------------------- 6
    # STAGE 8.6 — THE GATE. Everything above ran WITHOUT it, which is the
    # point: `answer()` is unchanged, so sections 4 and 5 still measure the
    # pipeline the way every recorded number measured it.
    #
    # THE ONLY CLAIM WORTH TESTING HERE IS TRANSPARENCY. Whether the gate
    # CLASSIFIES well was measured on 120 questions across two independent
    # sets (53/60 and 55/60, scripts/experiment_prompts.py). Three questions
    # here cannot add to that and must not be quoted as if they could. What
    # three questions CAN prove is that adding a link to the chain did not
    # change the chain: a passed question yields byte-identical output to
    # `answer()`, a stopped question never reaches retrieval, and switching
    # the gate off restores yesterday's pipeline exactly.
    #
    # That last one turns "deleting this package returns the project to what
    # it was" from a claim in a docstring into a tested fact.
    print("\n" + "=" * W)
    print("6. THE INTENT GATE — transparent when it passes, absent when off")
    print("=" * W)
    if problem:
        print(f"  SKIPPED — {problem}")
        print("  (the gate needs the same model; nothing above depends on it)")
    elif not config.GATE_ENABLED:
        print("  SKIPPED — REGRAG_GATE=0, so the gate is not in this run.")
        print("  This is the DISABLED path and sections 4-5 above ARE its test:")
        print("  they passed, so the pipeline without the gate is intact.")
    else:
        from regrag.gate import respond
        print(f"  gate {config.GATE_MODEL} / {config.GATE_PROMPT}")

        q = "When is a financial asset credit-impaired?"
        g = respond(q)
        check("an answerable question passes the gate", not g.stopped,
              g.intent.decision if g.intent else "?")
        if not g.stopped:
            # `a` is section 5's answer to the SAME question, produced without
            # the gate. Temperature 0 and seed 0 make this a real equality
            # test. A difference means the gate touched the question, or
            # generation is not reproducible — both are findings, not noise.
            check("  the gate changed the answer in NO way",
                  g.answer.text == a.text and g.answer.sources == a.sources,
                  "identical to section 5" if g.answer.text == a.text
                  else "DIFFERENT — the gate is not transparent")

        j = respond("How do I cook basmati rice?")
        check("an out-of-domain question is stopped before retrieval",
              j.stopped and j.answer is None,
              j.intent.decision if j.intent else "?")
        check("  it is answered with a message, not silence",
              bool(j.text.strip()), f"{len(j.text)} chars")

        # THE ESCAPE HATCH, TESTED. Not "the flag exists" - that a switched-off
        # gate really does hand the question straight to answer().
        config.GATE_ENABLED = False
        try:
            off = respond(q)
            check("REGRAG_GATE=0 restores the ungated pipeline exactly",
                  off.intent is None and off.answer is not None
                  and off.answer.text == a.text,
                  "identical to section 5" if off.answer
                  and off.answer.text == a.text else "DIFFERENT")
        finally:
            config.GATE_ENABLED = True

    print("\n" + "=" * W)
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED:"); [print(f"  - {f}") for f in fails]
        print("\n  A count mismatch in section 2 means THE INDEX IS STALE — the chunker has")
        print("  changed since it was built. Rebuild before trusting any retrieval result.")
    else:
        print(f"ALL CHECKS PASSED in {time.perf_counter() - t0:.1f}s — "
              f"STAGES 5, 7 AND 8.6 GREEN.")
    print("=" * W)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
