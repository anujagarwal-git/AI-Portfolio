"""regrag — RAG Regulatory Assistant.

Package layout (see MENTOR_PROGRESS.md for the full design rationale):

    regrag/
        config.py       calibrated values: held constant, visible, versioned
        domain.py       typed objects shared everywhere — kills circular imports
        registry.py     THE SEAM between corpus and code; the verified gate
        validation.py   runtime post-checks                        (deferred)
        api.py          composition root — one entry point for eval AND prod
        ingestion/      parser.py  sections.py  clean.py  chunker.py
        index/          embedder.py  vector_store.py  lexical_store.py
        retrieval/      planner.py  search.py  reranker.py
        generation/     prompts.py  generate.py

Modules were merged on "same reason to change" (28 -> 17). Four must NOT be
merged into anything else:
    registry.py   it is the seam; corpus changes must be DATA changes
    domain.py     shared types have no dependencies, so nothing imports cycles
    prompts.py    prompt edits should show up as isolated git diffs
    api.py        a single composition root makes eval/prod drift impossible
"""

__version__ = "0.1.0"
