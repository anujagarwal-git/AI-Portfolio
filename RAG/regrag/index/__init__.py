"""Indexing — turning chunks into a searchable store.

    embedder.py       text -> vectors (bge-small, CPU)
    vector_store.py   Qdrant: collections, payload indexes, delete_by_doc
    lexical_store.py  BM25                                   (deferred from v1)
"""
