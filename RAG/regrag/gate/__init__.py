"""Stage 8.6 — the intent gate. A model in front of retrieval, not inside it.

`respond("question")` is the entry point WITH the gate.
`answer("question")`  is the entry point WITHOUT it, unchanged.

Both still exist on purpose: evaluation must be able to measure the pipeline
without a model deciding what gets measured.
"""
from regrag.gate.intent import DECISIONS, Intent, classify, decide
from regrag.gate.respond import Response, respond

__all__ = ["DECISIONS", "Intent", "Response", "classify", "decide", "respond"]
