"""THE ENTRY POINT WITH THE GATE IN FRONT. `respond("question") -> Response`.

    The gate runs BEFORE retrieval — `classify()` returns before
    `retrieve()` is called at all. 

WHAT A CALLER GETS
    `Response.text` is always what a person should read. On the pass-through
    path it is the cited answer; otherwise it is the gate's one sentence.
    `Response.answer` is None whenever the gate stopped the question — so a
    caller can tell "no answer was produced" from "an answer was produced and
    it was empty", which are different failures.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from regrag import config
from regrag.gate.intent import Intent, classify, locked_documents
from regrag.generation.answer import Answer, answer
from regrag.observability.tracing import tracer


@dataclass
class Response:
    question: str
    text: str
    intent: Intent | None = None       # None when the gate is switched off
    answer: Answer | None = None       # None when the gate stopped the question
    seconds: float = 0.0
    # What the PIPELINE was actually asked. Differs from `question` only when
    # the gate added a document name it had already determined - see
    # intent.locked_documents(). Kept so the audit record can show both.
    question_sent: str = ""

    @property
    def stopped(self) -> bool:
        return self.answer is None

    def __str__(self) -> str:
        return self.text


def respond(question: str, **kw) -> Response:
    """Gate, then the pipeline. `kw` is forwarded to `answer()` untouched."""
    t0 = time.perf_counter()

    if not config.GATE_ENABLED:
        a = answer(question, **kw)
        return Response(question, str(a), None, a, time.perf_counter() - t0,
                        question)

    with tracer().start_as_current_span(
            "gate", openinference_span_kind="chain") as _sp:
        _sp.set_input(question)
        intent = classify(question)
        _sp.set_attribute("decision", str(intent.decision))
        _sp.set_attribute("reason", str(intent.reason))
        _sp.set_attribute("passes", bool(intent.passes))
        _sp.set_attribute("extraction", json.dumps(intent.extraction,
                                                   default=str))
        _sp.set_output(intent.message or "")
    if not intent.passes:
        return Response(question, intent.message, intent, None,
                        time.perf_counter() - t0, question)

    # *** THE GATE PUTS BACK WHAT IT ALREADY KNEW (2026-09-17). ***
    # On a two-jurisdiction question the gate names the locked document to the
    # USER ("UK: using SS1/23.") but the question text carried only the one the
    # user typed, so planner rule 2 fired and threw that name away. Naming both
    # here makes the question take rule 1 - document fan-out - which is the
    # plan it was always meant to have. Nothing is invented: every name comes
    # from a registry cell holding exactly one document.
    asked = question
    extra = [d for d in locked_documents(intent.extraction)
             if d.lower() not in question.lower()]
    if extra:
        asked = f"{question} ({', '.join(extra)})"

    a = answer(asked, **kw)
    return Response(question, str(a), intent, a, time.perf_counter() - t0, asked)
