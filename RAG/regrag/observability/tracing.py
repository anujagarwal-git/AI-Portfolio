"""Turns tracing on, and stays out of the way when it is off.

    WHAT THIS OWNS
    One tracer, and the decision of where its spans go. 

    WHY init() IS CALLED BY THE SCRIPT AND NOT BY THE LIBRARY
    A library module that starts a tracer on import makes every test and every
    script depend on a docker being up. So
    regrag/ only ever calls tracer(), and the SCRIPT decides whether tracing
    was ever switched on.

    WHY THERE IS A NO-OP TRACER BELOW
    With tracing off, tracer() must return something that accepts exactly the
    same calls and does nothing. Not None, not a raise error. 

    It cannot be opentelemetry's own no-op tracer. Phoenix's tracer takes
    `openinference_span_kind=...`; the plain OTel one would raise TypeError on
    that keyword. The no-op here swallows any keyword on purpose.

    batch=False is kept. The batching exporter holds spans for a few seconds
    before sending; for one question at a time that means staring at an empty
    UI and wondering what broke. Immediate export costs a localhost round trip
    per span. Switch to batch=True only if a long run makes that cost visible.

    set_global_tracer_provider=False is deliberate. RAGAS and langchain are in
    this environment and may set up their own tracing. Writing our provider
    into OTel's global slot would make whichever ran last the winner. We hold
    our own provider and hand out our own tracer.
"""
from __future__ import annotations

import contextlib
import os

DEFAULT_ENDPOINT = os.environ.get(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
).rstrip("/") + "/v1/traces"

PROJECT = "regrag"

_TRACER = None


class _NoopSpan:
    """Accepts every call a real span accepts. Records nothing."""

    def set_attribute(self, *a, **k): pass
    def set_attributes(self, *a, **k): pass
    def set_input(self, *a, **k): pass
    def set_output(self, *a, **k): pass
    def set_status(self, *a, **k): pass
    def add_event(self, *a, **k): pass
    def record_exception(self, *a, **k): pass
    def is_recording(self) -> bool: return False


class _NoopTracer:
    @contextlib.contextmanager
    def start_as_current_span(self, name, **kwargs):
        yield _NoopSpan()


def init(*, project: str = PROJECT, endpoint: str = DEFAULT_ENDPOINT,
         verbose: bool = False):
    """Point spans at a Phoenix collector. Safe to call twice.

    Raises if the package is missing or the collector address is malformed -
    an entry point that ASKED for tracing should hear that it did not get it.
    Silence belongs in tracer(), not here.
    """
    global _TRACER
    if _TRACER is not None:
        return _TRACER
    from phoenix.otel import register            # imported here, never at module scope
    provider = register(
        endpoint=endpoint,
        project_name=project,
        protocol="http/protobuf",
        batch=False,
        set_global_tracer_provider=False,
        auto_instrument=False,
        verbose=verbose,
    )
    _TRACER = provider.get_tracer("regrag")
    return _TRACER


def tracer():
    """The tracer if init() ran, otherwise a no-op with the same interface."""
    return _TRACER if _TRACER is not None else _NoopTracer()


def enabled() -> bool:
    return _TRACER is not None


def current_trace_id() -> str:
    """Hex id of the trace in progress, or "" when tracing is off.

    This is the ONLY link between the durable audit line and the Phoenix view.
    Reading it must never fail: no package, no active span, both give "".
    """
    try:
        from opentelemetry import trace as _t
        ctx = _t.get_current_span().get_span_context()
        return format(ctx.trace_id, "032x") if ctx and ctx.trace_id else ""
    except Exception:                                          # noqa: BLE001
        return ""
