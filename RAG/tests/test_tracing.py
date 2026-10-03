"""Stage 10 smoke tests. NOTHING HERE NEEDS PHOENIX, AND THAT IS THE POINT.

The claim being tested is not "tracing works" - it is the harder one:
TRACING OFF CHANGES NOTHING. A span call with no collector must be a no-op,
not an exception and not a None that blows up one line later. If these tests
ever need a running container, the design has broken.

The one thing deliberately NOT tested here is a live export. That needs a
collector, so it belongs in a manual check (`ask.py --trace`), not in a suite
that has to pass on a machine with Docker stopped.
"""
from __future__ import annotations

import json

import pytest

from regrag.observability import audit, tracing


# --------------------------------------------------------------------------
# THE NO-OP TRACER
# --------------------------------------------------------------------------
def test_tracer_is_available_without_init():
    """No init(), no collector, and tracer() still returns something usable."""
    assert tracing.tracer() is not None


def test_noop_span_accepts_the_openinference_keyword():
    """The keyword that opentelemetry's OWN no-op tracer would reject.

    Phoenix's tracer takes `openinference_span_kind`; the plain OTel one does
    not. Falling back to OTel's no-op would raise TypeError here, in
    production, only on the path where tracing is off - the worst place to
    find out.
    """
    with tracing.tracer().start_as_current_span(
            "unit", openinference_span_kind="retriever") as span:
        span.set_input("q")
        span.set_output("a")
        span.set_attribute("n", 1)
        span.set_attributes({"m": 2})
        span.add_event("e")
        span.set_status("ok")
        assert span.is_recording() is False


def test_noop_span_does_not_swallow_a_real_error():
    """Silence about TRACING must not become silence about the pipeline."""
    with pytest.raises(ValueError):
        with tracing.tracer().start_as_current_span("unit"):
            raise ValueError("the pipeline failed")


def test_trace_id_is_empty_when_tracing_is_off():
    assert tracing.current_trace_id() == ""
    assert tracing.enabled() is False


def test_endpoint_is_the_collector_path_not_the_ui_root():
    """http://localhost:6006 serves the UI; spans go to /v1/traces.

    Posting to the root silently gets an HTML page back, which looks like
    nothing happening rather than like an error.
    """
    assert tracing.DEFAULT_ENDPOINT.endswith("/v1/traces")


# --------------------------------------------------------------------------
# THE AUDIT LINE
# --------------------------------------------------------------------------
class _Intent:
    decision, reason, passes, seconds, message = "PASS", "named doc", True, 0.4, ""


class _Response:
    """The two fields _record() branches on, and nothing else."""

    def __init__(self, answer=None, intent=None):
        self.question = "what does SR 11-7 require?"
        self.text = "..."
        self.intent = intent
        self.answer = answer
        self.seconds = 1.25

    @property
    def stopped(self) -> bool:
        return self.answer is None


def _read(path):
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(x) for x in lines]


def test_audit_writes_one_line_per_call(tmp_path):
    p = tmp_path / "regrag.jsonl"
    audit.log(_Response(), path=p)
    audit.log(_Response(), path=p)
    recs = _read(p)
    assert len(recs) == 2                      # appended, never overwritten
    assert recs[0]["question"].startswith("what does SR 11-7")
    assert recs[0]["stopped_at_gate"] is True


def test_audit_records_how_the_answer_was_made(tmp_path):
    """Model, temperature and window are what make two records comparable."""
    from regrag import config
    from regrag.generation.answer import Answer

    p = tmp_path / "regrag.jsonl"
    a = Answer("q", text="an answer", config_hash="abc123")
    audit.log(_Response(answer=a, intent=_Intent()), path=p)
    rec = _read(p)[0]
    assert rec["model"] == config.GEN_MODEL
    assert rec["temperature"] == config.TEMPERATURE
    assert rec["num_ctx"] == config.GEN_NUM_CTX
    assert rec["config_hash"] == "abc123"
    assert rec["gate"]["decision"] == "PASS"
    assert rec["stopped_at_gate"] is False
    assert rec["n_claims"] == 0


def test_audit_never_raises_into_the_caller(tmp_path):
    """A log that cannot be written must not cost the user their answer."""
    bad = tmp_path / "not-a-dir" / "x.jsonl"
    bad.parent.write_text("I am a file, not a directory", encoding="utf-8")
    audit.log(_Response(), path=bad)           # must return, not raise


def test_audit_creates_the_log_directory(tmp_path):
    p = tmp_path / "logs" / "regrag.jsonl"
    audit.log(_Response(), path=p)
    assert p.exists()
