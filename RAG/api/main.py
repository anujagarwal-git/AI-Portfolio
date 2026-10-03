"""FastAPI in front of the pipeline. Stage 11.

WHY POLLING AND NOT A PLAIN BLOCKING CALL
    One question takes ~90 seconds on this machine. A request that hangs that
    long looks broken to a browser, to a load balancer, and to curl's default
    timeout. So POST /ask returns an id immediately and the client asks GET
    /ask/{id} until it says done. The `stage` field carries the progress a
    stream would have carried.

WHY NOT STREAM THE MODEL'S TOKENS
    The citations do not exist while the model is writing. `answer()` gets JSON
    claims back, THEN swaps each label for a real reference. Streaming raw
    tokens would show the user machinery, not an answer. Decided 2026-09-12.

WHY AN IN-PROCESS DICT AND NOT A QUEUE
    One user, one machine, a portfolio demo. Redis or Celery here would be
    architecture for an audience that does not exist. What it costs: jobs die
    with the process, and two workers would not see each other's jobs - so run
    ONE worker. Both are stated in the README rather than hidden.

WHAT THIS MODULE OWNS
    HTTP shape, the job store, and the entry-point duties: tracing.init() and
    audit.log(). No pipeline logic. If a rule about retrieval or citation ever
    appears in this file, it is in the wrong place.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from regrag import config
from regrag.gate.respond import respond
from regrag.generation import llm
from regrag.observability import audit, tracing

TRACE_PROJECT = "regrag-api"


# ---------------------------------------------------------------------------
# THE WIRE TYPES. Pydantic models ARE the API documentation - /docs is built
# from them, so a field named badly here is a field named badly forever.
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    label: str                       # S1, S2 - the model's pointer
    citation: str                    # the reference a reader sees
    locator: str
    overlap: float
    align: str


class Evidence(BaseModel):
    """What was DELIVERED to the model, not what merely matched.

    This is the panel a validator reads. Without it the answer is a claim; with
    it the answer is a claim plus its source.
    """

    short_name: str
    heading: str
    chars: int
    windowed: bool
    cites: list[str]


class AskResult(BaseModel):
    question: str
    text: str
    stopped_at_gate: bool
    gate_decision: str = ""
    gate_reason: str = ""
    refused: bool = False
    refusal_reason: str = ""
    citations: list[Citation] = []
    evidence: list[Evidence] = []
    seconds: float = 0.0
    # STAMPED ON EVERY RESPONSE. An answer that cannot say how it was produced
    # cannot be compared with one produced differently.
    config_hash: str = ""
    model: str = config.GEN_MODEL
    trace_id: str = ""


class Job(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error"]
    stage: str = ""
    error: str = ""
    result: AskResult | None = None


# ---------------------------------------------------------------------------
# THE JOB STORE
# ---------------------------------------------------------------------------
_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def _set(job_id: str, **kw) -> None:
    with _LOCK:
        j = _JOBS[job_id]
        _JOBS[job_id] = j.model_copy(update=kw)


def _to_result(r, trace_id: str) -> AskResult:
    a = r.answer
    out = AskResult(question=r.question, text=r.text,
                    stopped_at_gate=bool(r.stopped),
                    seconds=round(r.seconds, 2), trace_id=trace_id)
    if r.intent is not None:
        out.gate_decision = str(r.intent.decision)
        out.gate_reason = str(r.intent.reason)
    if a is None:
        return out
    out.refused = bool(a.refused)
    out.refusal_reason = a.refusal_reason or ""
    out.config_hash = a.config_hash
    out.citations = [Citation(label=c.label, citation=c.citation,
                              locator=c.locator or "", overlap=c.overlap,
                              align=c.align_verdict) for c in a.claims]
    if a.retrieval is not None:
        out.evidence = [Evidence(short_name=p.short_name, heading=p.heading,
                                 chars=len(p.text), windowed=bool(p.windowed),
                                 cites=[h.cite for h in p.children])
                        for p in a.retrieval.passages]
    return out


def _run(job_id: str, question: str) -> None:
    """One question, on a worker thread. Never lets an exception escape.

    A crash here must land in the job as an error the client can read, not in a
    thread nobody is watching.
    """
    _set(job_id, status="running", stage="gating")
    try:
        with tracing.tracer().start_as_current_span(
                "api.ask", openinference_span_kind="chain") as sp:
            sp.set_input(question)
            # The stage label is best-effort: respond() does gate -> retrieve ->
            # generate in one call, so "generating" is set optimistically once
            # the gate cannot still be the answer. Honest naming: this is a
            # progress HINT, not a measurement.
            _set(job_id, stage="retrieving and generating")
            r = respond(question)
            sp.set_output(r.text)
            sp.set_attribute("stopped", bool(r.stopped))
            tid = tracing.current_trace_id()
        audit.log(r, trace_id=tid)
        _set(job_id, status="done", stage="done", result=_to_result(r, tid))
    except Exception as exc:                                   # noqa: BLE001
        _set(job_id, status="error", stage="failed",
             error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# STARTUP - load everything ONCE
# ---------------------------------------------------------------------------
WARMED: dict[str, Any] = {"ready": False, "seconds": 0.0, "error": ""}


def _warm() -> None:
    """One throwaway retrieval, to pay the load costs before a user arrives.

    MEASURED, NOT ASSUMED: the first question of a process spent 17.7s inside
    embed - almost certainly loading bge-small, not embedding eight words. The
    BM25 corpus is read out of Qdrant and cached the same way. Paying that on
    request one means the demo's first question looks twice as slow as the
    system is.
    """
    t0 = time.perf_counter()
    try:
        from regrag.retrieval.search import retrieve
        retrieve("model validation")
        WARMED["ready"] = True
    except Exception as exc:                                   # noqa: BLE001
        WARMED["error"] = f"{type(exc).__name__}: {exc}"
    WARMED["seconds"] = round(time.perf_counter() - t0, 1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tracing is the ENTRY POINT's job, and this is an entry point - but it is
    # OPT IN, like ask.py --trace. Set REGRAG_TRACE=1 before starting uvicorn to
    # send API traffic to the regrag-api project; unset, nothing is exported and
    # no collector is needed.
    if os.environ.get("REGRAG_TRACE", "0") != "0":
        try:
            tracing.init(project=TRACE_PROJECT)
        except Exception:                                      # noqa: BLE001
            pass      # no collector is not a reason to refuse to serve
    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="RegRAG", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Can this thing actually answer right now? Names WHICH part is down."""
    problems = []
    ollama = llm.why_unavailable()
    if ollama:
        problems.append(ollama)
    try:
        import requests
        r = requests.get(f"{config.QDRANT_URL}/readyz", timeout=5)
        if r.status_code != 200:
            problems.append(f"Qdrant returned HTTP {r.status_code}")
    except Exception as exc:                                   # noqa: BLE001
        problems.append(f"cannot reach Qdrant at {config.QDRANT_URL}: {exc}")
    return {"ok": not problems and WARMED["ready"],
            "warm": WARMED["ready"], "warm_seconds": WARMED["seconds"],
            "warm_error": WARMED["error"],
            "problems": problems,
            "model": config.GEN_MODEL, "gate": config.GATE_ENABLED,
            "tracing": tracing.enabled()}


@app.post("/ask", response_model=Job, status_code=202)
def ask(req: AskRequest) -> Job:
    """Accept the question, answer later. 202 means accepted, not finished."""
    job_id = uuid.uuid4().hex[:12]
    with _LOCK:
        _JOBS[job_id] = Job(id=job_id, status="queued", stage="queued")
    threading.Thread(target=_run, args=(job_id, req.question),
                     daemon=True).start()
    return _JOBS[job_id]


@app.get("/ask/{job_id}", response_model=Job)
def poll(job_id: str) -> Job:
    with _LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job (jobs do not survive a restart)")
    return job
