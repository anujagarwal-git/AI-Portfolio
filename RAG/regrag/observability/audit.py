"""The audit line. One JSON record per question, appended to logs/regrag.jsonl.

WHY THIS EXISTS ALONGSIDE PHOENIX
    They answer different questions. Phoenix is for a HUMAN debugging one
    question now - it is a viewer, its store is a container volume, and nobody
    greps it. This file is the durable record: what was asked, what the gate
    decided, which model and which prompt produced the answer, what it cited,
    and how long each step took. It is the thing you would hand an examiner,
    and the thing you can still read in six months with `cat`.

    `trace_id` ties one line here to one trace there, so the durable record and
    the debugging view are never two separate stories.

WHY IT NEVER RAISES
    A failure to WRITE A LOG must not cost the user their answer. Every error
    here is swallowed and reported once on stderr. That is the opposite of the
    rule in tracing.init(), and for the opposite reason: init() is asked for
    explicitly, this is a side effect of answering.

WHAT IS DELIBERATELY NOT IN THE RECORD
    The passage TEXT. It is large, it is already in Qdrant, and the locator
    plus config hash is enough to reproduce exactly what was delivered. What
    is kept is what could not be recovered later: the decisions.
"""
from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

from regrag import config

LOG_DIR = pathlib.Path(__file__).resolve().parents[2] / "logs"
LOG_PATH = LOG_DIR / "regrag.jsonl"


def _record(r, trace_id: str) -> dict:
    a = r.answer
    rec: dict = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trace_id": trace_id,
        "question": r.question,
        # What the pipeline was asked. Equal to `question` unless the gate
        # added a document name it had already determined.
        "question_sent": getattr(r, "question_sent", "") or r.question,
        "stopped_at_gate": bool(r.stopped),
        "seconds": round(r.seconds, 2),
        # HOW IT WAS MADE. Without these three a record cannot be compared to
        # any other record - the same question answers differently under a
        # different model or prompt, and the hash is what says which.
        "model": config.GEN_MODEL,
        "temperature": config.TEMPERATURE,
        "num_ctx": config.GEN_NUM_CTX,
    }
    if r.intent is not None:
        rec["gate"] = {"decision": r.intent.decision,
                       "reason": r.intent.reason,
                       "passes": bool(r.intent.passes),
                       "seconds": round(r.intent.seconds, 2)}
    if a is None:
        return rec
    rec["config_hash"] = a.config_hash
    rec["refused"] = bool(a.refused)
    rec["refusal_reason"] = a.refusal_reason or ""
    rec["n_claims"] = len(a.claims)
    rec["n_invalid_claims"] = len(a.invalid)
    rec["citations"] = [{"label": c.label, "citation": c.citation,
                         "locator": c.locator, "overlap": c.overlap,
                         "align": c.align_verdict} for c in a.claims]
    if a.retrieval is not None:
        ret = a.retrieval
        rec["retrieval"] = {
            "mode": str(ret.plan.mode),
            "facets": [str(f.label) for f in ret.plan.facets],
            "n_passages": len(ret.passages),
            "total_chars": ret.total_chars,
            "n_dropped": len(ret.dropped),
            "timings_ms": {k: round(v * 1000) for k, v in ret.timings.items()},
            "notes": [str(n) for n in ret.notes],
        }
    return rec


def log(r, *, trace_id: str = "", path: pathlib.Path | None = None) -> None:
    """Append one line for one Response. Never raises."""
    try:
        p = path or LOG_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(_record(r, trace_id), ensure_ascii=False) + "\n")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [audit log not written: {type(exc).__name__}: {exc}]",
              file=sys.stderr)
