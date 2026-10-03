"""The generation call. Raw HTTP to Ollama, nothing between us and the model.

WHY NOT langchain_ollama.ChatOllama
    A wrapper decides defaults on your behalf and hides them. The one default
    that matters here is `num_ctx`, and getting it wrong loses evidence with no
    error. Fifteen lines of `requests` keeps every knob at the call site, where
    it can be read in a review.

      1. sets num_ctx explicitly (config.GEN_NUM_CTX);
      2. returns `prompt_eval_count` to the caller so truncation is OBSERVABLE
         rather than assumed away.

DETERMINISM
    temperature=0 alone is not always enough - a tie between two equally likely
    tokens still has to be broken. `seed` fixes that. Both are set, and
    scripts/smoke_llm.py verifies it by running the same request twice and
    comparing byte for byte. An experiment on a non-deterministic generator
    measures the generator's noise, not the thing being tested.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests

from regrag import config


@dataclass
class Completion:
    """One generation, with the numbers needed to trust it."""

    text: str
    prompt_tokens: int          # what the server ACTUALLY read - see truncation note
    output_tokens: int
    seconds: float
    prompt_seconds: float = 0.0
    eval_seconds: float = 0.0
    load_seconds: float = 0.0
    model: str = ""
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def maybe_truncated(self) -> bool:
        """Did the prompt come within a hair of the window? Then suspect loss.

        Not proof. Ollama does not report that it dropped anything, so the only
        signal available is the prompt landing suspiciously close to the cap.
        """
        return self.prompt_tokens >= config.GEN_NUM_CTX - 64


def chat(system: str, user: str, *, model: str | None = None,
         max_tokens: int | None = None, timeout: int | None = None,
         think: bool | None = None, fmt: str | dict | None = None) -> Completion:
    
    """One non-streaming chat completion. Raises on any HTTP failure.

    `max_tokens` caps generation (`num_predict`). Set it to 1 to measure PROMPT
    processing alone - otherwise a timing includes both reading the prompt and
    writing the answer, and a slow result cannot be attributed to either.

    `think=False` disables the reasoning phase on hybrid Qwen3 models. Those
    models write a <think> block BEFORE the answer, and on CPU that block is
    often longer than the answer itself - pure latency for a task where the
    passages already contain the reasoning. 
    """
    t0 = time.perf_counter()
    r = requests.post(
        config.OLLAMA_CHAT_URL,
        json={
            "model": model or config.GEN_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
       
            **({"format": fmt} if fmt else {}),
            # Defaults to config.GEN_THINK so no caller can forget it and pay

            # for a reasoning block by accident.
            "think": config.GEN_THINK if think is None else think,
            "options": {
                "temperature": config.TEMPERATURE,
                "seed": config.GEN_SEED,
                "num_ctx": config.GEN_NUM_CTX,
                **({"num_predict": max_tokens} if max_tokens else {}),
            },
        },
        timeout=timeout or config.GEN_TIMEOUT_S,
    )
    r.raise_for_status()
    body = r.json()
    return Completion(
        text=body.get("message", {}).get("content", ""),
        prompt_tokens=int(body.get("prompt_eval_count") or 0),
        output_tokens=int(body.get("eval_count") or 0),
        seconds=time.perf_counter() - t0,
        # Ollama reports these in NANOSECONDS. They separate reading the prompt
        # from writing the answer, which wall-clock cannot.
        prompt_seconds=int(body.get("prompt_eval_duration") or 0) / 1e9,
        eval_seconds=int(body.get("eval_duration") or 0) / 1e9,
        load_seconds=int(body.get("load_duration") or 0) / 1e9,
        model=str(body.get("model", "")),
        raw=body,
    )


def available() -> bool:
    """Is Ollama up AND does it hold the model we intend to use?"""
    return not why_unavailable()


def why_unavailable() -> str:
    """Empty string if all is well, otherwise WHICH failure it was.

    The first version of this returned a bare False for every failure, so
    "Ollama is not running", "wrong host" and "model not pulled" all printed the
    same sentence: `<model> not in /api/tags`. That message named the wrong cause
    two times out of three. A check that cannot distinguish its own failures is
    not a check - it is a guess with a confident voice.
    """
    try:
        r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=10)
    except requests.exceptions.ConnectionError:
        return (f"cannot reach Ollama at {config.OLLAMA_HOST} - it is not running, "
                f"or REGRAG_OLLAMA_HOST points somewhere else. Try: ollama serve")
    except requests.exceptions.Timeout:
        return f"{config.OLLAMA_HOST} did not answer within 10s (busy or hung?)"
    except Exception as exc:                                     # noqa: BLE001
        return f"{type(exc).__name__} talking to {config.OLLAMA_HOST}: {exc}"
    if r.status_code != 200:
        return f"{config.OLLAMA_HOST}/api/tags returned HTTP {r.status_code}"
    names = sorted(m.get("name", "") for m in r.json().get("models", []))
    if config.GEN_MODEL in names or f"{config.GEN_MODEL}:latest" in names:
        return ""
    return (f"Ollama IS running and holds {len(names)} model(s), but "
            f"{config.GEN_MODEL!r} is not among them.\n  present: "
            + ", ".join(names) + f"\n  fix: ollama pull {config.GEN_MODEL}  "
            f"(or correct config.GEN_MODEL)")
