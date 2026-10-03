"""WHY DOES THE JUDGE NEVER RETURN? Four arms, one variable at a time.

THE OBSERVATION THAT STARTED THIS (2026-09-08)
    Every RAGAS job timed out at EXACTLY 300.01s - six of six - on smoke-test
    contexts three sentences long. A 4B model does not take five minutes on
    three sentences. Uniform timeouts on trivial input means the call never
    returns, not that it is slow. That rules out prompt size before any test.

THE HYPOTHESIS, AND WHY IT IS NOT A GUESS
    qwen3 is a HYBRID model that writes a <think> block before answering.
    regrag/generation/llm.py already handles this and says so:

        "think=False disables the reasoning phase on hybrid Qwen3 models.
         VERIFY IT WORKED: the caller must check the output for <think>,
         because an Ollama build that does not know this field will ignore it
         silently and thinking stays on."

    config.GEN_THINK = False, verified 2026-08-26. The GENERATOR turns thinking
    off over raw HTTP. The JUDGE goes through ChatOllama, which was never told.

WHAT EACH ARM ISOLATES - change ONE thing between adjacent arms
    A  raw HTTP, think unset      is the model slow by itself?
    B  raw HTTP, think=false      does think=false fix it?          (A vs B)
    C  ChatOllama, default        does langchain reach the same place as A?
    D  ChatOllama, reasoning=Fa.. can langchain pass think=false?   (C vs D)

    A vs B answers "is thinking the cause". C vs D answers "can the judge be
    configured to avoid it". They are different questions and a single
    before/after would conflate them - the 2026-08-23 rule.

EVERY ARM CHECKS THE OUTPUT FOR <think>, per llm.py's own warning. An arm that
returns fast but still carries <think> means the flag was silently ignored and
the speed came from somewhere else.

    uv run python scripts/diagnose_judge.py
    uv run python scripts/diagnose_judge.py --model qwen3:8b
"""
from __future__ import annotations

import argparse
import json
import time

W = 78
LINE = "=" * W

# Deliberately trivial. If this is slow, size is not the problem.
PROMPT = ('Context: "Model validation verifies that models perform as expected."\n'
          'Statement: "Validation must be repeated every eighteen months."\n'
          'Can the statement be attributed to the context? Answer yes or no.')


def show(arm: str, secs: float, text: str, err: str = "") -> bool:
    """Returns True if the arm looks healthy."""
    if err:
        print(f"  {arm:<34}{secs:>7.1f}s   FAILED: {err[:70]}")
        return False
    thinking = "<think>" in text.lower()
    flat = " ".join(text.split())
    tag = "THINKING PRESENT" if thinking else "no <think>"
    ok = (not thinking) and secs < 60
    print(f"  {arm:<34}{secs:>7.1f}s   {tag:<18}{'ok' if ok else '<-- '}")
    print(f"       raw: {flat[:150]!r}")
    return ok


def raw_http(model: str, think: bool | None, timeout: int) -> tuple[float, str, str]:
    """Same route the generator uses: Ollama's HTTP API, no langchain."""
    import requests
    body: dict = {"model": model, "prompt": PROMPT, "stream": False,
                  "options": {"temperature": 0}}
    if think is not None:
        body["think"] = think
    t0 = time.perf_counter()
    try:
        r = requests.post("http://localhost:11434/api/generate", json=body,
                          timeout=timeout)
        r.raise_for_status()
        d = r.json()
        # Ollama returns the reasoning separately on builds that know `think`.
        out = (d.get("response") or "") + (d.get("thinking") or "")
        return time.perf_counter() - t0, out, ""
    except Exception as exc:                                  # noqa: BLE001
        return time.perf_counter() - t0, "", f"{type(exc).__name__}: {exc}"


def via_langchain(model: str, reasoning: bool | None, timeout: int
                  ) -> tuple[float, str, str]:
    from langchain_ollama import ChatOllama
    kw: dict = {"model": model, "temperature": 0}
    if reasoning is not None:
        kw["reasoning"] = reasoning
    t0 = time.perf_counter()
    try:
        llm = ChatOllama(**kw)
    except TypeError as exc:
        return 0.0, "", (f"ChatOllama does not accept reasoning= on this "
                         f"langchain-ollama version ({exc})")
    try:
        return time.perf_counter() - t0, str(llm.invoke(PROMPT).content), ""
    except Exception as exc:                                  # noqa: BLE001
        return time.perf_counter() - t0, "", f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b")
    ap.add_argument("--timeout", type=int, default=90,
                    help="per call. Short ON PURPOSE - a 300s wall was already "
                         "paid six times to learn nothing.")
    ns = ap.parse_args()

    print(LINE)
    print(f"  JUDGE DIAGNOSIS — {ns.model!r}, trivial prompt, {ns.timeout}s cap")
    print(LINE)
    results = {}
    for arm, fn, arg in (
            ("A  raw HTTP, think unset",   raw_http,      None),
            ("B  raw HTTP, think=false",   raw_http,      False),
            ("C  ChatOllama, default",     via_langchain, None),
            ("D  ChatOllama, reasoning=F", via_langchain, False)):
        secs, text, err = fn(ns.model, arg, ns.timeout)
        results[arm[0]] = show(arm, secs, text, err)

    print("\n" + LINE)
    print("  READING")
    if results.get("A") is False and results.get("B") is True:
        print("  A slow / B fast -> THINKING IS THE CAUSE, confirmed.")
    if results.get("D"):
        print("  D healthy -> ChatOllama CAN turn it off. Use reasoning=False")
        print("               in the judge and re-run; nothing else needs changing.")
    elif results.get("B"):
        print("  B healthy but D not -> the model is fine and langchain cannot")
        print("               reach the flag. Options: (1) append '/no_think' to")
        print("               the prompt, which Qwen3 honours, or (2) wrap")
        print("               regrag.generation.llm in a RAGAS LLM adapter and")
        print("               skip ChatOllama entirely - it already sets think.")
    if not any(results.values()):
        print("  NOTHING returned healthy. Do not tune RAGAS - the model is not")
        print("  answering a three-sentence prompt. Check `ollama ps` for a")
        print("  stuck load, and whether this tag is pulled at all.")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
