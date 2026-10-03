"""Owns building the LLM-as-judge for RAGAS — with thinking OFF, and PROVEN off.

WHY THIS MODULE EXISTS
    qwen3 is a HYBRID model: it writes a <think> block before answering. The
    generator already handles this - `generation/llm.py` sends `think` at the
    top level of /api/chat and `config.GEN_THINK` is False, verified 2026-08-26.
    The JUDGE went through langchain's ChatOllama, which was never told, and on
    2026-09-08 every RAGAS job timed out at EXACTLY 300.01s - six of six, on
    contexts three sentences long. Uniform timeouts on trivial input is not
    slowness, it is a call that never returns.

    So there is now ONE place that knows how to build a judge, and it refuses
    to hand back one whose thinking is still on.

THE WARNING THIS MODULE ENFORCES, taken from generation/llm.py verbatim:
    "VERIFY IT WORKED: the caller must check the output for <think>, because an
     Ollama build that does not know this field will ignore it silently and
     thinking stays on."

    A flag that is silently ignored is worse than no flag - it produces a run
    that looks configured and behaves as though it is not. `verify()` sends one
    trivial prompt and fails loudly BEFORE a multi-hour evaluation, rather than
    after it.
"""
from __future__ import annotations

import time


class ThinkingStillOn(RuntimeError):
    """The judge emitted <think>, or took absurdly long on a trivial prompt."""


PROBE = ("Context: \"Model validation verifies that models perform as "
         "expected.\"\nStatement: \"Validation must be repeated every eighteen "
         "months.\"\nCan the statement be attributed to the context? "
         "Answer yes or no.")


# WHAT EACH CONFIGURATION HAS ACTUALLY SCORED, so the choice is evidence.
#
#   judge                 think  faithfulness        context_recall
#   qwen2.5:3b-instruct   n/a    PASS 1.0 / 0.0      FAIL  R3 = 1.000
#   qwen3:8b              ON     timeout, all NaN    WORKED gs-02 = 0.00
#   qwen3:4b              OFF    FAIL R1 = 0.000     FAIL  R3 = 1.000
#   qwen3:4b              ON     timed out on Q1     timed out on Q1
#   qwen2.5:7b-instruct   n/a    PASS 1.000/0.333    PASS R1 1.0, R3 0.0  <- USE THIS
#
# qwen2.5:7b-instruct PASSED ALL THREE CONDITIONS (2026-09-08), and every
# cell is right, not just the separations: R2's 0.333 is 1 of 3 claims
# supported - it counted the two fabricated figures rather than flagging
# the row binary. That is the strongest single signal in this table.
#
# SO THE 'RECALL NEEDS REASONING' HYPOTHESIS IS DISPROVED, not sidestepped.
# A model with no reasoning phase at all does attribution correctly at 7b.
# What recall needed was CAPACITY. Every earlier failure - 3b, 4b - was a
# size problem wearing a thinking problem's clothes, and the mentor read it
# the wrong way round twice.
#
# REASONING JUDGES ARE OFF THE TABLE (Anuj, 2026-09-08). Thinking ON times out
# even at 4b; thinking OFF breaks both metrics. So the judge must be a model
# that is capable WITHOUT a reasoning phase - which is what qwen2.5 is. The 3b
# of that family already passed faithfulness cleanly, so the open question is
# only whether 7b has the capacity for attribution that 3b lacked.
#
# NOTE WHAT IS STILL UNISOLATED: those rows differ in SIZE and in THINKING at
# once. "Recall needs reasoning" was a hypothesis and it is now MOOT rather
# than disproved - the reasoning path was abandoned on cost, not on evidence.
# If qwen2.5:7b passes, that hypothesis was simply wrong, and the honest
# reading is that recall needed CAPACITY, not a reasoning phase.


# *** THE JUDGE'S CONTEXT WINDOW ***
# `ollama ps` showed CONTEXT 4096 while golden-set contexts reach ~20,000
# characters (~5,000 tokens) BEFORE the metric's own prompt scaffolding. A
# judge whose window silently truncates the context it is grading would score
# recall against a fragment and report a LOW number for a retrieval that
# actually worked - a false negative that looks exactly like a real miss.
# 32,768 is qwen2.5:7b's native window. The generator runs 16,384 by its own
# config; the judge needs more because it reads the context AND the answer AND
# the reference AND the rubric in one prompt.
JUDGE_NUM_CTX = 32_768


def build(model: str, *, think: bool = False, fmt: str | None = "json",
          temperature: float = 0.0, num_ctx: int = JUDGE_NUM_CTX):
    """A ChatOllama with reasoning disabled, or the closest thing available.

    `reasoning=False` is langchain-ollama's route to Ollama's `think` field.
    It is passed positively rather than hoped for: if this langchain version
    does not accept it, that is reported, not swallowed - because the fallback
    judge is one whose thinking is ON, and the caller must know.
    """
    from langchain_ollama import ChatOllama

    kw: dict = {"model": model, "temperature": temperature, "num_ctx": num_ctx}
    if fmt:
        kw["format"] = fmt
    if think:
        # Reasoning left ON. Expect minutes per call, not seconds - budget for
        # it rather than discovering it at a timeout.
        return ChatOllama(**kw), True
    try:
        return ChatOllama(reasoning=False, **kw), True
    except TypeError:
        return ChatOllama(**kw), False


def verify(llm, *, budget_s: float = 60.0, expect_think: bool = False) -> None:
    """One trivial call. Raise if thinking is on or the model is not answering.

    Two failure shapes, both fatal and both distinguished in the message:
      * <think> in the output  -> the flag was ignored; scores will be NaN
      * slow with no <think>   -> something else is wrong; do not blame RAGAS
    """
    t0 = time.perf_counter()
    out = str(llm.invoke(PROBE).content)
    secs = time.perf_counter() - t0
    if expect_think:
        print(f"  judge probe: {secs:.1f}s, reasoning ON by request "
              f"({'<think> seen' if '<think>' in out.lower() else 'no <think> tag'})")
        return
    if "<think>" in out.lower():
        raise ThinkingStillOn(
            f"the judge emitted <think> after {secs:.0f}s. The reasoning flag "
            f"was IGNORED, not applied. Every metric will time out. See "
            f"generation/llm.py - this is the failure it warns about.")
    if secs > budget_s:
        raise ThinkingStillOn(
            f"the judge took {secs:.0f}s on a three-sentence prompt (budget "
            f"{budget_s:.0f}s) and emitted no <think>. Thinking is not the "
            f"cause here - check `ollama ps` for a stuck load and whether this "
            f"tag is pulled, before touching RAGAS settings.")
    print(f"  judge verified: {secs:.1f}s on the probe, no <think> in output")


def make(model: str, *, think: bool = False, fmt: str | None = "json",
         budget_s: float = 60.0, num_ctx: int = JUDGE_NUM_CTX):
    """Build, verify, and wrap for RAGAS. The only entry point callers need."""
    from ragas.llms import LangchainLLMWrapper

    print(f"  judge {model!r}  num_ctx {num_ctx:,}")
    llm, flag_ok = build(model, think=think, fmt=fmt, num_ctx=num_ctx)
    if not flag_ok:
        print("  WARNING: this langchain-ollama does not accept reasoning=. "
              "Thinking is ON. The probe below will almost certainly fail.")
    try:
        verify(llm, budget_s=budget_s if not think else 1e9, expect_think=think)
    except ThinkingStillOn:
        raise
    except Exception as exc:                                  # noqa: BLE001
        # A model with NO reasoning mode (the qwen2.5 family) can REJECT the
        # think field rather than ignore it. Retry once without it, because
        # "the judge refused a flag it has no use for" must not read as "the
        # judge is broken" - two different findings, and only one is real.
        print(f"  probe failed with reasoning disabled ({type(exc).__name__}); "
              f"retrying without the flag - this model may have no think mode")
        from langchain_ollama import ChatOllama
        kw: dict = {"model": model, "temperature": 0.0, "num_ctx": num_ctx}
        if fmt:
            kw["format"] = fmt
        llm = ChatOllama(**kw)
        verify(llm, budget_s=budget_s, expect_think=False)
    return LangchainLLMWrapper(llm)
