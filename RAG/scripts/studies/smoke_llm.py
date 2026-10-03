"""Look at what the model ACTUALLY returns before building anything on it.

Four questions, answered by observation rather than by assumption:

  1. IS IT THERE?          model name present in /api/tags, or fail now rather
                           than at question 19 of 31.
  2. WHAT COMES BACK?      the raw response body, printed. qwen3 hybrid models
                           emit <think>...</think> blocks; the -instruct-2507
                           line is documented as non-reasoning. Printing settles
                           it in one call instead of designing a parser around a
                           guess.
  3. IS IT DETERMINISTIC?  same request twice, compared byte for byte. If two
                           identical runs disagree, every number the refusal
                           experiment produces is noise and we stop here.
  4. IS num_ctx REAL?      send a deliberately long prompt and read
                           prompt_eval_count. If the server is honouring 16,384
                           the count keeps rising past 4,096. If it caps near
                           4,096, evidence is being dropped silently and the
                           whole of Stage 7 is built on sand.

    uv run python scripts/smoke_llm.py
"""
from __future__ import annotations

import json

from regrag import config
from regrag.generation import llm

LINE = "=" * 96


def main() -> int:
    print(LINE)
    print(f"model     {config.GEN_MODEL}")
    print(f"endpoint  {config.OLLAMA_CHAT_URL}")
    print(f"num_ctx   {config.GEN_NUM_CTX:,}   temperature {config.TEMPERATURE}   "
          f"seed {config.GEN_SEED}")
    print(LINE)

    # 1 -------------------------------------------------------------------
    ok = llm.available()
    print(f"\n1. MODEL PRESENT: {ok}")
    if not ok:
        print("   Not in /api/tags. Check `ollama list` and the name in config.GEN_MODEL.")
        return 1

    # 2 -------------------------------------------------------------------
    print("\n2. RAW RESPONSE")
    c = llm.chat("You are terse.", "Reply with exactly the word: ready")
    shown = {k: v for k, v in c.raw.items() if k != "message"}
    print("   body keys minus message:")
    print("   " + json.dumps(shown, indent=2)[:900].replace("\n", "\n   "))
    print(f"\n   message.content = {c.text!r}")
    thinking = "<think>" in c.text or "</think>" in c.text
    print(f"   contains <think> tags: {thinking}"
          + ("   -> the parser MUST strip them" if thinking else "   -> nothing to strip"))
    print(f"   {c.prompt_tokens} prompt tok, {c.output_tokens} out tok, {c.seconds:.1f}s")

    # 3 -------------------------------------------------------------------
    print("\n3. DETERMINISM (same request twice)")
    a = llm.chat("You are a regulatory assistant.",
                 "In one sentence, what is model risk?")
    b = llm.chat("You are a regulatory assistant.",
                 "In one sentence, what is model risk?")
    same = a.text == b.text
    print(f"   identical byte for byte: {same}")
    if not same:
        print(f"   run 1: {a.text[:150]!r}")
        print(f"   run 2: {b.text[:150]!r}")
        print("   *** STOP. Fix this before running the experiment. ***")

    # 4 -------------------------------------------------------------------
    print("\n4. IS num_ctx HONOURED?")
    filler = ("The bank shall maintain documentation sufficient to permit an "
              "independent party to evaluate the model. ") * 700          # ~8k tokens
    d = llm.chat("You are terse.", filler + "\n\nReply with the word: seen")
    print(f"   sent a deliberately long prompt")
    print(f"   prompt_eval_count = {d.prompt_tokens:,}")
    if d.prompt_tokens > 5000:
        print(f"   -> above 4,096, so the 16,384 window is real.")
    else:
        print(f"   -> AT OR BELOW ~4,096. The window setting is NOT taking effect.")
        print(f"      Evidence would be dropped silently. Do not proceed.")
    print(f"   maybe_truncated flag: {d.maybe_truncated}")

    print(f"\n{LINE}\nAll four must pass before the experiment runs.\n{LINE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
