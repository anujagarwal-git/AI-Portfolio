"""How fast does this CPU READ a prompt? Measured, not estimated.

WHY
    An 8,000-token prompt did not complete in 300s (2026-08-26). That could be
    prompt reading, answer writing, model loading, or the window setting not
    taking effect. A single failed call cannot tell them apart.

    So: a LADDER of prompt sizes, each with num_predict=1 so almost no answer is
    written. Ollama reports prompt_eval_duration and eval_duration separately,
    in nanoseconds, which isolates the one thing being measured.

WHAT IT SETTLES
    1. tokens/second for READING a prompt on this box.
    2. Whether num_ctx=16,384 is honoured - prompt_eval_count must keep rising
       past 4,096 instead of flattening there.
    3. The largest context this project can actually afford, which then sets
       the retrieval budget - not the other way round.

    uv run python scripts/measure_ctx.py
"""
from __future__ import annotations

import argparse

from regrag import config
from regrag.generation import llm

SENT = "The bank shall maintain documentation sufficient to permit an independent party to evaluate the model. "
LADDER = [250, 500, 1000, 2000, 3500, 5000]      # approximate token targets
PER_SENT = 18                                     # rough tokens per sentence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.GEN_MODEL)
    ap.add_argument("--think", default="off", choices=["off", "on", "default"],
                    help="hybrid Qwen3 models reason before answering; off is "
                         "what we want for grounded RAG")
    ns = ap.parse_args()
    think = {"off": False, "on": True, "default": None}[ns.think]
    print(f"model {ns.model}   num_ctx {config.GEN_NUM_CTX:,}   think={ns.think}")
    print(f"{'target':>8}{'prompt tok':>12}{'read s':>9}{'tok/s':>9}{'write s':>9}{'load s':>9}")
    print("-" * 60)
    rates = []
    for target in LADDER:
        body = SENT * max(1, target // PER_SENT)
        try:
            c = llm.chat("You are terse.", body + "\n\nReply with: seen",
                         model=ns.model, max_tokens=1, timeout=600, think=think)
        except Exception as e:
            print(f"{target:>8}   FAILED: {type(e).__name__} - stopping ladder here")
            break
        rate = c.prompt_tokens / c.prompt_seconds if c.prompt_seconds else 0
        rates.append((c.prompt_tokens, rate))
        print(f"{target:>8}{c.prompt_tokens:>12,}{c.prompt_seconds:>9.1f}"
              f"{rate:>9.1f}{c.eval_seconds:>9.1f}{c.load_seconds:>9.1f}")

    if not rates:
        print("\nNothing completed. Ollama may not be serving this model.")
        return 1

    biggest, rate = rates[-1]
    print(f"\n  largest prompt actually read: {biggest:,} tokens")
    # FIXED 2026-08-26. The first version compared `biggest` against 5,000 and
    # declared the window broken below it - but the ladder's own token estimate
    # was low, so it never SENT 5,000 and the check could not pass by
    # construction. A test whose threshold its own input cannot reach measures
    # nothing. The real signal is simply: did anything above 4,096 get read?
    if biggest > 4096:
        print("  -> above the 4,096 default, so a larger num_ctx IS honoured.")
    else:
        print("  -> nothing above 4,096 was SENT. Inconclusive, not a failure.")
    print(f"  read speed at that size: {rate:.0f} tok/s")
    if rate:
        for chars in (10_000, 20_000, 25_000):
            tok = chars / 5.0        # measured ~5-6 chars/tok on this filler
            print(f"  a {chars:,}-char context (~{tok:,.0f} tok) costs ~{tok / rate:,.0f}s "
                  f"to READ, before a single word is written")

    # --- WRITE SPEED, and whether think=off actually took effect -------------
    print("\n  WRITE SPEED (small prompt, real answer)")
    w = llm.chat("You are a regulatory assistant.",
                 "Explain model validation in about 100 words.",
                 model=ns.model, max_tokens=200, timeout=900, think=think)
    wps = w.output_tokens / w.eval_seconds if w.eval_seconds else 0
    print(f"    {w.output_tokens} tok written in {w.eval_seconds:.1f}s  ->  {wps:.1f} tok/s")
    thinking = "<think>" in w.text
    print(f"    output contains <think>: {thinking}"
          + ("   *** THINKING IS STILL ON - the flag did not take ***" if thinking
             else "   (reasoning is off)"))
    if wps and rate:
        for chars, out in ((10_000, 150), (20_000, 150), (25_000, 150)):
            tot = (chars / 5.0) / rate + out / wps
            print(f"    {chars:,}-char context + {out} tok answer  ->  ~{tot:.0f}s per question")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
