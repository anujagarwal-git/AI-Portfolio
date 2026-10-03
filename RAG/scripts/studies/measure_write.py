"""Two numbers the ladder could not give: WRITE speed, and the big-prompt cost.

The ladder set num_predict=1, so eval_duration was ~0 and generation speed was
never measured. A generation costs READ + WRITE, and we only have READ.

It also leaves the 300s timeout unexplained. So this sends a deliberately large
prompt ONCE, with a 900s ceiling and separate timings, so the cost can be
attributed instead of guessed.

    uv run python scripts/measure_write.py
"""
from __future__ import annotations

from regrag import config
from regrag.generation import llm

SENT = ("The bank shall maintain documentation sufficient to permit an "
        "independent party to evaluate the model. ")


def show(tag, c):
    print(f"  {tag}")
    print(f"    prompt {c.prompt_tokens:,} tok  read {c.prompt_seconds:.1f}s  "
          f"({c.prompt_tokens / c.prompt_seconds if c.prompt_seconds else 0:.0f} tok/s)")
    print(f"    output {c.output_tokens:,} tok  write {c.eval_seconds:.1f}s  "
          f"({c.output_tokens / c.eval_seconds if c.eval_seconds else 0:.1f} tok/s)")
    print(f"    load {c.load_seconds:.1f}s   wall {c.seconds:.1f}s\n")


def main() -> int:
    print(f"model {config.GEN_MODEL}   num_ctx {config.GEN_NUM_CTX:,}\n")

    print("1. WRITE SPEED - small prompt, real answer")
    show("small prompt, ~120 tok answer",
         llm.chat("You are a regulatory assistant.",
                  "Explain model validation in about 100 words.",
                  max_tokens=120, timeout=900))

    print("2. THE BIG PROMPT THAT TIMED OUT - one token of output, 900s ceiling")
    big = SENT * 700
    show("~11k token prompt, 1 tok answer",
         llm.chat("You are terse.", big + "\n\nReply with: seen",
                  max_tokens=1, timeout=900))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
