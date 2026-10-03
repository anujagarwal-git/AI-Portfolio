"""IS FLASH ATTENTION ON, AND DOES IT HELP? One variable, measured twice.

WHAT THIS CAN AND CANNOT PROVE — read this before trusting a number.

  CANNOT: Ollama's HTTP API does not report whether flash attention is active.
  There is no endpoint to ask. So a script cannot READ the setting, and any
  script claiming to has guessed. The only two honest sources of truth are the
  SERVER LOG at model load, and a MEASUREMENT that changes.

  CAN: run a fixed prompt at two context sizes, several times, and record read
  and write speed. Do that once with the setting off and once with it on, and
  the difference is the answer. If nothing moves, the setting did nothing here
  - which is a real result, not a failed test.

WHY TWO CONTEXT SIZES. Flash attention changes how the attention matrix is read
and written. Whatever effect it has should GROW with context length. A single
size can miss it entirely, and a single size is also how you end up crediting
noise. Small and large together give the shape, not just a point.

WHY REPEATS AND A MEDIAN. One call on a busy laptop is noise. The first call
after a model load also pays the load cost, which is reported separately and
excluded here.

THE FALSIFYING CONDITION, WRITTEN BEFORE THE RUN:
    Flash attention helped ONLY IF median read tok/s at the LARGE size improves
    by more than 10% AND the small size does not get worse. Anything under 10%
    on a CPU with other things running is not distinguishable from noise.
    Write that down now; do not soften it after seeing the number.

USE — three commands, in this order
    uv run python scripts/check_flash.py --label off
    ... turn the setting on and RESTART OLLAMA (see below) ...
    uv run python scripts/check_flash.py --label on
    uv run python scripts/check_flash.py --compare off on

HOW TO TURN IT ON — WINDOWS. This is where it usually goes wrong.
    Ollama runs as a BACKGROUND APPLICATION. Setting a variable in your shell
    does NOT reach it. `set OLLAMA_FLASH_ATTENTION=1` in a terminal changes
    nothing at all, and the run will look like "flash attention does not help"
    when in fact it was never enabled.

    Do this instead:
      1. Quit Ollama completely from the system tray (right-click -> Quit).
      2. Set a USER environment variable:
           setx OLLAMA_FLASH_ATTENTION 1
         (or Settings -> Environment Variables -> New, under User variables)
      3. Start Ollama again from the Start menu.
      4. Confirm it took: see VERIFY below. Do not skip this.

VERIFY IT IS ACTUALLY ON — the log is the only direct evidence
    Ollama writes a server log. On Windows:
        %LOCALAPPDATA%\\Ollama\\server.log
    Load the model once (this script does that), then search the log for a line
    mentioning flash attention, printed by llama.cpp at model load. The wording
    varies between Ollama versions - look for `flash_attn` or `flash attention`
    and read what value follows it.
        powershell: Select-String -Path "$env:LOCALAPPDATA\\Ollama\\server.log" `
                      -Pattern "flash" | Select-Object -Last 20
    IF THE LINE IS ABSENT, DRAW NO CONCLUSION EITHER WAY. Some builds do not
    log it. In that case the measurement below is your only evidence, and you
    should say so rather than claiming the setting was on.

RELATED, AND PROBABLY MORE USEFUL AT YOUR CONTEXT SIZE
    OLLAMA_KV_CACHE_TYPE=q8_0 shrinks the key/value cache, and it REQUIRES
    flash attention to be on. At num_ctx 16,384 that memory saving may matter
    more than raw speed. Test it the same way, as a THIRD label, and only after
    this one - one variable at a time.
        uv run python scripts/check_flash.py --label on_q8

I have not verified any of this against your Ollama build. The env var names
and the log path are what I believe to be current; confirm them against the
Ollama documentation for your installed version before concluding anything.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import uuid

import requests

from regrag import config
from regrag.generation import llm

OUT = pathlib.Path("evaluation") / "flash"
SENT = ("The bank shall maintain documentation sufficient to permit an "
        "independent party to evaluate the model and its limitations. ")
# A LADDER, not two points. Two points cannot show WHERE reading stops - and on
# 2026-09-02 the large point reported 8,194 tokens read in 2.5s on a box that
# reads 820 tokens in 11s. Something discarded the prompt silently. A ladder
# shows the cliff: wall time should rise roughly linearly with tokens, and the
# rung where it stops rising is the real ceiling, whatever config says.
SIZES = {"1k": 55, "2k": 110, "4k": 220, "8k": 440, "12k": 660, "16k": 880}
REPEATS = 2          # a ladder is 6 rungs; 2 timed runs each is enough
IMPROVE_MIN = 0.10                        # the falsifying threshold, fixed here


def build(n_sent: int) -> str:
    """A prompt with a UNIQUE PREFIX, so Ollama's prompt cache cannot serve it.

    THE BUG THIS FIXES (2026-09-02, caught by Anuj on the first run). Sending
    the identical prompt three times produced 12,304 / 12,707 / 11,717 tok/s on
    a CPU that reads at ~160. Those were CACHE HITS: `prompt_eval_duration`
    reported a lookup, not a read. Only the warm-up was a real measurement.

    The nonce goes FIRST because the cache is a PREFIX cache. A nonce at the end
    still matches the cached prefix and most of the prompt is skipped - the
    number would come down but stay fake, which is worse than obviously fake.
    """
    return f"Reference {uuid.uuid4()}. " + SENT * n_sent


def one(model: str, text: str) -> dict:
    """One call, num_predict=1, so the timing isolates PROMPT READING.

    TWO RATES ARE REPORTED, AND THEY MUST AGREE.
      api   = prompt_eval_count / prompt_eval_duration, as Ollama reports it.
      wall  = the same token count over WALL-CLOCK time, minus load and minus
              the one token written.
    When Ollama serves a prompt from its cache it still reports the full
    prompt_eval_count but a near-zero duration, so `api` explodes to tens of
    thousands of tokens per second while `wall` stays honest. A large gap
    between the two IS the cache-hit detector. Trust `wall`.
    """
    out = llm.chat("You are a helpful assistant.", text,
                   model=model, max_tokens=1, think=False)
    read = out.prompt_seconds
    wall = max(out.seconds - out.load_seconds - out.eval_seconds, 1e-6)
    return {"prompt_tokens": out.prompt_tokens,
            "read_s": round(read, 3),
            "api_tok_s": round(out.prompt_tokens / read, 1) if read else None,
            "wall_s": round(wall, 3),
            "read_tok_s": round(out.prompt_tokens / wall, 1),
            "load_s": round(out.load_seconds, 2),
            "total_s": round(out.seconds, 2)}


def measure(model: str) -> dict:
    res: dict = {"model": model, "num_ctx": config.GEN_NUM_CTX, "sizes": {}}
    for name, n in SIZES.items():
        print(f"\n  {name}: {n} sentences, ~{len(build(n)):,} chars")
        runs = []
        last = None
        for i in range(REPEATS + 1):          # first call absorbs model load
            # BUILD INSIDE THE LOOP. Built once outside it (the bug on
            # 2026-09-02), every repeat sent identical text and was served from
            # the prompt cache at ~12,000 tok/s on a CPU that reads at ~85.
            r = one(model, build(n))
            tag = "warmup" if i == 0 else f"run {i}"
            # Every run now has a unique prefix, so the warm-up is a
            # genuine read as well - it is excluded only because it
            # also pays the model LOAD cost.
            gap = (r["api_tok_s"] or 0) / max(r["read_tok_s"], 1e-6)
            print(f"    {tag:<7} {r['prompt_tokens']:>6} tok  "
                  f"wall {r['wall_s']:>6.2f}s = {r['read_tok_s']:>7.1f} tok/s"
                  f"   (api says {r['api_tok_s'] or 0:>8.1f})"
                  + ("  <- CACHE HIT" if gap > 3 else ""))
            last = r
            if gap > 3:
                print("      !! api and wall disagree by more than 3x — the "
                      "prompt was cached, not read. Discarded.")
                continue
            if i:
                runs.append(r)
        rates = [r["read_tok_s"] for r in runs if r["read_tok_s"]]
        toks = [r["prompt_tokens"] for r in runs]
        res["sizes"][name] = {
            "runs": runs,
            "median_read_tok_s": round(statistics.median(rates), 1) if rates else None,
            "prompt_tokens": toks[0] if toks else None,
        }
        # num_ctx sanity: a prompt that should exceed 4,096 must report more.
        est = len(build(n)) // 4
        seen = toks or ([last["prompt_tokens"]] if last else [])
        if seen and seen[0] < est * 0.6:
            toks = toks or seen
        if seen and seen[0] < est * 0.6:
            print(f"    !! {toks[0]:,} tokens reported for ~{est:,} expected. "
                  f"Ollama is TRUNCATING. Every rate here is measured on a "
                  f"shorter prompt than the one sent, and num_ctx is not doing "
                  f"what config says.")
        if name == "large" and toks and toks[0] <= 4200:
            print(f"    !! prompt_eval_count is {toks[0]} — Ollama may be "
                  f"truncating at its 4,096 default. num_ctx is NOT taking "
                  f"effect, and every number here is measured on a short prompt.")
    return res


def compare(a: str, b: str) -> int:
    fa, fb = OUT / f"{a}.json", OUT / f"{b}.json"
    for f in (fa, fb):
        if not f.exists():
            sys.exit(f"missing {f} — run: check_flash.py --label {f.stem}")
    da, db = json.loads(fa.read_text()), json.loads(fb.read_text())

    print(f"\n{'=' * 78}\n{a}  vs  {b}\n{'=' * 78}")
    print(f"  {'size':<8}{'prompt tok':>12}{a[:10]:>12}{b[:10]:>12}{'change':>11}")
    verdict = {}
    for name in SIZES:
        ra = da["sizes"].get(name, {}).get("median_read_tok_s")
        rb = db["sizes"].get(name, {}).get("median_read_tok_s")
        tok = da["sizes"].get(name, {}).get("prompt_tokens")
        if not (ra and rb):
            print(f"  {name:<8}{'-':>12}{'-':>12}{'-':>12}{'no data':>11}")
            continue
        pct = (rb - ra) / ra
        verdict[name] = pct
        print(f"  {name:<8}{tok:>12,}{ra:>12.1f}{rb:>12.1f}{pct:>10.1%}")

    print(f"\n  RULE FIXED BEFORE THE RUN: helped only if LARGE improves by more")
    print(f"  than {IMPROVE_MIN:.0%} AND small does not get worse.")
    lg, sm = verdict.get("large"), verdict.get("small", 0)
    if lg is None:
        print("\n  VERDICT: cannot say — no large-size data.")
    elif lg > IMPROVE_MIN and sm >= 0:
        print(f"\n  VERDICT: HELPED. Large {lg:+.1%}, small {sm:+.1%}.")
    elif lg > IMPROVE_MIN:
        print(f"\n  VERDICT: MIXED. Large {lg:+.1%} but small {sm:+.1%}. "
              f"Read both before keeping it.")
    else:
        print(f"\n  VERDICT: NO MEASURABLE BENEFIT. Large {lg:+.1%}, "
              f"below the {IMPROVE_MIN:.0%} bar. That is a result — record it "
              f"and stop paying attention to this setting.")
    print("\n  This compares two RUNS, not two settings. It is only a valid")
    print("  attribution if the ONLY thing you changed between them was the")
    print("  setting — same machine, nothing else heavy running, same model.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", help="name this run, e.g. off / on / on_q8")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--model", default=config.GEN_MODEL)
    ns = ap.parse_args()

    if ns.compare:
        return compare(*ns.compare)
    if not ns.label:
        ap.error("give --label NAME or --compare A B")

    try:
        v = requests.get(f"{config.OLLAMA_HOST}/api/version", timeout=10).json()
    except Exception as exc:                                       # noqa: BLE001
        sys.exit(f"Ollama unreachable at {config.OLLAMA_HOST}: {exc}")

    print(f"{'=' * 78}\nRUN {ns.label!r}   ollama {v.get('version', '?')}   "
          f"model {ns.model}   num_ctx {config.GEN_NUM_CTX:,}\n{'=' * 78}")
    print("  The API cannot report whether flash attention is on. This run")
    print("  records SPEED only. Check the server log for the setting itself —")
    print("  see this file's docstring.")

    res = measure(ns.model)
    res["label"] = ns.label
    res["ollama_version"] = v.get("version")

    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{ns.label}.json"
    f.write_text(json.dumps(res, indent=1), encoding="utf-8")

    print(f"\n  median read speed (wall clock)")
    for name, d in res["sizes"].items():
        if d.get("median_read_tok_s") is None:
            print(f"    {name:<8}  no usable run — every call was cached. "
                  f"Nothing measured.")
            continue
        print(f"    {name:<8}{d['prompt_tokens']:>8,} tok   "
              f"{d['median_read_tok_s']} tok/s")
    print(f"\n  saved -> {f}")
    print(f"  next:  check_flash.py --compare off {ns.label}"
          if ns.label != "off" else
          "  next:  enable the setting, restart Ollama, then --label on")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
