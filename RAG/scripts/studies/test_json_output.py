"""APPROACH B on the five questions where v2 produced NO citation at all.

WHY THESE FIVE
    They are the failures, and only the failures. Running on all ten would mix
    questions v2 already handled with the ones it did not, and the number would
    not answer the question being asked: does structured output rescue the cases
    prose formatting lost?

    That also means this is NOT a fair overall comparison - it is a targeted
    probe of the worst cases. If it works here, the fair test on all ten comes
    next.

WHAT IS CHECKED - all mechanical
    parses      Ollama constrains decoding to JSON, so this should always pass.
                If it fails, the constraint is not working on this build.
    shape       is there a "claims" list of {text, source} objects?
    labels      does every "source" name a label that actually exists?
    grounded    do the claim's words appear in the passage it cites?

    uv run python scripts/test_json_output.py
"""
from __future__ import annotations

import json
import pathlib
import re
import textwrap
from datetime import datetime, timezone

from regrag import config
from regrag.evaluation.refusal_set import true_positives
from regrag.generation import llm, render
from regrag.retrieval.search import retrieve

# The five v2 scored 0 labels on (evaluation/citation_run_20260829T175126Z).
FAILED = [3, 4, 5, 7, 9]          # 0-indexed into true_positives()
STOP = set("the a an of to in for and or is are be as that this which with on by must "
           "should may not from at it its their under section".split())
W = 96


def grounded(text: str, passage: str) -> float:
    ws = [w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in STOP]
    return sum(w in passage.lower() for w in ws) / len(ws) if ws else 0.0


def main() -> int:
    problem = llm.why_unavailable()
    if problem:
        raise SystemExit("GENERATION UNAVAILABLE: " + problem)
    sysmsg = pathlib.Path("prompts/answer_v3_json.md").read_text(encoding="utf-8")
    qs = true_positives()
    tot = {"parsed": 0, "claims": 0, "valid": 0, "invalid": 0, "weak": 0}
    # SAVE THE RAW OUTPUT. Every other study tonight wrote its answers to disk
    # and this one did not, so its first run could not be re-read or
    # re-classified. A result that exists only in a terminal is not a result.
    saved: list[dict] = []
    weak: list[tuple] = []

    for n, idx in enumerate(FAILED, 1):
        c = qs[idx]
        r = retrieve(c.text)
        ctx, labels = render.render(r)
        ptext = {f"S{j}": p.text for j, p in enumerate(r.passages, 1)}
        out = llm.chat(sysmsg, render.user_message(c.text, ctx), fmt="json")
        saved.append({"q": c.text, "raw": out.text, "labels": labels,
                      "n_passages": len(r.passages), "chars": len(ctx),
                      "seconds": round(out.seconds, 1)})

        print("=" * W)
        print(f"[{n}/5] {c.text}")
        print(f"      {len(r.passages)} passages, {len(ctx):,} chars, {out.seconds:.0f}s")
        print("=" * W)
        try:
            data = json.loads(out.text)
            tot["parsed"] += 1
        except Exception as e:
            print(f"  !! DID NOT PARSE: {e}\n  raw: {out.text[:300]}\n")
            continue
        claims = data.get("claims")
        if not isinstance(claims, list):
            print(f"  !! WRONG SHAPE - no 'claims' list. keys: {list(data)}\n")
            continue
        if not claims:
            print("  (empty claims list - the model declined to answer)\n")
            continue
        # --- diagnostics: OURS, not the reader's -------------------------
        for cl in claims:
            src = str(cl.get("source", ""))
            tot["claims"] += 1
            if src not in labels:
                tot["invalid"] += 1
                continue
            tot["valid"] += 1
            if grounded(str(cl.get("text", "")), ptext.get(src, "")) < 0.45:
                tot["weak"] += 1
                weak.append((c.text, str(cl.get("text", ""))[:70], labels[src]))

        # --- the answer as a READER sees it ------------------------------
        body, sources = render.format_answer(claims, labels)
        print(render.as_text(body, sources))
        print()

    print("=" * W)
    print(f"  parsed as JSON      {tot['parsed']}/5")
    print(f"  claims produced     {tot['claims']}")
    print(f"  valid labels        {tot['valid']}")
    print(f"  INVALID labels      {tot['invalid']}   <- must be 0")
    print(f"  weakly grounded     {tot['weak']}   <- read these before trusting them")
    print("=" * W)
    print("  Compare: prompt v2 produced ZERO citations on these same five.")
    if weak:
        print("\n  WEAK MATCHES — reviewer view only, never shown in the answer:")
        for q_, claim, cite in weak:
            print(f"    {q_[:40]}\n      {claim}...\n      cited: {cite}")
    f = pathlib.Path("evaluation") / (
        f"json_run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    f.write_text(json.dumps({"model": config.GEN_MODEL, "prompt": "v3_json",
                             "totals": tot, "rows": saved}, indent=1),
                 encoding="utf-8")
    print(f"  raw output -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
