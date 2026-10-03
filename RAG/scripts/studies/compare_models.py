"""1.7B vs 4B ON THE FIVE ANSWERS THAT WENT WRONG. Quality and latency, one variable.

WHY THESE FIVE
    Each one is a graded defect, verified against the parsed PDFs on
    2026-09-02. They are not a sample of anything — they are the specific
    failures a bigger model would have to fix to be worth its latency:

      1 WRONG LOCATOR        five claims, all cited CRE53.56, none from it
      2 RE-LABELLED SCOPE    CAP30.18-30.31 is a TLAC range, presented as the
                             general CET1 cross-reference
      3 QUESTION AS PREMISE  "not implemented" appears NOWHERE in Basel CRE;
                             the model echoed the question back with citations
      4 LIST BOUNDARY        retail sub-classes and QRRE folded into
                             specialised lending, which they are not
      5 INVERTED MECHANISM   "conserve earnings AS dividends and buybacks" —
                             the exact distributions the buffer RESTRICTS

    THE FIRST IS A CODE DEFECT, NOT A MODEL ONE (parent-level locators), so a
    bigger model may not fix it and should not be credited if it does. The
    other four are reading failures, which is what this test is actually about.

THE ONE VARIABLE
    RETRIEVAL IS COMPUTED ONCE PER QUESTION AND HANDED TO BOTH MODELS.
    `answer(question, retrieval=r)` exists precisely for this. Without it the
    two arms would differ in retrieval AND generation, and any difference would
    be unattributable — the 2026-08-23 lesson, applied on purpose.

    The generator is switched by setting `config.GEN_MODEL`, which `llm.chat`
    reads at call time. That is a test-script liberty, not a pattern to copy;
    it is restored in a finally block.

WHAT IS MEASURED
    QUALITY  is not scored here. A rubric written by the same party that wrote
             the questions would be theatre. Both answers are printed side by
             side against the recorded defect; YOU read them and say whether
             the defect survived.
    LATENCY  is measured: read tok/s, write tok/s, seconds per answer. On the
             same fixed retrieval, so prompt sizes match to the token.

WHAT IS ALREADY KNOWN, AND MAY REPEAT
    2026-08-29: qwen3:4b-2507 read at 62 tok/s against 1.7b's 162, and an
    ~11k-token prompt DID NOT FINISH IN 300s. It was rejected on latency, never
    on accuracy — which is the gap this test closes. If a call times out, that
    IS the result: the script records it and moves on rather than dying.

    uv run python scripts/compare_models.py --save
    uv run python scripts/compare_models.py --models qwen3:1.7b,qwen3:4b-2507
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import textwrap
from datetime import datetime, timezone

import requests

from regrag import config
from regrag.generation import llm
from regrag.generation.answer import answer
from regrag.retrieval.search import retrieve

W = 100
LINE = "=" * W
THIN = "-" * W

CASES = [
    ("BASEL guidelines: what must a bank do before putting a model in production?",
     "WRONG LOCATOR — all claims cited CRE53.56; they came from 53.50/53.51/53.55. "
     "This one is a CODE defect (parent-level locators), so do not credit the model."),
    ("Which paragraphs does the CAP standard cross-refer to when defining "
     "regulatory adjustments to CET1?",
     "RE-LABELLED SCOPE — CAP30.18-30.31 is the TLAC/own-holdings range, quoted "
     "from CAP30.6 and CAP99.8, presented as the general CET1 cross-reference."),
    ("Under CRE, in which cases does the treatment differ for banks in "
     "jurisdictions that have not implemented the framework?",
     "QUESTION AS PREMISE — 'not implemented' appears nowhere in Basel CRE. The "
     "model restated the question and attached real CRE54/CRE20 citations. The "
     "honest answer is that CRE does not address this."),
    ("Which asset classes fall under specialised lending?",
     "LIST BOUNDARY — the five SL sub-classes are right, but the retail three "
     "and corporate purchased receivables/QRRE are SEPARATE asset classes, not "
     "part of specialised lending. Also cited CRE30.7 where the names are 30.8."),
    ("How do Basel CAP and Basel RBC treat minimum capital ratios?",
     "INVERTED MECHANISM — said banks must 'conserve earnings AS dividends, "
     "share buybacks or discretionary bonus payments'. Those are the "
     "distributions the conservation buffer RESTRICTS. Also missed CAP's actual "
     "minimums (CET1 4.5 / Tier 1 6 / Total 8)."),
]


def one(model: str, question: str, retrieval) -> dict:
    """Generate with `model` on a FIXED retrieval. Never raises."""
    before = config.GEN_MODEL
    config.GEN_MODEL = model
    try:
        a = answer(question, retrieval=retrieval)
        out = {
            "model": model, "seconds": round(a.seconds, 1),
            "refused": a.refusal_reason if a.refused else None,
            "text": a.text, "sources": a.sources,
            "n_claims": len(a.claims), "n_invalid": len(a.invalid),
            "n_weak": len(a.weak), "config_hash": a.config_hash,
            "claims": [{"text": c.text, "citation": c.citation,
                        "overlap": c.overlap} for c in a.claims],
        }
    except Exception as exc:                                       # noqa: BLE001
        # A timeout is a RESULT, not a crash. 4B failed to finish an ~11k-token
        # prompt in 300s on 2026-08-29; recording that is the point.
        out = {"model": model, "error": f"{type(exc).__name__}: {exc}",
               "seconds": None, "claims": []}
    finally:
        config.GEN_MODEL = before
    return out


def speed(model: str, retrieval) -> dict:
    """Read and write speed on THIS prompt, isolated.

    num_predict=1 makes `prompt_seconds` a clean read measurement; the write
    rate comes from the real answer above. Reported per model per question
    because prompt size drives everything on CPU.
    """
    from regrag.generation import render
    try:
        ctx, _labels = render.render(retrieval)
        out = llm.chat("You are a helpful assistant.",
                       render.user_message(retrieval.question, ctx),
                       model=model, max_tokens=1, think=False)
        r = out.prompt_seconds or 0.0
        return {"prompt_tokens": out.prompt_tokens,
                "read_s": round(r, 2),
                "read_tok_s": round(out.prompt_tokens / r, 1) if r else None}
    except Exception as exc:                                       # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3:1.7b,qwen3:4b")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--pick", default="", metavar="1,3,5")
    ns = ap.parse_args()
    models = [m.strip() for m in ns.models.split(",") if m.strip()]

    try:
        have = {m["name"] for m in requests.get(
            f"{config.OLLAMA_HOST}/api/tags", timeout=10).json().get("models", [])}
    except Exception as exc:                                       # noqa: BLE001
        sys.exit(f"Ollama unreachable: {exc}")
    for m in models:
        if not any(h == m or h.startswith(m + ":") for h in have):
            sys.exit(f"{m} not pulled. have: {sorted(have)}\n"
                     f"  ollama pull {m}")

    cases = CASES
    if ns.pick:
        cases = [CASES[int(i) - 1] for i in ns.pick.split(",") if i.strip()]

    print(f"{LINE}\nMODEL COMPARISON — {', '.join(models)}\n{LINE}")
    print("  RETRIEVAL IS FIXED per question and shared by both models, so the")
    print("  generator is the only thing that differs. Quality is NOT scored")
    print("  here — read the two answers against the recorded defect yourself.")

    rows = []
    for i, (q, defect) in enumerate(cases, 1):
        print(f"\n{LINE}\n[{i}/{len(cases)}] {q}\n{LINE}")
        print(textwrap.indent(textwrap.fill("RECORDED DEFECT: " + defect, W - 4), "  "))
        r = retrieve(q)
        if r.refused or not r.passages:
            print(f"\n  retrieval refused: {r.refused}")
            rows.append({"question": q, "defect": defect, "refused": r.refused})
            continue
        print(f"\n  retrieval: {len(r.passages)} passage(s), {r.total_chars:,} chars"
              f"   [shared by every model below]")

        row = {"question": q, "defect": defect,
               "n_passages": len(r.passages), "chars": r.total_chars,
               "runs": [], "speed": {}}
        for m in models:
            row["speed"][m] = speed(m, r)
            res = one(m, q, r)
            row["runs"].append(res)
            print(f"\n{THIN}\n  {m}"
                  f"   {res.get('seconds') if res.get('seconds') is not None else 'FAILED'}s"
                  f"   read {row['speed'][m].get('read_tok_s')} tok/s"
                  f"   prompt {row['speed'][m].get('prompt_tokens')} tok\n{THIN}")
            if res.get("error"):
                print(f"    ERROR: {res['error']}")
                continue
            if res.get("refused"):
                print(f"    REFUSED: {res['refused']}")
                continue
            print(textwrap.indent(res["text"], "    "))
            print(f"\n    SOURCES: {res['sources']}")
            print(f"    {res['n_claims']} claim(s), {res['n_invalid']} invalid, "
                  f"{res['n_weak']} weak")
        rows.append(row)

    # ---- latency, the half that IS measured ------------------------------
    print(f"\n{LINE}\nLATENCY\n{LINE}")
    print(f"  {'model':<22}{'answers':>9}{'median s':>11}{'max s':>9}"
          f"{'median read tok/s':>20}{'failures':>10}")
    for m in models:
        secs = [x["seconds"] for r in rows for x in r.get("runs", [])
                if x["model"] == m and x.get("seconds") is not None]
        rates = [r["speed"][m]["read_tok_s"] for r in rows
                 if r.get("speed", {}).get(m, {}).get("read_tok_s")]
        fails = sum(1 for r in rows for x in r.get("runs", [])
                    if x["model"] == m and x.get("error"))
        print(f"  {m:<22}{len(secs):>9}"
              f"{statistics.median(secs) if secs else 0:>11.0f}"
              f"{max(secs) if secs else 0:>9.0f}"
              f"{statistics.median(rates) if rates else 0:>20.0f}{fails:>10}")
    print("\n  A FAILURE IS A RESULT. qwen3:4b-2507 did not finish an ~11k-token")
    print("  prompt in 300s on 2026-08-29; if that repeats, latency decides this")
    print("  before quality gets a vote.")
    print("\n  READ THE ANSWERS AGAINST THE DEFECT LINE. Question 1's wrong")
    print("  locator is a CODE defect — parent-level citations — so a bigger")
    print("  model fixing it would be luck, not capability.")

    if ns.save:
        f = (pathlib.Path("evaluation") /
             f"models_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps({"models": models, "rows": rows}, indent=1),
                     encoding="utf-8")
        print(f"\n  raw -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
