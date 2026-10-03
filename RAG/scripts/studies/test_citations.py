"""Does prompt v2 fix the citation failure? One variable, both arms, same run.

THE FAILURE BEING FIXED
    Measured 2026-08-26 on the ten verified answerable questions: the model got
    the SUBSTANCE right 10 times out of 10 and cited nothing 7 times out of 10.
    A correct answer with no citation is not usable by a reviewer - it is the
    audit trail, not decoration.

WHY BOTH PROMPTS RUN AGAIN INSTEAD OF REUSING THE SAVED v0 NUMBERS
    Retrieval CHANGED since that run: config.RERANK_DROP_BELOW now removes
    chunks scoring below zero, so the passages are not the same passages. The
    old 7-of-10 was measured on a different context. Comparing v2-on-new-context
    against v0-on-old-context would differ by TWO things and prove nothing.
    So v0 is re-run here. Only the prompt varies.

WHAT IS MEASURED - all mechanical, no judgement
    cited        answer contains at least one [S] label
    valid        every label used exists in the label table
    grounded     the cited excerpt actually contains wording from the sentence
                 (crude overlap check - a screen, not a verdict; read the misses)

    uv run python scripts/test_citations.py
"""
from __future__ import annotations

import json
import pathlib
import re
from datetime import datetime, timezone

from regrag import config
from regrag.evaluation.refusal_set import true_positives
from regrag.generation import llm, render
from regrag.retrieval.search import retrieve

RX = re.compile(r"\[S(\d+)\]")
STOP = set("the a an of to in for and or is are be as that this which with on by "
           "must should may not from at it its their under section".split())
LINE = "=" * 100


def grounded(sentence: str, passage: str) -> bool:
    """Crude: do the sentence's content words appear in the cited passage?

    NOT a verdict. It cannot tell paraphrase from fabrication, and it will pass
    a sentence that borrows the right vocabulary to say the wrong thing. It is a
    SCREEN that narrows what a human has to read. Treat a failure as 'look at
    this one', never as proof, and treat a pass as nothing at all.
    """
    words = [w for w in re.findall(r"[a-z]{4,}", sentence.lower()) if w not in STOP]
    if not words:
        return True
    low = passage.lower()
    return sum(w in low for w in words) / len(words) >= 0.5


def score(text: str, labels: dict, passages: dict) -> dict:
    used = sorted(set(RX.findall(text or "")), key=int)
    bad = [u for u in used if f"S{u}" not in labels]
    # sentence-level grounding, only for sentences that carry a label
    ok = miss = 0
    for sent in re.split(r"(?<=[.!?])\s+", text or ""):
        labs = RX.findall(sent)
        if not labs:
            continue
        body = RX.sub("", sent)
        if any(grounded(body, passages.get(f"S{l}", "")) for l in labs):
            ok += 1
        else:
            miss += 1
    return {"n_labels": len(used), "invalid": len(bad),
            "grounded": ok, "ungrounded": miss, "labels": used}


def main() -> int:
    prompts = {"v0": pathlib.Path("prompts/answer_v0_norule.md").read_text(encoding="utf-8"),
               "v2": pathlib.Path("prompts/answer_v2.md").read_text(encoding="utf-8")}
    problem = llm.why_unavailable()
    if problem:
        raise SystemExit("GENERATION UNAVAILABLE: " + problem)

    rows = []
    qs = true_positives()
    print(f"{len(qs)} verified answerable questions x 2 prompts. "
          f"relevance floor {config.RERANK_DROP_BELOW:+.1f}\n{LINE}")
    for i, c in enumerate(qs, 1):
        r = retrieve(c.text)                      # rerank + floor now default
        if r.refused or not r.passages:
            print(f"  [{i:>2}] *** RETRIEVAL REFUSED a verified positive: {r.refused} ***")
            rows.append({"q": c.text, "refused": r.refused})
            continue
        ctx, labels = render.render(r)
        ptext = {f"S{j}": p.text for j, p in enumerate(r.passages, 1)}
        user = render.user_message(c.text, ctx)
        rec = {"q": c.text, "n_passages": len(r.passages), "chars": len(ctx),
               "labels": labels}
        for tag, sysmsg in prompts.items():
            out = llm.chat(sysmsg, user)
            rec[tag] = {"text": out.text, "seconds": round(out.seconds, 1),
                        **score(out.text, labels, ptext)}
        rows.append(rec)
        v0, v2 = rec["v0"], rec["v2"]
        print(f"  [{i:>2}] {len(r.passages)} psg {len(ctx):>6,}c   "
              f"v0 {v0['n_labels']:>2} labels ({v0['ungrounded']} ungrounded)   "
              f"v2 {v2['n_labels']:>2} labels ({v2['ungrounded']} ungrounded)   {c.text[:38]}")

    live = [r for r in rows if "v0" in r]
    print(f"\n{LINE}\nCITATION TEST — {len(live)} questions\n{LINE}")
    print(f"  {'':<6}{'cited':>10}{'total labels':>15}{'invalid':>10}{'ungrounded sent':>18}")
    for tag in ("v0", "v2"):
        cited = sum(1 for r in live if r[tag]["n_labels"] > 0)
        print(f"  {tag:<6}{cited:>7}/{len(live):<3}"
              f"{sum(r[tag]['n_labels'] for r in live):>15}"
              f"{sum(r[tag]['invalid'] for r in live):>10}"
              f"{sum(r[tag]['ungrounded'] for r in live):>18}")
    print("\n  'invalid' = a label pointing at no passage (hallucinated citation).")
    print("  'ungrounded' = a cited sentence whose words are not in the cited passage.")
    print("  Ungrounded is a SCREEN. Read those sentences before calling them wrong.")

    out = pathlib.Path("evaluation") / (
        f"citation_run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    out.write_text(json.dumps({"model": config.GEN_MODEL,
                               "floor": config.RERANK_DROP_BELOW, "rows": rows},
                              indent=1), encoding="utf-8")
    print(f"\n  raw answers -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
