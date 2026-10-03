"""STAGE 7 — one command, every check. Raw question -> cited answer.

Replaces the one-off studies written while Stage 7 was being worked out
(test_citations, test_json_output, preview_autocite, measure_ctx). Those did
their job and their results are recorded in MENTOR_PROGRESS.md; this is the
repeatable check that has to keep passing.

WHAT IT VERIFIES, IN ORDER — each gate stops the run rather than letting a
later number be measured on a broken foundation.

  A. PREFLIGHT
     1 Ollama reachable and holding config.GEN_MODEL. Says WHICH failed.
     2 Determinism: the same question twice, byte for byte. Without this every
       number below is the generator's noise rather than a measurement.

  B. RETRIEVAL — the two filters, and proof they FIRED
     3 The facet margin drops dead slices (config.FACET_DROP_MARGIN).
     4 The relevance floor drops weak chunks (config.RERANK_DROP_BELOW).
     Both are printed per question. A filter that silently does nothing is the
     bug that cost two runs on 2026-08-26: `mode == "fallback"` never matched
     the fan-out it was written for, so the rule was dead for weeks and looked
     fine. PROVE IT FIRED. Do not infer it from the config value.

  C. GENERATION
     5 Structured JSON output (prompts/answer_v3_json.md). Prose formatting was
       measured and failed twice: a refusal rule refused everything, and harder
       citation wording produced the same five labels rearranged.

  D. THE ANSWER
     6 Every claim's source must be a label that exists   -> INVALID is a bug.
     7 Word overlap between claim and cited passage       -> a SCREEN, not proof.
     8 Rendered with inline markers and a SOURCES block.

WHAT THIS CANNOT TELL YOU — stated so nobody reads more into a green run.
  * REFUSAL. Measured 2026-08-26 and unsolved: a score gate is dead (ranges
    overlap by 3.62) and a prompt rule refused 10 of 10 answerable questions.
    The system has no reliable way to say "I don't have that". Documented, not
    fixed.
  * RELEVANCE. Overlap compares a claim to its PASSAGE. Nothing compares it to
    the QUESTION, so a correctly cited, perfectly grounded, off-topic claim
    passes every check here.
  * Overlap is substring, not whole-word, and ignores word order.

    uv run python scripts/stage7_check.py                  # 10 answerable questions
    uv run python scripts/stage7_check.py --quick          # first 3, smoke test
    uv run python scripts/stage7_check.py --all            # + junk + hard negatives
    uv run python scripts/stage7_check.py --pick 4,5,6,8,10
        the five prose prompting could not cite on 2026-08-29 - the set the
        facet margin and structured output were built against
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import textwrap
from datetime import datetime, timezone

from regrag import config
from regrag.evaluation.refusal_set import HARD_NEGATIVE, JUNK, true_positives
from regrag.generation import llm, render
from regrag.retrieval.search import retrieve

W = 96
LINE = "=" * W
RX = re.compile(r"\[S(\d+)\]")
STOP = set("""the a an of to in for and or is are be as that this which with on by must
should may not from at it its their under section any all such other than when where
have has been were was will can could would each both same these those there here into
over more most less about after before during between within without upon""".split())
OVERLAP_MIN = 0.45


def overlap(claim: str, passage: str) -> float:
    ws = [w for w in re.findall(r"[a-z]{4,}", claim.lower()) if w not in STOP]
    return sum(w in passage.lower() for w in ws) / len(ws) if ws else 0.0


# --- A. PREFLIGHT ----------------------------------------------------------
def preflight(sysmsg: str) -> None:
    print(LINE)
    print(f"A. PREFLIGHT   model {config.GEN_MODEL}   num_ctx {config.GEN_NUM_CTX:,}   "
          f"temp {config.TEMPERATURE}   seed {config.GEN_SEED}")
    print(f"   floor {config.RERANK_DROP_BELOW:+.1f}   "
          f"facet margin {config.FACET_DROP_MARGIN:.1f}   "
          f"context cap {config.GEN_MAX_CONTEXT_CHARS:,}")
    print(LINE)

    problem = llm.why_unavailable()
    if problem:
        sys.exit("   FAIL: " + problem)
    print("   1 model reachable and present            OK")

    a = llm.chat(sysmsg, "QUESTION\nWhat is model risk?\n\nEXCERPTS\n(none)")
    b = llm.chat(sysmsg, "QUESTION\nWhat is model risk?\n\nEXCERPTS\n(none)")
    if a.text != b.text:
        print("   2 determinism                            FAIL")
        print(f"     run 1: {a.text[:120]!r}")
        print(f"     run 2: {b.text[:120]!r}")
        sys.exit("   STOP — a non-deterministic generator makes every number below noise.")
    print("   2 determinism (same request twice)       OK")


# --- B..D. ONE QUESTION ----------------------------------------------------
def run_one(i: int, n: int, kind: str, question: str, sysmsg: str) -> dict:
    rec: dict = {"kind": kind, "question": question}
    print(f"\n{'#' * W}\n[{i}/{n}] {kind}  {question}\n{'#' * W}")

    r = retrieve(question)
    rec["plan_mode"] = r.plan.mode
    rec["inferred_fanout"] = r.plan.inferred_fanout
    print(f"  plan [{r.plan.mode}]  inferred_fanout={r.plan.inferred_fanout}  "
          f"{len(r.plan.facets)} facet(s)")

    # PROVE the filters fired — read the notes, do not assume from config.
    fired = {"facet margin": False, "relevance floor": False}
    for note in r.notes:
        print(f"  ! {note}")
        for k in fired:
            if note.startswith(k):
                fired[k] = True
    rec["filters_fired"] = fired

    if r.refused:
        print(f"  REFUSED BY RETRIEVAL: {r.refused}")
        rec.update(refused=r.refused, claims=[])
        return rec
    if not r.passages:
        print("  NO PASSAGES")
        rec.update(refused="no passages", claims=[])
        return rec

    try:
        ctx, labels = render.render(r)
    except render.ContextTooLarge as e:
        print(f"  CONTEXT TOO LARGE: {e}")
        rec["error"] = str(e)
        return rec

    ptext = {f"S{j}": p.text for j, p in enumerate(r.passages, 1)}
    rec.update(n_passages=len(r.passages), chars=len(ctx), labels=labels)
    print(f"  {len(r.passages)} passage(s), {len(ctx):,} chars")

    out = llm.chat(sysmsg, render.user_message(question, ctx), fmt="json")
    rec["raw"] = out.text
    rec["seconds"] = round(out.seconds, 1)
    try:
        claims = json.loads(out.text).get("claims", [])
        if not isinstance(claims, list):
            raise ValueError("no 'claims' list")
    except Exception as e:
        print(f"  !! JSON PROBLEM: {e}")
        rec["json_error"] = str(e)
        return rec

    checked = []
    for c in claims:
        text, src = str(c.get("text", "")).strip(), str(c.get("source", ""))
        valid = src in labels
        checked.append({"text": text, "source": src, "valid": valid,
                        "overlap": round(overlap(text, ptext.get(src, "")), 3)
                        if valid else None})
    rec["claims"] = checked

    body, sources = render.format_answer(claims, labels)
    print()
    print(textwrap.indent(render.as_text(body, sources), "  "))

    bad = [c for c in checked if not c["valid"]]
    weak = [c for c in checked if c["valid"] and c["overlap"] < OVERLAP_MIN]
    print(f"\n  {len(checked)} claim(s), {out.seconds:.0f}s"
          f" | INVALID {len(bad)} | weak {len(weak)}")
    for c in weak:
        print(f"    weak {c['overlap']:.0%}  {c['text'][:74]}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="first 3 answerable only")
    ap.add_argument("--all", action="store_true", help="also junk and hard negatives")
    ap.add_argument("--pick", default="", metavar="1,4,7",
                    help="run only these answerable questions (1-based). "
                         "--pick 4,5,6,8,10 = the five that prose prompting "
                         "could not cite on 2026-08-29.")
    ns = ap.parse_args()

    sysmsg = pathlib.Path("prompts/answer_v3_json.md").read_text(encoding="utf-8")
    preflight(sysmsg)

    qs = [("ANSWERABLE", c.text) for c in true_positives()]
    if ns.pick:
        want = [int(x) - 1 for x in ns.pick.split(",") if x.strip()]
        qs = [qs[i] for i in want]
    elif ns.quick:
        qs = qs[:3]
    if ns.all:
        qs += ([("JUNK", q.text) for q in JUNK]
               + [("HARD_NEG", q.text) for q in HARD_NEGATIVE])

    print(f"\n{LINE}\nB/C/D. {len(qs)} question(s)\n{LINE}")
    rows = [run_one(i, len(qs), k, q, sysmsg) for i, (k, q) in enumerate(qs, 1)]

    # ---- summary ----------------------------------------------------------
    ans = [r for r in rows if r["kind"] == "ANSWERABLE"]
    claims = [c for r in rows for c in r.get("claims", [])]
    invalid = [c for c in claims if not c["valid"]]
    weak = [c for c in claims if c["valid"] and (c["overlap"] or 0) < OVERLAP_MIN]
    fired_f = sum(1 for r in rows if r.get("filters_fired", {}).get("facet margin"))
    fired_c = sum(1 for r in rows if r.get("filters_fired", {}).get("relevance floor"))

    print(f"\n{LINE}\nSUMMARY\n{LINE}")
    print(f"  questions                     {len(rows)}")
    print(f"  answerable refused            {sum(1 for r in ans if r.get('refused'))}"
          f"/{len(ans)}   <- must be 0")
    print(f"  facet margin fired on         {fired_f}/{len(rows)} question(s)")
    print(f"  relevance floor fired on      {fired_c}/{len(rows)} question(s)")
    print(f"  claims produced               {len(claims)}")
    print(f"  INVALID sources               {len(invalid)}   <- must be 0")
    print(f"  weakly grounded (<{OVERLAP_MIN:.0%})         {len(weak)}   <- read these")
    print(f"  total generation time         {sum(r.get('seconds', 0) for r in rows):.0f}s")
    print("\n  NOT CHECKED HERE: refusal (unsolved) and relevance to the question.")
    print("  See this file's docstring before quoting any of these numbers.")

    f = pathlib.Path("evaluation") / f"stage7_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    f.write_text(json.dumps({
        "model": config.GEN_MODEL, "prompt": "answer_v3_json",
        "floor": config.RERANK_DROP_BELOW, "facet_margin": config.FACET_DROP_MARGIN,
        "rows": rows}, indent=1), encoding="utf-8")
    print(f"\n  raw -> {f}")
    return 1 if invalid or any(r.get("refused") for r in ans) else 0


if __name__ == "__main__":
    raise SystemExit(main())
