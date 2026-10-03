"""DOES THE ALIGNER STILL EARN ITS PLACE? A measurement of SHIPPED behaviour.

WHAT CHANGED ON 2026-09-03
    The aligner was promoted into `regrag/generation/align.py` and is on by
    default. So this script no longer prototypes an idea — it MEASURES THE CODE
    THAT SHIPS, by importing it. Nothing here re-implements scoring, splitting
    or the guards; a private copy would drift from the real thing within a week
    and then report on a system nobody runs. (Same rule as `decide_b` in the
    intent experiments, for the same reason.)

    Because alignment is now inside `answer()`, `c.citation` in these runs is
    ALREADY aligned. CURRENT is reconstructed here — the child that won
    retrieval — so the two can still be compared.

THE DEFECT IT TARGETS (five examples, graded against the PDFs, 2026-09-02)
    `compose_citation` used `p.children[0]` — the child that WON RETRIEVAL. The
    model reads the whole parent WINDOW and draws claims from anywhere in it,
    so the citation named the chunk the search found while the claim came from
    a different paragraph. Verified: an answer citing CRE53.56 drew its five
    claims from CRE53.50, 53.51 and 53.55.

WHAT TO USE IT FOR NOW
    * a floor sweep on fresh questions, before changing config.ALIGN_FLOOR
    * checking a corpus change has not broken paragraph markers (a passage
      showing one unit where it should show a dozen is a PARSING regression,
      not an aligner one)
    * regenerating the evidence behind the 0.65 floor

WHY THERE IS NO --plain ANY MORE
    It switched to unweighted overlap, which is not what ships. A knob that
    measures code nobody runs is how a test script starts lying.

    uv run python scripts/test_aligner.py                 # 15 multi-paragraph
    uv run python scripts/test_aligner.py --cases graded  # the 5 verified
    uv run python scripts/test_aligner.py --floor 0.3 --save
"""
from __future__ import annotations

import argparse
import json
import pathlib
import textwrap
from datetime import datetime, timezone

from regrag import config
from regrag.generation import llm
# answer() DIRECTLY, not respond(). This tests CITATION, not the gate — and on
# the first run the gate stopped 3 of 5 questions on a mis-read subject, which
# is a finding about the gate and pure noise here. One thing under test at a
# time.
from regrag.generation.answer import answer

W = 100
LINE = "=" * W
THIN = "-" * W

# The questions whose true locators were verified against the parsed PDFs on
# 2026-09-02. `expect` is what a correct citation WOULD say; it is printed for
# comparison and never used to score, because the model's claims vary between
# runs and a hard assertion would fail for the wrong reason.
CASES = [
    ("BASEL guidelines: what must a bank do before putting a model in production?",
     "claims verified to come from CRE53.50 / 53.51 / 53.55 — cited CRE53.56"),
    ("Which asset classes fall under specialised lending?",
     "the five SL sub-class NAMES are CRE30.8 — cited CRE30.7"),
    ("Which paragraphs does the CAP standard cross-refer to when defining "
     "regulatory adjustments to CET1?",
     "CAP30.1 correct; the 30.18-30.31 range is quoted from CAP30.6 / CAP99.8"),
    ("How do Basel CAP and Basel RBC treat minimum capital ratios?",
     "content came from RBC30 conservation table and a CAP Footnotes bin"),
    ("What transitional arrangements apply to the deduction of significant "
     "investments in the capital of financial institutions?",
     "source is a CAP FAQ on the 2013-2022 phase-out, pointing at CAP30.21-30.30"),
]

# 15 QUESTIONS CHOSEN SO THE PARENT HAS MANY NUMBERED PARAGRAPHS (Anuj,
# 2026-09-02). The first five graded cases were too thin to benchmark a floor:
# several passages held one or two units, where max() has no real choice. These
# aim at Basel chapters — CRE 20/30/31/32/36/53, CAP 30, RBC 30, LEX, SCO,
# SRP32 — whose parents run to a dozen or more marked paragraphs. That is the
# only regime where a floor means anything.
#
# NO GROUND TRUTH IS ASSERTED FOR THESE. They exist to produce a SCORE
# DISTRIBUTION over realistic multi-unit passages, not to be graded one by one.
# The floor is chosen from that distribution — and then, per the lesson from
# the prompt variants, CONFIRMED ON A SET IT WAS NOT CHOSEN ON.
CASES_MULTI = [
    "What risk weight applies to exposures to sovereigns under the standardised approach?",
    "How is effective maturity calculated for corporate exposures under the IRB approach?",
    "What are the requirements for recognising financial collateral under the comprehensive approach?",
    "How is exposure at default determined for off-balance sheet commitments?",
    "What conditions must a netting agreement meet to be recognised for capital purposes?",
    "What are the requirements for using the internal models method for counterparty credit risk?",
    "How are supervisory haircuts applied to collateral?",
    "What are the criteria for an instrument to qualify as Additional Tier 1 capital?",
    "How are defined benefit pension fund assets treated in regulatory capital?",
    "How are minority interests included in regulatory capital?",
    "What is the treatment of investments in the capital of banking entities outside the scope of consolidation?",
    "What is the large exposure limit for exposures between global systemically important banks?",
    "How is the capital conservation buffer calculated and what constraints apply?",
    "What is the scope of application of the framework to holding companies?",
    "What risk types must be considered under the supervisory review process?",
]

# THE SCORER, THE SPLITTER AND BOTH GUARDS COME FROM THE SHIPPED MODULE.
# Nothing is redefined here on purpose — see the header.
from regrag.generation.align import (        # noqa: E402
    align, build_idf, paragraph_units, similarity,
)

ROWS: list[dict] = []      # everything the run saw, for --save


def run(question: str, note: str, floor: float, margin: float) -> None:
    print(f"\n{LINE}\n{question}\n{LINE}")
    if note:
        print(f"  verified: {note}\n")
    a = answer(question)
    if a.refused or not a.claims:
        print(f"  no claims: {a.refusal_reason}")
        ROWS.append({"question": question, "note": note,
                     "refused": a.refusal_reason, "claims": []})
        return

    # The units are built by the SHIPPED splitter. If a passage that should
    # show a dozen numbered paragraphs shows one, that is a PARSING regression
    # upstream, not an aligner one — read the passage before touching the floor.
    by_label = {f"S{i}": paragraph_units(p)
                for i, p in enumerate(a.retrieval.passages, 1)}

    changed = 0
    rows: list[dict] = []
    for n, c in enumerate(a.claims, 1):
        kids = by_label.get(c.label, [])
        print(f"  [{n}] {textwrap.shorten(c.text, 150)}")
        if not kids:
            print(f"      label {c.label!r} has no paragraphs — skipped\n")
            continue
        # WHAT WOULD HAVE SHIPPED BEFORE THE ALIGNER: the matched child that
        # won retrieval. Reconstructed here because `c.citation` is now already
        # aligned, so the comparison would otherwise be against itself.
        psg = a.retrieval.passages[int(c.label[1:]) - 1]
        cur = (psg.children[0].payload.get("locator") if psg.children else "") or "(none)"
        # Re-run at THIS script's floor/margin rather than reading c.locator,
        # so a sweep can explore settings the answer was not produced under.
        al = align(c.text, kids, cur, floor=floor, margin=margin)
        verdict, loc, best, second = al.verdict, al.locator, al.best, al.runner_up
        same = loc == cur
        changed += not same
        print(f"      paragraphs in this passage : {len(kids)}"
              f"   locators: {sorted({k.locator for k in kids if k.locator})}")
        print(f"      CURRENT  (children[0])   : {cur}")
        print(f"      SHIPPED  [{c.align_verdict:<5}]        : {c.locator}"
              f"   (config floor {config.ALIGN_FLOOR})")
        print(f"      ALIGNED  [{verdict:<5}]        : {loc}"
              f"   best {best:.2f}  runner-up {second:.2f}"
              f"{'' if same else '   <-- DIFFERENT'}")
        print()
        idf = build_idf([k.text for k in kids])
        rows.append({
            "n": n, "claim": c.text, "label": c.label,
            "citation_shipped": c.citation, "overlap_to_passage": c.overlap,
            "shipped_locator": c.locator, "shipped_verdict": c.align_verdict,
            "current_locator": cur, "aligned_locator": loc,
            "verdict": verdict, "best": round(best, 3),
            "runner_up": round(second, 3), "differs": not same,
            "n_units": len(kids),
            "unit_locators": [k.locator for k in kids],
            # The winning paragraph, so a disagreement can be judged from the
            # file without re-running 30-60s of CPU generation.
            "best_unit_text": max(
                kids, key=lambda k: similarity(c.text, k.text, idf)).text[:600],
        })
    print(f"  {changed} of {len(a.claims)} claim(s) would get a different citation")
    ROWS.append({"question": question, "note": note, "refused": None,
                 "seconds": round(a.seconds, 1), "config_hash": a.config_hash,
                 "answer_text": a.text, "sources": a.sources,
                 "n_changed": changed, "claims": rows,
                 "passages": [{"label": f"S{i}", "short_name": p.short_name,
                               "heading": p.heading, "chars": len(p.text),
                               "full_chars": p.full_chars,
                               "windowed": p.windowed,
                               "matched_locators": [h.payload.get("locator")
                                                    for h in p.children]}
                              for i, p in enumerate(a.retrieval.passages, 1)]})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question", action="append", default=[])
    ap.add_argument("--floor", type=float, default=0.65,
                    help="below this, defer to the shipped citation. 0.65 was "
                         "CHOSEN FROM THE 20-QUESTION RUN of 2026-09-02: every "
                         "correct re-citation scored >= 0.731, every wrong one "
                         "<= 0.574, and nothing landed in between. 0.65 sits in "
                         "the middle of that empty band rather than on its edge.")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="top two closer than this are indistinguishable — range")
    ap.add_argument("--cases", default="multi",
                    choices=["multi", "graded", "all"],
                    help="multi (default) = the 15 many-paragraph questions; "
                         "graded = the 5 with verified locators; all = both")
    ap.add_argument("--save", nargs="?", const="auto", default=None,
                    metavar="NAME", help="write the whole run to evaluation/")
    ns = ap.parse_args()

    problem = llm.why_unavailable()
    if problem:
        raise SystemExit(problem)

    print(f"{LINE}\nALIGNER TEST   floor {ns.floor}   margin {ns.margin}   "
          f"idf-weighted (shipped scorer)\n{LINE}")
    print("  CURRENT is what shipped BEFORE the aligner: the child that won")
    print("  retrieval. SHIPPED is what this build actually cited, at the")
    print("  config floor. ALIGNED re-runs it at THIS script's floor, so a")
    print("  sweep can explore settings the answer was not produced under.")
    print("  Read the DIFFERENT rows against the verified note on each question.")

    if ns.question:
        cases = [(q, "") for q in ns.question]
    elif ns.cases == "graded":
        cases = CASES
    elif ns.cases == "multi":
        cases = [(q, "") for q in CASES_MULTI]
    else:
        cases = CASES + [(q, "") for q in CASES_MULTI]
    for q, note in cases:
        run(q, note, ns.floor, ns.margin)

    print(f"\n{LINE}")
    print("THE ALIGNER FIXES WHICH PARAGRAPH IS CITED, NEVER WHETHER THE CLAIM")
    print("IS TRUE. The inverted-mechanism, list-boundary and should/must")
    print("defects still ship — better cited. If ALIGNED == CURRENT everywhere")
    print("on a corpus that HAS numbered paragraphs, something upstream stopped")
    print("producing markers: read a passage before touching the floor.")
    print(LINE)

    # ---- FLOOR SWEEP — the point of the 15-question run -------------------
    claims = [c for r in ROWS for c in r.get("claims", [])]
    multi = [c for c in claims if c["n_units"] >= 4]
    if claims:
        print(f"\n{LINE}\nFLOOR SWEEP — {len(claims)} claim(s), "
              f"{len(multi)} in passages with 4+ paragraphs\n{LINE}")
        print("  Only the 4+ column is meaningful: with one or two units the")
        print("  aligner has no real choice and a high score proves nothing.\n")
        print(f"  {'floor':>7}{'aligned':>10}{'deferred':>10}"
              f"{'changed':>10}   {'aligned (4+ units)':>20}")
        for f in (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8):
            ok = [c for c in claims if c["best"] >= f
                  and (c["best"] - c["runner_up"]) >= ns.margin]
            okm = [c for c in multi if c["best"] >= f
                   and (c["best"] - c["runner_up"]) >= ns.margin]
            ch = [c for c in ok if c["differs"]]
            print(f"  {f:>7.2f}{len(ok):>10}{len(claims) - len(ok):>10}"
                  f"{len(ch):>10}   {len(okm):>13}/{len(multi)}")
        print("\n  READ IT AS A TRADE, NOT A MAXIMUM. A low floor aligns more")
        print("  claims AND ships more confident-wrong locators — 0.574 named")
        print("  the wrong paragraph on 2026-09-02. A high floor defers to the")
        print("  current citation, which is wrong in a known way rather than a")
        print("  new one. Pick the lowest floor at which you would defend every")
        print("  aligned locator by reading it, not the one that aligns most.")

        best_scores = sorted((c["best"] for c in multi), reverse=True)
        if best_scores:
            print(f"\n  best-score distribution (4+ units): "
                  f"max {best_scores[0]:.2f}  "
                  f"median {best_scores[len(best_scores) // 2]:.2f}  "
                  f"min {best_scores[-1]:.2f}")

    if ns.save:
        name = ("aligner" if ns.save == "auto" else ns.save)
        f = (pathlib.Path("evaluation") /
             f"{name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
        f.parent.mkdir(exist_ok=True)
        f.write_text(json.dumps(
            {"tool": "test_aligner.py", "floor": ns.floor,
             "margin": ns.margin, "rows": ROWS},
            indent=1), encoding="utf-8")
        print(f"\n  raw -> {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
