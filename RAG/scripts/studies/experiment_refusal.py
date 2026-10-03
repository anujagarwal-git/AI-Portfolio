"""THE REFUSAL EXPERIMENT. Which mechanism actually says "I don't have that"?

THE PROBLEM
    Retrieval never returns nothing. Ask this index anything and it hands back
    passages with scores. So the model will write a confident, cited answer to a
    question the corpus cannot answer. `_facet_can_match` cannot catch the worst
    case: a UK document DOES exist, so a UK question about ECL passes the plan
    gate and retrieves SS1/23.

THE TWO CANDIDATES, AND THE CONFOUND THEY ARRIVED WITH
    (a) the PROMPT refuses      - reranking stays off
    (b) a SCORE GATE refuses    - reranking becomes load-bearing
    As written those differ by TWO things: who decides, and whether reranking
    runs. Comparing them directly proves nothing (see the 2026-08-23 note in
    MENTOR_PROGRESS.md - a before/after across two edits).

    So RERANKING RUNS FOR EVERY ARM HERE. Scores are always computed. The only
    thing that varies is who acts on them. Reranking's latency cost is already
    measured and is not re-litigated by this experiment.

        arm 0    no rule, no gate     does the model refuse UNPROMPTED?
        arm A    rule, no gate        option (a)
        arm B    no rule, gate        option (b)
        arm A+B  rule and gate        both

    Generation runs TWICE per question (rule / no rule). The gate is pure code
    on the scores, so all four arms compose from those two runs.

THE DECISION RULE - FIXED BEFORE THE FIRST RUN
    (b) is viable ONLY if the score ranges do not overlap AT ALL: highest
        hard-negative score strictly below lowest true-positive score. Not
        "mostly". A gate is one number applied to every future question; if the
        ranges overlap on questions we chose ourselves, there is no boundary.
    (a) is viable if it refuses >= 80% of hard negatives while wrongly refusing
        <= 10% of true positives.
    BOTH viable    -> (a) wins. No rerank latency, no second failure mode.
    NEITHER viable -> neither ships as a refusal guarantee. Document the
                      limitation. That is a real result, not a failure.

    80% on 10 questions is a SCREENING bar, not proof - 8/10 and 6/10 are not
    statistically distinguishable at this sample size. It is enough to
    eliminate an option that fails badly. It is not enough to call a close race.

WHY EVERY RAW ANSWER IS WRITTEN TO DISK
    62 CPU generations is an expensive dataset. Classification is a cheap,
    fallible, re-runnable step. Keeping the raw answers means a better
    classifier can be applied later WITHOUT paying for generation again - and
    means the classifier can be audited instead of trusted.

    uv run python scripts/experiment_refusal.py
    uv run python scripts/experiment_refusal.py --classify evaluation/refusal_run_X.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import time
from datetime import datetime, timezone

from regrag import config
from regrag.evaluation.refusal_set import HARD_NEGATIVE, JUNK, true_positives
from regrag.generation import llm, render
from regrag.retrieval.search import retrieve

PROMPT_NORULE = pathlib.Path("prompts/answer_v0_norule.md")
PROMPT_RULE = pathlib.Path("prompts/answer_v1_rule.md")
OUT_DIR = pathlib.Path("evaluation")
LINE = "=" * 100

MARKER = "INSUFFICIENT CONTEXT"


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

# Arm 0 has never been told HOW to refuse, so it refuses in its own words.
# These patterns catch the common shapes. Anything they do not settle is
# printed for a human to judge - a regex pretending to be a measurement is
# worse than an explicit review list.
REFUSAL_PATTERNS = [
    r"\bdo(?:es)? not (?:contain|provide|include|address|answer|cover)\b",
    r"\bcannot be answered\b", r"\bcan(?:not|'t) answer\b",
    r"\bno information\b", r"\bnot (?:enough|sufficient) (?:information|context)\b",
    r"\binsufficient\b", r"\bnot (?:addressed|covered|mentioned) in the (?:excerpts|passages|provided)\b",
    r"\bthe excerpts? (?:do|does) not\b", r"\bunable to answer\b",
]
_RX = re.compile("|".join(REFUSAL_PATTERNS), re.I)


def classify(text: str) -> tuple[bool, bool]:
    """(refused, ambiguous). Ambiguous rows are printed for human judgement."""
    t = (text or "").strip()
    if not t:
        return True, True                       # empty output is not an answer
    if MARKER.lower() in t[:200].lower():
        return True, False
    hit = bool(_RX.search(t[:400]))
    cited = bool(re.search(r"\[S\d+\]", t))
    if hit and not cited:
        return True, False                      # refused and cited nothing
    if hit and cited:
        return True, True                       # hedged AND answered - judge it
    return False, False


# ---------------------------------------------------------------------------
def run() -> pathlib.Path:
    if not PROMPT_NORULE.exists() or not PROMPT_RULE.exists():
        sys.exit("prompts/ missing. Both prompt files must exist and be frozen.")
    problem = llm.why_unavailable()
    if problem:
        sys.exit("GENERATION UNAVAILABLE: " + problem)

    sys_norule = PROMPT_NORULE.read_text(encoding="utf-8")
    sys_rule = PROMPT_RULE.read_text(encoding="utf-8")

    questions = (
        [("JUNK", q.text, q.why) for q in JUNK]
        + [("HARD_NEG", q.text, f"{q.family}: {q.why}") for q in HARD_NEGATIVE]
        + [("TRUE_POS", c.text, c.note) for c in true_positives()]
    )
    print(LINE)
    print(f"{len(questions)} questions x 2 generations. model {config.GEN_MODEL}, "
          f"num_ctx {config.GEN_NUM_CTX:,}, temp {config.TEMPERATURE}, seed {config.GEN_SEED}")
    print(LINE)

    rows, t0 = [], time.perf_counter()
    for i, (kind, q, why) in enumerate(questions, 1):
        rec: dict = {"kind": kind, "question": q, "why": why}
        r = retrieve(q, rerank=True)            # rerank ON for every arm

        rec["plan_refused"] = r.refused
        rec["n_passages"] = len(r.passages)
        scores = [h.rerank for p in r.passages for h in p.children if h.rerank is not None]
        rec["best_ce"] = max(scores) if scores else None

        # --- PER FACET and PER PARENT, captured because the scores already
        # exist. The facet numbers answer a DIFFERENT and easier question than
        # the refusal gate: not "is this good in absolute terms" but "is this
        # facet far worse than the best facet in THIS query". A cross-encoder
        # can answer the second even where it fails the first, because the
        # comparison is relative and inside one query. Evidence it works is
        # already in search.py's docstring: US -7.6, UK -11.0, Basel +7.7 on a
        # question only Basel could answer.
        # NO MARGIN IS CHOSEN HERE. This run collects the distribution; picking
        # a number before seeing it is how the uncalibrated 0.30 gate happened.
        facet_best: dict[str, float] = {}
        for pas in r.passages:
            for h in pas.children:
                if h.rerank is None:
                    continue
                if h.facet not in facet_best or h.rerank > facet_best[h.facet]:
                    facet_best[h.facet] = h.rerank
        rec["facet_best"] = {k: round(v, 3) for k, v in
                             sorted(facet_best.items(), key=lambda kv: -kv[1])}
        ranked = sorted(facet_best.values(), reverse=True)
        # The margin between the winning facet and the next one. Large margin =
        # the losers are plausibly dead weight. Small = every facet is in play.
        rec["facet_margin"] = round(ranked[0] - ranked[1], 3) if len(ranked) > 1 else None

        rec["passages"] = [{
            "short_name": pas.short_name,
            "heading": (pas.heading or "")[:70],
            "facets": pas.facets,
            "chars": len(pas.text),
            "windowed": pas.windowed,
            "best_ce": round(max((h.rerank for h in pas.children
                                  if h.rerank is not None), default=float("nan")), 3),
            "n_children": len(pas.children),
        } for pas in r.passages]

        if r.refused or not r.passages:
            # Links 1 and 2 of the refusal chain already fired. Nothing to
            # generate, and every arm agrees, so this question carries no
            # information about the two candidates.
            rec["answer_norule"] = rec["answer_rule"] = ""
            rec["upstream_refusal"] = True
            rows.append(rec)
            print(f"  [{i:>2}/{len(questions)}] {kind:<9} upstream refusal: {r.refused or 'no passages'}")
            continue

        rec["upstream_refusal"] = False
        try:
            ctx, table = render.render(r)
        except render.ContextTooLarge as e:
            rec["error"] = str(e)
            rows.append(rec)
            print(f"  [{i:>2}/{len(questions)}] {kind:<9} CONTEXT TOO LARGE - {e}")
            continue
        rec["labels"] = table
        rec["context_chars"] = len(ctx)
        user = render.user_message(q, ctx)

        a = llm.chat(sys_norule, user)
        b = llm.chat(sys_rule, user)
        rec.update(answer_norule=a.text, answer_rule=b.text,
                   prompt_tokens=a.prompt_tokens, maybe_truncated=a.maybe_truncated,
                   seconds=round(a.seconds + b.seconds, 1))
        rows.append(rec)
        ce = f"{rec['best_ce']:+.2f}" if rec["best_ce"] is not None else "  n/a"
        print(f"  [{i:>2}/{len(questions)}] {kind:<9} ce {ce}  {rec['seconds']:>5.1f}s  {q[:52]}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUT_DIR / f"refusal_run_{stamp}.json"
    out.write_text(json.dumps({
        "model": config.GEN_MODEL, "num_ctx": config.GEN_NUM_CTX,
        "temperature": config.TEMPERATURE, "seed": config.GEN_SEED,
        "rerank_model": config.RERANK_MODEL,
        # sha256, not hash() - Python randomises hash() per process, so the
        # fingerprint would differ between two runs of the SAME prompt.
        "prompt_v0_sha": _sha(PROMPT_NORULE),
        "prompt_v1_sha": _sha(PROMPT_RULE),
        "rows": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nraw answers -> {out}   ({time.perf_counter() - t0:.0f}s total)")
    return out


# ---------------------------------------------------------------------------
def report(path: pathlib.Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data["rows"] if not r.get("error")]

    live = [r for r in rows if not r.get("upstream_refusal")]
    up = len(rows) - len(live)
    print(f"\n{LINE}\nREFUSAL EXPERIMENT - {path.name}\n{LINE}")
    print(f"{len(rows)} questions, {up} settled upstream by the planner "
          f"(no information about the two candidates), {len(live)} live.")

    # ---- the gate, first: it can be killed without any generation ----------
    hn = [r["best_ce"] for r in live if r["kind"] == "HARD_NEG" and r["best_ce"] is not None]
    tp = [r["best_ce"] for r in live if r["kind"] == "TRUE_POS" and r["best_ce"] is not None]
    print(f"\n--- CROSS-ENCODER SCORES ---")
    print(f"  hard negatives  n={len(hn):<3} min {min(hn):+7.2f}  max {max(hn):+7.2f}")
    print(f"  true positives  n={len(tp):<3} min {min(tp):+7.2f}  max {max(tp):+7.2f}")
    gate_ok = max(hn) < min(tp)
    if gate_ok:
        b = (max(hn) + min(tp)) / 2
        print(f"  -> NO OVERLAP. A boundary at {b:+.2f} separates them. GATE IS VIABLE.")
    else:
        print(f"  -> THEY OVERLAP by {max(hn) - min(tp):.2f}. NO boundary separates them.")
        print(f"     OPTION (b) IS DEAD on this evidence - same verdict the bi-encoder got.")
        b = None

    # ---- facets: a free by-product of the same scores -----------------------
    multi = [r for r in live if len(r.get("facet_best") or {}) > 1]
    if multi:
        print(f"\n--- FACET SCORES ({len(multi)} multi-facet questions) ---")
        print(f"  {'kind':<10}{'margin':>8}  winner -> losers")
        margins = []
        for r in sorted(multi, key=lambda x: -(x["facet_margin"] or 0)):
            fb = r["facet_best"]
            margins.append(r["facet_margin"])
            items = list(fb.items())
            tail = "  ".join(f"{k}={v:+.1f}" for k, v in items[1:])
            print(f"  {r['kind']:<10}{r['facet_margin']:>8.1f}  "
                  f"{items[0][0]}={items[0][1]:+.1f}  ->  {tail}")
        margins.sort()
        mid = margins[len(margins) // 2]
        print(f"\n  margin: min {margins[0]:.1f}  median {mid:.1f}  max {margins[-1]:.1f}")
        print("  A LARGE margin means the losing facets are plausibly dead weight.")
        print("  Do NOT pick a cut-off from this run alone - read which document each")
        print("  losing facet actually returned before calling it noise.")

    # ---- parents: which passage carried the score --------------------------
    print(f"\n--- TOP PARENT PER QUESTION ---")
    print(f"  {'kind':<10}{'ce':>7}{'chars':>8}  document / heading")
    for r in sorted(live, key=lambda x: -(x["best_ce"] or -99))[:12]:
        ps = sorted(r.get("passages") or [], key=lambda p: -(p["best_ce"] or -99))
        if not ps:
            continue
        t = ps[0]
        print(f"  {r['kind']:<10}{t['best_ce']:>+7.1f}{t['chars']:>8,}  "
              f"[{t['short_name']}] {t['heading'][:52]}")

    # ---- the arms ----------------------------------------------------------
    def arm(rule: bool, gate: bool) -> dict:
        out = {"TRUE_POS": [0, 0], "HARD_NEG": [0, 0], "JUNK": [0, 0]}   # [answered, refused]
        amb = []
        for r in live:
            if gate and b is not None and (r["best_ce"] or -99) < b:
                refused, a = True, False
            else:
                refused, a = classify(r["answer_rule"] if rule else r["answer_norule"])
            out[r["kind"]][1 if refused else 0] += 1
            if a:
                amb.append(r["question"])
        out["_ambiguous"] = amb
        return out

    arms = {"0  no rule, no gate": arm(False, False),
            "A  rule only": arm(True, False),
            "B  gate only": arm(False, True),
            "A+B both": arm(True, True)}

    print(f"\n--- THE ARMS ---")
    print(f"  {'arm':<20}{'TP answered':>13}{'TP refused':>12}{'HN refused':>12}"
          f"{'JUNK refused':>14}{'verdict':>26}")
    ntp = sum(1 for r in live if r["kind"] == "TRUE_POS")
    nhn = sum(1 for r in live if r["kind"] == "HARD_NEG")
    njk = sum(1 for r in live if r["kind"] == "JUNK")
    for name, a in arms.items():
        if name.strip().startswith("B") and b is None:
            print(f"  {name:<20}{'--- gate has no boundary; arm not evaluable ---':>77}")
            continue
        hn_r, tp_r = a["HARD_NEG"][1], a["TRUE_POS"][1]
        ok = (nhn and hn_r / nhn >= 0.80) and (not ntp or tp_r / ntp <= 0.10)
        print(f"  {name:<20}{a['TRUE_POS'][0]:>8}/{ntp:<4}{tp_r:>7}/{ntp:<4}"
              f"{hn_r:>7}/{nhn:<4}{a['JUNK'][1]:>9}/{njk:<4}"
              f"{('MEETS THE BAR' if ok else 'below the bar'):>26}")

    amb = sorted({q for a in arms.values() for q in a["_ambiguous"]})
    if amb:
        print(f"\n--- {len(amb)} ANSWER(S) THE CLASSIFIER COULD NOT SETTLE - READ THEM ---")
        for q in amb:
            print(f"    {q}")
        print("    (hedged AND cited: the model refused and answered at the same time)")

    print(f"\n{LINE}\nApply the decision rule from this file's docstring. Do not adjust it now.\n{LINE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify", type=pathlib.Path, help="report on a saved run")
    ns = ap.parse_args()
    report(ns.classify if ns.classify else run())
