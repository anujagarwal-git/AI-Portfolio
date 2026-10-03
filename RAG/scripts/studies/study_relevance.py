"""POINT 4 — WHY DO IRRELEVANT PASSAGES REACH THE PROMPT? A study, not a fix.

THE QUESTION
    Retrieval works. Citations resolve. And a passage that has nothing to do
    with the question still arrives, correctly cited, and gets used. Nothing in
    the pipeline compares anything to the QUESTION: `overlap` compares a claim
    to the passage it CITES, so a perfectly grounded, correctly cited,
    completely off-topic claim scores 100%.

    This does not fix that. It looks for a PATTERN in the retrieval scores that
    separates the passages you judge irrelevant from the ones you judge useful.
    If a pattern exists, it becomes a gate. If it does not, that is the result
    and you stop paying for it.

THE ONE DESIGN DECISION THAT MATTERS — RELATIVE, NOT ABSOLUTE
    An ABSOLUTE cross-encoder threshold is already dead here, measured
    2026-08-26: hard negatives reached +6.99 while true positives fell to
    +3.38, ranges overlapping by 3.62. Cross-encoder scores are not calibrated
    ACROSS questions, so a cut that works on one fails on the next.

    What DOES work in this project is relative and within one question: the
    facet margin compares a facet's best against the winning facet's best, and
    it was chosen from a distribution. So the features below are mostly
    WITHIN-QUESTION: a passage's distance from the best passage of ITS OWN
    question, its rank, which arms found it. The absolute score is computed too
    — not because it is expected to work, but so its failure is visible in the
    same table rather than assumed.

WHAT MAKES THIS HONEST, AND WHAT WOULD MAKE IT WORTHLESS
    THE LABELS ARE THE STUDY. Not the code. You read the passage and say
    whether it helps answer the question. A proxy — "short chunks are
    irrelevant", "footnote bins are irrelevant" — would only ever move the
    flattering direction, which is the trap already recorded twice in this
    project. Label by reading. 30 questions is roughly 150 passages.

    THE FALSIFYING CONDITION IS FIXED HERE, BEFORE YOU LOOK. With ~150 rows and
    six score columns, SOMETHING will look like a pattern. A feature counts as
    a candidate rule ONLY if:
      1. the middle 80% of relevant and the middle 80% of irrelevant DO NOT
         OVERLAP (p10..p90 of one clear of the other), and
      2. a cut placed between them would drop at most 5% of RELEVANT passages.
    Anything else is a correlation to note, not a rule to build. Do not soften
    these after seeing the numbers.

THREE PHASES, RUN IN ORDER
    1  --collect   run all 30 questions, dump every delivered passage with its
                   dense rank, BM25 rank, RRF, rerank score and flags. No LLM.
    2  --label     read each passage, mark it relevant / irrelevant / unsure.
                   Saves after every keystroke; resume any time.
    3  --analyse   report each feature's separation against the rule above.

    uv run python scripts/study_relevance.py --collect
    uv run python scripts/study_relevance.py --label
    uv run python scripts/study_relevance.py --analyse

COST. Collection is retrieval only — no generation — so roughly 45 x 1.5s plus
model load. LABELLING IS THE REAL COST: 45 questions is roughly 200-230
passages. Budget two to three hours. Do each SITTING in one go — a standard
that drifts between the first passage and the last is worse than fewer labels,
so `--label` saves after every keystroke and resumes exactly where you stopped.
If you want it smaller, label the 15 multi-facet questions FIRST: they are
where irrelevance is expected, so they carry more information per label.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import textwrap
from datetime import datetime, timezone

from regrag.retrieval.search import is_bin, retrieve

OUT = pathlib.Path("evaluation") / "relevance_study.json"
W = 92
LINE = "=" * W
THIN = "-" * W

# 30 questions, tagged with the document each one TARGETS. Two for the
# content-rich documents, one for the thin and reference-data ones — 19
# documents do not divide into 30 any other way.
#
# THE TARGET TAG IS A HINT, NOT A LABEL. A passage arriving from a different
# document is often correct: a question about model validation legitimately
# pulls SR 11-7 and SS1/23 together. `off_target` is reported so you can see
# whether it tracks your judgement — it is NOT used as a substitute for it.
QUESTIONS: list[tuple[str, str, str]] = [
    ("SR 11-7", "named", "What does SR 11-7 say about effective challenge?"),
    ("SR 11-7", "named", "Under SR 11-7, what are the core elements of model validation?"),
    ("SR 26-2", "named", "What does SR 26-2 expect on model risk governance?"),
    ("SR 26-2", "named", "What does SR 26-2 say about maintaining a model inventory?"),
    ("SS1/23", "named", "Under SS1/23, what is expected of the model risk management framework?"),
    ("SS1/23", "named", "What does SS1/23 say about senior management responsibilities for models?"),
    ("IFRS 9", "named", "When is a financial asset credit-impaired under IFRS 9?"),
    ("IFRS 9", "named", "What does IFRS 9 say about the 30 days past due rebuttable presumption?"),
    ("BCBS 239", "named", "What does BCBS 239 require on the accuracy and integrity of risk data?"),
    ("BCBS 239", "named", "What does BCBS 239 say about the timeliness of risk reporting?"),
    ("BCBS d450", "named", "What principles does BCBS d450 set for a stress testing framework?"),
    ("BCBS d450", "named", "What does BCBS d450 say about stress testing governance?"),
    ("BCBS d403", "named", "How does BCBS d403 define a non-performing exposure?"),
    ("BCBS d403", "named", "What does BCBS d403 say about forbearance?"),
    ("Basel CAP", "named", "What are the criteria for Common Equity Tier 1 instruments?"),
    ("Basel CAP", "named", "What does Basel CAP say about regulatory adjustments to CET1?"),
    ("Basel CRE", "named", "What does Basel CRE say about eligible financial collateral?"),
    ("Basel CRE", "named", "What does Basel CRE say about the standardised approach to credit risk mitigation?"),
    ("Basel RBC", "named", "What minimum risk-based capital ratios does Basel RBC set?"),
    ("Basel LEX", "named", "What is the large exposure limit to a single counterparty?"),
    ("Basel SCO", "named", "What does Basel SCO say about the scope of application of the framework?"),
    ("Basel SRP32", "named", "What risk types does Basel SRP32 cover under Pillar 2?"),
    ("12 CFR Part 252", "named", "Under 12 CFR Part 252, what are the company-run stress test requirements?"),
    ("12 CFR Part 252", "named", "What does 12 CFR Part 252 require on capital plan submission?"),
    ("12 CFR 225.8", "named", "What must a firm include in its capital plan under 12 CFR 225.8?"),
    ("12 CFR 225.8", "named", "Under 12 CFR 225.8, when must a firm resubmit its capital plan?"),
    ("SR 15-18", "named", "What does SR 15-18 expect of capital planning at large firms?"),
    ("SR 15-19", "named", "What does SR 15-19 say about capital planning at smaller firms?"),
    ("2026 Stress Test Scenarios", "named", "What does the 2026 severely adverse scenario assume?"),
    ("2026 DFAST Results", "named", "What do the 2026 DFAST results report on capital ratios?"),

    # ------------------------------------------------------------------
    # THE MULTI-FACET HALF (added 2026-09-02, Anuj). The 30 above ALL name a
    # document, so planner rule 4 fires every time and retrieval is a single
    # filtered search. That is the easy path, and measuring only it would have
    # produced a flattering answer to a question about irrelevant passages.
    #
    # A fan-out spends the context budget on two or three slices at once. The
    # recorded failure is exactly here: on "which asset classes fall under
    # specialised lending?" the US facet had 0 of 4,019 eligible chunks on
    # topic and the UK facet 0 of 248, and the per-facet quota still handed
    # them two thirds of a 24,984-char window. If irrelevance concentrates
    # anywhere, it concentrates in these.
    #
    # NOTE ON THE VERSION SET: weak wording ("differ", "compare") no longer
    # triggers rule 3 — REGRAG_WEAK_VERSION defaults to 0 since 2026-09-02. So
    # these use STRONG wording ("previous", "what changed", "earlier",
    # "superseded"), which fires on its own and does not depend on the flag.

    # inferred fan-out — nothing named, the US/UK/GLOBAL split is OUR guess
    ("", "inferred", "What must a bank do before putting a model into production?"),
    ("", "inferred", "What are the expectations for ongoing monitoring of models?"),
    ("", "inferred", "How should a firm document its capital planning process?"),
    ("", "inferred", "What governance is required over risk data?"),
    ("", "inferred", "Which asset classes fall under specialised lending?"),

    # two jurisdictions named — the split is what the USER asked for
    ("", "jurisdiction", "How do US and UK expectations on model validation compare?"),
    ("", "jurisdiction", "What do US and UK regulators require on model inventories?"),
    ("", "jurisdiction", "How does model risk governance in the US differ from the UK?"),

    # two documents named — rule 1, which also separates their versions
    ("", "document", "How does SR 26-2 differ from SR 11-7 on model validation?"),
    ("", "document", "What do SR 11-7 and SS1/23 each say about effective challenge?"),
    ("", "document", "How do Basel CAP and Basel RBC treat minimum capital ratios?"),

    # version fan-out — STRONG wording, no document named
    ("", "version", "What was the previous supervisory expectation on model risk management?"),
    ("", "version", "What changed in the supervisory guidance on capital planning?"),
    ("", "version", "What did the earlier guidance say about model documentation?"),
    ("", "version", "Which requirements were superseded in the model risk framework?"),
]


# ---------------------------------------------------------------------------
# 1. COLLECT
# ---------------------------------------------------------------------------
def collect() -> int:
    rows: list[dict] = []
    drops: list[dict] = []
    for i, (target, want_mode, q) in enumerate(QUESTIONS, 1):
        r = retrieve(q)
        # EXPECTED vs ACTUAL plan mode. A mismatch is a finding in itself —
        # that is precisely how the CRE bug surfaced, where "differ" +
        # "framework" sent a named-document question down the version path.
        print(f"[{i:>2}/{len(QUESTIONS)}] {want_mode:<13}->{r.plan.mode:<20}"
              f"{len(r.passages)} psg  {len(r.plan.facets)} facet  "
              f"{r.timings.get('total', 0) * 1000:>5.0f}ms  {q[:34]}")
        if want_mode != "named" and not r.plan.mode.startswith(want_mode):
            print(f"        ! expected a {want_mode} plan, got {r.plan.mode!r}")
        if r.refused:
            print(f"        REFUSED: {r.refused}")
            continue

        # The best rerank in THIS question — every relative feature is measured
        # from here, because a score only means something next to its siblings.
        scores = [h.rerank for p in r.passages for h in p.children
                  if h.rerank is not None]
        best = max(scores) if scores else None

        # --- THE CASUALTIES ------------------------------------------
        # Every candidate a cut discarded. Most cost NOTHING: a chunk the
        # quota dropped whose parent still arrived through a sibling was
        # read by the model anyway. `loss` separates those out, and only
        # loss != "none" can possibly have cost the answer something.
        #
        # `label_me` is a LABELLING BUDGET, not a claim about importance.
        # A cut that fires at rank 5 is decided at ranks 5-7; rank 18 was
        # never close. Reading a thousand passages would drift the standard
        # between the first and the last, which is worse than reading fewer
        # carefully. Everything is COUNTED; the bounded subset is READ.
        margin_seen: dict[str, int] = {}
        for d in r.dropped:
            keep_for_label = False
            if d.loss != "none":
                if d.cut == "relevance_floor":
                    keep_for_label = True
                elif d.cut == "context_budget":
                    keep_for_label = True
                elif d.cut == "fallback_pool":
                    keep_for_label = True          # facet winners; few
                elif d.cut == "quota":
                    keep_for_label = (d.rank_in_facet or 0) < 8
                elif d.cut == "facet_margin":
                    n = margin_seen.get(d.facet, 0)
                    keep_for_label = n < 2         # best two of a killed slice
                    margin_seen[d.facet] = n + 1
            drops.append({
                "qid": i, "question": q, "target_doc": target,
                "want_mode": want_mode, "plan_mode": r.plan.mode,
                "inferred_fanout": r.plan.inferred_fanout,
                "cut": d.cut, "loss": d.loss, "label_me": keep_for_label,
                "facet": d.facet, "short_name": d.short_name,
                "heading": d.heading, "parent_id": d.parent_id,
                "chars": d.chars, "rank_in_facet": d.rank_in_facet,
                "rerank": d.rerank, "rrf": d.rrf,
                "dense_rank": d.dense_rank, "lexical_rank": d.lexical_rank,
                "locator": d.locator, "text": d.text,
                "verdict": None, "verdict_claude": None,
            })

        for label_i, p in enumerate(r.passages, 1):
            # A parent is delivered once but may hold several matched children.
            # Score the BEST child: that is the one that earned the parent its
            # place, so it is the one a gate would have to judge.
            kid = max(p.children,
                      key=lambda h: (h.rerank if h.rerank is not None else h.rrf))
            rows.append({
                "qid": i, "question": q, "target_doc": target,
                "want_mode": want_mode,
                "plan_mode": r.plan.mode,
                "inferred_fanout": r.plan.inferred_fanout,
                "n_facets": len(r.plan.facets),
                "label_id": f"S{label_i}",
                "short_name": p.short_name, "heading": p.heading,
                "parent_id": p.parent_id,
                "chars": len(p.text), "full_chars": p.full_chars,
                "windowed": p.windowed, "is_bin": is_bin(p.heading),
                # Only meaningful when the question named ONE document.
                "off_target": (target.lower() not in p.short_name.lower()
                               if target else None),
                "facet": kid.facet,
                "dense_rank": kid.dense_rank,
                "lexical_rank": kid.lexical_rank,
                "rrf": round(kid.rrf, 5),
                "rerank": kid.rerank,
                "gap_to_best": round(best - kid.rerank, 3)
                               if (best is not None and kid.rerank is not None) else None,
                "n_children": len(p.children),
                "locator": kid.payload.get("locator"),
                "text": p.text,
                # TWO LABEL FIELDS, KEPT APART ON PURPOSE.
                # `verdict` is Anuj's, entered through --label.
                # `verdict_claude` is the mentor's first pass, produced from the
                # passage TEXT ONLY with every score hidden — the features being
                # tested were designed by the same party writing these labels,
                # so they must never be merged into one column. Their AGREEMENT
                # RATE is the check: high agreement makes the first pass usable,
                # low agreement means the judgement is subjective and no cut
                # built on it means much.
                "verdict": None,
                "verdict_claude": None,
            })

    # --- CARRY EXISTING LABELS FORWARD -------------------------------
    # A re-collection must never silently destroy hours of labelling. The key
    # is (question, parent) - which is exactly what a label is ABOUT, so a
    # label follows its passage even when scores, ranks or plan shape have
    # changed underneath it. A passage retrieval no longer delivers simply
    # loses its label, which is correct: there is nothing left to label.
    carried = 0
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            sys.exit(f"{OUT} exists but could not be read ({exc}). REFUSING to "
                     f"overwrite it - move it aside by hand if you mean to.")
        old_labels = {(r.get("qid"), r.get("parent_id")):
                      (r.get("verdict"), r.get("verdict_claude"))
                      for r in prev.get("rows", []) + prev.get("dropped", [])}
        for r in rows + drops:
            hit_ = old_labels.get((r["qid"], r["parent_id"]))
            if hit_ and any(hit_):
                r["verdict"], r["verdict_claude"] = hit_
                carried += 1
        print(f"\n  carried {carried} existing label(s) forward from {OUT.name}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "collected": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
        "rows": rows, "dropped": drops}, indent=1), encoding="utf-8")
    print(f"\n  {len(rows)} passage(s) across {len(QUESTIONS)} question(s) -> {OUT}")

    print(f"\n  {len(drops)} candidate(s) discarded by a cut")
    print(f"  {'cut':<18}{'total':>8}{'cost nothing':>15}{'real loss':>12}{'to label':>10}")
    cuts: dict[str, list] = {}
    for d in drops:
        cuts.setdefault(d["cut"], []).append(d)
    for cut, g in sorted(cuts.items(), key=lambda kv: -len(kv[1])):
        none_ = sum(1 for d in g if d["loss"] == "none")
        print(f"  {cut:<18}{len(g):>8}{none_:>15}{len(g) - none_:>12}"
              f"{sum(1 for d in g if d['label_me']):>10}")
    print(f"\n  A 'real loss' is a drop whose parent never reached the prompt,")
    print(f"  or was trimmed so this text is not in the delivered window.")
    print(f"  IT IS NOT A MISS: a chunk the index search never surfaced has no")
    print(f"  row here at any effort. That is the golden set's question.")
    print("\n  next: uv run python scripts/study_relevance.py --label")
    return 0


# ---------------------------------------------------------------------------
# 2. LABEL — the part that decides whether any of this is worth anything
# ---------------------------------------------------------------------------
def _select(rows: list[dict], only: str) -> list[dict]:
    """Which rows to put in front of you, in what order.

    DEFAULT IS MULTI-FACET FIRST, and not for convenience. Labelling is the
    expensive half of this study and attention degrades across a long sitting,
    so the passages most likely to carry a finding should be judged while the
    standard is sharpest. The 30 named-document questions each run ONE filtered
    search — the easy path, and the one least likely to deliver anything you
    would call irrelevant. The 15 fan-out questions spend the budget across two
    or three slices at once, which is where the recorded failure lives.

    If you label only the multi-facet half, `--analyse` still runs. It will say
    so, and its one-facet-vs-fan-out comparison will be missing a side — read
    that as an incomplete study, not a result about fan-out.
    """
    if only == "all":
        return rows
    if only == "multi":
        return [r for r in rows if r.get("want_mode") != "named"]
    if only == "named":
        return [r for r in rows if r.get("want_mode") == "named"]
    return [r for r in rows if r.get("want_mode") == only]


def label(show_chars: int, only: str) -> int:
    d = json.loads(OUT.read_text(encoding="utf-8"))
    rows = _select(d["rows"], only)
    if not rows:
        sys.exit(f"no rows match --only {only!r}")
    todo = [r for r in rows if r["verdict"] is None]
    if not todo:
        print(f"  every row in --only {only!r} is already labelled.")
        rest = [r for r in d["rows"] if r["verdict"] is None]
        print(f"  {len(rest)} unlabelled elsewhere — try --only all")
        return 0
    print(f"{LINE}\n{len(todo)} of {len(rows)} left to label"
          f"   [--only {only}]\n{LINE}")
    print("  r = relevant     it helps answer THIS question")
    print("  i = irrelevant   it does not, whatever else is true of it")
    print("  u = unsure       excluded from the analysis, and that is fine")
    print("  s = skip for now      q = save and stop")
    print("\n  SCORES ARE HIDDEN ON PURPOSE. Seeing 'ce+6.8' before you read the")
    print("  passage is how a label becomes a description of the score you are")
    print("  trying to test. Read the text, then judge.\n")

    for n, row in enumerate(todo, 1):
        print(f"\n{LINE}")
        print(f"[{n}/{len(todo)}]  {row['question']}")
        print(THIN)
        print(f"  {row['short_name']}  {row['heading'][:70]!r}"
              f"   {row['chars']:,}c")
        print(THIN)
        print(textwrap.indent(textwrap.fill(row["text"][:show_chars], W - 4), "  "))
        if len(row["text"]) > show_chars:
            print(f"  ... [{len(row['text']) - show_chars:,} more chars]")
        try:
            ans = input("\n  r / i / u / s / q > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "q"
        if ans == "q":
            break
        if ans == "s":
            continue
        row["verdict"] = {"r": "relevant", "i": "irrelevant",
                          "u": "unsure"}.get(ans)
        # Save after EVERY answer. An hour of labelling lost to a closed
        # terminal is an hour nobody labels again.
        OUT.write_text(json.dumps(d, indent=1), encoding="utf-8")

    done = sum(1 for r in rows if r["verdict"])
    allrows = d["rows"]
    print(f"\n  {done}/{len(rows)} labelled in this selection")
    print(f"  {sum(1 for r in allrows if r['verdict'])}/{len(allrows)} overall"
          f" -> {OUT}")
    return 0


# ---------------------------------------------------------------------------
# 3. ANALYSE
# ---------------------------------------------------------------------------
FEATURES = [
    ("gap_to_best", "distance below this question's best score", "lower=better"),
    ("rerank", "absolute cross-encoder score (EXPECTED TO FAIL)", "higher=better"),
    ("rrf", "reciprocal rank fusion", "higher=better"),
    ("chars", "passage length", "-"),
    ("n_children", "matched children in this parent", "-"),
]
DROP_MAX = 0.05          # a cut may lose at most 5% of relevant passages
P_LO, P_HI = 10, 90      # "middle 80%"


def pct(xs: list[float], p: int) -> float:
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def agreement(rows: list[dict]) -> None:
    """Where the two label passes differ, and how much."""
    both = [r for r in rows if r.get("verdict") in ("relevant", "irrelevant")
            and r.get("verdict_claude") in ("relevant", "irrelevant")]
    if not both:
        return
    same = [r for r in both if r["verdict"] == r["verdict_claude"]]
    print(f"\n{THIN}\n  LABEL AGREEMENT — {len(same)}/{len(both)} "
          f"({len(same) / len(both):.0%})\n{THIN}")
    for r in both:
        if r["verdict"] != r["verdict_claude"]:
            print(f"  q{r['qid']:>2} {r['label_id']}  you={r['verdict']:<11}"
                  f"mentor={r['verdict_claude']:<11}"
                  f"{r['short_name']} {r['heading'][:34]!r}")
    if len(same) / len(both) < 0.85:
        print("\n  BELOW 85%. The two passes are judging different things, so a")
        print("  cut fitted to either one is fitted to a judgement call rather")
        print("  than to relevance. Settle the disagreements before believing")
        print("  any separation below.")


def analyse(which: str) -> int:
    d = json.loads(OUT.read_text(encoding="utf-8"))
    key = "verdict_claude" if which == "claude" else "verdict"
    if which == "agreed":
        rows = [r for r in d["rows"]
                if r.get("verdict") in ("relevant", "irrelevant")
                and r.get("verdict") == r.get("verdict_claude")]
        print("  USING ONLY ROWS BOTH PASSES AGREE ON — the cleanest labels,")
        print("  and a smaller, biased sample: the easy calls survive and the")
        print("  hard ones are dropped. Read any separation as optimistic.")
    else:
        rows = [r for r in d["rows"] if r.get(key) in ("relevant", "irrelevant")]
        for r in rows:
            r["verdict"] = r[key]
    agreement(d["rows"])
    if len(rows) < 20:
        sys.exit(f"only {len(rows)} labelled rows — label more before reading "
                 f"anything into a separation.")

    rel = [r for r in rows if r["verdict"] == "relevant"]
    irr = [r for r in rows if r["verdict"] == "irrelevant"]
    print(f"{LINE}\nRELEVANCE STUDY — {len(rel)} relevant, {len(irr)} irrelevant, "
          f"{len(d['rows']) - len(rows)} unlabelled/unsure\n{LINE}")
    if not irr:
        print("  NO IRRELEVANT PASSAGES LABELLED. That is a real finding: on")
        print("  these 30 questions, retrieval delivered nothing you judged")
        print("  useless — and point 4 may be rarer than the CRE run suggested.")
        return 0

    print(f"\n  {'feature':<16}{'relevant p10..p90':>26}{'irrelevant p10..p90':>26}"
          f"{'verdict':>16}")
    print("  " + "-" * (W - 2))
    candidates = []
    for key, desc, _dir in FEATURES:
        a = [r[key] for r in rel if r.get(key) is not None]
        b = [r[key] for r in irr if r.get(key) is not None]
        if len(a) < 5 or len(b) < 5:
            print(f"  {key:<16}{'too few values':>26}")
            continue
        a_lo, a_hi = pct(a, P_LO), pct(a, P_HI)
        b_lo, b_hi = pct(b, P_LO), pct(b, P_HI)
        clear = a_hi < b_lo or b_hi < a_lo
        if clear:
            # A cut sits between the two middles. How many relevant does it lose?
            cut = (a_hi + b_lo) / 2 if a_hi < b_lo else (b_hi + a_lo) / 2
            lost = (sum(1 for x in a if x > cut) if a_hi < b_lo
                    else sum(1 for x in a if x < cut)) / len(a)
            passed = lost <= DROP_MAX
            verdict = f"CANDIDATE cut {cut:.2f}" if passed else f"loses {lost:.0%} rel"
            if passed:
                candidates.append((key, cut, lost))
        else:
            verdict = "overlaps"
        print(f"  {key:<16}{a_lo:>12.2f}..{a_hi:<12.2f}"
              f"{b_lo:>12.2f}..{b_hi:<12.2f}{verdict:>16}")

    # --- categorical signals, reported as rates not ranges ----------------
    print(f"\n{THIN}\n  CATEGORICAL — rate among relevant vs irrelevant\n{THIN}")
    for key in ("is_bin", "windowed", "off_target"):
        ra = sum(1 for r in rel if r.get(key)) / len(rel)
        rb = sum(1 for r in irr if r.get(key)) / len(irr)
        print(f"  {key:<16}{ra:>8.0%}{rb:>12.0%}"
              f"    {'notable' if abs(ra - rb) > 0.25 else ''}")

    print(f"\n  arms that found the passage:")
    for name, test in (("dense only", lambda r: r["lexical_rank"] is None),
                       ("bm25 only", lambda r: r["dense_rank"] is None),
                       ("both", lambda r: r["dense_rank"] is not None
                                          and r["lexical_rank"] is not None)):
        ra = sum(1 for r in rel if test(r)) / len(rel)
        rb = sum(1 for r in irr if test(r)) / len(irr)
        print(f"    {name:<14}{ra:>8.0%}{rb:>12.0%}"
              f"    {'notable' if abs(ra - rb) > 0.25 else ''}")

    # --- THE SPLIT THIS SET WAS ADDED FOR --------------------------------
    print(f"\n{THIN}\n  IRRELEVANCE BY PLAN SHAPE — single search vs fan-out\n{THIN}")
    print(f"  {'plan mode':<24}{'passages':>10}{'irrelevant':>12}{'rate':>8}")
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r.get("plan_mode", "?"), []).append(r)
    for mode, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        bad = sum(1 for r in g if r["verdict"] == "irrelevant")
        print(f"  {mode:<24}{len(g):>10}{bad:>12}{bad / len(g):>8.0%}")
    single = [r for r in rows if r.get("n_facets", 1) <= 1]
    multi = [r for r in rows if r.get("n_facets", 1) > 1]
    if not single or not multi:
        have = "fan-out only" if multi else "single-facet only"
        print(f"\n  ONE SIDE IS UNLABELLED ({have}). The single-vs-fan-out")
        print("  comparison below is NOT AVAILABLE, and no claim about fan-out")
        print("  being worse can be made from this run — there is nothing to")
        print("  compare it against. Label the other half, or say so when")
        print("  quoting these numbers.")
    if single and multi:
        rs = sum(1 for r in single if r["verdict"] == "irrelevant") / len(single)
        rm = sum(1 for r in multi if r["verdict"] == "irrelevant") / len(multi)
        print(f"\n  one facet   {rs:>6.0%} irrelevant   ({len(single)} passages)")
        print(f"  fan-out     {rm:>6.0%} irrelevant   ({len(multi)} passages)")
        print("\n  IF FAN-OUT IS MUCH WORSE, the cheapest fix is not a relevance")
        print("  gate at all — it is tightening the facet margin, which already")
        print("  drops dead slices and needs no new model call. Check that")
        print("  before building anything that reads passages.")
        print("  UNISOLATED EITHER WAY: fan-out questions are also broader, so a")
        print("  higher rate may be the QUESTIONS rather than the fan-out.")

    print(f"\n{LINE}")
    if candidates:
        print(f"{len(candidates)} FEATURE(S) MEET THE RULE FIXED BEFORE THE RUN:")
        for k, cut, lost in candidates:
            print(f"  {k} at {cut:.2f} — middles clear, loses {lost:.0%} of relevant")
        print("\n  NOT A RULE YET. This was chosen ON these 30 questions, so its")
        print("  numbers here are optimistic by construction — the same reason")
        print("  the intent prompt needed a second question set. Write 30 fresh")
        print("  questions and confirm the cut there before building anything.")
    else:
        print("NO FEATURE MEETS THE RULE. The middles overlap, or a cut between")
        print("them costs more than 5% of the relevant passages.")
        print("\n  That is a RESULT, not a failed study. It says the same thing")
        print("  the score gate said in August: these signals do not know what")
        print("  the question was about. Record it and stop paying for it —")
        print("  a relevance gate would then need a signal that reads the")
        print("  QUESTION and the PASSAGE together, which is a different build.")
    print(LINE)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--label", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--chars", type=int, default=1200,
                    help="passage characters shown while labelling")
    ap.add_argument("--only", default="multi",
                    choices=["multi", "named", "all", "inferred", "jurisdiction",
                             "document", "version"],
                    help="which questions to label. DEFAULT 'multi' — the 15 "
                         "fan-out questions, where a finding is most likely and "
                         "so where fresh attention is worth most.")
    ap.add_argument("--labels", default="human",
                    choices=["human", "claude", "agreed"],
                    help="which label pass to analyse. 'human' is yours and is "
                         "the default; 'claude' is the mentor's first pass; "
                         "'agreed' keeps only rows both passes match on.")
    ns = ap.parse_args()
    if ns.collect:
        return collect()
    if ns.label:
        return label(ns.chars, ns.only)
    if ns.analyse:
        return analyse(ns.labels)
    ap.error("choose --collect, --label or --analyse")


if __name__ == "__main__":
    raise SystemExit(main())
