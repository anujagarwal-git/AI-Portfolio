"""RETRIEVAL SCORED AGAINST THE RELEVANCE LABELS. A soft golden test, not a golden set.

WHY "SOFT"
    A golden set says, for a question, WHICH PASSAGE IS THE RIGHT ONE — written
    before retrieval runs, independent of what retrieval happens to find. It can
    therefore measure RECALL: did we fetch the thing we were supposed to fetch?

    This is the other thing. The labels in relevance_study.json were made on the
    passages RETRIEVAL ALREADY DELIVERED. So every question is graded on its own
    output, and the only question that can be answered is PRECISION:

        of what the model was made to read, how much was worth reading?

    THIS CANNOT SEE A MISS. If retrieval never fetched the right section, no row
    exists for it, so nothing here is marked wrong. A question that quietly
    returned three plausible-but-useless passages scores 0% precision; a question
    that returned nothing at all does not appear. Do not read a good score here
    as "retrieval is fine" — read it as "what it delivered was mostly usable".
    Recall stays unknown until the golden set exists. That is the whole reason
    the golden set is the next stage and not this one.

WHAT IT IS ACTUALLY GOOD FOR
    Deciding whether retrieval needs work BEFORE paying for a golden set. Three
    findings would each change what you build next, and they are different:
      * low HIT RATE  -> questions come back with nothing usable at all. That is
                         a retrieval defect and a golden set will only confirm it
                         more expensively.
      * low PRECISION, high hit rate -> the right passage is in there with junk
                         around it. That is a QUOTA or FACET MARGIN problem, and
                         it has a cheap fix that needs no new model call.
      * junk concentrated in ONE PLAN SHAPE -> fan-out is the culprit, not
                         retrieval as a whole. Tighten the margin, not the floor.

THE BAR, FIXED BEFORE READING THE OUTPUT
    HIT RATE   >= 90%   every question should deliver at least one usable passage
    PRECISION  >= 60%   under this the model reads more noise than evidence
    These are judgement, not measurement. They are written here so the run cannot
    be graded against whatever it happens to produce.

TWO BIASES YOU MUST CARRY INTO EVERY NUMBER
    1. THE LABELLED HALF IS THE HARD HALF. The 15 multi-facet questions were
       labelled first BECAUSE irrelevance was expected there. Scores over that
       subset are pessimistic for the corpus as a whole and say nothing about the
       30 named-document questions.
    2. WHOSE LABELS. `--labels claude` is the mentor's first pass and grades the
       system against the same party that built it. `--labels human` is yours.
       `--labels agreed` keeps only rows you both called the same way: the
       cleanest labels and a biased sample, because the easy calls survive.

ORDER IS DELIVERY ORDER, NOT RERANK ORDER
    P@1 asks about the FIRST PASSAGE THE MODEL READS. On a fan-out plan the
    passages are grouped by facet, so delivery order is not descending rerank on
    21 of the 45 questions. Both are reported; they answer different questions.

    uv run python scripts/soft_golden.py
    uv run python scripts/soft_golden.py --labels agreed
    uv run python scripts/soft_golden.py --by want_mode
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import statistics
import sys

OUT = pathlib.Path("evaluation") / "relevance_study.json"
W = 92
LINE = "=" * W
THIN = "-" * W

HIT_BAR = 0.90           # fixed before the run — see the header
PRECISION_BAR = 0.60


def load(which: str) -> tuple[list[dict], dict]:
    """Rows carrying a usable label, plus coverage facts about the rest."""
    d = json.loads(OUT.read_text(encoding="utf-8"))
    rows = d["rows"]
    if which == "agreed":
        keep = [r for r in rows
                if r.get("verdict") in ("relevant", "irrelevant")
                and r.get("verdict") == r.get("verdict_claude")]
        for r in keep:
            r["_label"] = r["verdict"]
    else:
        key = "verdict_claude" if which == "claude" else "verdict"
        keep = [r for r in rows if r.get(key) in ("relevant", "irrelevant")]
        for r in keep:
            r["_label"] = r[key]

    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["qid"]].append(r)
    cov = collections.Counter()
    for g in byq.values():
        n = sum(1 for r in g if r.get("_label"))
        cov["full" if n == len(g) else ("none" if n == 0 else "partial")] += 1
    return keep, {"all_rows": len(rows), "all_questions": len(byq),
                  "coverage": cov, "collected": d.get("collected")}


def per_question(rows: list[dict]) -> list[dict]:
    """One record per question, over its LABELLED passages only.

    A question whose passages are only half labelled is scored on that half and
    flagged. It is not dropped — dropping it would quietly bias the set toward
    whatever was easy to label.
    """
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["qid"]].append(r)

    out = []
    for qid, g in sorted(byq.items()):
        g = sorted(g, key=lambda r: int(r["label_id"][1:]))     # delivery order
        rel = [r for r in g if r["_label"] == "relevant"]
        first = next((i for i, r in enumerate(g, 1)
                      if r["_label"] == "relevant"), None)
        by_score = sorted(g, key=lambda r: -(r.get("rerank") or -99))
        junk_chars = sum(r["chars"] for r in g if r["_label"] == "irrelevant")
        out.append({
            "qid": qid, "question": g[0]["question"],
            "want_mode": g[0].get("want_mode"), "plan_mode": g[0].get("plan_mode"),
            "n_facets": g[0].get("n_facets", 1),
            "labelled": len(g), "relevant": len(rel),
            "precision": len(rel) / len(g),
            "hit": bool(rel),
            "p_at_1": g[0]["_label"] == "relevant",
            "p_at_1_by_rerank": by_score[0]["_label"] == "relevant",
            "first_relevant": first,
            "rr": 1 / first if first else 0.0,
            "chars": sum(r["chars"] for r in g),
            "junk_chars": junk_chars,
        })
    return out


def bar(x: float, width: int = 22) -> str:
    n = int(round(x * width))
    return "#" * n + "." * (width - n)


def breakdown(qs: list[dict], field: str) -> None:
    print(f"\n{THIN}\n  BY {field.upper()}\n{THIN}")
    print(f"  {'group':<22}{'questions':>10}{'hit':>8}{'precision':>11}"
          f"{'P@1':>8}{'junk chars':>12}")
    groups: dict[str, list[dict]] = {}
    for q in qs:
        groups.setdefault(str(q.get(field)), []).append(q)
    for name, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"  {name:<22}{len(g):>10}"
              f"{sum(q['hit'] for q in g) / len(g):>8.0%}"
              f"{statistics.mean(q['precision'] for q in g):>11.0%}"
              f"{sum(q['p_at_1'] for q in g) / len(g):>8.0%}"
              f"{sum(q['junk_chars'] for q in g):>12,}")
    if len(groups) < 2:
        print("\n  ONE GROUP ONLY — this breakdown compares nothing. It is here")
        print("  so that fact is visible rather than inferred from a tidy table.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="claude",
                    choices=["human", "claude", "agreed"],
                    help="whose pass to grade against. 'claude' is the mentor's "
                         "first pass and grades the system against the party "
                         "that built it — say so when quoting the number.")
    ap.add_argument("--by", action="append", default=[],
                    choices=["want_mode", "plan_mode", "n_facets"],
                    help="extra breakdowns; plan_mode and n_facets always print")
    ap.add_argument("--worst", type=int, default=8,
                    help="how many weakest questions to print in full")
    ap.add_argument("--save", action="store_true")
    ns = ap.parse_args()

    if not OUT.exists():
        sys.exit(f"{OUT} not found — run study_relevance.py --collect first")
    rows, meta = load(ns.labels)

    print(f"{LINE}\nSOFT GOLDEN TEST — retrieval only, labels: {ns.labels}\n{LINE}")
    print(f"  collected {meta['collected']}")
    print(f"  {len(rows)} labelled passage(s) of {meta['all_rows']} delivered, "
          f"across {meta['all_questions']} question(s)")
    c = meta["coverage"]
    print(f"  question coverage: {c['full']} fully labelled, "
          f"{c['partial']} partial, {c['none']} unlabelled")

    if len(rows) < 20:
        sys.exit(f"\n  only {len(rows)} labelled row(s). Label more before "
                 f"reading anything into this — a rate over a handful of "
                 f"passages moves several points per label.")

    qs = per_question(rows)
    print(f"\n  PRECISION ONLY. Every row here was delivered BY retrieval, so a")
    print(f"  passage retrieval never fetched cannot be marked missing. Recall")
    print(f"  is unknown until the golden set exists.")

    hit = sum(q["hit"] for q in qs) / len(qs)
    prec = statistics.mean(q["precision"] for q in qs)
    p1 = sum(q["p_at_1"] for q in qs) / len(qs)
    p1r = sum(q["p_at_1_by_rerank"] for q in qs) / len(qs)
    mrr = statistics.mean(q["rr"] for q in qs)
    junk = sum(q["junk_chars"] for q in qs)
    total = sum(q["chars"] for q in qs)

    print(f"\n{LINE}\nHEADLINE — {len(qs)} question(s)\n{LINE}")
    print(f"  {'hit rate':<30}{bar(hit)} {hit:>6.0%}   "
          f"(bar {HIT_BAR:.0%})  at least one usable passage")
    print(f"  {'mean precision':<30}{bar(prec)} {prec:>6.0%}   "
          f"(bar {PRECISION_BAR:.0%})  share of delivered that was usable")
    print(f"  {'P@1 (delivery order)':<30}{bar(p1)} {p1:>6.0%}   "
          f"           first passage the model reads")
    print(f"  {'P@1 (rerank order)':<30}{bar(p1r)} {p1r:>6.0%}   "
          f"           highest-scoring passage")
    print(f"  {'MRR':<30}{bar(mrr)} {mrr:>6.2f}   "
          f"           1/rank of the first usable one")
    print(f"\n  wasted context: {junk:,} of {total:,} chars "
          f"({junk / total:.0%}) — that is prompt the model paid to read and")
    print(f"  should not have. On an 11k-token prompt at 77 tok/s it is seconds "
          f"per answer.")

    # --- the verdict, against the bar written before the run --------------
    print(f"\n{LINE}\nVERDICT\n{LINE}")
    if hit < HIT_BAR:
        print(f"  HIT RATE {hit:.0%} IS BELOW {HIT_BAR:.0%}. Questions are coming")
        print("  back with nothing usable. That is a retrieval defect, and a")
        print("  golden set will confirm it more expensively than fixing it")
        print("  will. Read the zero-hit questions below before tuning anything.")
    elif prec < PRECISION_BAR:
        print(f"  HIT RATE {hit:.0%} IS FINE; PRECISION {prec:.0%} IS NOT. The")
        print("  right passage is being found and shipped with junk around it.")
        print("  That is a QUOTA or FACET MARGIN problem — both cheap, neither")
        print("  needs a new model call. Check the plan-shape table first: if")
        print("  the junk is concentrated in fan-outs, the margin is the fix.")
    else:
        print(f"  BOTH BARS MET ({hit:.0%} hit, {prec:.0%} precision) ON THE")
        print("  LABELLED SUBSET. That is a reason to move to the golden set,")
        print("  NOT a finding that retrieval is good: this test cannot see a")
        print("  miss, and the labelled half is the half where irrelevance was")
        print("  expected. The golden set exists to measure what this cannot.")

    for f in ["plan_mode", "n_facets"] + [x for x in ns.by
                                          if x not in ("plan_mode", "n_facets")]:
        breakdown(qs, f)

    # --- the questions worth reading --------------------------------------
    weak = sorted(qs, key=lambda q: (q["hit"], q["precision"]))[:ns.worst]
    print(f"\n{THIN}\n  WEAKEST {len(weak)} QUESTION(S) — read these, do not tune "
          f"on the averages\n{THIN}")
    for q in weak:
        flag = "  NO USABLE PASSAGE" if not q["hit"] else ""
        print(f"  [{q['qid']:>2}] {q['precision']:>4.0%}  "
              f"{q['relevant']}/{q['labelled']} usable   "
              f"{q['plan_mode']}{flag}")
        print(f"       {q['question']}")

    if ns.save:
        f = OUT.with_name(f"soft_golden_{ns.labels}.json")
        f.write_text(json.dumps(
            {"labels": ns.labels, "collected": meta["collected"],
             "bars": {"hit": HIT_BAR, "precision": PRECISION_BAR},
             "headline": {"questions": len(qs), "hit_rate": round(hit, 3),
                          "precision": round(prec, 3), "p_at_1": round(p1, 3),
                          "p_at_1_by_rerank": round(p1r, 3),
                          "mrr": round(mrr, 3), "junk_chars": junk,
                          "total_chars": total},
             "questions": qs}, indent=1), encoding="utf-8")
        print(f"\n  raw -> {f}")

    print(f"\n{LINE}")
    print("NEXT: this measures PRECISION on delivered passages. The golden set")
    print("measures RECALL against passages named in advance. They are not")
    print("substitutes and a good score here does not buy you out of that one.")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
