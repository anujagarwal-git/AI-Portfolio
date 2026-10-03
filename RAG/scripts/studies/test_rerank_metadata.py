"""THREE ARMS, ONE VARIABLE BETWEEN B AND C.

    A  no reranking                RRF order
    B  rerank on chunk TEXT        what was measured on 2026-08-23
    C  rerank on LOCATOR + HEADING + TEXT      <- the thing being tested

HYPOTHESIS (Anuj, 2026-08-23): Stage 6 failed not because the model is wrong but
because the input is impoverished — we hand it a bare paragraph and hide every
field that identifies it.

THE FALSIFIER, NAMED BEFORE RUNNING: `what does CRE36.122 require?`.
RRF puts that paragraph in the top five; arm B demotes it out. If arm C ALSO
demotes it with the locator sitting in front of the text, the model cannot use an
exact identifier even when handed it — the hypothesis is refuted and the problem
is the MODEL, not the input. Watch `IRB minimum requirements` too: it was the one
clear win in arm B, and metadata breaking it would be a bad trade either way.

`expect` is a PROBE, not a golden set: a substring we look for in the citations of
a handful of hand-picked questions. There are no labelled chunks and no ground
truth here, and nothing in this file may ever be used to tune retrieval
parameters — the moment it is, it becomes leakage.

    uv run python scripts/test_rerank_metadata.py
"""
from __future__ import annotations

import time

from regrag import config
from regrag.retrieval.reranker import pair_text
from regrag.retrieval.search import retrieve

# (question, substring we hope survives — None where nothing should)
CASES = [
    # ---- the five from 2026-08-23, unchanged ----
    ("what did SR 11-7 say about spreadsheets?",                    "SR 11-7"),
    ("what is model validation?",                                   "SS1/23"),
    ("what does CRE36.122 require?",                                "CRE36.122"),
    ("how often must a bank review its model inventory?",           "Model inventory"),
    ("what are the minimum requirements for the IRB approach?",     "CRE36"),
    # ---- five new probes ----
    # 2nd identifier, different chapter — does the CRE36.122 result generalise?
    ("what does CRE53.54 say about the modelling process?",         "CRE53.54"),
    # the signal is in the HEADING, may be absent from the child text
    ("what are the roles and responsibilities for model risk?",     "Roles and responsibilities"),
    # answer is likely a SHORT list item — does the heading rescue a tiny child?
    ("which asset classes fall under specialised lending?",         "Specialised lending"),
    # must NOT collapse onto one document
    ("what governance is required over model risk management?",     None),
    # NEGATIVE CONTROL — the corpus has no Indian material at all
    ("what are the RBI norms for model risk management?",           None),
]

ARMS = [
    ("A  no rerank",        dict(rerank=False)),
    ("B  text only",        dict(rerank=True, rerank_metadata=False)),
    ("C  text + metadata",  dict(rerank=True, rerank_metadata=True)),
]
W = 104
TRUNC_CHARS = 2_000     # ~500 tokens, roughly where MiniLM cuts


def cites(r):
    return [(h.cite, h.rerank) for pas in r.passages for h in pas.children]


def main() -> int:
    print(f"model {config.RERANK_MODEL} | candidates/facet {config.RERANK_CANDIDATES}\n")
    retrieve("warm up every model", rerank=True)

    # how much does metadata cost in length? measured on real candidates.
    probe = retrieve("what is model validation?", rerank=False)
    pl = [h.payload for pas in probe.passages for h in pas.children]
    b_over = sum(1 for p in pl if len(pair_text(p, metadata=False)) > TRUNC_CHARS)
    c_over = sum(1 for p in pl if len(pair_text(p, metadata=True)) > TRUNC_CHARS)
    grew = sum(len(pair_text(p, metadata=True)) - len(pair_text(p, metadata=False)) for p in pl)
    print(f"TRUNCATION RISK on {len(pl)} real candidates: "
          f"over {TRUNC_CHARS:,} chars  B={b_over}  C={c_over}  "
          f"(metadata adds {grew // max(1, len(pl))} chars each on average)\n")

    summary = []
    for q, expect in CASES:
        print("=" * W); print(q); print("=" * W)
        got = {}
        for label, kw in ARMS:
            t = time.perf_counter(); r = retrieve(q, **kw); el = time.perf_counter() - t
            got[label] = cites(r)
            print(f"  {label:<20}{el * 1000:>6.0f}ms   [{r.plan.mode}]")
            for i, (c, sc) in enumerate(got[label][:6], 1):
                tag = "" if sc is None else f"  ce{sc:+.1f}"
                print(f"     {i}. {c[:64]}{tag}")
            if not got[label]:
                print("     (nothing)")
            print()

        a, b, c = (set(x for x, _ in got[l]) for l, _ in ARMS)
        row = {"q": q, "expect": expect,
               "A": expect is not None and any(expect.lower() in x.lower() for x in a),
               "B": expect is not None and any(expect.lower() in x.lower() for x in b),
               "C": expect is not None and any(expect.lower() in x.lower() for x in c),
               "BvC": len(b & c), "kept": len(a & c)}
        if expect:
            mark = lambda v: "YES" if v else "no "
            print(f"  probe {expect!r:<34} A:{mark(row['A'])} B:{mark(row['B'])} C:{mark(row['C'])}")
        print(f"  B and C share {row['BvC']} of their citations | C keeps {row['kept']} of A's\n")
        summary.append(row)

    print("=" * W); print("SUMMARY — arm C is the change under test"); print("=" * W)
    print(f"  {'question':<50}{'probe':>6}{'A':>4}{'B':>4}{'C':>4}{'B∩C':>6}")
    for r in summary:
        e = "-" if r["expect"] is None else "yes"
        f = lambda v: ("Y" if v else ".") if r["expect"] else "-"
        print(f"  {r['q'][:48]:<50}{e:>6}{f(r['A']):>4}{f(r['B']):>4}{f(r['C']):>4}{r['BvC']:>6}")

    cre = next(r for r in summary if "CRE36.122" == r["expect"])
    print(f"\n  FALSIFIER — CRE36.122 present?  A:{cre['A']}  B:{cre['B']}  C:{cre['C']}")
    if cre["A"] and not cre["B"] and cre["C"]:
        print("  -> HYPOTHESIS SUPPORTED on this case: the locator rescued it.")
    elif cre["A"] and not cre["C"]:
        print("  -> HYPOTHESIS REFUTED on this case: handed the identifier, the model still")
        print("     demoted it. The problem is the MODEL, not the input. Go test BGE.")
    else:
        print("  -> inconclusive on this case — read the lists above before concluding.")
    print("\n  Then judge the rest BY READING. A count of shared citations is not quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
