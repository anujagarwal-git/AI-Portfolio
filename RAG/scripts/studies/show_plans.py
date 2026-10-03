"""Print the branch the planner takes for every case in tests/fixtures.

Reading, not deciding: shows WHICH rule fired for each question so a wrong
route is visible. Add -v to see the facets and notes too.

    uv run python scripts/show_plans.py
    uv run python scripts/show_plans.py -v
    uv run python scripts/show_plans.py --mode version_fanout
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

from regrag.retrieval.planner import plan

CASES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "planner_cases.yaml"


def main() -> int:
    verbose = "-v" in sys.argv
    only = sys.argv[sys.argv.index("--mode") + 1] if "--mode" in sys.argv else None

    cases = yaml.safe_load(CASES.read_text(encoding="utf-8"))
    tally: Counter[str] = Counter()
    group = ""

    for c in cases:
        p = plan(c["q"])
        tally[p.mode] += 1
        if only and p.mode != only:
            continue
        if c["id"][0] != group:
            group = c["id"][0]
            print()
        print(f"  {c['id']:<5}{p.mode:<22}{len(p.facets)}  {c['q'][:58]!r}")
        if verbose:
            for f in p.facets:
                print(f"          {f}")
            for n in p.notes:
                print(f"        ! {n}")

    print("\n" + "=" * 70)
    for mode, n in tally.most_common():
        print(f"  {mode:<24}{n:>4}")
    print(f"  {'TOTAL':<24}{sum(tally.values()):>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
