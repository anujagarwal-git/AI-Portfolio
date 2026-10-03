"""PREVIEW ONLY — what would answer.py's auto-citation do to the saved v2 answers?

Changes nothing. Generates nothing. It reloads the answers already produced by
scripts/test_citations.py, re-runs retrieval (deterministic, so the same
passages come back), strips whatever labels the model wrote, and attaches
citations by MATCHING each sentence back to the passage its words came from.

THE METHOD, AND ITS LIMITS
    For each sentence, take its distinctive words (4+ letters, not stopwords)
    and ask which passage contains most of them. Best passage above the
    threshold gets the citation; nothing above it is marked UNSUPPORTED.

    This is WORD OVERLAP, not meaning. It will:
      - flag a true sentence the model paraphrased heavily        (false alarm)
      - accept a sentence that borrows the right words to say
        the wrong thing                                            (false pass)
    So it is a SCREEN that tells a reviewer where to look. It is not proof, and
    an UNSUPPORTED mark is a question, not a verdict.

    uv run python scripts/preview_autocite.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import textwrap

from regrag.generation import render
from regrag.retrieval.search import retrieve

STOP = set("""the a an of to in for and or is are be as that this which with on by must
should may not from at it its their under section any all such other than when where
have has been were was will can could would each both same these those there here into
over more most less about after before during between within without upon""".split())
THRESHOLD = 0.45          # fraction of a sentence's distinctive words a passage must hold
W = 98


def words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]{4,}", s.lower()) if w not in STOP]


def best_passage(sentence: str, passages: list[str]) -> tuple[int | None, float]:
    ws = words(sentence)
    if len(ws) < 3:
        return None, 0.0                       # too short to attribute honestly
    scores = [sum(w in p.lower() for w in ws) / len(ws) for p in passages]
    i = max(range(len(scores)), key=lambda k: scores[k])
    return (i, scores[i]) if scores[i] >= THRESHOLD else (None, scores[i])


def main() -> int:
    f = sorted(glob.glob("evaluation/citation_run_*.json"))[-1]
    d = json.load(open(f, encoding="utf-8"))
    out, stats = [], {"cited": 0, "unsupported": 0, "sentences": 0}

    out += ["=" * W, "PREVIEW — the SAME v2 answers, with citations attached BY CODE",
            f"source {f}   threshold {THRESHOLD}", "",
            "Labels the model wrote are stripped first. Everything in [] below was",
            "attached by matching the sentence back to a passage.", "=" * W]

    for i, r in enumerate(d["rows"], 1):
        if "v2" not in r:
            continue
        rt = retrieve(r["q"])                    # deterministic; same passages
        _, table = render.render(rt)
        ptexts = [p.text for p in rt.passages]
        cites = [table[f"S{j}"] for j in range(1, len(ptexts) + 1)]

        raw = re.sub(r"\[S\d+\]", "", r["v2"]["text"] or "")
        raw = re.sub(r"\[A fact sentence[^\]]*\]", "", raw)      # v2 echoed the template
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(raw.split())) if s.strip()]

        out += ["", "#" * W, f"[{i}/10] {r['q']}", "#" * W, ""]
        for s in sents:
            k, sc = best_passage(s, ptexts)
            stats["sentences"] += 1
            body = textwrap.fill(s, W - 6, initial_indent="   ", subsequent_indent="   ")
            if k is None:
                stats["unsupported"] += 1
                out += [body, f"        !! UNSUPPORTED  (best overlap {sc:.0%}) — read this one"]
            else:
                stats["cited"] += 1
                out += [body, f"        [{cites[k]}]  ({sc:.0%} overlap)"]
            out.append("")

    out += ["=" * W,
            f"{stats['sentences']} sentences: {stats['cited']} auto-cited, "
            f"{stats['unsupported']} flagged UNSUPPORTED",
            "Compare: the model itself cited 5 of 10 ANSWERS, and one of those labels",
            "pointed at a passage that did not exist.", "=" * W]

    p = pathlib.Path("evaluation/autocite_preview.txt")
    p.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[-5:]))
    print(f"\nfull preview -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
