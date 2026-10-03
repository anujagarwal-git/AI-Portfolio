"""WHICH PARAGRAPH DID THIS CLAIM COME FROM? Sentence-to-paragraph alignment.

THE DEFECT THIS FIXES (graded against the parsed PDFs.)
    `compose_citation` cited `p.children[0]` — the child that WON RETRIEVAL.
    But the model reads the whole parent WINDOW and draws claims from anywhere
    inside it. So the citation named the chunk the search found while the claim
    came from a different paragraph. Verified: an answer citing CRE53.56 drew
    its five claims from CRE53.50, CRE53.51 and CRE53.55.

SCORING — IDF-weighted overlap, deliberately not raw word overlap
    Rare words carry the signal. "the bank must" appears in every paragraph;
    "backtesting" appears in one. Tokens are weighted by inverse frequency
    ACROSS THE PARAGRAPHS OF THAT PARENT — computed per passage, no corpus
    statistics, no model, no stemming. It is BM25's idf idea without term
    saturation or length normalisation, because there is nothing here to
    normalise: one claim against a dozen paragraphs of similar size.

THE FLOOR IS MEASURED, NOT CHOSEN BY FEEL
    0.65 comes from the 20-question run of 2026-09-02: every correct
    re-citation scored >= 0.731, every wrong one <= 0.574, and nothing landed
    between. 0.65 sits in the middle of that empty band rather than on its
    edge. 
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from regrag import config

TOKEN = re.compile(r"[a-z0-9]+")
STOP = set("""the a an of to in for and or is are be as that this which with on by must
should may not from at it its their under section any all such other than when where have
has been were was will can could would each both same these those there here into over more
most less about after before during between within without upon shall may bank banks""".split())

RX_PREFIX = re.compile(r"([A-Z]{2,6})\d+\.\d+[a-z]?$")
RX_MARKER = re.compile(r"(\d+\.\d+[a-z]?)\b")

MIN_UNIT_CHARS = 40          # below this it is a heading or a stray fragment


@dataclass(frozen=True)
class Unit:
    """One paragraph of the delivered passage, with the locator it announces."""
    locator: str             # "" when the paragraph carries no usable marker
    text: str


@dataclass(frozen=True)
class Alignment:
    verdict: str             # MATCH | FLOOR | TIE | NOLOC | EMPTY
    locator: str             # never "" — falls back to what was shipped
    best: float
    runner_up: float

    @property
    def moved(self) -> bool:
        return self.verdict == "MATCH"


def tokens(s: str) -> list[str]:
    return [t for t in TOKEN.findall(s.lower()) if len(t) > 2 and t not in STOP]


def build_idf(texts: list[str]) -> dict[str, float]:
    """Rare-within-this-parent words weigh more. No corpus statistics needed."""
    n = len(texts) or 1
    df: Counter[str] = Counter()
    for t in texts:
        df.update(set(tokens(t)))
    return {t: math.log((n + 1) / (d + 0.5)) for t, d in df.items()}


def similarity(claim: str, unit: str, idf: dict[str, float]) -> float:
    """Weighted fraction of the CLAIM's tokens present in this paragraph."""
    ct = tokens(claim)
    if not ct:
        return 0.0
    have = set(tokens(unit))
    num = sum(idf.get(t, 1.0) for t in ct if t in have)
    den = sum(idf.get(t, 1.0) for t in ct)
    return num / den if den else 0.0


def paragraph_units(passage) -> list[Unit]:
    """Split the DELIVERED text into paragraphs and label each from its marker.

    The body text prints "53.50" while a citation reads "CRE53.50", so the
    chapter prefix is taken from a matched child's locator. A paragraph with no
    marker, or a passage with no usable prefix, yields a unit with an empty
    locator — which NOLOC then turns into "keep what shipped".
    """
    prefix = ""
    for h in passage.children:
        m = RX_PREFIX.fullmatch((h.payload.get("locator") or "").strip())
        if m:
            prefix = m.group(1)
            break

    units: list[Unit] = []
    for para in re.split(r"\n\s*\n", passage.text):
        para = para.strip()
        if len(para) < MIN_UNIT_CHARS:
            continue
        m = RX_MARKER.match(para)
        units.append(Unit(f"{prefix}{m.group(1)}" if (m and prefix) else "", para))

    # SR letters, BCBS papers and FAQ bins number nothing, so every unit comes
    # back with an empty locator and the aligner can only ever say NOLOC. The
    # matched children DO carry locators ("Footnotes", "Principle 1.2"), so
    # they are the better units in that case.
 
    if any(u.locator for u in units):
        return units
    kids = [Unit((h.payload.get("locator") or ""), (h.payload.get("text") or ""))
            for h in passage.children if (h.payload.get("locator") or "").strip()]
    return kids or units


def align(claim: str, units: list[Unit], current: str, *,
          floor: float | None = None, margin: float | None = None) -> Alignment:
    """Best-matching paragraph for `claim`, or `current` when unconvinced."""
    floor = config.ALIGN_FLOOR if floor is None else floor
    margin = config.ALIGN_MARGIN if margin is None else margin

    if not units:
        return Alignment("EMPTY", current, 0.0, 0.0)

    idf = build_idf([u.text for u in units])
    scored = sorted(((similarity(claim, u.text, idf), u) for u in units),
                    key=lambda x: -x[0])
    best, top = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0

    if best < floor:
        return Alignment("FLOOR", current, best, second)
    if len(scored) > 1 and (best - second) < margin:
        return Alignment("TIE", current, best, second)
    if not top.locator:
        return Alignment("NOLOC", current, best, second)
    return Alignment("MATCH", top.locator, best, second)
