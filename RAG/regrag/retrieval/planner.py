"""Turn ONE question into a SEARCH PLAN. Runs no search of its own.

THE RULE, in one line: AT MOST ONE AXIS FANS OUT. Never more than 3 facets.

    1. two or more DOCUMENTS named   -> one search each
    2. two or more JURISDICTIONS named -> one search each
    3. old-vs-new wording            -> one search for current, one for superseded
    4. exactly one DOCUMENT named    -> one search, filtered to that document ONLY
    5. exactly one JURISDICTION named -> one search, that jurisdiction, current
    6. a SUBJECT recognised, nothing else -> one search per jurisdiction, current
    7. nothing recognised            -> one search per jurisdiction, current

WHY FAN OUT AT ALL — measured, not assumed
    SR 11-7 uses "validat*" ~101 times in 21 pages; Basel CRE ~39 times in 323.
    One pooled top-k hands the whole budget to the densest document, and
    `context_precision` cannot report the loss — it is computed only over what
    WAS retrieved. A fixed share per facet is the only guarantee.

WHY RULES AND NOT AN LLM
    Generation here is not reproducible at temperature 0 (identical retrieved
    context, two different responses, 2026-08-19). An LLM planner would make the
    pipeline non-reproducible at step ONE, evaluation included. Rules are
    brittle instead — so every facet records the words that triggered it.

KNOWN LIMITS
    - Document aliases are generated from the registry and are deliberately
      loose ("SR 11" finds SR 11-7). Loose means false matches are possible;
      the `why` field always names the alias that fired.
    - The query text is identical in every facet. Only the filter differs.
    - 12 CFR 217 is a known_gap, not an indexable row, so it has no alias and
      will not be detected. Refusal for it belongs at generation time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from regrag import config
from regrag.domain import Jurisdiction, Status, Subject
from regrag.registry import load as load_registry

CURRENT: tuple[str, ...] = (Status.IN_FORCE.value, Status.PARTIALLY_REVISED.value)
HISTORIC: tuple[str, ...] = (Status.SUPERSEDED.value,)
DEFAULT_JURISDICTIONS = (Jurisdiction.US, Jurisdiction.UK, Jurisdiction.GLOBAL)

# ---------------------------------------------------------------------------
# TERM TABLES
# ---------------------------------------------------------------------------
SUBJECT_TERMS: dict[Subject, tuple[str, ...]] = {
    Subject.MODEL_RISK_MANAGEMENT: (
        "model risk", "model validation", "validation", "validate", "validating",
        "model inventory", "model governance", "model development", "model tiering",
        "effective challenge", "backtest", "backtesting", "back-testing",
        "benchmarking", "model owner", "model use", "model performance",
        "outcomes analysis", "challenger model", "challenger models",
        "revalidate", "revalidated", "revalidation", "model",
    ),
    Subject.CAPITAL_ADEQUACY: (
        "capital", "capital adequacy", "risk weight", "risk weights", "risk-weight",
        "rwa", "tier 1", "tier 2", "irb", "internal ratings", "standardised approach",
        "standardized approach", "exposure at default", "ead", "lgd", "pd",
        "leverage ratio", "large exposure", "securitisation", "securitization",
    ),
    Subject.CREDIT_IMPAIRMENT: (
        "impairment", "ecl", "expected credit loss", "expected credit losses",
        "provision", "provisions", "provisioning", "stage 2", "stage 3", "sicr",
        "significant increase in credit risk", "non-performing", "npa", "npl",
        "problem asset", "problem assets", "write-off", "forbearance",
    ),
    Subject.DATA_QUALITY: (
        "data quality", "data governance", "data aggregation", "risk data",
        "lineage", "data architecture", "accuracy and integrity", "completeness",
    ),
    Subject.STRESS_TESTING: (
        "stress test", "stress tests", "stress-test", "stress testing",
        "stress scenario", "ccar", "capital plan", "capital planning",
        "adverse scenario", "severely adverse", "scenario design", "baseline scenario",
    ),
}

JURISDICTION_TERMS: dict[Jurisdiction, tuple[str, ...]] = {
    Jurisdiction.US: (
        "us", "u.s.", "usa", "united states", "american", "federal reserve",
        "fed", "occ", "fdic", "cfr", "regulation y", "regulation yy", "reg yy",
    ),
    Jurisdiction.UK: (
        "uk", "u.k.", "united kingdom", "british", "pra", "bank of england",
    ),
    Jurisdiction.GLOBAL: ("basel", "bcbs", "bis", "international", "iasb"),
}

# STRONG: unambiguously points at a version other than the current one.
# Fires on its own.
STRONG_VERSION_TERMS = (
    "superseded", "replaced", "replaces", "no longer", "used to", "formerly",
    "former", "previously", "previous", "prior", "earlier", "originally", "old",
    "what changed", "what's changed", "what has changed", "has changed",
    "have changed", "had changed", "was changed", "were changed",
)

# WEAK: means "difference" but not necessarily between VERSIONS. Needs a second
# signal — a word naming the guidance itself — before it may fan out.
# "what changes are required to the model inventory?" is a question about
# current requirements, and must NOT drag superseded guidance into the answer.

WEAK_VERSION_TERMS = (
    "change", "changed", "changes", "changing", "revised", "revision",
    "amended", "amendment", "differ", "differs", "difference", "differences",
    "compare", "compared", "comparison", "contrast", "versus", "vs",
)

DOC_WORDS = (
    "guidance", "guideline", "guidelines", "rule", "rules", "regulation",
    "regulations", "standard", "standards", "statement", "version", "edition",
    "framework",
)

# Words about RECENCY, not about change. "what is the latest guidance?" wants
# the current document ONLY. These must never trigger the version fan-out.
RECENCY_TERMS = ("new", "newer", "latest", "current", "updated", "recent", "today")

assert not (set(RECENCY_TERMS) & (set(STRONG_VERSION_TERMS) | set(WEAK_VERSION_TERMS))), \
    "a recency word leaked into the version tables — it would pull superseded " \
    "documents into a question that asked for the current one"


def _matches(text: str, terms) -> list[str]:
    """Terms present in `text`, matched on whole words.

    Word boundaries are load-bearing: a substring match on "us" fires inside
    "thus", "because" and "must", which would tag nearly every regulatory
    question as US-scoped.
    """
    return [t for t in terms
            if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", text)]


# ---------------------------------------------------------------------------
# DOCUMENT ALIASES — generated FROM THE REGISTRY, never hand-listed here.
#
# People type "SR 11", "SR 11-7", "sr11-7". All three must find the same row.
# Generating from the registry means a new document gets aliases for free and
# a renamed one cannot go stale — the failure mode a hand-written table has.
# ---------------------------------------------------------------------------
_CODE_LIKE = re.compile(r"^[a-z]{2,4}\d{0,3}$")


def _aliases_for(short_name: str) -> set[str]:
    base = short_name.lower().strip()
    out: set[str] = set()

    inside = re.findall(r"\(([^)]+)\)", base)          # "(PAP)", "(credit risk extract)"
    stem = re.sub(r"\s*\([^)]*\)", "", base).strip()   # short_name minus the bracket
    out.add(stem)
    for got in inside:
        got = got.strip()
        if " " not in got and len(got) >= 3:
            out.add(got)

    out.add(re.sub(r"[^a-z0-9]", "", stem))            # sr117, ss123, bcbs239

    # "SR 11-7" -> "sr 11", "sr11".  "SS1/23" -> "ss1", "ss 1"
    m = re.match(r"^([a-z]{2,4})\s*(\d+)\s*[-/]\s*(\d+)$", stem)
    if m:
        out.update({f"{m.group(1)} {m.group(2)}", f"{m.group(1)}{m.group(2)}"})

    # "Basel CRE" -> "cre";  "BCBS d403" -> "d403"
    parts = stem.split()
    if len(parts) == 2 and parts[0] in {"basel", "bcbs"} and _CODE_LIKE.match(parts[1]):
        out.add(parts[1])

    # "12 CFR Part 252" -> "12 cfr 252", "part 252"
    if "cfr" in stem:
        out.add(stem.replace(" part ", " "))
        m = re.search(r"(part\s+\d+[\.\d]*)", stem)
        if m:
            out.add(m.group(1))

    return {a for a in out if len(a) >= 3}


@lru_cache(maxsize=1)
def _doc_index() -> tuple[tuple[str, str, str, re.Pattern], ...]:
    """(doc_id, short_name, alias, compiled pattern) for every indexable row."""
    rows = []
    for row in load_registry().indexable():
        for alias in _aliases_for(row.short_name):
            # A bare Basel/BCBS code may carry a chapter number: cre, cre36, cre 36.
            tail = r"\s?\d{0,3}" if _CODE_LIKE.match(alias) and " " not in alias else ""
            pat = re.compile(rf"(?<![a-z0-9]){re.escape(alias)}{tail}(?![a-z0-9])")
            rows.append((row.doc_id, row.short_name, alias, pat))

    # *** AN ALIAS OWNED BY TWO DOCUMENTS IDENTIFIES NEITHER. ***
    #
    # The shortening itself is worth keeping: "sr 11" is a legitimate handle
    # for SR 11-7 because no other document answers to it. So the rule is
    # AMBIGUITY, not shortening - drop an alias only where a second document
    # generates the identical string. Against the current registry that
    # removes exactly two aliases ("sr 15", "sr15") and leaves every one of
    # the 19 documents with at least one unique alias, so nothing becomes
    # unnameable. A future registry addition that collides is handled for
    # free, which a hand-written exception for SR 15 would not be.
    owners: dict[str, set[str]] = {}
    for doc_id, _sn, alias, _pat in rows:
        owners.setdefault(alias, set()).add(doc_id)
    rows = [r for r in rows if len(owners[r[2]]) == 1]

    # longest alias first, so "sr 11-7" is reported rather than "sr 11"
    return tuple(sorted(rows, key=lambda r: -len(r[2])))


def _documents_in(text: str) -> dict[str, tuple[str, str]]:
    """doc_id -> (short_name, the alias that matched). One entry per document."""
    found: dict[str, tuple[str, str]] = {}
    for doc_id, short_name, alias, pat in _doc_index():
        if doc_id not in found and pat.search(text):
            found[doc_id] = (short_name, alias)
    return found


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Facet:
    """ONE search: a filter, a fixed share of the budget, and its reason."""

    label: str
    query: str
    where: dict[str, object]
    quota: int
    why: str

    def __str__(self) -> str:
        filt = ", ".join(f"{k}={v}" for k, v in sorted(self.where.items())) or "no filter"
        return f"{self.label:<28} quota {self.quota:<3} [{filt}]  <- {self.why}"


@dataclass(frozen=True)
class Plan:
    question: str
    mode: str
    facets: tuple[Facet, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    # WHY A FIELD AND NOT A STRING MATCH ON `notes`: the notes are prose written
    # for a human. Matching them makes a rule depend on wording nobody thinks of
    # as an interface. A field carries the fact independently of how it is
    # described - the same reason sections.py prefers Docling's layout labels
    # over text regexes.
    #
    # True  -> the split is OUR guess; a facet with nothing on topic is waste
    #          and may be dropped (config.FACET_DROP_MARGIN).
    # False -> the user named the slices; coverage is what they asked for, and
    #          dropping one answers a different question.
    inferred_fanout: bool = False

    @property
    def budget(self) -> int:
        return sum(f.quota for f in self.facets)

    @property
    def fans_out(self) -> bool:
        return len(self.facets) > 1

    def __str__(self) -> str:
        head = (f"PLAN [{self.mode}] for {self.question!r}\n"
                f"  {len(self.facets)} facet(s), {self.budget} chunks")
        body = "\n".join(f"  - {f}" for f in self.facets)
        tail = "\n".join(f"  ! {n}" for n in self.notes)
        return "\n".join(p for p in (head, body, tail) if p)


def plan(question: str,*,quota: int | None = None, requirement_only: bool = False,) -> Plan:
    """Read the question, decide what searches to run. At most one axis fans out.
    `subject` is never a filter. It is a DOCUMENT-level registry label, so it
    describes the document, not the chunk 
    """
    q = question.lower()
    quota = config.FACET_QUOTA_DEFAULT if quota is None else quota
    notes: list[str] = []

    subjects = {s: h for s, t in SUBJECT_TERMS.items() if (h := _matches(q, t))}
    jurisdictions = {j: h for j, t in JURISDICTION_TERMS.items() if (h := _matches(q, t))}
    documents = _documents_in(q)

    strong = _matches(q, STRONG_VERSION_TERMS)
    weak = _matches(q, WEAK_VERSION_TERMS)
    docwords = _matches(q, DOC_WORDS)
    recency = _matches(q, RECENCY_TERMS)
    if not config.WEAK_VERSION_TERMS_ENABLED:
        weak = []          # experiment switch — see config.py
    version_question = bool(strong) or bool(weak and docwords)

    base: dict[str, object] = {}
    if requirement_only:
        base["doc_type"] = list(config.REQUIREMENT_DOC_TYPES)
        notes.append("requirement lookup — guidance and reference data excluded")

    if subjects:
        notes.append("topic read as " + ", ".join(
            f"{s.value} ({h[0]})" for s, h in subjects.items()) + " — recorded, NOT filtered")
    if recency and not version_question:
        notes.append(f"recency wording {recency} — read as 'the current one', "
                     f"not as 'what changed'")

    def facet(label, where, why):
        w = dict(base); w.update(where)
        return Facet(label, question, w, quota, why)

    # -- 1. two or more documents named ------------------------------------
    if len(documents) >= 2:
        notes.append("more than one document named — one search each; this also "
                     "separates their versions, so no status split is needed")
        facets = [facet(sn, {"doc_id": did}, f"named as {alias!r}")
                  for did, (sn, alias) in documents.items()]
        return Plan(question, "document_fanout", tuple(facets), tuple(notes))

    # -- 2. two or more jurisdictions named --------------------------------
    if len(jurisdictions) >= 2:
        notes.append("more than one jurisdiction named — one search each")
        facets = [facet(f"{j.value}+current",
                        {"jurisdiction": j.value, "status": list(CURRENT)},
                        f"named: {h[:2]}")
                  for j, h in jurisdictions.items()]
        return Plan(question, "jurisdiction_fanout", tuple(facets), tuple(notes))

    # -- 3. old vs new -----------------------------------------------------
    if version_question:
        notes.append(f"version question — {'strong wording ' + str(strong[:3]) if strong else 'change wording ' + str(weak[:2]) + ' with document words ' + str(docwords[:2])}; "
                     f"searching each version separately so neither can crowd out the other")
        where_extra: dict[str, object] = {}
        if len(jurisdictions) == 1:
            j = next(iter(jurisdictions))
            where_extra["jurisdiction"] = j.value
        facets = [facet("current", {**where_extra, "status": list(CURRENT)}, "the version in force"),
                  facet("historic", {**where_extra, "status": list(HISTORIC)}, "the version it replaced")]
        return Plan(question, "version_fanout", tuple(facets), tuple(notes))

    # -- 4. exactly one document named -> that document, nothing else ------
    if len(documents) == 1:
        did, (sn, alias) = next(iter(documents.items()))
        notes.append(f"one document named and no comparison asked for — searching "
                     f"{sn} only, with NO status filter, so its own version is whatever "
                     f"the document is")
        return Plan(question, "single_document",
                    (facet(sn, {"doc_id": did}, f"named as {alias!r}"),), tuple(notes))

    # -- 5. exactly one jurisdiction ---------------------------------------
    if len(jurisdictions) == 1:
        j, h = next(iter(jurisdictions.items()))
        return Plan(question, "single_jurisdiction",
                    (facet(f"{j.value}+current",
                           {"jurisdiction": j.value, "status": list(CURRENT)},
                           f"named: {h[:2]}"),), tuple(notes))

    # -- 6. a subject and nothing else -> spread across the legal families --
    if subjects:
        notes.append("a topic but no jurisdiction — fanning out over US / UK / GLOBAL "
                     "so no one legal family can take the whole budget")
        facets = [facet(f"{j.value}+current",
                        {"jurisdiction": j.value, "status": list(CURRENT)},
                        "no jurisdiction named — guaranteed share")
                  for j in DEFAULT_JURISDICTIONS]
        return Plan(question, "jurisdiction_fanout", tuple(facets), tuple(notes),
                    inferred_fanout=True)

    # -- 7. nothing recognised -> STILL fan out -----------------------------
    # Knowing LESS must widen the search, never narrow it. One pooled search
    # over eighteen current documents is the weakest thing this module can do:
    # Basel CRE holds 2,941 chunks against SR 26-2's 44, so the densest document
    # takes the whole budget on a question we already failed to read.
    # The mode still reads `fallback`, so the term-table miss stays visible.
    notes.append("nothing recognised — fanning out over US / UK / GLOBAL anyway. "
                 "A term-table miss WIDENS the search, it never empties it.")
    facets = [facet(f"{j.value}+current",
                    {"jurisdiction": j.value, "status": list(CURRENT)},
                    "no term matched — guaranteed share")
              for j in DEFAULT_JURISDICTIONS]
    return Plan(question, "fallback", tuple(facets), tuple(notes),
                inferred_fanout=True)
