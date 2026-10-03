"""THE INTENT GATE. One small model in front of retrieval.

WHAT IT IS FOR
    Four things happen to a question before the pipeline sees it:
      OUT_OF_DOMAIN  refuse, one generic sentence
      INCOMPLETE     ask ONE clarifying question instead of answering
      OUT_OF_CORPUS  decline, one generic sentence
      ANSWERABLE     pass through UNCHANGED to retrieve() and answer()


    WHY A MODEL EXTRACTS AND CODE DECIDES — THE ONE DESIGN CHOICE THAT MATTERS
    Asking a small model "can this corpus answer the question?" was measured
    twice and failed twice: a prompt refusal rule refused 10 of 10 ANSWERABLE
    questions (2026-08-26), and prompt A of the intent experiment refused
    almost everything (2026-09-02, 14/15 out-of-corpus caught while keeping
    1/15 answerable). A model does not know what is in the corpus.

    So the model is asked only what it can read off the sentence — subject,
    jurisdiction, a named document. `decide()` then answers the corpus question
    from `registry.yaml`, where an empty (subject x jurisdiction) cell is a
    FACT, not an estimate. 

    NOT A RELEVANCE CHECK. Nothing here reads a passage. A question that passes
    the gate can still retrieve on-topic-looking passages that do not answer it.

    KNOWN OPEN — do not "fix" this without re-running both sets
    THE DOCUMENT MATCHER IS TOO LOOSE. `_names_hit` matches a bare substring,
    so "Basel" matches "Basel CAP" and a question about an ABSENT Basel volume
    is called ANSWERABLE. It accounts for the single OUT_OF_CORPUS -> ANSWERABLE
    leak on each set. It is shipped AS MEASURED on purpose: tightening it now
    would mean the numbers above no longer describe this code. Fix it as its own
    change, with its own before/after run.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from regrag import config, registry
from regrag.generation import llm

DECISIONS = ("OUT_OF_DOMAIN", "INCOMPLETE", "OUT_OF_CORPUS", "ANSWERABLE")

MESSAGES = {
    "OUT_OF_DOMAIN":
        "I only answer questions about the banking regulations in this corpus "
        "(Basel, IFRS 9, SR 11-7, SS1/23, 12 CFR and related supervisory "
        "material). I cannot help with this one.",
    "OUT_OF_CORPUS":
        "This corpus does not hold a source that answers that. I will not "
        "answer from a document that does not govern the point you asked "
        "about.",
}

@dataclass(frozen=True)
class Intent:
    """What the gate decided, and everything needed to audit the decision."""

    question: str
    decision: str                  # one of DECISIONS
    reason: str                    # WHY, in code's words — for logs, not users
    message: str = ""              # what a person reads; "" when ANSWERABLE
    extraction: dict = field(default_factory=dict)   # the model's raw JSON
    seconds: float = 0.0
    config_hash: str = ""

    @property
    def passes(self) -> bool:
        return self.decision == "ANSWERABLE"

    def __str__(self) -> str:
        return f"[{self.decision}] {self.reason}"


# ---------------------------------------------------------------------------
# THE CORPUS GRID — built from registry.yaml so it cannot drift from the index
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def corpus_grid() -> tuple[dict, tuple[str, ...], tuple[str, ...], dict]:
    """(subject, jurisdiction) -> short_names, plus the controlled vocabularies.

    `indexable()` and not `documents`: an unverified row is refused by the
    indexer, so it is not in the corpus, so the gate must not treat it as
    coverage. Using the wrong list here would make the gate promise a document
    the retriever cannot reach.
    """
    docs = registry.load_cached().indexable()
    g: dict[tuple[str, str], list[str]] = {}
    for d in docs:
        su = d.subject.value if hasattr(d.subject, "value") else str(d.subject)
        ju = d.jurisdiction.value if hasattr(d.jurisdiction, "value") else str(d.jurisdiction)
        g.setdefault((su, ju), []).append(d.short_name)
    subjects = tuple(sorted({k[0] for k in g}))
    jurisdictions = tuple(sorted({k[1] for k in g}))
    names = {d.short_name.lower(): d.short_name for d in docs}
    return g, subjects, jurisdictions, names


# THE STATUS VIEW - the same documents, bucketed by version instead of by
# jurisdiction. `corpus_grid()` is keyed (subject, jurisdiction) and is used in
# a dozen places, so status gets its OWN cached view rather than a fifth tuple
# member that every caller would have to unpack.
#
# `indexable()` for the same reason as the grid: an unverified row is refused by
# the indexer, so it is not in the corpus, so the gate must not offer it.
@lru_cache(maxsize=1)
def status_view() -> tuple[dict, tuple[str, ...], tuple[str, ...]]:
    """((subject, jurisdiction, bucket) -> short_names, CURRENT, HISTORIC).

    The two buckets are the planner's own, imported rather than re-typed, so a
    status added to `Status` cannot mean one thing to the planner and another
    to the gate.
    """
    from regrag.retrieval.planner import CURRENT, HISTORIC

    docs = registry.load_cached().indexable()
    v: dict[tuple[str, str, str], list[str]] = {}
    for d in docs:
        st = d.status.value if hasattr(d.status, "value") else str(d.status)
        if st in CURRENT:
            bucket = "current"
        elif st in HISTORIC:
            bucket = "historic"
        else:
            continue          # a status in neither list is not a version fact
        su = d.subject.value if hasattr(d.subject, "value") else str(d.subject)
        ju = d.jurisdiction.value if hasattr(d.jurisdiction, "value") else str(d.jurisdiction)
        v.setdefault((su, ju, bucket), []).append(d.short_name)
    return v, CURRENT, HISTORIC

# Words that appear in several short_names and therefore identify NONE of them
# on their own. Derived from the registry at import time rather than typed, so
# adding "Basel XYZ" cannot leave a hand-written list behind.
@lru_cache(maxsize=1)
def _ambiguous_tokens() -> frozenset[str]:
    _g, _s, _j, names = corpus_grid()
    seen: dict[str, int] = {}
    for low in names:
        for tok in set(re.findall(r"[a-z0-9]+", low)):
            seen[tok] = seen.get(tok, 0) + 1
    return frozenset(t for t, n in seen.items() if n > 1)

# only a document we can NAME and do not hold may produce OUT_OF_CORPUS. 
# KEPT DELIBERATELY SMALL. Every word added here is a word that can no longer
# trigger a decline, so the list holds descriptors and nothing else. It must
# never grow to cover a real title.
_GENERIC_DOC_WORDS = frozenset("""
    guidance guidelines guideline supervisory supervision regulator regulators
    regulation regulations regulatory rule rules rulebook requirement
    requirements standard standards framework frameworks guide handbook letter
    letters circular circulars policy policies statement statements principle
    principles document documents paper papers consultation directive act law
    laws code manual bulletin notice expectation expectations text texts
    version versions earlier previous prior current
    the a an of for on and or
""".split())


@lru_cache(maxsize=1)
def _generic_tokens() -> frozenset[str]:
    """Descriptors, plus every word that names a JURISDICTION or a REGULATOR.

    The second half comes from the planner so the two modules cannot disagree:
    "Bank of England" and "Federal Reserve" name a supervisor, not one of its
    documents, and a question that names only a supervisor must reach the
    (subject x jurisdiction) grid rather than a decline.
    """
    from regrag.retrieval.planner import JURISDICTION_TERMS

    out = set(_GENERIC_DOC_WORDS)
    for terms in JURISDICTION_TERMS.values():
        for t in terms:
            out.update(re.findall(r"[a-z0-9]+", t))
    return frozenset(out)


def document_seen(doc: str, question: str) -> bool:
    """Did the QUESTION actually contain the document the model extracted as JSON output?

    This function exists so that an INVENTED name cannot cost the answer.
    """
    content = set(re.findall(r"[a-z0-9]+", doc.lower())) - _generic_tokens()
    if not content:
        return False
    q = question.lower()
    return all(re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", q)
               for t in content)


def _names_hit(doc: str, names: dict) -> tuple[str, str]:
    """Does the question name a document we hold — SPECIFICALLY?

    "Basel" matches "Basel CAP" and a question like "What are the Basel liquidity monitoring tools?"
    was called ANSWERABLE. The corpus holds six Basel volumes and LCR is not
    one of them.

    THE RULE NOW: a match must rest on at least one token that identifies ONE
    document. "basel" appears in six short_names, so it identifies none of them
    and is ignored; "cre" appears in one, so "Basel CRE" matches. A question
    naming only the family falls through to the subject x jurisdiction check,
    which is where an absent volume is correctly declined.

    FOUR OUTCOMES, because only ONE of them is a decline:
      EXACT   a document we hold -> ANSWERABLE
      FAMILY  the question named only a shared word. NOT a decline: "What does
              Basel say about capital?" names no volume, but CAPITAL_ADEQUACY x
              GLOBAL is populated and the question is answerable. Fall through
              to the grid, exactly as if no document had been named.

      ABSENT  a real name we do not hold ("Solvency II") -> OUT_OF_CORPUS
      NONE    nothing that can be a title: an empty field, or descriptors only
              ("US guidance"). A FAILED EXTRACTION, not a decline. Falls
              through to the grid exactly like FAMILY. See _GENERIC_DOC_WORDS
              for the 2026-09-09 evidence.
                  """
    ambiguous = _ambiguous_tokens()
    doc_tokens = set(re.findall(r"[a-z0-9]+", doc))
    if not doc_tokens:
        return "NONE", ""

    best: tuple[int, str] | None = None
    family = False
    for low, full in names.items():
        name_tokens = set(re.findall(r"[a-z0-9]+", low))
        shared = doc_tokens & name_tokens
        if not shared:
            continue
        if not (shared - ambiguous):
            # Overlaps ONLY on a shared word: "basel", "sr", "cfr". That names
            # a family, not a document.
            family = True
            continue
        # Most tokens in common wins, so "Basel CRE" beats a one-token overlap.
        if best is None or len(shared) > best[0]:
            best = (len(shared), full)

    if best:
        return "EXACT", best[1]
    if family:
        return "FAMILY", ""
    if not (doc_tokens - _generic_tokens()):
        # Descriptors only: "us guidance", "the supervisory rules", "PRA
        # expectations". A KIND of document, not one. Treated exactly like
        # FAMILY — no document was named, so the grid decides.
        return "NONE", ""
    return "ABSENT", ""


# ---------------------------------------------------------------------------
# THE RULE. Every branch is a registry fact, not a score and not a judgement.
# ---------------------------------------------------------------------------
def decide(x: dict) -> tuple[str, str]:
    """Extraction -> (decision, reason). No model, no network, deterministic.

    This is the half of the gate that can be unit-tested without Ollama, and
    the half that carries the corpus knowledge.
    """
    g, _subjects, _jurisdictions, names = corpus_grid()

    if not x.get("banking"):
        return "OUT_OF_DOMAIN", "not banking regulation"

    doc = str(x.get("document") or "").strip().lower()
    if doc:
        kind, hit = _names_hit(doc, names)
        if kind == "EXACT":
            return "ANSWERABLE", f"names a document we hold ({hit})"
        if kind == "ABSENT":
            if x.get("document_seen") is False:
                pass
            else:
                return "OUT_OF_CORPUS", (f"names {doc!r}, which is not in the "
                                         f"registry")
        # FAMILY / NONE: no specific document was named, so the subject x
        # jurisdiction check below decides. Do NOT decline here.

    # VALIDATE AGAINST THE VOCABULARY BEFORE USING IT (fixed 2026-09-02).
    # THE BUG: asked "GLOBAL" the model returned {"subject": "GLOBAL"} — a
    # JURISDICTION in the subject field. The grid held nothing at that key, so
    # `decide` reported "no document at all on GLOBAL" and DECLINED, confidently
    # and wrongly. A value the model invented became a statement about the
    # corpus.
    
    _g2, subjects, jurisdictions, _n2 = corpus_grid()
    su = x.get("subject") if x.get("subject") in subjects else None
    ju = x.get("jurisdiction") if x.get("jurisdiction") in jurisdictions else None

    # A jurisdiction that landed in the subject field is still usable — the
    # model read the word correctly and filed it wrongly.
    if su is None and x.get("subject") in jurisdictions and ju is None:
        ju = x.get("subject")

    # THE RULE NOW: a wrongly-extracted subject may cost a CLARIFYING TURN. It
    # may never cost a DECLINE. Only a document we can name and do not hold
    # produces OUT_OF_CORPUS, because that is the one branch resting on a fact
    # the model READ rather than a label it CHOSE.
    #
    # EVERY PATH RETURNS. Nothing below may re-decide a sentence that named two
    # jurisdictions - falling through is the bug this block exists to fix.
    asked = [j for j in (x.get("jurisdictions") or []) if j in jurisdictions]
    su_det = su if x.get("subject_from") == "question text" else None
    if len(asked) > 1:
        if not su_det:
            return "INCOMPLETE", (f"question names {asked}, but no subject was "
                                  f"read from the question text")
        covered = [j for j in asked if g.get((su_det, j))]
        if covered:
          
            # The fan-out reserves a slot per jurisdiction, but WHICH document
            # fills it is decided by RRF alone - which is how 12 CFR and
            # SR 15-18 took the reserved US slots on question related to SR 26-2 (tested).
            # Where the cell holds exactly one document there is nothing to
            # decide and nothing to ask: it is named and locked. Where it holds
            # two or more, the user knows which one they mean and the score
            # does not, so ASK - and ask only about the ambiguous jurisdiction.
    
            ambiguous = [j for j in covered if len(g[(su_det, j)]) > 1]
            if ambiguous:
                return "INCOMPLETE", (f"{su_det} has more than one document for "
                                      f"{ambiguous} - which one?")
            if len(covered) > 1:
                return "ANSWERABLE", (f"question names {covered} - one search "
                                      f"per jurisdiction, not a choice to make")
            return "ANSWERABLE", (f"of {asked}, the corpus holds {su_det} only "
                                  f"for {covered[0]}")
        elsewhere = sorted({j for (sub, j) in g if sub == su_det and g[(sub, j)]})
        if elsewhere:
            return "INCOMPLETE", (f"question names {asked}, but {su_det} is "
                                  f"held only for {elsewhere}")
        return "INCOMPLETE", (f"question names {asked}; nothing held on "
                              f"{su_det} at all")

    # Planner rule 3 splits an old-versus-new question into a CURRENT facet and
    # a HISTORIC one so neither version can crowd out the other. It reserves a
    # slot for each, but WHICH document fills it is decided by RRF alone - the
    # same gap the jurisdiction rule closed. So: a bucket holding one document
    # is settled and is NAMED; a bucket holding two or more is a choice the
    # user can make and the score cannot, so it is ASKED.
    
    if su_det and x.get("version"):
        vv, _cur_st, _hist_st = status_view()

        def _bucket(name: str) -> list[str]:
            return sorted({n for (s2, j2, b), ns in vv.items()
                           if b == name and s2 == su_det
                           and (ju is None or j2 == ju)
                           for n in ns})

        cur, hist = _bucket("current"), _bucket("historic")
        if cur and hist:
            crowded = [b for b, ns in (("current", cur), ("historic", hist))
                       if len(ns) > 1]
            if crowded:
                return "INCOMPLETE", (f"version question on {su_det}: more than "
                                      f"one document in {crowded} - which one?")
            return "ANSWERABLE", (f"version question on {su_det}: {cur[0]} now, "
                                  f"{hist[0]} before")

    if not su:
        return "INCOMPLETE", "banking, but no subject identified"


    if not ju:
        have = sorted({j for (sub, j) in g if sub == su and g[(sub, j)]})
        if not have:
            # Was OUT_OF_CORPUS. Now asks — the subject may simply be wrong.
            return "INCOMPLETE", (f"nothing held on {su}; the subject may be "
                                  f"mis-read — name a document or regulator")
        if len(have) == 1:
            return "ANSWERABLE", f"{su} exists only for {have[0]} — no ambiguity"
        return "INCOMPLETE", f"{su} exists for {have} — which one?"

    if not g.get((su, ju)):
        return "INCOMPLETE", (f"({su} x {ju}) is an empty cell — either the "
                              f"corpus lacks it, or the subject was mis-read")
    return "ANSWERABLE", f"({su} x {ju}) has {len(g[(su, ju)])} document(s)"


def locked_documents(x: dict) -> list[str]:
    """Documents the GRID already determines, for a multi-jurisdiction question.

    Returns only documents the grid determines UNIQUELY for a jurisdiction the
    SENTENCE named, and only when the subject was read from the question text -
    the same two conditions clarifying_question() uses before it says "using
    <doc>". A cell with two candidates is a choice for the user, never a guess
    made here.
    """
    g, _s, jurisdictions, _names = corpus_grid()
    su = x.get("subject")
    if not su or x.get("subject_from") != "question text":
        return []
    asked = [j for j in (x.get("jurisdictions") or []) if j in jurisdictions]
    if len(asked) < 2:                      # one jurisdiction cannot crowd out
        return []                           # a named document - rule 4 handles it
    out: list[str] = []
    for j in asked:
        here = sorted(g.get((su, j)) or [])
        if len(here) == 1:
            out.append(here[0])
    return out


def clarifying_question(x: dict, reason: str) -> str:
    """The INCOMPLETE message — the only one that is not a fixed sentence.

    It NAMES the choice the corpus actually offers, because "could you be more
    specific?" makes the user guess at a menu only the registry knows. The
    options come from the grid, so they can never offer something absent.
    """
    g, _s, jurisdictions, names = corpus_grid()
    su = x.get("subject")

    
    su_det0 = su if x.get("subject_from") == "question text" else None
    if su_det0 and x.get("version"):
        vv, _c, _h = status_view()
        ju0 = x.get("jurisdiction") if x.get("jurisdiction") in jurisdictions \
            else None

        def _b(name: str) -> list[str]:
            return sorted({n for (s2, j2, b), ns in vv.items()
                           if b == name and s2 == su_det0
                           and (ju0 is None or j2 == ju0)
                           for n in ns})

        cur, hist = _b("current"), _b("historic")
        if cur and hist:
            pretty0 = su_det0.lower().replace("_", " ")
            out = [f"I read that as a question about what changed in "
                   f"{pretty0}."]
            for label, docs_here in (("in force now", cur),
                                     ("the version it replaced", hist)):
                if len(docs_here) == 1:
                    out.append(f"  {label}: using {docs_here[0]}.")
                else:
                    out.append(f"  {label}: {', '.join(docs_here)} - which "
                               f"one?")
            out.append("Name the document for each side I asked about.")
            return "\n".join(out)

    asked = [j for j in (x.get("jurisdictions") or []) if j in jurisdictions]
    if len(asked) > 1:
        su_det = su if x.get("subject_from") == "question text" else None
        lines: list[str] = []
        if su_det:
            pretty = su_det.lower().replace("_", " ")
            # THE CELL IS COVERED, THE CHOICE IS INSIDE IT (2026-09-10). One
            # document is stated as locked so the user does not answer a
            # question nobody asked; two or more is the only thing asked about.
            covered = [j for j in asked if g.get((su_det, j))]
            if covered:
                lines.append(f"I read that as {pretty}.")
                for j in covered:
                    docs_here = sorted(g[(su_det, j)])
                    if len(docs_here) == 1:
                        lines.append(f"  {j}: using {docs_here[0]}.")
                    else:
                        lines.append(f"  {j}: {', '.join(docs_here)}"
                                     f" - which one?")
                for j in asked:
                    if j not in covered:
                        lines.append(f"  {j}: nothing on {pretty}"
                                     f" - that part will not be searched.")
                lines.append("Name the document for each jurisdiction I asked "
                             "about.")
                return "\n".join(lines)
            elsewhere = sorted({j for (sub, j) in g
                                if sub == su_det and g[(sub, j)]})
            if elsewhere:
                held = sorted({n for j in elsewhere for n in g[(su_det, j)]})
                lines.append(f"I read that as {pretty}. This corpus holds "
                             f"{pretty} only for {', '.join(elsewhere)} - "
                             f"{', '.join(held)}. It holds nothing on {pretty} "
                             f"for {' or '.join(asked)}.")
            else:
                lines.append(f"I read that as {pretty}, which this corpus does "
                             f"not hold for any jurisdiction.")
        else:
            lines.append("I could not read a subject from your question.")
        lines.append("Under the jurisdictions you named it holds -")
        for j in asked:
            here = sorted({n for (sub, jj), ns in g.items() if jj == j
                           for n in ns})
            lines.append(f"  {j}: " + (", ".join(here) if here else "nothing"))
        lines.append("Which document should I use under each?")
        return "\n".join(lines)

    # ASK FOR A DOCUMENT FIRST, ALWAYS. The document field has resolved
    # correctly every time it fired; the subject field has been wrong eight
    # times. So the question a user can answer usefully is "which document",
    # not "UK or US" — and "UK or US" is actively misleading when the subject
    # behind it was mis-read, which is how a Basel question got a UK/US menu.
    docs = ", ".join(sorted(names.values()))
    if su and su in {s for (s, _j) in g}:
        have = sorted({j for (s, j) in g if s == su and g[(s, j)]})
        if len(have) > 1:
            pretty = ", ".join(have[:-1]) + " or " + have[-1]
            return (f"I read that as {su.lower().replace('_', ' ')}, which this "
                    f"corpus covers for {pretty}. Which one do you mean — or, "
                    f"more reliably, which document?\n  {docs}")
    return ("I could not pin down what to search. Which document or regulator "
            f"should I use?\n  {docs}")


# ---------------------------------------------------------------------------
# THE MODEL CALL
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _prompt(version: str) -> tuple[str, str]:
    """(text, sha). Vocabularies are filled from the registry at load time.

    The prompt FILE holds placeholders, not a hard-coded list. Adding a
    jurisdiction to registry.yaml must not require editing a prompt — same rule
    the registry header states for the pipeline: a corpus change is a DATA
    change, never a code change.
    """
    path = config.PROJECT_ROOT / "prompts" / f"{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"no gate prompt {version!r} at {path}")
    raw = path.read_bytes()
    _g, subjects, jurisdictions, _n = corpus_grid()
    text = raw.decode("utf-8").format(subjects=list(subjects),
                                      jurisdictions=list(jurisdictions))
    return text, hashlib.sha256(raw).hexdigest()[:16]


def config_hash() -> str:
    _t, sha = _prompt(config.GATE_PROMPT)
    return f"{config.GATE_MODEL}|{config.GATE_PROMPT}@{sha}|t{config.TEMPERATURE}"


def version_named(question: str) -> bool:
    """Does the QUESTION TEXT ask old-versus-new? No model, same vocabulary.

    Planner rule 3's own test, read from the same term tables so the gate and
    the planner cannot disagree about what a version question is - the third
    "one vocabulary, two readers" pair after subject and jurisdiction.

    `config.WEAK_VERSION_TERMS_ENABLED` is honoured here for the same reason it
    exists in the planner: it is an experiment switch, and a switch that moves
    the planner but not the gate would make the gate promise a fan-out that
    never happens.
    """
    from regrag.retrieval.planner import (DOC_WORDS, STRONG_VERSION_TERMS,
                                          WEAK_VERSION_TERMS, _matches)

    q = question.lower()
    strong = _matches(q, STRONG_VERSION_TERMS)
    weak = _matches(q, WEAK_VERSION_TERMS)
    if not config.WEAK_VERSION_TERMS_ENABLED:
        weak = []
    return bool(strong) or bool(weak and _matches(q, DOC_WORDS))


def subject_named(question: str) -> str:
    """The subject THE QUESTION STATES, read from the text. No model.

    Reuses the planner's SUBJECT_TERMS, so the gate and the planner cannot
    disagree about the topic - one vocabulary, one answer.

    EXACTLY ONE MATCH, OR NOTHING. `SUBJECT_TERMS` overlaps by design (MRM
    lists the bare word "model", so "capital planning model" hits MRM AND
    CAPITAL_ADEQUACY). Picking a winner there would move the vagueness rather
    than remove it, so two matches fall back to the model - which is the
    behaviour that already exists and is already capped at one clarifying turn.
    """
    from regrag.retrieval.planner import SUBJECT_TERMS, _matches

    q = question.lower()
    hits = [s for s, terms in SUBJECT_TERMS.items() if _matches(q, terms)]
    return hits[0].value if len(hits) == 1 else ""


def jurisdiction_named(question: str) -> str:
    """The jurisdiction THE QUESTION STATES. Same rule, same vocabulary.

    EXACTLY ONE, OR NOTHING - and here the multi-match case is not an edge:
    "How do US and UK expectations compare?" names two on purpose. Two matches
    must fall through to the model, which is also what makes the planner build
    a jurisdiction fan-out for the same question. Collapsing them to one would
    make the gate reason about a narrower question than the one being searched.

    MORE CONSEQUENTIAL THAN THE SUBJECT, so state the effect plainly: the gate
    checks (subject x jurisdiction) for an EMPTY CELL. A correctly-read "US" on
    a topic the corpus only holds GLOBALLY now produces a clarifying question
    where a vaguer extraction might have answered. That is the intended trade -
    "minimum CET1 for a US bank holding company" SHOULD ask, because Regulation
    Q is a known_gap and the corpus cannot answer it for the US.
    """
    hits = jurisdictions_named(question)
    return hits[0] if len(hits) == 1 else ""


def jurisdictions_named(question: str) -> list[str]:
    """EVERY jurisdiction the question states. The multi-value read.

    `jurisdiction_named` collapses to one or nothing because the extraction
    dict has ONE jurisdiction field. This one does not collapse, because
    "UK and US" is a real and common shape - six of the 48 golden questions -
    and the planner already fans out on it. `decide()` reads this to stop
    asking a question the sentence has already answered.
    """
    from regrag.retrieval.planner import JURISDICTION_TERMS, _matches

    q = question.lower()
    return [j.value for j, terms in JURISDICTION_TERMS.items()
            if _matches(q, terms)]


def documents_named(question: str) -> list[str]:
    """Documents THE QUESTION NAMES, read from the text itself. No model.

    Reuses the planner's registry-derived alias matcher, so the gate and the
    planner cannot disagree about whether a document was named - one matcher,
    one answer. Imported lazily to keep the gate importable on its own.
    """
    from regrag.retrieval.planner import _documents_in

    return [sn for _did, (sn, _alias) in _documents_in(question.lower()).items()]


def classify(question: str) -> Intent:
    """Question -> Intent. NEVER raises: a broken gate must not break the app.

    FAIL OPEN, DELIBERATELY. If Ollama is down, the model returns junk, or the
    JSON will not parse, the question is passed through as ANSWERABLE with the
    failure recorded in `reason`. 
    """
    t0 = time.perf_counter()

    named = documents_named(question)
    if named:
        x: dict = {"jurisdictions": jurisdictions_named(question)}
        su = subject_named(question)
        if su:
            x["subject"], x["subject_from"] = su, "question text"
        return Intent(question, "ANSWERABLE",
                      f"question names {', '.join(sorted(named))} — "
                      f"no gate needed", extraction=x,
                      seconds=time.perf_counter() - t0)

    ch = ""
    try:
        system, _sha = _prompt(config.GATE_PROMPT)
        ch = config_hash()
        out = llm.chat(system, question, model=config.GATE_MODEL, fmt="json",
                       max_tokens=config.GATE_MAX_TOKENS, think=False)
        x = json.loads(out.text)
        if not isinstance(x, dict):
            raise ValueError("model returned JSON that is not an object")
    except Exception as exc:                                       # noqa: BLE001
        return Intent(question, "ANSWERABLE",
                      f"gate unavailable, passed through: "
                      f"{type(exc).__name__}: {exc}",
                      seconds=time.perf_counter() - t0, config_hash=ch)

    # *** THE REGEX OVERWRITES THE MODEL, IT DOES NOT SUGGEST TO IT. ***
    # Only the subject is overwritten. `banking` and `jurisdiction` still come
    # from the model, and an ambiguous subject still does too.
    
    su = subject_named(question)
    if su:
    # `decide()`refuses to use a subject it cannot PROVE came from SUBJECT_TERMS
        
        x["subject"], x["subject_from"] = su, "question text"
    
    # The jurisdiction is read the same way, with one extra case: TWO named.
    # There the regex knows something the single `jurisdiction` field cannot
    # hold, so it is recorded separately AND the model's single value is
    # CLEARED. Leaving it would let a coin-flip "US" silently narrow a question
    # that said "UK and US" - the failure gs-08 and gs-09 showed.
    
    jus = jurisdictions_named(question)
    if len(jus) > 1:
        x["jurisdictions"] = jus
        x["jurisdiction"], x["jurisdiction_from"] = None, "question text (two)"
    elif len(jus) == 1 and x.get("jurisdiction") != jus[0]:
        x["jurisdiction"], x["jurisdiction_from"] = jus[0], "question text"
    
    # THE THIRD DETERMINISTIC READING (2026-09-10). Old-versus-new is planner
    # rule 3's own test, read here from the same term tables. The model is not
    # asked and cannot override it - it never had this field.
    
    x["version"] = version_named(question)

    # AND THE FOURTH: is the document field something the model READ, or
    # something it WROTE? Only an explicit False suppresses the decline below;
    # a dict built by hand carries no claim either way.
    
    _doc = str(x.get("document") or "").strip()
    if _doc:
        x["document_seen"] = document_seen(_doc, question)

    decision, reason = decide(x)
    if decision == "INCOMPLETE":
        msg = clarifying_question(x, reason)
    else:
        msg = MESSAGES.get(decision, "")
    return Intent(question, decision, reason, msg, x,
                  round(time.perf_counter() - t0, 2), ch)
