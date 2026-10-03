"""Question in, cited answer out.

WHAT THIS IS FOR
    The five steps - retrieve, render, call, parse, format. This is those steps, once.

THE TWO THINGS ONLY THIS MODULE DOES
1.  RESOLVES THE LABEL. The model writes "S2". A reader needs
    "SS1/23, Principle 1.2 Model inventory (in_force; eff. 2024-05-17)". The
    swap happens here, in code, from the table render() already built.
    DELIBERATELY NOT THE MODEL'S JOB: a small model asked to reproduce a full
    reference verbatim will eventually get a digit wrong, and a wrong citation
    that LOOKS right is the same failure as a wrong table value - absent data
    becomes "I don't know", wrong data becomes a confident answer with a clean
    reference. The model points; the code cites.

2.  STAMPS THE ANSWER WITH HOW IT WAS MADE. `config_hash` carries model,
    prompt version AND its content sha, temperature, relevance floor and facet
    margin. 

    RELEVANCE OF A CLAIM. `overlap` compares a claim to the passage it
    cites, never to the question. 
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from regrag import config
from regrag.generation import align as aligner
from regrag.generation import llm, prompts, render
from regrag.observability.tracing import tracer
from regrag.retrieval.search import Retrieval, retrieve

RX_LABEL = re.compile(r"^S\d+$")
STOP = set("""the a an of to in for and or is are be as that this which with on by must
should may not from at it its their under section any all such other than when where
have has been were was will can could would each both same these those there here into
over more most less about after before during between within without upon""".split())


@dataclass(frozen=True)
class Claim:
    text: str
    label: str                 # the model's "S2"
    citation: str              # resolved, or "" when the label was invalid
    overlap: float             # claim words found in the cited passage, 0..1
    # --- where inside the passage, decided by the aligner (2026-09-03) ------
    # `locator` is the paragraph in the citation. `align_verdict` says how it
    # was arrived at: MATCH means the aligner moved it, anything else means it
    # kept what retrieval would have shipped. Diagnostics for the REVIEW view -
    # they never reach the reader, for the same reason `overlap` does not.
    locator: str = ""
    align_verdict: str = ""
    align_score: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.citation)

    @property
    def weak(self) -> bool:
        """Below the screen's bar. NOT a verdict - see `overlap_of`."""
        return self.valid and self.overlap < config.CLAIM_OVERLAP_MIN


@dataclass
class Answer:
    question: str
    text: str = ""                       # prose with inline [1] markers
    sources: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""
    retrieval: Retrieval | None = None   # the evidence, kept WITH the answer
    config_hash: str = ""
    seconds: float = 0.0

    @property
    def invalid(self) -> list[Claim]:
        """Labels pointing at no passage. A bug, not a warning."""
        return [c for c in self.claims if not c.valid]

    @property
    def weak(self) -> list[Claim]:
        return [c for c in self.claims if c.weak]

    def __str__(self) -> str:
        if self.refused:
            return f"REFUSED {self.question!r}: {self.refusal_reason}"
        return render.as_text(self.text, self.sources)


def overlap_of(claim: str, passage: str) -> float:
    """Fraction of the claim's distinctive words appearing in the passage.

    A SCREEN, NOT A VERDICT. Substring matching, so "data" matches inside
    "database"; word order and meaning are ignored, so "banks validate models"
    and "models validate banks" score the same. It answers "did these words
    come from here?" and never "is this true?". Low overlap means read it.
    """
    ws = [w for w in re.findall(r"[a-z]{4,}", claim.lower()) if w not in STOP]
    return sum(w in passage.lower() for w in ws) / len(ws) if ws else 0.0


def config_hash(prompt: prompts.Prompt) -> str:
    """Everything that changes an answer without changing the question.

    THE RENDER VERSION IS IN HERE ON PURPOSE. The aligner changed the citations
    without touching the model, the prompt or retrieval; an eval run made
    before it is not comparable to one made after, and nothing else in this
    string would have said so.
    """
    al = (f"align{config.ALIGN_FLOOR}/{config.ALIGN_MARGIN}"
          if config.ALIGN_ENABLED else "align-off")
    return (f"{config.GEN_MODEL}|{prompt.version}@{prompt.sha}"
            f"|t{config.TEMPERATURE}|ctx{config.GEN_NUM_CTX}"
            f"|floor{config.RERANK_DROP_BELOW}|facet{config.FACET_DROP_MARGIN}"
            f"|render{render.RENDER_VERSION}|{al}")


def _answer_impl(question: str, *, prompt_version: str = prompts.CURRENT,
                 retrieval: Retrieval | None = None) -> Answer:
    """One question -> one Answer. Never raises on a bad model response.

    `retrieval` lets a caller reuse a Retrieval it already has, so an
    experiment can hold retrieval fixed and vary only the prompt. Passing it is
    the difference between comparing two prompts and comparing two runs.
    """
    t0 = time.perf_counter()
    p = prompts.load(prompt_version)
    a = Answer(question, config_hash=config_hash(p))

    if retrieval is not None:
        r = retrieval                      # reused: not this question's work, not traced
    else:
        with tracer().start_as_current_span(
                "retrieve", openinference_span_kind="retriever") as _sp:
            _sp.set_input(question)
            r = retrieve(question)
            _describe(_sp, r)
    a.retrieval = r

    if r.refused or not r.passages:
        a.refused = True
        a.refusal_reason = r.refused or "retrieval returned no passages"
        a.seconds = time.perf_counter() - t0
        return a

    try:
        with tracer().start_as_current_span(
                "render", openinference_span_kind="chain") as _sp:
            context, labels = render.render(r)
            _sp.set_attribute("context_chars", len(context))
            _sp.set_attribute("labels", list(labels))
            _sp.set_attribute("render_version", render.RENDER_VERSION)
    except render.ContextTooLarge as exc:
        a.refused = True
        a.refusal_reason = str(exc)
        a.seconds = time.perf_counter() - t0
        return a

    by_label = {f"S{i}": pas for i, pas in enumerate(r.passages, 1)}
    passage_text = {k: v.text for k, v in by_label.items()}
    # Split once per PASSAGE, not once per claim - the same dozen paragraphs
    # would otherwise be re-split for every sentence the model wrote.
    units = ({k: aligner.paragraph_units(v) for k, v in by_label.items()}
             if config.ALIGN_ENABLED else {})
    with tracer().start_as_current_span(
            "generate", openinference_span_kind="llm") as _sp:
        user_msg = render.user_message(question, context)
        out = llm.chat(p.text, user_msg, fmt="json")
        _sp.set_input(user_msg)
        _sp.set_output(out.text)
        _sp.set_attribute("model", out.model)
        _sp.set_attribute("prompt_tokens", out.prompt_tokens)
        _sp.set_attribute("output_tokens", out.output_tokens)
        _sp.set_attribute("num_ctx", config.GEN_NUM_CTX)
        # THE TRUNCATION ALARM, now on every trace instead of only when
        # someone thinks to look. See llm.py for why Ollama cannot tell us.
        _sp.set_attribute("maybe_truncated", out.maybe_truncated)
        _sp.set_attribute("prompt_seconds", round(out.prompt_seconds, 3))
        _sp.set_attribute("eval_seconds", round(out.eval_seconds, 3))

    # A malformed response is a BAD ANSWER, not a crash.
    try:
        raw_claims = json.loads(out.text).get("claims", [])
        if not isinstance(raw_claims, list):
            raise ValueError("'claims' is not a list")
    except Exception as exc:                                   # noqa: BLE001
        a.refused = True
        a.refusal_reason = f"model response unusable: {type(exc).__name__}: {exc}"
        a.seconds = time.perf_counter() - t0
        return a

    for c in raw_claims:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text", "")).strip()
        label = str(c.get("source", "")).strip()
        if not text:
            continue
        ok = bool(RX_LABEL.match(label)) and label in labels
        if not ok:
            # A label pointing at no passage is a BUG, not a weak citation.
            # It gets no locator and no alignment - there is nothing to align
            # against - and format_answer marks it [unattributed].
            a.claims.append(Claim(text=text, label=label, citation="", overlap=0.0))
            continue

        # WHICH PARAGRAPH. The model reads a whole parent; retrieval only knows
        # which CHILD it found. 
        
        psg = by_label[label]
        current = (psg.children[0].payload.get("locator") or "") if psg.children else ""
        al = (aligner.align(text, units.get(label, []), current)
              if config.ALIGN_ENABLED else None)
        loc = al.locator if al else current

        a.claims.append(Claim(
            text=text,
            label=label,
            # Recomposed per claim: same document, same status, possibly a
            # different paragraph. Without a locator this falls back to the
            # heading exactly as it did before.
            citation=render.compose_citation(psg, loc or None),
            overlap=round(overlap_of(text, passage_text.get(label, "")), 3),
            locator=loc,
            align_verdict=al.verdict if al else "off",
            align_score=round(al.best, 3) if al else 0.0,
        ))

    a.text, a.sources = render.format_answer(
        [{"text": c.text, "source": c.label, "citation": c.citation}
         for c in a.claims], labels)
    a.seconds = time.perf_counter() - t0
    return a


# ---------------------------------------------------------------------------
# TRACING - the root span, and one helper that copies what is already measured
# ---------------------------------------------------------------------------
def _describe(sp, r: Retrieval) -> None:
    """Put what retrieve() ALREADY measured onto its span. Measures nothing new.

    Retrieval carries the plan, the notes, the per-step timings and the
    casualties. Copying them here is what lets a trace answer "which facet took
    the slot" without opening dump_retrieval.py.
    """
    sp.set_attribute("plan.mode", str(r.plan.mode))
    sp.set_attribute("plan.facets", [str(f.label) for f in r.plan.facets])
    sp.set_attribute("plan.inferred_fanout", bool(r.plan.inferred_fanout))
    sp.set_attribute("n_passages", len(r.passages))
    sp.set_attribute("total_chars", r.total_chars)
    sp.set_attribute("refused", r.refused or "")
    sp.set_attribute("n_dropped", len(r.dropped))
    if r.notes:
        sp.set_attribute("notes", [str(n) for n in r.notes])
    for k, v in r.timings.items():
        sp.set_attribute(f"timing.{k}_ms", round(v * 1000))
    # str(Passage) is the same one-line summary the evidence panel prints.
    sp.set_attribute("passages", [str(x) for x in r.passages])
    sp.set_output(f"{len(r.passages)} passage(s), {r.total_chars:,} chars")


def answer(question: str, *, prompt_version: str = prompts.CURRENT,
           retrieval: Retrieval | None = None) -> Answer:
    """One question -> one Answer, wrapped in the root span.

    WHY A WRAPPER INSTEAD OF A `with` INSIDE THE FUNCTION
        The implementation has five early returns. A wrapper closes the span on
        every one of them without touching a single `return`, and the diff stays
        small enough to read. With tracing off this costs one no-op object.
    """
    with tracer().start_as_current_span(
            "answer", openinference_span_kind="chain") as sp:
        sp.set_input(question)
        a = _answer_impl(question, prompt_version=prompt_version,
                         retrieval=retrieval)
        sp.set_output(a.text or a.refusal_reason)
        sp.set_attribute("prompt_version", prompt_version)
        sp.set_attribute("config_hash", a.config_hash)
        sp.set_attribute("refused", bool(a.refused))
        sp.set_attribute("refusal_reason", a.refusal_reason or "")
        sp.set_attribute("n_claims", len(a.claims))
        sp.set_attribute("n_invalid_claims", len(a.invalid))
        sp.set_attribute("seconds", round(a.seconds, 2))
        return a
