"""Passages -> the text the model reads. Plus the citation the code composes.

TWO DESIGN DECISIONS WORTH DEFENDING

1.  THE MODEL WRITES A LABEL; THE CODE WRITES THE CITATION.
    Each passage is shown as [S1], [S2]. The model cites [S1]. Afterwards the
    code swaps S1 for the real reference. A small model asked to reproduce
    "Basel CRE, CRE36, in_force, eff. 2023-01-01" verbatim will eventually get
    a digit wrong, and a wrong citation. The model points, the
    code cites.

2.  THE CITATION IS COMPOSED FROM STRUCTURED FIELDS, NEVER FREE TEXT.
    short_name + volume + status + effective_from, from the payload the
    registry denormalised onto every chunk. 

THE BUDGET IS A CORRECTNESS CHECK, NOT A PERFORMANCE ONE.
    Ollama silently drops whatever does not fit in num_ctx. So the renderer RAISES rather
    than hand over a context it knows is too big.
"""
from __future__ import annotations

from regrag import config
from regrag.retrieval.search import Passage, Retrieval


class ContextTooLarge(RuntimeError):
    """Raised instead of letting the server trim the evidence in silence."""

RENDER_VERSION = "v2"

def compose_citation(p: Passage, locator: str | None = None) -> str:
    """Full reference for one passage, built from registry fields.

    `locator` overrides the paragraph reference — that is the aligner's whole
    output. Everything else in the citation (document, volume, status,
    effective date) is a property of the PASSAGE and cannot move with a claim.
    Passing nothing reproduces exactly what shipped before the aligner.
    """
    pay = p.children[0].payload if p.children else {}
    bits = [str(pay.get("short_name") or "?")]
    vol = pay.get("volume")
    if vol and str(vol) not in bits[0]:
        bits.append(str(vol))
    loc = locator or (p.children[0].payload.get("locator") if p.children else None)
    bits.append(str(loc) if loc else (p.heading or "")[:60])
    tail = []
    if pay.get("status"):
        tail.append(str(pay["status"]))
    if pay.get("effective_from"):
        tail.append(f"eff. {pay['effective_from']}")
    head = ", ".join(b for b in bits if b)
    return f"{head} ({'; '.join(tail)})" if tail else head


def render(retr: Retrieval) -> tuple[str, dict[str, str]]:
    """Returns (context_text, {label: full_citation}).

    Raises ContextTooLarge above config.GEN_MAX_CONTEXT_CHARS.
    """
    blocks, table = [], {}
    for i, p in enumerate(retr.passages, 1):
        label = f"S{i}"
        table[label] = compose_citation(p)
        blocks.append(f"[{label}] {table[label]}\n{p.text}")
    text = "\n\n---\n\n".join(blocks)
    if len(text) > config.GEN_MAX_CONTEXT_CHARS:
        raise ContextTooLarge(
            f"{len(text):,} chars > {config.GEN_MAX_CONTEXT_CHARS:,}. "
            f"Ollama would drop the overflow with no error. Reduce the quota or "
            f"raise config.GEN_NUM_CTX and this cap together."
        )
    return text, table


def user_message(question: str, context: str) -> str:
    return f"EXCERPTS\n\n{context}\n\n---\n\nQUESTION\n{question}"


# ---------------------------------------------------------------------------
# THE FINAL ANSWER — what a reader sees.
# ---------------------------------------------------------------------------
def format_answer(claims: list[dict], labels: dict[str, str]) -> tuple[str, list[str]]:
    """Claims -> prose with inline markers, plus a numbered SOURCES block.

    TWO AUDIENCES, TWO OUTPUTS - do not mix them.
        The READER gets prose and references. Overlap percentages, label ids
        and weak-match warnings are OUR diagnostics; putting them in the answer
        makes a working note look like a finding and asks the reader to
        interpret a number we told them not to trust.

        The REVIEWER gets those diagnostics in the review view, on demand.

    WHY MARKERS AND NOT A SOURCES LIST ALONE
        A list at the bottom says which documents were consulted. It cannot say
        which sentence rests on which one - so a reader checking a single claim
        must read every passage and work it out. It also hides an unattributed
        sentence completely: the answer still ends with a tidy list, and nothing
        marks the claim that had no source. Markers keep the link; the block
        keeps the metadata out of the prose.
    """
    order: list[str] = []          # CITATION STRINGS, in first-appearance order
    out: list[str] = []
    for c in claims:
        src = str(c.get("source", ""))
        text = str(c.get("text", "")).strip()
        if not text:
            continue
        # THE KEY IS THE CITATION, NOT THE LABEL. Since the aligner, two claims
        # reading the same passage can cite two different paragraphs, and they
        # must get two different numbers. Keying on the label would collapse
        # them back to one and put the wrong paragraph under half the markers.
        # With alignment off, one label still yields one citation, so the
        # numbering is identical to what it was before.
        cite = str(c.get("citation") or "") or (labels.get(src, "") if src in labels else "")
        # THE MARKER GOES INSIDE THE SENTENCE, BEFORE THE FULL STOP.
        # "...boundaries [1]." not "...boundaries. [1]" - a marker after the
        # stop floats between two sentences and the reader cannot tell which
        # one it belongs to. Inside the stop, it is unambiguous.
        stem, dot = (text[:-1], text[-1]) if text[-1:] in ".!?" else (text, "")
        if cite:
            if cite not in order:
                order.append(cite)
            out.append(f"{stem} [{order.index(cite) + 1}]{dot or '.'}")
        else:
            # No usable source. Say so IN PLACE rather than dropping the claim
            # or letting it pass unmarked - an unattributed sentence hidden in
            # clean prose is the failure this whole design exists to prevent.
            out.append(f"{stem} [unattributed]{dot or '.'}")
    body = " ".join(out)
    sources = [f"[{i + 1}] {s}" for i, s in enumerate(order)]
    return body, sources


def as_text(body: str, sources: list[str]) -> str:
    import textwrap
    lines = [textwrap.fill(body, 92)]
    if sources:
        lines += ["", "SOURCES"] + [f"  {s}" for s in sources]
    return "\n".join(lines)
