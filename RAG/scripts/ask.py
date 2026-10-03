"""ASK — type a question, read the answer, then read the evidence behind it.

ONE question, with the evidence laid out. It is the tool for the four evidence-quality
    studies, and it MEASURES NOTHING and DECIDES NOTHING. Every flag it prints
    is a pointer to something worth reading, never a verdict.

    Nothing is written to disk unless you type `:save`.

THE FOUR STUDIES, AND WHERE EACH ONE SHOWS UP HERE
  1 FOOTNOTE CHUNKS   -> a passage marked `BIN`. That is a bag of unrelated
                         notes, not a section. Ask: did it reach the prompt at
                         all, and did a claim cite it? A citation reading
                         "Basel CAP, Footnotes" is useless to a validator.
  2 OVERSIZED PARENTS -> a passage marked `WIN` shows `window/full` chars. The
                         question is not the size, it is whether the window
                         landed on the right paragraph. `:p N` prints what the
                         model actually saw. Known-open bug: `_window` falls
                         back to paragraph 0 when the child text is not found
                         inside the parent.
  3 TABLES            -> `pipes=N` counts lines holding two or more `|`. It is
                         a ROUGH MARKER that table-shaped text is present, NOT
                         a table detector and NOT a check that the row is
                         correct. Use it to find a passage worth reading, then
                         read it with `:p N` and compare against the PDF. 
  4 QUESTION SETS     -> `:junk N`, `:hn N`, `:tp N` load a question from
                         `regrag/evaluation/refusal_set.py` instead of typing
                         it. Same pipeline, no re-typing, no transcription slip.

USE
    uv run python scripts/ask.py
    uv run python scripts/ask.py --no-preflight     # skip the model check
    uv run python scripts/ask.py "what is a model?" # answer one, then exit

COMMANDS (inside the loop)
    :h              this list
    :p N            print passage N in full — exactly the text the model saw
    :ctx            print the whole rendered prompt context
    :ev             re-print the evidence panel for the last question
    :claims         re-print claims with label, overlap, aligner verdict, citation
    :cfg            current config (model, floor, facet margin, caps)
    :tp N / :junk N / :hn N     load question N from the refusal set
    :list tp|junk|hn            list that set with its numbers
    :save [name]    write this whole session to evaluation/ as JSON
    :q              quit
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import textwrap
from datetime import datetime, timezone

from regrag import config
from regrag.evaluation.refusal_set import HARD_NEGATIVE, JUNK, true_positives
from regrag.gate import respond
from regrag.observability import audit, tracing
from regrag.observability.tracing import tracer
from regrag.generation import prompts, render
from regrag.generation.answer import Answer
from regrag.retrieval.search import is_bin

W = 92
LINE = "=" * W
THIN = "-" * W


# --------------------------------------------------------------------------
# EVIDENCE — the point of this script
# --------------------------------------------------------------------------
def pipe_lines(text: str) -> int:
    """Lines holding two or more `|`. A MARKER, not a table detector.

    It cannot tell a real table from prose containing pipes, it says nothing
    about whether the row survived parsing intact, and a table that lost its
    pipes is invisible to it. It exists to point you at a passage to READ.
    """
    return sum(1 for ln in text.splitlines() if ln.count("|") >= 2)


def flags_for(p) -> str:
    """Three pointers, in the order the four studies care about."""
    out = []
    if is_bin(p.heading):
        out.append("BIN")                       # study 1
    if p.windowed:
        out.append(f"WIN {len(p.text):,}/{p.full_chars:,}")   # study 2
    elif p.full_chars > 12_000:
        out.append(f"BIG {p.full_chars:,}")     # study 2, delivered whole
    n = pipe_lines(p.text)
    if n:
        out.append(f"pipes={n}")                # study 3
    return "  ".join(out)


def evidence(a: Answer) -> None:
    r = a.retrieval
    print(f"\n{THIN}\nEVIDENCE\n{THIN}")
    if r is None:
        print("  none — no retrieval was run")
        return

    print(f"  plan [{r.plan.mode}]  inferred_fanout={r.plan.inferred_fanout}  "
          f"{len(r.plan.facets)} facet(s)")
    for f in r.plan.facets:
        print(f"    {f}")

    # The notes are the ONLY proof a filter fired. Do not infer it from config.
    for note in r.notes:
        print(f"  ! {note}")
    if not r.notes:
        print("  ! no filter notes — neither the facet margin nor the floor "
              "dropped anything")

    if r.refused:
        print(f"  REFUSED BY RETRIEVAL: {r.refused}")
        return

    # Which labels the model actually cited, so an unused passage is visible.
    cited = {c.label for c in a.claims if c.valid}
    print(f"\n  {len(r.passages)} passage(s), {r.total_chars:,} chars "
          f"({', '.join(f'{k} {v * 1000:.0f}ms' for k, v in r.timings.items())})")
    for i, p in enumerate(r.passages, 1):
        mark = "cited" if f"S{i}" in cited else "  -  "
        print(f"\n  S{i} [{mark}] {p.short_name}  {p.heading[:60]!r}")
        fl = flags_for(p)
        print(f"       {len(p.text):,} chars" + (f"   {fl}" if fl else ""))
        for h in p.children:
            print(f"       <- {h.cite}  [{h.arms}]  facet={h.facet}")
    print(f"\n  (`:p N` prints a passage in full — that is what the model read)")


def claims_panel(a: Answer) -> None:
    print(f"\n{THIN}\nCLAIMS\n{THIN}")
    if not a.claims:
        print("  none")
        return
    for i, c in enumerate(a.claims, 1):
        if not c.valid:
            print(f"  {i}. INVALID LABEL {c.label!r} — this is a BUG, not a warning")
        else:
            bar = "weak" if c.weak else "    "
            # MATCH means the aligner MOVED the paragraph off the one retrieval
            # found. Anything else means it kept what retrieval would have
            # cited, and the old defect may still be in that line.
            moved = "*" if c.align_verdict == "MATCH" else " "
            print(f"  {i}. {bar} overlap {c.overlap:>5.0%}  [{c.label}] "
                  f"{moved}{c.align_verdict.lower():<6} {c.align_score:>4.2f}  "
                  f"-> {c.citation}")
        print(textwrap.indent(textwrap.fill(c.text, W - 8), "        "))
    print(f"\n  overlap = claim words found in the CITED PASSAGE. It is a screen for "
          f"'did these\n  words come from here'. It says nothing about whether the "
          f"claim answers the question.")
    print("  match/floor/tie/noloc = the ALIGNER. `match` (starred) means it "
          "picked the\n  paragraph itself; the rest mean it deferred to the "
          "child that won retrieval.\n  It fixes WHICH paragraph is cited, "
          "never whether the claim is true.")


# --------------------------------------------------------------------------
# ONE QUESTION
# --------------------------------------------------------------------------
PENDING: str | None = None      # the question the gate asked us to clarify


def ask(question: str) -> Answer | None:
    """Through the GATE now (Stage 8.6), not straight into the pipeline.

    Returns None when the gate stopped the question — there is no Answer to
    inspect, and `:p` / `:ctx` have nothing to show. That is a different thing
    from an empty answer, and the loop keeps them apart.
    """
    # THE GATE ASKED A QUESTION AND NOTHING COULD HEAR THE ANSWER
    # (fixed 2026-09-02). The clarifying turn was built, the reply was not:
    # typing "GLOBAL" started a brand-new classification of the word "GLOBAL".
    # A follow-up that cannot be answered is worse than no follow-up — it asks
    # the user for something and then throws it away.
    #
    # THE STATE LIVES HERE, NOT IN THE GATE. `classify()` and `respond()` stay
    # pure functions of one question, so evaluation still gets one input and
    # one output. Conversation is a property of this loop.
    global PENDING
    if PENDING:
        question = f"{PENDING} ({question})"
        print(f"\n  [combining with the question the gate asked about]")
        PENDING = None

    print(f"\n{LINE}\n{question}\n{LINE}")
    print("  ... classifying intent, then retrieving and generating",
          flush=True)
    # ONE SPAN AROUND THE WHOLE QUESTION, so the gate and the answer are
    # one trace and not two unrelated ones.
    with tracer().start_as_current_span(
            "ask", openinference_span_kind="chain") as _sp:
        _sp.set_input(question)
        r = respond(question)
        _sp.set_output(r.text)
        _sp.set_attribute("stopped", bool(r.stopped))
        _sp.set_attribute("seconds", round(r.seconds, 2))
        _tid = tracing.current_trace_id()
    # The durable record. Written for EVERY question, traced or not.
    audit.log(r, trace_id=_tid)
    if r.intent is not None and r.intent.decision == "INCOMPLETE":
        PENDING = question

    if r.intent is not None:
        print(f"\n{THIN}\nGATE   [{r.intent.decision}]   "
              f"{r.intent.seconds:.1f}s\n{THIN}")
        print(f"  reason      {r.intent.reason}")
        print(f"  extraction  {json.dumps(r.intent.extraction)}")

    if r.stopped:
        print(f"\n{THIN}\nSTOPPED BEFORE RETRIEVAL   {r.seconds:.1f}s\n{THIN}")
        print(textwrap.indent(textwrap.fill(r.text, W - 4), "  "))
        if PENDING:
            print("\n  Type just the missing piece — it is added to your "
                  "question, not treated as a new one.")
        print("\n  Nothing was retrieved, so there is no evidence panel.")
        return None

    a = r.answer
    print(f"\n{THIN}\nANSWER   {a.seconds:.0f}s\n{THIN}")
    if a.refused:
        print(f"  REFUSED: {a.refusal_reason}")
    else:
        print(textwrap.indent(str(a), "  "))

    claims_panel(a)
    evidence(a)

    bad, weak = a.invalid, a.weak
    print(f"\n  {len(a.claims)} claim(s) | INVALID {len(bad)} | weak {len(weak)} "
          f"| {a.config_hash}")
    if bad:
        print("  INVALID > 0 — a claim points at a label that does not exist. Bug.")
    return a


# --------------------------------------------------------------------------
# THE LOOP
# --------------------------------------------------------------------------
SETS = {"tp": ("ANSWERABLE", lambda: [c.text for c in true_positives()]),
        "junk": ("JUNK", lambda: [q.text for q in JUNK]),
        "hn": ("HARD_NEG", lambda: [q.text for q in HARD_NEGATIVE])}


def show_set(key: str) -> None:
    name, get = SETS[key]
    print(f"\n  {name} — :{key} N to run one")
    for i, t in enumerate(get(), 1):
        print(f"   {i:>3}. {t}")


def show_cfg() -> None:
    print(f"\n  model            {config.GEN_MODEL}  think={config.GEN_THINK}")
    print(f"  temp / seed      {config.TEMPERATURE} / {config.GEN_SEED}")
    print(f"  num_ctx          {config.GEN_NUM_CTX:,}")
    print(f"  relevance floor  {config.RERANK_DROP_BELOW:+.1f}")
    print(f"  facet margin     {config.FACET_DROP_MARGIN:.1f}")
    print(f"  context cap      {config.GEN_MAX_CONTEXT_CHARS:,} chars")
    print(f"  claim overlap    {config.CLAIM_OVERLAP_MIN:.0%} (screen only)")
    print(f"  prompt           {prompts.CURRENT}")


def save(log: list[Answer], name: str) -> None:
    if not log:
        print("  nothing to save")
        return
    rows = []
    for a in log:
        r = a.retrieval
        rows.append({
            "question": a.question,
            "refused": a.refusal_reason if a.refused else None,
            "seconds": round(a.seconds, 1),
            "config_hash": a.config_hash,
            "text": a.text,
            "sources": a.sources,
            "claims": [{"text": c.text, "label": c.label, "citation": c.citation,
                        "overlap": c.overlap, "valid": c.valid,
                        "locator": c.locator, "align_verdict": c.align_verdict,
                        "align_score": c.align_score} for c in a.claims],
            "plan_mode": r.plan.mode if r else None,
            "inferred_fanout": r.plan.inferred_fanout if r else None,
            "notes": list(r.notes) if r else [],
            "passages": [{
                "label": f"S{i}", "short_name": p.short_name, "heading": p.heading,
                "chars": len(p.text), "full_chars": p.full_chars,
                "windowed": p.windowed, "is_bin": is_bin(p.heading),
                "pipe_lines": pipe_lines(p.text),
                "children": [h.cite for h in p.children],
                "text": p.text,
            } for i, p in enumerate(r.passages, 1)] if r else [],
        })
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    f = pathlib.Path("evaluation") / f"ask_{name or 'session'}_{stamp}.json"
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps({"tool": "ask.py", "model": config.GEN_MODEL,
                             "rows": rows}, indent=1), encoding="utf-8")
    print(f"  {len(rows)} question(s) -> {f}")


def handle(cmd: str, last: Answer | None, log: list[Answer]) -> bool:
    """A `:` command. Returns False to quit."""
    parts = cmd.split(maxsplit=1)
    head, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")

    if head in (":q", ":quit", ":exit"):
        return False
    if head in (":h", ":help", ":?"):
        print(__doc__[__doc__.index("COMMANDS"):])
    elif head == ":cfg":
        show_cfg()
    elif head == ":list":
        key = arg.lower()
        show_set(key) if key in SETS else print(f"  :list tp|junk|hn")
    elif head == ":save":
        save(log, re.sub(r"[^a-z0-9]+", "_", arg.lower()))
    elif last is None:
        print("  ask a question first")
    elif head == ":ev":
        evidence(last)
    elif head == ":claims":
        claims_panel(last)
    elif head == ":ctx":
        if last.retrieval is None or last.retrieval.refused:
            print("  no context — retrieval refused")
        else:
            ctx, labels = render.render(last.retrieval)
            print(f"\n{THIN}\nCONTEXT SENT TO THE MODEL — {len(ctx):,} chars\n{THIN}")
            print(ctx)
            print(f"\n  labels: {json.dumps(labels, indent=1)}")
    elif head == ":p":
        ps = last.retrieval.passages if last.retrieval else []
        try:
            p = ps[int(arg) - 1]
        except (ValueError, IndexError):
            print(f"  :p N   where N is 1..{len(ps)}")
        else:
            fl = flags_for(p)
            print(f"\n{THIN}\nS{arg}  {p.short_name}  {p.heading!r}\n"
                  f"  parent_id {p.parent_id}\n"
                  f"  {len(p.text):,} chars of {p.full_chars:,}"
                  + (f"   {fl}" if fl else "") + f"\n{THIN}")
            print(p.text)
    else:
        print(f"  unknown command {head!r} — :h for the list")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="answer one question, then exit")
    ap.add_argument("--no-preflight", action="store_true")
    # Tracing is OPT IN. Without this flag nothing is exported and no
    # collector is needed - the tracer is a no-op object.
    ap.add_argument("--trace", action="store_true",
                    help="send spans to the local Phoenix (localhost:6006)")
    ns = ap.parse_args()

    # THE ENTRY POINT DECIDES, not the library. init() raises if the
    # collector address is wrong - an opt-in that silently did nothing
    # would be worse than an error.
    if ns.trace:
        from regrag.observability import tracing
        tracing.init(verbose=True)

    if not ns.no_preflight:
        from regrag.generation import llm
        problem = llm.why_unavailable()
        if problem:
            sys.exit(f"FAIL: {problem}\n(Is Ollama running? Is Qdrant up? "
                     f"`docker compose up -d`)")

    if ns.question:
        ask(" ".join(ns.question))
        return 0

    print(LINE)
    print("ASK — one question at a time, with the evidence behind it.")
    print(f"  {config.GEN_MODEL}  temp {config.TEMPERATURE}  "
          f"floor {config.RERANK_DROP_BELOW:+.1f}  "
          f"facet margin {config.FACET_DROP_MARGIN:.1f}")
    print(f"  gate {'ON' if config.GATE_ENABLED else 'OFF'} "
          f"({config.GATE_MODEL} / {config.GATE_PROMPT})   REGRAG_GATE=0 to disable")
    print("  :h for commands, :q to quit. Nothing is saved unless you type :save.")
    print("  REMINDER: the gate is NOT a refusal guarantee. Measured 12-13 of 15")
    print("  out-of-corpus questions declined — the rest still come back cited.")
    print(LINE)

    last: Answer | None = None
    log: list[Answer] = []
    while True:
        try:
            line = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith(":"):
            key = line[1:].split(maxsplit=1)[0].lower()
            if key in SETS:                      # :tp 3 / :junk 1 / :hn 7
                _, get = SETS[key]
                items = get()
                arg = line.split(maxsplit=1)[1].strip() if " " in line else ""
                try:
                    line = items[int(arg) - 1]
                except (ValueError, IndexError):
                    show_set(key)
                    continue
            else:
                if not handle(line, last, log):
                    break
                continue

        try:
            got = ask(line)
            if got is not None:          # a stopped question has no Answer to
                last, _ = got, log.append(got)   # inspect or to save
        except KeyboardInterrupt:
            print("\n  interrupted")
        except Exception as exc:                                   # noqa: BLE001
            # A crash here must not lose the session. Print and carry on.
            print(f"\n  ERROR {type(exc).__name__}: {exc}")

    if log:
        print(f"\n  {len(log)} question(s) this session, not written to disk. "
              f"Use :save next time if you want the raw evidence kept.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
