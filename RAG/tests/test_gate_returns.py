"""THE TWELVE QUESTIONS THE GATE RETURNED. NO OLLAMA, NO QDRANT.

Anuj read the gate column of the 48-question run on 2026-09-09 and named two
defects. This file is the falsifying condition for both, fixed before the
re-run:

  1. NINE questions were blocked with
         "MODEL_RISK_MANAGEMENT exists for ['UK', 'US'] - which one?"
     followed by all 19 document names. Every one of them (gs-08, gs-09 and
     their kind) NAMES BOTH jurisdictions in the sentence. Anuj: "For multi
     jurisdiction in the question, it should seek for one applicable document
     under each jurisdiction rather than listing out all the available
     documents."

  2. THREE questions were DECLINED - gs-13, gs-14, gs-44 - because the
     extractor put "us supervisory guidance" / "us guidance" in the document
     field and `_names_hit` read a phrase it did not recognise as a document
     it does not hold. Anuj: "The 3 questions that were returned had the word
     guidance in the question, which is read as document."

WHAT IS ACTUALLY PROTECTED HERE
    Not "does the gate get 48/48" - no unit test can say that. What is
    protected is the module's own standing rule, which both defects broke from
    opposite sides:

        only a document we can NAME and do not hold may produce OUT_OF_CORPUS,
        and a clarifying question may only be asked about something the
        question has not already said.

    So the regression guards matter as much as the fixes. "Solvency II" must
    still decline, and a subject spanning two jurisdictions must still ask when
    the question named neither. A fix that widens ANSWERABLE far enough to
    swallow those has traded a cheap failure for the expensive one.

Every expected value is derived from the grid, never typed, so adding a row to
registry.yaml cannot make a test silently wrong.
"""
from __future__ import annotations

import pytest

from regrag.gate import intent as gate


@pytest.fixture(scope="module")
def grid():
    return gate.corpus_grid()


@pytest.fixture(scope="module")
def multi(grid):
    """A subject the corpus holds for MORE THAN ONE jurisdiction, and its list.

    This is the shape the "which one?" branch fires on. On the current registry
    it is MODEL_RISK_MANAGEMENT x {UK, US} - the exact cell in the nine blocked
    questions - but it is looked up, not typed.
    """
    g, subjects, _j, _n = grid
    for su in subjects:
        have = sorted({j for (s, j) in g if s == su and g[(s, j)]})
        if len(have) > 1:
            return su, have
    pytest.skip("no multi-jurisdiction subject in this registry")


# ---------------------------------------------------------------------------
# DEFECT 1 - two jurisdictions named is a FAN-OUT, not a choice
# ---------------------------------------------------------------------------
def test_the_question_that_names_two_jurisdictions_is_read_as_naming_two():
    """gs-08's own words. If this fails, nothing below it can pass."""
    q = ("What do UK and US supervisory expectations each say about "
         "maintaining a model inventory?")
    assert sorted(gate.jurisdictions_named(q)) == ["UK", "US"]
    # ... and the single-value read still refuses to pick a winner, because the
    # extraction dict has one field and neither value is the whole truth.
    assert gate.jurisdiction_named(q) == ""


def test_naming_both_jurisdictions_never_asks_the_jurisdiction_again(multi):
    """DEFECT 1's guard, NARROWED 2026-09-10.

    The question the gate must never re-ask is the JURISDICTION one - the
    sentence already answered it. Whether it answers or asks now depends on the
    cell sizes (see DEFECT 4 below), so this asserts the thing defect 1 was
    about and nothing more. Asserting ANSWERABLE here would encode a second
    claim the fix never made.
    """
    su, have = multi
    d, why = gate.decide({"banking": True, "subject": su,
                          "subject_from": "question text",
                          "jurisdiction": None, "jurisdictions": have})
    assert "exists for" not in why, why      # the old "UK or US - which one?"
    assert d in ("ANSWERABLE", "INCOMPLETE"), why
    if d == "INCOMPLETE":
        assert "document" in why, why        # asking about DOCUMENTS, not
                                             # about the jurisdiction


def test_a_jurisdiction_the_corpus_lacks_costs_that_facet_and_nothing_else(multi):
    """One named jurisdiction covered, one not. The covered one still answers.

    The alternative - refusing because part of the question cannot be served -
    would decline a question the corpus half-answers, which is the failure this
    gate's whole design is arranged against.
    """
    g, _subjects, jurisdictions, _names = gate.corpus_grid()
    su, have = multi
    absent = [j for j in jurisdictions if not g.get((su, j))]
    if not absent:
        pytest.skip(f"{su} is held for every jurisdiction")
    d, why = gate.decide({"banking": True, "subject": su,
                          "subject_from": "question text",
                          "jurisdiction": None,
                          "jurisdictions": [have[0], absent[0]]})
    assert d == "ANSWERABLE", why
    assert have[0] in why


def test_one_jurisdiction_named_still_asks_which_one(multi):
    """THE REGRESSION GUARD. The ask is not gone, it is narrowed.

    A question that names a two-jurisdiction subject and NO jurisdiction has
    told us nothing to fan out on, so the clarifying turn is still the right
    outcome - and still cheaper than a wrong answer.
    """
    su, have = multi
    d, why = gate.decide({"banking": True, "subject": su, "jurisdiction": None})
    assert d == "INCOMPLETE", why
    assert "which one" in why


def test_two_named_but_neither_covered_asks_and_never_falls_through(multi):
    """`covered`, not `asked`. The registry decides, the sentence only asks.

    RENAMED 2026-09-10. The old name said "falls through to the old path", and
    the falling through WAS the defect: it landed in the `if not ju:` branch,
    which is written for a question that named NO jurisdiction. Now the block
    returns here, and the reason names the jurisdictions the sentence asked
    for - which is the evidence that the new branch, not the old one, decided.
    """
    g, _s, _j, _n = gate.corpus_grid()
    su, have = multi
    absent = [j for j in ("US", "UK", "GLOBAL") if not g.get((su, j))]
    if len(absent) < 2:
        pytest.skip(f"{su} lacks fewer than two jurisdictions")
    d, why = gate.decide({"banking": True, "subject": su,
                          "subject_from": "question text",
                          "jurisdiction": None, "jurisdictions": absent[:2]})
    assert d == "INCOMPLETE", why
    assert str(absent[:2]) in why, why


def test_an_unrecognised_jurisdiction_value_cannot_smuggle_in_an_answer(multi):
    """`jurisdictions` is filtered against the vocabulary like every other
    field. A model that returns ["EU", "APAC"] must change nothing."""
    su, _have = multi
    d, why = gate.decide({"banking": True, "subject": su, "jurisdiction": None,
                          "jurisdictions": ["EU", "APAC"]})
    assert d == "INCOMPLETE", why


# ---------------------------------------------------------------------------
# DEFECT 2 - "guidance" is a KIND of document, not a document
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("doc", ["us guidance", "us supervisory guidance",
                                 "the previous us guidance", "supervisory guidance",
                                 "pra expectations", "federal reserve guidance",
                                 "bank of england rules", ""])
def test_a_phrase_of_descriptors_names_no_document(doc, grid):
    _g, _s, _j, names = grid
    kind, _hit = gate._names_hit(doc, names)
    assert kind == "NONE", f"{doc!r} was read as {kind}"


def test_a_phrase_of_descriptors_never_declines(grid):
    """The whole point. gs-13, gs-14 and gs-44 died here.

    The subject x jurisdiction cell now decides, exactly as if the document
    field had been empty - which, in substance, it was.
    """
    g, _s, _j, _n = grid
    su, ju = next((s, j) for (s, j) in g if g[(s, j)])
    d, why = gate.decide({"banking": True, "document": "us supervisory guidance",
                          "subject": su, "jurisdiction": ju})
    assert d == "ANSWERABLE", why


def test_a_real_title_we_do_not_hold_still_declines(grid):
    """THE REGRESSION GUARD, and the one that matters most.

    OUT_OF_CORPUS is the branch the gate exists for. If the generic-word list
    ever grows far enough to swallow a real title, this fails - which is why
    the list holds descriptors and nothing else.
    """
    _g, _s, _j, names = grid
    for doc in ("solvency ii", "dodd-frank act", "mifid ii"):
        kind, _hit = gate._names_hit(doc, names)
        assert kind == "ABSENT", f"{doc!r} was read as {kind}"
        d, why = gate.decide({"banking": True, "document": doc})
        assert d == "OUT_OF_CORPUS", why


def test_a_document_we_hold_is_still_matched_exactly(grid):
    _g, _s, _j, names = grid
    for full in names.values():
        kind, hit = gate._names_hit(full.lower(), names)
        assert kind in ("EXACT", "FAMILY"), f"{full!r} was read as {kind}"
        if kind == "EXACT":
            assert hit in names.values()


def test_no_document_we_hold_is_built_only_from_generic_words(grid):
    """A STRUCTURAL GUARD ON THE LIST ITSELF.

    If a registry short_name were ever composed entirely of words on the
    generic list, that document would become permanently unnameable - the gate
    would read its own title as "not a title". Cheaper to catch here than in a
    run.
    """
    import re

    _g, _s, _j, names = grid
    generic = gate._generic_tokens()
    for low, full in names.items():
        toks = set(re.findall(r"[a-z0-9]+", low))
        assert toks - generic, f"{full!r} is entirely generic words"


# ---------------------------------------------------------------------------
# DEFECT 3 (2026-09-10) - two jurisdictions named, subject held under NEITHER
#
# Anuj: "What do UK and US supervisory expectations each say about maintaining
# capital?" was ANSWERED. The subject read correctly as capital adequacy, which
# this corpus holds only for GLOBAL, so neither named jurisdiction covered it -
# and the block fell through to a branch written for a question that named no
# jurisdiction at all. Anuj's rule: only answer if the subject exists UNDER A
# NAMED JURISDICTION; otherwise ask, listing the documents held under each one
# and saying outright where the subject IS held.
#
# Every expected value below is derived from the grid, never typed.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def elsewhere_only(grid):
    """A subject held ONLY outside {UK, US}, and the jurisdictions holding it.

    On the current registry that is CAPITAL_ADEQUACY x GLOBAL - Anuj's own
    failing question - but it is looked up, so a registry change cannot leave
    this test quietly asserting about a cell that no longer exists.
    """
    g, subjects, _j, _n = grid
    named = ("UK", "US")
    for su in subjects:
        have = sorted({j for (s, j) in g if s == su and g[(s, j)]})
        if have and not set(have) & set(named):
            return su, have, list(named)
    pytest.skip("every subject is held for UK or US")


def test_subject_held_only_elsewhere_must_not_answer(elsewhere_only):
    """THE FALSIFYING CONDITION for Anuj's capital question.

    If this ever returns ANSWERABLE again, the fall-through is back.
    """
    su, _have, named = elsewhere_only
    d, why = gate.decide({"banking": True, "subject": su,
                          "subject_from": "question text",
                          "jurisdiction": None, "jurisdictions": named})
    assert d == "INCOMPLETE", why
    assert "no ambiguity" not in why, why


def test_the_ask_says_where_the_subject_is_actually_held(elsewhere_only):
    """The follow-up must state the fact the registry knows.

    Listing UK and US documents alone would send the user hunting through a
    menu that cannot answer their question.
    """
    su, have, named = elsewhere_only
    g, _s, _j, _n = gate.corpus_grid()
    x = {"banking": True, "subject": su, "subject_from": "question text",
         "jurisdiction": None, "jurisdictions": named}
    d, why = gate.decide(x)
    msg = gate.clarifying_question(x, why)
    for j in have:                       # where it IS held
        assert j in msg, msg
    for doc in g[(su, have[0])]:         # and the documents there, by name
        assert doc in msg, msg


def test_the_ask_lists_per_named_jurisdiction_not_the_whole_registry(
        elsewhere_only):
    """The nineteen-document dump is what Anuj objected to. It must be gone
    HERE, and (see the guard below) still present everywhere else."""
    su, _have, named = elsewhere_only
    g, _s, _j, names = gate.corpus_grid()
    x = {"banking": True, "subject": su, "subject_from": "question text",
         "jurisdiction": None, "jurisdictions": named}
    msg = gate.clarifying_question(x, gate.decide(x)[1])
    for j in named:
        assert f"  {j}: " in msg, msg
    # a document held under NEITHER named jurisdiction nor the subject's own
    # cell must not appear - that is the whole-registry dump by another name.
    listed = {n for (_s2, j), ns in g.items() if j in named for n in ns}
    listed |= {n for j in _have for n in g[(su, j)]}
    strays = [n for n in names.values() if n not in listed and n in msg]
    assert not strays, strays


def test_two_jurisdictions_with_a_model_read_subject_must_not_answer(multi):
    """Anuj: "The subject inferred by LLM should not go as a tie breaker."

    Same rule, applied one step earlier. Without `subject_from` the subject is
    the 1.7b extractor's own label - wrong eight consecutive times - so it may
    cost a clarifying turn and must never decide coverage. This is the SAME
    dict as the ANSWERABLE test above, minus the provenance.
    """
    su, have = multi
    d, why = gate.decide({"banking": True, "subject": su,
                          "jurisdiction": None, "jurisdictions": have})
    assert d == "INCOMPLETE", why


def test_anuj_capital_sentence_end_to_end_without_a_model():
    """His exact words, read by the deterministic readers only.

    `subject_named` and `jurisdictions_named` are pure regex over the planner's
    vocabulary, so this runs the real path with no Ollama: read the sentence,
    build the dict the way `classify()` does, decide.
    """
    q = ("What do UK and US supervisory expectations each say about "
         "maintaining capital?")
    su = gate.subject_named(q)
    jus = gate.jurisdictions_named(q)
    if not su or len(jus) < 2:
        pytest.skip(f"vocabulary no longer reads this sentence: {su!r} {jus!r}")
    x = {"banking": True, "subject": su, "subject_from": "question text",
         "jurisdiction": None, "jurisdictions": jus}
    d, why = gate.decide(x)
    g, _s, _j, _n = gate.corpus_grid()
    covered = [j for j in jus if g.get((su, j))]
    if covered:
        pytest.skip(f"registry now holds {su} for {covered}")
    assert d == "INCOMPLETE", why


# ---------------------------------------------------------------------------
# DEFECT 4 (2026-09-10) - a covered cell holding TWO documents is a CHOICE
#
# Anuj, as an ADDED FEATURE, not a fix: "If two jurisdictions are named and the
# subject and jurisdiction are identified but the subject has multiple document
# hits, then ask the user which document to look for. If single hit then
# nothing required - just let the user know which document is locked for that
# jurisdiction."
#
# WHY IT MATTERS BEYOND THE GATE. The fan-out reserves a slot per jurisdiction,
# but which document fills it is decided by RRF alone - the crowding that put
# 12 CFR and SR 15-18 in the reserved US slots on gs-09 instead of SR 26-2.
# Asking removes the guess for the one case where the user already knows.
#
# THE LIST THIS ASKS FROM IS THE CELL, not the jurisdiction. That is what
# separates it from the no-subject branch, which has no cell to narrow to.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def crowded(grid):
    """A subject, a covered jurisdiction holding >1 document, and a second
    named jurisdiction. On the current registry: MODEL_RISK_MANAGEMENT, US
    (SR 26-2 + SR 11-7) and UK (SS1/23 alone) - looked up, never typed."""
    g, subjects, _j, _n = grid
    for su in subjects:
        have = sorted({j for (s, j) in g if s == su and g[(s, j)]})
        many = [j for j in have if len(g[(su, j)]) > 1]
        if many and len(have) > 1:
            other = [j for j in have if j != many[0]]
            return su, many[0], other[0]
    pytest.skip("no subject with a crowded cell and a second jurisdiction")


def test_a_crowded_cell_asks_which_document(crowded):
    su, many, other = crowded
    d, why = gate.decide({"banking": True, "subject": su,
                          "subject_from": "question text",
                          "jurisdiction": None, "jurisdictions": [many, other]})
    assert d == "INCOMPLETE", why
    assert many in why, why
    assert "document" in why, why


def test_the_ask_names_the_locked_one_and_only_asks_about_the_other(crowded):
    """A jurisdiction with one document is SETTLED. Asking about it would make
    the user answer a question nobody had."""
    g, _s, _j, _n = gate.corpus_grid()
    su, many, other = crowded
    if len(g[(su, other)]) != 1:
        pytest.skip(f"{su} x {other} is not a single-document cell")
    x = {"banking": True, "subject": su, "subject_from": "question text",
         "jurisdiction": None, "jurisdictions": [many, other]}
    msg = gate.clarifying_question(x, gate.decide(x)[1])
    assert f"{other}: using {g[(su, other)][0]}" in msg, msg
    for doc in g[(su, many)]:
        assert doc in msg, msg
    assert "which one?" in msg, msg


def test_the_ask_offers_the_cell_not_the_whole_jurisdiction(crowded):
    """THE REGRESSION GUARD FOR THIS FEATURE. A document that sits under the
    crowded jurisdiction but under a DIFFERENT subject must not be offered -
    that would be the whole-jurisdiction list, which is the no-subject case."""
    g, _s, _j, _n = gate.corpus_grid()
    su, many, other = crowded
    x = {"banking": True, "subject": su, "subject_from": "question text",
         "jurisdiction": None, "jurisdictions": [many, other]}
    msg = gate.clarifying_question(x, gate.decide(x)[1])
    off_subject = {n for (s, j), ns in g.items()
                   if j == many and s != su for n in ns}
    strays = sorted(n for n in off_subject if n in msg)
    assert not strays, strays


def test_every_covered_cell_single_still_answers(multi):
    """The feature must not turn a settled question into a question. When each
    covered cell holds exactly one document there is nothing to choose."""
    g, _s, _j, _n = gate.corpus_grid()
    su, have = multi
    singles = [j for j in have if len(g[(su, j)]) == 1]
    if len(singles) < 2:
        pytest.skip(f"{su} has fewer than two single-document cells")
    d, why = gate.decide({"banking": True, "subject": su,
                          "subject_from": "question text",
                          "jurisdiction": None, "jurisdictions": singles[:2]})
    assert d == "ANSWERABLE", why


# ---------------------------------------------------------------------------
# DEFECT 5 (2026-09-10) - the version fan-out gets the same treatment
#
# Anuj: "Apply the logic to version fanout too. If superseded or current has
# one document, lock it and tell it explicitly. If one side has one document
# (lock it) and the other has multiple, ask for that one. If both have
# multiple, ask for applicable documents under both."
#
# Planner rule 3 reserves a CURRENT slot and a HISTORIC slot; which document
# fills each is RRF's guess. A bucket of one is not a guess.
#
# THE BUCKET IS NARROWED BY SUBJECT - and that is an OFFER, not a filter. The
# planner's `where` clauses are untouched; `subject` still filters nothing,
# because Basel CRE is CAPITAL_ADEQUACY and holds CRE36's validation section.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def versioned():
    """A subject with BOTH a current and a superseded document, and the
    jurisdiction they share. Currently MODEL_RISK_MANAGEMENT x US
    (SR 26-2 in force, SR 11-7 superseded) - looked up, never typed."""
    vv, _c, _h = gate.status_view()
    for (su, ju, bucket), _ns in sorted(vv.items()):
        if bucket != "current":
            continue
        if vv.get((su, ju, "historic")):
            return su, ju
    pytest.skip("no subject with both a current and a superseded document")


def test_the_version_reader_is_deterministic_and_shared():
    """One vocabulary, two readers - the third pair after subject and
    jurisdiction. If the planner stops calling this a version question, so
    must the gate."""
    from regrag.retrieval import planner

    q = "What changed in the model risk guidance compared with the old one?"
    assert gate.version_named(q) is True
    assert planner.plan(q).mode == "version_fanout"
    assert gate.version_named("What is a model inventory?") is False


def test_both_buckets_single_is_answerable_and_names_both(versioned):
    su, ju = versioned
    vv, _c, _h = gate.status_view()
    if len(vv[(su, ju, "current")]) > 1 or len(vv[(su, ju, "historic")]) > 1:
        pytest.skip("a bucket holds more than one document")
    d, why = gate.decide({"banking": True, "subject": su,
                          "subject_from": "question text",
                          "jurisdiction": ju, "version": True})
    assert d == "ANSWERABLE", why
    assert vv[(su, ju, "current")][0] in why, why
    assert vv[(su, ju, "historic")][0] in why, why


def test_a_crowded_bucket_asks_and_the_single_one_is_stated(versioned):
    """Whichever side is crowded, the settled side must be NAMED, not asked."""
    su, ju = versioned
    vv, _c, _h = gate.status_view()
    cur, hist = vv[(su, ju, "current")], vv[(su, ju, "historic")]
    if len(cur) == 1 and len(hist) == 1:
        pytest.skip("neither bucket is crowded on this registry")
    x = {"banking": True, "subject": su, "subject_from": "question text",
         "jurisdiction": ju, "version": True}
    d, why = gate.decide(x)
    assert d == "INCOMPLETE", why
    msg = gate.clarifying_question(x, why)
    for label, docs_here in (("in force now", cur),
                             ("the version it replaced", hist)):
        if len(docs_here) == 1:
            assert f"{label}: using {docs_here[0]}" in msg, msg
        else:
            assert "which" in msg, msg


def test_a_subject_with_no_superseded_document_falls_through(versioned):
    """ONE-SIDED CORPORA MUST STAY SILENT. A subject the corpus holds only in
    force has no comparison to offer, and inventing a refusal for it would be
    the empty-cell decline this gate spent two sessions removing."""
    vv, _c, _h = gate.status_view()
    one_sided = [(su, ju) for (su, ju, b) in vv
                 if b == "current" and not vv.get((su, ju, "historic"))]
    if not one_sided:
        pytest.skip("every subject has a superseded document")
    su, ju = one_sided[0]
    d, why = gate.decide({"banking": True, "subject": su,
                          "subject_from": "question text",
                          "jurisdiction": ju, "version": True})
    assert d != "OUT_OF_CORPUS", why
    assert "version question" not in why, why


def test_a_model_read_subject_cannot_trigger_the_version_branch(versioned):
    """Same rule as everywhere else: only the subject read from SUBJECT_TERMS
    may decide. Without `subject_from` this must not produce a version
    verdict."""
    su, ju = versioned
    d, why = gate.decide({"banking": True, "subject": su,
                          "jurisdiction": ju, "version": True})
    assert "version question" not in why, why


# ---------------------------------------------------------------------------
# DEFECT 6 (2026-09-10) - the model INVENTS document names
#
# Anuj typed "US" as the reply to a clarifying question. `ask.py` appended it as
# "(US)". The 1.7b extractor returned {"document": "US GAAP"} and the gate
# DECLINED. "gaap" is in nothing he typed.
#
# The 2026-09-09 descriptor rule could not catch it: "US GAAP" is not built
# entirely of descriptors, because "gaap" is a real title word.
#
# THE RULE RESTORED: `decide()` says OUT_OF_CORPUS is safe because it rests on
# a fact the model READ rather than a label it CHOSE. Nothing checked that.
# Now every content token of the document field must appear in the question.
# THIS GATES THE DECLINE ONLY - an EXACT match still answers.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("doc,question,seen", [
    ("US GAAP", "What validation activities are expected? (US)", False),
    ("Solvency II", "What does Solvency II require?", True),
    ("US", "anything at all", False),          # all generic -> nothing named
    ("SR 11-7", "What does SR 11-7 say about validation?", True),
])
def test_document_seen_reads_the_question_not_the_model(doc, question, seen):
    assert gate.document_seen(doc, question) is seen


def test_an_invented_title_cannot_decline(grid):
    """THE FALSIFYING CONDITION for Anuj's "(US)" turn."""
    _g, subjects, jurisdictions, _n = grid
    d, why = gate.decide({"banking": True, "document": "US GAAP",
                          "document_seen": False,
                          "subject": subjects[0], "jurisdiction": None})
    assert d != "OUT_OF_CORPUS", why


def test_a_title_the_question_did_contain_still_declines(grid):
    """THE REGRESSION GUARD THAT MATTERS MOST. Suppressing the decline for an
    invented name must not suppress it for a real one the user actually wrote.
    """
    _g, subjects, _j, _n = grid
    d, why = gate.decide({"banking": True, "document": "solvency ii",
                          "document_seen": True,
                          "subject": subjects[0], "jurisdiction": None})
    assert d == "OUT_OF_CORPUS", why


def test_an_unstamped_dict_keeps_the_old_behaviour(grid):
    """Only an explicit False suppresses. A dict with no `document_seen` makes
    no claim either way, so every existing test and caller is unchanged."""
    _g, subjects, _j, _n = grid
    d, why = gate.decide({"banking": True, "document": "solvency ii",
                          "subject": subjects[0], "jurisdiction": None})
    assert d == "OUT_OF_CORPUS", why


def test_a_document_we_hold_answers_even_if_the_model_resolved_it(grid):
    """The check gates the DECLINE, never the answer. A user who wrote "the PRA
    supervisory statement" and a model that resolved it to SS1/23 has been
    READ, not invented, and downgrading that would cost an answer to save
    nothing."""
    _g, _s, _j, names = grid
    real = sorted(names.values())[0]
    d, why = gate.decide({"banking": True, "document": real,
                          "document_seen": False,
                          "subject": None, "jurisdiction": None})
    assert d == "ANSWERABLE", why
