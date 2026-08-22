# Regulatory RAG — a retrieval assistant for banking regulation

A retrieval-augmented generation system over a 19-document corpus of banking
regulation: Basel Framework chapters, US model risk and capital planning
guidance, the PRA's UK equivalent, IFRS 9 impairment, and the CCAR rules.

**Status: ingestion pipeline complete and tested. Retrieval and generation are
being rebuilt for this corpus.** See [Current state](#current-state).

---

## The problem this is built around

Ask a generic RAG system *"what is the minimum CET1 capital requirement for a
US bank?"* over this corpus and it will answer **4.5%**, confidently, with two
citations.

Both are wrong:

- **Basel RBC** states *"Common Equity Tier 1 must be at least 4.5% of
  risk-weighted assets"* — but BCBS text binds nobody until a jurisdiction
  adopts it. It is not US law.
- **The 2026 DFAST results** contain the numeral 4.5 in a CET1 table — but
  those are stress-test *outcomes*, not requirements.

Two independent-looking sources agreeing on the same number, and neither
answers the question. The actual requirement lives in 12 CFR 217, which this
corpus does not hold — 12 CFR Part 252 defines CET1 only by cross-reference to
`217.20(b)`.

**The correct answer is a refusal that says where the answer lives.**

Retrieval alone cannot produce that. It needs curated metadata that the
documents do not carry about themselves.

---

## What makes this different from a document Q&A demo

### Legal force is metadata, not content

A regulation never states its own authority. Nothing in the Basel sentence
above says *"I bind nobody."* So it cannot be embedded, cannot be inferred, and
cannot be recovered downstream — it has to be curated.

Every document carries an `authority_rank` doing two structurally different
jobs:

| Job | Mechanism |
|---|---|
| **Eligibility** | rank 0 (reference data) is inadmissible for a requirement question, whatever it scores |
| **Precedence** | among admissible sources, 5 (binding rule) outranks 3 (supervisory guidance) outranks 2 (non-binding international standard) |

Separately, `doc_type` distinguishes **standards** from **guidelines** — BCBS
publishes both, and the guidelines say *"banks should"*, never *"banks must"*.
Retrieved as a requirement, *should* silently becomes *must*.

### Version bleed produces inverted answers, not vague ones

SR 11-7 (2011) and SR 26-2 (2026) are two versions of the same US model risk
guidance. On spreadsheets they say opposite things:

> **SR 11-7:** "User-developed applications, such as spreadsheets … are
> particularly prone to model risk."
>
> **SR 26-2:** excludes "simple arithmetic calculations, such as those found
> within spreadsheets."

Both quotable. A version-bleed failure yields a fluent, well-cited, **inverted**
answer. Both documents share a `framework` key, so
`(framework=US_MRM_GUIDANCE, status=in_force)` resolves the current version by
construction rather than by hope.

### Effective dates live at chapter level, not document level

Basel's consolidated chapters each carry their own version stamp. `BASEL_SCO`
spans **15 Dec 2019 to 01 Jan 2026** inside one PDF, so any document-level date
would be false for five of its six chapters. The registry deliberately leaves
`effective_from` null there, and the parser extracts the real date per chapter —
verified across all six Basel files, 48 chapters in total.

### The system knows the boundary of its own corpus

`known_gaps` records instruments the corpus *references* but does not *hold*,
with the 12 sections of 12 CFR 217 that Parts 252 and 225.8 actually
cross-reference — extracted from the corpus, not chosen by hand. A refusal that
names the gap is auditable; *"I don't know"* is indistinguishable from a
retrieval failure.

---

## Architecture

```
data/registry.yaml          19 curated documents — the corpus map
        │
   registry.py              THE SEAM. Nothing downstream opens the YAML.
        │                   THE GATE. verified:false is refused, not warned about.
        ▼
   parser.py  ──►  cache.py        PDF → DoclingDocument, cached by
        │                          content hash + Docling version
        ▼
   clean.py                 artifacts out, content preserved
        ▼
   sections.py              chapter codes + per-chapter effective dates
        ▼
   chunker.py               ← next
        ▼
   embedder / vector_store / planner / search / prompts / generate
        ▼
   api.py                   single composition root: eval and prod cannot drift
```

**Corpus changes are data changes.** Adding a jurisdiction adds filter *values*
to the registry. No function signature moves.

### Three orthogonal retrieval axes

| Axis | Field | Type | Why |
|---|---|---|---|
| Framework | `subject` | **fan-out** | subjects overlap — model validation appears in SR 11-7 *and* in Basel CRE36 §8, as a capital eligibility condition |
| Jurisdiction | `jurisdiction` | filter | a clause cannot be both US and UK law |
| Version | `framework` + `status` | filter | exactly one member of a version family is in force |

Framework is a *fan-out* axis because scoping a validation query to model risk
would silently delete the capital-side half of the answer — and CRE is ~25×
less dense in that vocabulary per page, so it loses on raw similarity even when
relevant. Per-facet quota forces it into context.

---

## Current state

| Stage | Status |
|---|---|
| Registry + metadata schema | ✅ 19 documents, schema v4 |
| Parsing, caching, cleaning, sectioning | ✅ built and tested |
| Chunking | 🔜 next |
| Embedding, indexing, retrieval, generation | ⬜ prototype exists; being rebuilt for this corpus |
| Evaluation | ⚠️ see below |

**On the prototype evaluation results in `evaluation/`:** they are kept for
transparency and are **not** valid comparisons. The "semantic" run turned out to
be byte-identical to the "recursive" run on all 11 questions, and the LLM judge
proved non-deterministic — scoring identical context *and* identical response
0.5 in one run and 1.0 in another, and scoring a verbatim quotation of SR 11-7
as 0.00 faithful. The chunking strategy was chosen by reading the answers
by hand instead. Eval methodology is being reworked.

---

## Testing

63 tests, of which the interesting ones assert **failure**:

- **16 registry tests deliberately corrupt the YAML** — a vocabulary typo, a
  one-sided supersession link, a `reference_data` row typed as a `standard` —
  and assert the loader refuses. A validator nobody has watched fail is a
  comment, not a control.
- **Cleaning tests mostly assert what must SURVIVE.** CRE contains
  `M{{i}}, E{{i}}, S{{i}}` — mathematical subscripts — alongside `{{IRB}}`, a
  glossary link. Same syntax, two meanings. Deleting the braces yields
  "M, E, S", which still reads like a formula and no longer means anything.
  So the rule is unwrap, never delete.
- **A PDF preflight checks all 19 files** for structural integrity in two
  seconds. It exists because one download was caught half-finished — valid
  `%PDF-` header, no `%%EOF` trailer, still growing.

```bash
uv run pytest -q
```

---

## Setup

```bash
uv sync
uv pip install -e .
```

**The corpus is not included** (see [Corpus](#corpus)). Place the 19 source PDFs
at the paths recorded in `data/registry.yaml` — every row carries its
`source_url` — then:

```bash
docker compose up -d                              # Qdrant
uv run python -m regrag.registry                  # validate the corpus map
uv run python -m regrag.ingestion.cache --warm    # parse once (slow, once)
uv run pytest -q
```

Without the PDFs, `registry.py` refuses to load — it verifies every `file` path
resolves before returning a single document. That is deliberate: a corpus map
pointing at files that are not there should fail loudly, not half-load.

Requires Python 3.12+. Runs CPU-only: `bge-small-en-v1.5` for embeddings,
Qwen via Ollama for generation, Qdrant for vectors.

---

## Corpus

19 documents, ~1,100 pages.

| Subject | Documents |
|---|---|
| Capital adequacy | Basel CAP, CRE, LEX, RBC, SCO |
| Model risk management | SR 11-7, SR 26-2, PRA SS1/23 |
| Stress testing | Basel SRP32, BCBS d450, 12 CFR 225.8, 12 CFR 252, SR 15-18, SR 15-19, Fed 2026 scenarios + results |
| Credit impairment | IFRS 9 (credit-risk extract), BCBS d403 |
| Data quality | BCBS 239 |

**The source PDFs are not in this repository.** They are published by the BIS,
the Federal Reserve, the PRA and the IFRS Foundation and remain their
copyright, and redistribution terms differ by publisher — BIS covers permit
"brief excerpts", which a complete 323-page chapter is not.

What *is* here is `data/registry.yaml`, which records a `source_url` for every
document alongside its curated metadata. The corpus is described precisely
enough to be rebuilt from the registry; the files themselves are not
redistributed.
