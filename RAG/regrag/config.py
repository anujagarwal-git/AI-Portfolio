"""Calibrated values for regrag.

WHY THIS FILE EXISTS: every number below is a knob that changes results. If a
knob lives as a literal inside the function that uses it, then (a) nobody can
see what the system was configured to do, and (b) an eval run and a production
run can silently disagree. Putting them here makes the configuration VISIBLE,
VERSIONED (it is in git), and CONSTANT across runs.

IMPORTS NOTHING FROM regrag. This module sits at the bottom of the dependency
graph alongside domain.py, so it can never take part in a circular import.

Values marked UNCALIBRATED are placeholders that have not been measured yet.
Treat a placeholder that survives into a result as a bug.
"""

from __future__ import annotations

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
REGISTRY_PATH = DATA_DIR / "registry.yaml"
PROCESSED_DIR = DATA_DIR / "processed"

EVAL_DIR = PROJECT_ROOT / "evaluation"
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.jsonl"

# =============================================================================
# SERVICES — env-overridable, because Docker and bare-metal disagree on host
# =============================================================================
QDRANT_URL = os.getenv("REGRAG_QDRANT_URL", "http://localhost:6333")
OLLAMA_HOST = os.getenv("REGRAG_OLLAMA_HOST", "http://localhost:11434")
PHOENIX_URL = os.getenv("REGRAG_PHOENIX_URL", "http://localhost:6006")

# =============================================================================
# MODELS
# =============================================================================
# bge-small locally because this is a CPU box and 384 dims keeps the index
# cheap. The production upgrade path is BGE-M3 in-tenant — same family, much
# stronger multilingual/long-context behaviour, but not viable on CPU.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384

# BGE is an ASYMMETRIC model: queries get an instruction prefix, documents do
# not. Dropping this prefix silently degrades retrieval without any error.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# THE TRADE, 1.7B model is much faster at reading/writing tokens per second on a CPU. 
# Tried with 3B and 4b parameter model also. Answer quality did improve so does the answer time. Almost 
# doubled for 3B/4B model.

GEN_MODEL = "qwen3:1.7b"

# qwen3:1.7b is a HYBRID model: it writes a <think> block before answering
# unless told not to. On CPU that block can cost more than the answer, and the
# passages already contain the reasoning. Verified off 2026-08-26 - the output
# carried no <think> tags.

GEN_THINK = False
JUDGE_MODEL = "qwen2.5:7b-instruct"

# -- GENERATION CALL --------------------------------------------------------
# Called over raw HTTP, not through a framework wrapper. One dependency fewer,
# and every knob below is visible at the call site instead of buried in a
# client's defaults.
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"

# *** NUM_CTX IS A CORRECTNESS SETTING, NOT A TUNING KNOB. ***
# The MODEL supports 256k tokens. OLLAMA allocates 4,096 unless told otherwise,
# and anything past that is dropped with NO error and NO warning. A fan-out
# question has produced ~25,000 chars of passages (~7k tokens) and setting num_ctx=16,384 supports 
# ~80,000 chars, so the window stopped being the binding.
GEN_NUM_CTX = 16_384

# Reproducibility needs BOTH. temperature=0 alone is not always deterministic.
GEN_SEED = 0
GEN_TIMEOUT_S = 300          # CPU generation is slow; this is not a hot path.

# THIS IS NOW A TIME BUDGET, NOT A TRUNCATION GUARD.

# At read time of ~70 tok/s, every 10,000 chars costs ~30s before a word is
# written. 30,000 sits just above the largest fan-out context observed
# (~25,000) - high enough not to fire in normal use, low enough to catch a
# runaway before it costs a minute of CPU.
GEN_MAX_CONTEXT_CHARS = 30_000

# Temperature 0 is here for REPRODUCIBILITY first and faithfulness second.
# A non-zero temperature makes two eval runs incomparable, which destroys the
# ability to attribute a metric change to a code change.
TEMPERATURE = 0.0

# This exist for questions related to version comparison or change. Words like "superseded", "replaced" are considered 
#strong indicative words for comparison and old-vs-new wording rule fire if the question contains these kind of words.
#There are certain words which are considered as weak comparison words and would need a additional rule to fire. On internal 
# testing it was wrongly firing on general questions and retrieval was impacted. 
# Anyways, not removed outright but turned into a switch on/off indicator. Deliberately switched off. 

WEAK_VERSION_TERMS_ENABLED = os.getenv("REGRAG_WEAK_VERSION", "0") != "0"

# =============================================================================
# THE INTENT GATE
# =============================================================================
# One small model in FRONT of retrieval. It sorts a question into refuse /
# clarify / decline / pass-through. 

# REGRAG_GATE=0 turns it off. `respond()` then calls `answer()` directly and
# the fate is escaped
GATE_ENABLED = os.getenv("REGRAG_GATE", "1") != "0"

# The SAME model as generation, on purpose: it is already resident in Ollama,
# so the gate costs no second model load. Measured ~+0.2s median per question.
# It also beat qwen2.5:3b (44/60) and gemma3:1b (23/60, plus 32 unparsable
# responses on the harder prompt) on the same 60 questions.

GATE_MODEL = "qwen3:1.7b"

# the prompt version file. any change in prompt version can be tuned here to see the change in gate decisions.
GATE_PROMPT = "intent_v1"

# The reply is five short JSON fields. 200 is generous; capping it keeps the
# WRITE half of the latency near zero, which is what makes the gate affordable
# on every question including the ones that pass straight through.
GATE_MAX_TOKENS = 200

# RAGAS 0.2.15 with ChatOllama needs format="json" or the judge output fails to
# parse. 
JUDGE_FORMAT = "json"
RAGAS_TIMEOUT_S = 900
RAGAS_MAX_WORKERS = 1

# =============================================================================
# SECTIONING — Basel chapter detection
# =============================================================================
# How many text items after a chapter code to search for its "Version effective
# as of" stamp. In the BIS layout the stamp sits in the chapter's title block,
# a handful of blocks after the code.
#
# DO NOT widen this to make a stubborn chapter pass. A wider window can reach
# into the NEXT chapter's title block and attach the wrong date — and a wrong
# effective date is far worse than a loud failure, because it silently makes a
# superseded clause look current. If a chapter won't resolve, open the PDF.
SECTION_STAMP_WINDOW = 20

# =============================================================================
# CHUNKING — decision log 2026-07-23
# Even though the project uses parent child chunking strategy but the parameter placeholders are kept 
# for Semantic and recursvie chunking strategy also. The parent doc chunking strategy was decided empirically 
# after doing a protoytype on SR 11-7 document.

# =============================================================================
CHUNK_STRATEGIES = ("recursive", "semantic", "parentdoc")

RECURSIVE_CHUNK_SIZE = 800
RECURSIVE_CHUNK_OVERLAP = 100

SEMANTIC_BREAKPOINT_PERCENTILE = 90

# parentdoc: retrieve the PARAGRAPH (precise), deliver the SECTION (context).
PARENTDOC_PARENT = "section"
PARENTDOC_CHILD = "paragraph"

# CHILD_MIN_CHARS. Only applied to parent which has multiple chil.d. 
# To drop noisy children which does not contain any concrete information.
CHILD_MIN_CHARS = 120


#merge child when the fragment ends WITHOUT terminal punctuation AND the next item 
# starts lowercase. Neither signal mentions length, so a SHORT COMPLETE unit stays
# short on purpose. Upper bound on a repair, so a run of fragments cannot snowball into a 
# chunk the size of its parent.
CHILD_MERGE_CEILING = 700

# SPLIT above this. bge-small-en-v1.5 accepts 512 tokens (~1,800-2,000 chars);
# beyond that the tail is silently dropped, which is the failure mode with no
# error message. Only 6 children in the whole corpus (0.1%) are affected —
# 4 in the IFRS 9 extract, 1 in CRE53, 1 in 12 CFR 252.
CHILD_MAX_CHARS = 1_800

# =============================================================================
# VECTOR STORE — ONE COLLECTION PER CHUNKING STRATEGY
#
# Not one per framework. Framework is a payload FILTER; making it a collection
# would mean adding a jurisdiction required new infrastructure instead of a new
# filter value, which breaks "corpus change = data change".
# =============================================================================
COLLECTION_VERSION = "v1"


def collection_name(strategy: str) -> str:
    """e.g. 'regrag_recursive_v1'. Never hard-code a collection name."""
    if strategy not in CHUNK_STRATEGIES:
        raise ValueError(f"unknown chunk strategy {strategy!r}; expected one of {CHUNK_STRATEGIES}")
    return f"regrag_{strategy}_{COLLECTION_VERSION}"


# Payload indexes are what make filtered search fast instead of a full scan.
# These must match the filterable fields Document.payload() emits.
PAYLOAD_INDEX_FIELDS = (
    "doc_id",
    "subject",
    "framework",
    "jurisdiction",
    "issuer",
    "status",
    "applicability",
    "authority_rank",
    "doc_type",
)

# =============================================================================
# RETRIEVAL
# =============================================================================
DENSE_TOP_K = 20
LEXICAL_TOP_K = 20
RRF_K = 60  # the constant in sum(1/(k+rank)); rewards agreement across rankers
RERANK_TOP_N = 5

# Gate for dropping irrelevant chunks. Value arrived by testing on a sample of 30 questions. 
RERANK_DROP_BELOW = 0.0

# *** THE FACET MARGIN (ONLY FOR MULTI FACET QUESTIONS) or more precisely when the inferred fanout rule fires— drop a dead SLICE, not a weak chunk. ***
#if a facet's best chunk is far below the winning  facet's best chunk, 
# that facet had nothing on topic and should not spend
# budget or context.

#measured over 21 multi-facet questions:

# ONLY WHERE THE FAN-OUT WAS INFERRED. Same reasoning as the fallback pool: if
# the user NAMES two jurisdictions, coverage is what they asked for, and
# dropping the weaker one answers a different question.

FACET_DROP_MARGIN = 6.0

# *** HOW MANY CHUNKS THE FLOOR'S RESCUE KEEPS ***
# The rescue fires when EVERY chunk of a facet scores below RERANK_DROP_BELOW
# and the user NAMED the target - retrieval must not come back empty on a
# document someone asked for by name. It used to keep exactly two, the
# highest-scoring, and that is a documented failure:

# WHY TOP-2 AND NOT A LOWER FLOOR: this path only runs when the cross-encoder
# has already scored the whole facet below zero.

FLOOR_RESCUE_KEEP = 2

# The bar for the claim/passage word-overlap SCREEN. Below it, a claim is
# flagged for a human to read - it is never dropped and never shown to the
# reader. 0.45 was picked by eye and it does not have to be sharp: a wrong call
# costs one line of review, not an answer.
CLAIM_OVERLAP_MIN = 0.45

# --- CITATION ALIGNMENT
# The model reads a whole parent WINDOW; the shipped citation used to name the
# CHILD THAT WON RETRIEVAL, which is a different paragraph some time.
# The aligner scores each claim against each paragraph of the delivered text
# and cites the best match using IDF. 
ALIGN_ENABLED = os.getenv("REGRAG_ALIGN", "1") != "0"

# MEASURED, NOT CHOSEN BY FEEL. On the 20-question run of 2026-09-02 every
# correct re-citation scored >= 0.731 and every wrong one <= 0.574, with
# nothing in between; 0.65 sits in the middle of that empty band. 
# Below the floor the aligner keeps the citation that would have shipped.
ALIGN_FLOOR = float(os.getenv("REGRAG_ALIGN_FLOOR", "0.65"))

# Near-identical boilerplate paragraphs can score within a hair of each other.
# Closer than this and the aligner has not distinguished them — defer.
ALIGN_MARGIN = 0.05

# =============================================================================
# RERANKING — the senior underwriter after the filter
#
# A bi-encoder embeds question and chunk SEPARATELY and compares two points.
# A cross-encoder reads the pair TOGETHER, with attention across both. Far more
# accurate but time taking on CPU. So capped to top 20 chunks per facet.

# =============================================================================
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20        # per facet, into the cross-encoder
RERANK_BATCH_SIZE = 32

# How many chunks allowed for each facet. Per-facet quota. 
# Balance must be guaranteed BY CONSTRUCTION, not hoped for:

FACET_QUOTA_DEFAULT = 5

# =============================================================================
# CONTEXT SIZE — bounded at DELIVERY, not at chunking
# =============================================================================

# THE PARENT CAP IS A FAIRNESS DEVICE BETWEEN PASSAGES; THE BUDGET IS THE LIMIT.
# With one passage there is nobody to be fair to, so capping it at 5,000 while
# 20,000 of budget goes unused is pure loss — a precise question aimed at one
# CRE or CAP section should deliver that section. So each passage gets its FAIR
# SHARE of the budget, never less than the floor:
#     cap = max(PARENT_MAX_CHARS, CONTEXT_MAX_CHARS // passage_count)
#   1 passage -> 25,000   2 -> 12,500   3 -> 8,333   5+ -> 5,000

# Max characters a parent is allowed to retain before passing the context to LLM

PARENT_MAX_CHARS = 5_000    

# Retrieval's own budget. search.py uses it to divide space between passages 
# and to stop adding passages once the total would go over 25,000 
# before passing the context to LLM

CONTEXT_MAX_CHARS = 25_000  

# EXCEPT for bins. A bin never gets the fair-share exception; it is always windowed to the floor.
PARENT_BIN_HEADINGS = frozenset({"footnotes", "footnote", "notes", "table", "tables", ""})
# =============================================================================
# REQUIREMENT LOOKUP
#
# Only standards and rules state requirements. BCBS *guidelines* say "banks
# should" — retrieved as a requirement, "should" becomes "must" and the answer
# is wrong about the one thing that matters. reference_data (authority_rank 0)
# states no requirement at all.
# =============================================================================
REQUIREMENT_DOC_TYPES = ("standard", "rule")

# authority_rank >= this counts as legally binding in its jurisdiction.
BINDING_RANK_FLOOR = 4
