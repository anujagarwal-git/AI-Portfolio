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

GEN_MODEL = "qwen3:1.7b"
JUDGE_MODEL = "qwen2.5:3b-instruct"

# Temperature 0 is here for REPRODUCIBILITY first and faithfulness second.
# A non-zero temperature makes two eval runs incomparable, which destroys the
# ability to attribute a metric change to a code change.
TEMPERATURE = 0.0

# RAGAS 0.2.15 with ChatOllama needs format="json" or the judge output fails to
# parse. Run eval OUTSIDE Jupyter (py3.13 asyncio conflict).
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
# Both strategy families are kept deliberately and compared empirically in
# Stage 3. The comparison is a feature of the project, not indecision.
# =============================================================================
CHUNK_STRATEGIES = ("recursive", "semantic", "parentdoc")

RECURSIVE_CHUNK_SIZE = 800
RECURSIVE_CHUNK_OVERLAP = 100

SEMANTIC_BREAKPOINT_PERCENTILE = 90

# parentdoc: retrieve the PARAGRAPH (precise), deliver the SECTION (context).
PARENTDOC_PARENT = "section"
PARENTDOC_CHILD = "paragraph"

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

# Per-facet quota. Balance must be guaranteed BY CONSTRUCTION, not hoped for:
# a pooled reranker has no notion of coverage and will delete a whole facet to
# raise global precision, and no precision metric will report the loss.
FACET_QUOTA_DEFAULT = 5

# =============================================================================
# GATE THRESHOLDS — A DICT KEYED BY ISSUER, NOT A SCALAR
#
# Decision 2026-08-11. A single global threshold would contradict the very
# argument used to justify per-facet reranking: cross-encoder scores are not
# calibrated across corpora, so "0.5" does not mean the same thing on a BIS
# chapter as on an eCFR section.
#
# ALL VALUES BELOW ARE UNCALIBRATED PLACEHOLDERS.
# =============================================================================
GATE_THRESHOLD_DEFAULT = 0.30  # UNCALIBRATED
GATE_THRESHOLD: dict[str, float] = {
    "BCBS": 0.30,             # UNCALIBRATED
    "FED": 0.30,              # UNCALIBRATED
    "US_INTERAGENCY": 0.30,   # UNCALIBRATED
    "PRA": 0.30,              # UNCALIBRATED
    "IASB": 0.30,             # UNCALIBRATED
}


def gate_threshold(issuer: str) -> float:
    return GATE_THRESHOLD.get(issuer, GATE_THRESHOLD_DEFAULT)


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
