"""The cross-encoder. Reorders a shortlist; never searches, never filters.

WHY A SECOND MODEL AT ALL
    bge-small is a BI-encoder: it turns the question into one point and each
    chunk into another, then compares the two points. The chunk was embedded
    days ago, with no idea what would be asked of it.

    A CROSS-encoder reads the question and the chunk TOGETHER in one pass, with
    attention running across both. It can see that "validate" in the question
    and "validation of internal estimates" in the chunk are the same idea in
    this context. That is strictly more information — and strictly more
    expensive, because there is nothing to precompute: every (question, chunk)
    pair is a fresh forward pass.

    So it can never replace retrieval. 9,935 pairs per question is not a
    system. It reorders a shortlist of twenty. Cheap recall, expensive
    precision, in that order — the two-stage underwriting pattern.

TWO RULES THAT MATTER HERE
1.  SCORE CHILDREN, NEVER PARENTS. The model truncates at a few hundred tokens.
    A 5,000-char parent would be cut off, and the cut part is often the part
    that mattered. Children are capped at 1,800 chars by the chunker, so they
    fit. Rerank the child, deliver the parent — the same split the whole
    parentdoc design rests on.

2.  THE SCORE IS AN ORDERING, NOT A MEASUREMENT. It is not a probability and
    not calibrated across corpora. Never compare a BCBS score to an eCFR score.

    TESTED 2026-08-26, AND THE LINE ABOVE HELD WHERE IT MATTERED. Using the
    score to decide a WHOLE QUESTION failed exactly as the bi-encoder did:
    unanswerable questions reached +6.99 while answerable ones fell to +3.38.
    No threshold separates them. That idea is closed.

    What DID survive is narrower and does not contradict the rule: a per-CHUNK
    floor (config.RERANK_DROP_BELOW). Dropping a weak chunk removes evidence;
    it does not decide anything. Ten verified answerable questions kept at
    least one chunk each. The distinction to hold on to is BEST-score-decides-
    question (dead) versus each-chunk-earns-its-place (works).

The model is downloaded from HuggingFace on first use (~90MB) and cached.
"""

from __future__ import annotations

from functools import lru_cache

from regrag import config


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)


def available() -> bool:
    """Can the cross-encoder be loaded at all? Never raises.

    Retrieval must still work on a machine that cannot reach HuggingFace —
    reranking is an improvement, not a dependency.
    """
    try:
        _model()
        return True
    except Exception:
        return False


def pair_text(payload: dict, *, metadata: bool) -> str:
    """What the cross-encoder actually reads for one chunk.

    ONLY METADATA THAT VARIES INSIDE A FACET CAN DISCRIMINATE.
    The reranker orders candidates within ONE facet, and facets partition on
    doc_id / jurisdiction / status — so those fields are CONSTANT across every
    candidate it sees. For "what does CRE36.122 require?" all 20 candidates are
    Basel CRE, so prepending "Basel CRE" costs tokens and says nothing.
    That leaves two fields worth adding:

      locator         "CRE36.122"  — the citation form exists NOWHERE in the
                      chunk text (the document prints "36.122"), which is the
                      same gap that made BM25 rank the right paragraph 45th.
      parent_heading  "Principle 1.2 Model inventory" — a short child is often
                      meaningless alone while its heading carries the topic.
                      Measured 2026-08-23: the reranker put `Basel CRE Model
                      validation` above `SS1/23 Principle 1.2 Model inventory`
                      for a question about the model inventory. The phrase was
                      in the heading and we withheld it.

    Framed as prose, not pipe-delimited fields — the model was trained on
    (question, passage) pairs, so "CRE36.122. Section 8: validation of internal
    estimates. Banks must..." is closer to its training than a table row.
    """
    text = payload.get("text") or ""
    if not metadata:
        return text
    loc = (payload.get("locator") or "").strip()
    head = (payload.get("parent_heading") or "").strip()
    bits = [b for b in (loc, head if head.lower() != loc.lower() else "") if b]
    return f"{'. '.join(bits)}. {text}" if bits else text


def score(query: str, texts: list[str]) -> list[float]:
    """Relevance of each text to the query. ONE batched pass, not one per text.

    Batching is the whole game on CPU. Three facets of twenty candidates is
    sixty pairs; sixty separate calls pay the per-call overhead sixty times,
    while one call of sixty amortises it once. The caller therefore hands over
    EVERY facet's candidates together and splits the scores back afterwards.
    """
    if not texts:
        return []
    out = _model().predict([(query, t) for t in texts],
                           batch_size=config.RERANK_BATCH_SIZE,
                           show_progress_bar=False)
    return [float(x) for x in out]
