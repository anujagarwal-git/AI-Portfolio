"""Run a Plan against the index and assemble what the generator will read.

    question -> plan() -> embed ONCE -> per facet: dense + BM25 -> RRF
             -> round-robin merge -> resolve parents -> bound size -> passages

FOUR THINGS THAT ARE EASY TO GET WRONG, AND WHY THEY MATTER HERE
1.  FILTER BEFORE RANKING, ON BOTH ARMS.
    Qdrant filters server-side, so the dense arm is fine. BM25 has no filter, so
    the corpus is narrowed to the facet FIRST and ranked second. Ranking the
    whole corpus and dropping non-matches afterwards silently returns fewer than
    the quota — the facet is starved and nothing reports it.

2.  ONE CANONICAL CHUNK LIST.
    The BM25 corpus is read back OUT of Qdrant, not rebuilt from the chunker.
    Two independently built lists drift, and a drifted chunk_id resolves to
    nothing. Reading from the index makes disagreement impossible.

3.  RRF FUSES RANKS, NOT SCORES.
    Cosine similarity and BM25 scores are on different scales and cannot be
    added. RRF only asks "how high did each arm put this?", which needs no
    calibration — the reason it was chosen over a weighted sum.

4.  ROUND-ROBIN AFTER THE QUOTA, NOT A GLOBAL SORT.
    The quota guarantees every facet reaches the candidate set. Sorting all
    candidates by score and cutting to the budget would hand the window back to
    the densest facet and undo the quota at the last step, after paying for it.

WHY BM25 EARNS ITS PLACE IN THIS CORPUS
    Dense retrieval finds MEANING; BM25 finds the EXACT STRING. Regulatory text
    is full of terms where the string IS the concept — CRE36.122, SICR,
    12 CFR 252.54, SS1/23. A near-synonym is not an acceptable answer for an
    identifier. That is also why the tokenizer keeps dotted and hyphenated forms
    intact: split "36.122" into "36" and "122" and the lexical arm stops adding
    anything this corpus needs.

    Each BM25 document is the chunk's LOCATOR + its TEXT, because people cite a
    paragraph the way it is referenced ("CRE36.122") and the document prints it
    the way it is numbered ("36.122"). Indexing text alone left the citation
    form unfindable — see the comment in _corpus().
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache

from regrag import config
from regrag.index.embedder import embed_query
from regrag.index.vector_store import VectorStore
from regrag.registry import load as load_registry
from regrag.retrieval.planner import Facet, Plan, plan as make_plan

# Keeps "36.122", "sr", "11-7", "ss1/23", "252.54" whole. Identifiers are the
# whole reason the lexical arm exists; a tokenizer that shreds them is useless.
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase words, identifiers kept whole AND split into their parts.

    Both forms are emitted so "CRE36.122" answers a query for the exact string
    and also for the bare "36.122". The SAME function must run at build time and
    at query time — two tokenizers is the classic way to get an index that
    silently matches nothing.
    """
    out: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        out.append(tok)
        if any(c in tok for c in ".-/"):
            out.extend(p for p in re.split(r"[.\-/]", tok) if p)
    return out


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Hit:
    chunk_id: str
    payload: dict
    facet: str
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rrf: float = 0.0
    rerank: float | None = None

    @property
    def cite(self) -> str:
        p = self.payload
        return f"{p.get('short_name', '?')} {p.get('locator') or p.get('parent_heading') or ''}".strip()

    @property
    def arms(self) -> str:
        d = "-" if self.dense_rank is None else str(self.dense_rank + 1)
        l = "-" if self.lexical_rank is None else str(self.lexical_rank + 1)
        r = "" if self.rerank is None else f"/ce{self.rerank:+.1f}"
        return f"dense#{d}/bm25#{l}{r}"


@dataclass
class Passage:
    """One PARENT, delivered once, with every child that matched inside it."""

    parent_id: str
    short_name: str
    heading: str
    text: str
    windowed: bool
    full_chars: int
    children: list[Hit] = field(default_factory=list)

    @property
    def facets(self) -> list[str]:
        return sorted({h.facet for h in self.children})

    def __str__(self) -> str:
        w = f" (windowed from {self.full_chars:,})" if self.windowed else ""
        kids = "; ".join(f"{h.cite} [{h.arms}]" for h in self.children)
        return (f"[{self.short_name}] {self.heading[:52]!r} "
                f"{len(self.text):,}c{w}  <- {kids}")


@dataclass
class Dropped:
    """A candidate that was FETCHED AND SCORED, then discarded by a cut.

    RECORDED, NEVER READ BACK. Nothing in retrieval consults this list; it
    exists so a study can ask what the cuts cost. Collecting it changes no
    decision, which is the only way the measurement stays honest.

    WHAT IT CANNOT SEE. A chunk that dense top-20 and BM25 top-20 never
    surfaced from the index has no row here and never will. This measures
    loss AT THE CUTS, not retrieval recall. Recall needs the golden set.

    `loss` says whether the drop actually cost the prompt anything:
        none          this chunk's parent was delivered, text included
        windowed_out  parent delivered but trimmed, and this text is not in
                      the delivered window - HEURISTIC, see _resolve_loss
        absent        the parent never reached the prompt at all
    """

    cut: str
    facet: str
    parent_id: str
    short_name: str
    heading: str
    text: str
    rerank: float | None
    rrf: float
    dense_rank: int | None
    lexical_rank: int | None
    rank_in_facet: int | None
    locator: str | None
    chars: int
    loss: str = "?"


def _record(sink: list, cut: str, hits: list, ranks: list | None = None) -> None:
    for i, h in enumerate(hits):
        pay = h.payload
        sink.append(Dropped(
            cut=cut, facet=h.facet,
            parent_id=str(pay.get("parent_id") or ""),
            short_name=str(pay.get("short_name") or "?"),
            heading=str(pay.get("parent_heading") or ""),
            text=str(pay.get("text") or ""),
            rerank=h.rerank, rrf=h.rrf,
            dense_rank=h.dense_rank, lexical_rank=h.lexical_rank,
            rank_in_facet=(ranks[i] if ranks is not None else i),
            locator=pay.get("locator"),
            chars=len(str(pay.get("text") or "")),
        ))


def _norm(s: str) -> str:
    return " ".join((s or "").split())


def _contained_in(a: "Passage", b: "Passage") -> bool:
    """Is a's whole text already inside b's, in the SAME document?"""

    if a.short_name != b.short_name or a.parent_id == b.parent_id:
        return False
    x, y = _norm(a.text), _norm(b.text)
    return bool(x) and len(x) < len(y) and x in y


def _resolve_loss(casualties: list, delivered: list) -> None:
    """Did the drop actually cost the prompt anything?

    A chunk dropped by the quota whose PARENT still arrived via a sibling
    chunk cost nothing - the model read that text anyway. Counting those as
    losses would inflate every number here, so they are separated out.

   """
    by_pid = {p.parent_id: p for p in delivered}
    for d in casualties:
        if d.cut == "context_budget":
            d.loss = "absent"
            continue
        pas = by_pid.get(d.parent_id)
        if pas is None:
            d.loss = "absent"
        elif not pas.windowed:
            d.loss = "none"
        else:
            head = _norm(d.text)[:120]
            d.loss = "none" if head and head in _norm(pas.text) else "windowed_out"


@dataclass
class Retrieval:
    question: str
    plan: Plan
    passages: list[Passage]
    notes: list[str] = field(default_factory=list)
    refused: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    dropped: list[Dropped] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.passages)

    def __str__(self) -> str:
        if self.refused:
            return f"REFUSED {self.question!r}: {self.refused}"
        head = (f"{self.question!r}\n  [{self.plan.mode}] {len(self.plan.facets)} facet(s) "
                f"-> {len(self.passages)} passage(s), {self.total_chars:,} chars")
        body = "\n".join(f"  - {p}" for p in self.passages)
        tail = "\n".join(f"  ! {n}" for n in self.notes)
        t = ("  " + "  ".join(f"{k} {v * 1000:.0f}ms" for k, v in self.timings.items())
             if self.timings else "")
        return "\n".join(x for x in (head, body, tail, t) if x)


# ---------------------------------------------------------------------------
# THE CORPUS — read back out of Qdrant so there is exactly one chunk list.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=2)
def _corpus(strategy: str):
    """(payloads, tokenised texts, BM25 index) for the whole collection.

    IDF is a property of the WHOLE corpus, so BM25 is built once over every
    chunk and never per facet. Narrowing happens on the RESULT of scoring, by
    restricting which candidates are eligible — not by rebuilding the index,
    which would change every term's IDF and make facets incomparable.
    """
    from rank_bm25 import BM25Okapi

    store = VectorStore(strategy=strategy)
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection, limit=2_000,
            offset=offset, with_payload=True, with_vectors=False)
        payloads.extend(p.payload for p in points)
        if offset is None:
            break
    
    tokens = [tokenize(f"{p.get('locator') or ''} {p.get('text') or ''}") for p in payloads]
    return payloads, tokens, BM25Okapi(tokens)


def _matches(payload: dict, where: dict) -> bool:
    for key, want in where.items():
        got = payload.get(key)
        if isinstance(want, (list, tuple, set)):
            if got not in want:
                return False
        elif got != want:
            return False
    return True


@lru_cache(maxsize=1)
def _registry_payloads() -> tuple[dict, ...]:
    return tuple(row.payload() for row in load_registry().indexable())


def _facet_can_match(where: dict) -> bool:
    """Does ANY registry row satisfy this filter?

    The UK corpus is one document (SS1/23, model risk only) and there is no US
    credit-impairment document, because 12 CFR 217 is a recorded gap. A facet
    over those slices can only return noise. Saying so beats running it.
    """
    keys = {k: v for k, v in where.items() if k != "status"}
    return any(_matches(p, keys) for p in _registry_payloads()) if keys else True


# ---------------------------------------------------------------------------
def _rrf(dense: list[str], lexical: list[str], k: int) -> dict[str, float]:
    """sum(1 / (k + rank)). Rank-based, so the two arms need no common scale.

    k is large (60) on purpose: it flattens the difference between rank 1 and
    rank 5, so the fusion rewards a chunk BOTH arms liked over one that a single
    arm loved. Agreement is the signal.
    """
    scores: dict[str, float] = {}
    for ranked in (dense, lexical):
        for rank, cid in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _run_facet(facet: Facet, vector, payloads, tokens, bm25, store) -> list[Hit]:
    # The store is PASSED IN, not built here. Building one per facet opened a
    # fresh connection each time — ~379ms apiece, against 20ms for a search on
    # a connection that already exists. A three-facet question paid for three
    # connections to the same database in the same second.

    # --- dense arm: Qdrant filters server-side --------------------------
    dense_pts = store.search(vector, limit=config.DENSE_TOP_K, where=facet.where)
    dense_ids = [p.payload["chunk_id"] for p in dense_pts]
    by_id = {p.payload["chunk_id"]: p.payload for p in dense_pts}

    # --- lexical arm: narrow FIRST, rank SECOND -------------------------
    eligible = [i for i, p in enumerate(payloads) if _matches(p, facet.where)]
    scores = bm25.get_scores(tokenize(facet.query))
    eligible.sort(key=lambda i: -scores[i])
    lex_ids = []
    for i in eligible[:config.LEXICAL_TOP_K]:
        if scores[i] <= 0:
            break                      # no term overlap at all — not a candidate
        cid = payloads[i]["chunk_id"]
        lex_ids.append(cid)
        by_id.setdefault(cid, payloads[i])

    fused = _rrf(dense_ids, lex_ids, config.RRF_K)
    d_rank = {c: r for r, c in enumerate(dense_ids)}
    l_rank = {c: r for r, c in enumerate(lex_ids)}
    # CANDIDATES, not the final five. The quota is taken AFTER reranking —
    # cutting to 5 here would mean the cross-encoder never sees the candidates
    # it exists to rescue. Without reranking the caller simply takes the first
    # `quota` of this list, which is the old behaviour exactly.
    ordered = sorted(fused, key=lambda c: -fused[c])[:config.RERANK_CANDIDATES]
    return [Hit(cid, by_id[cid], facet.label, d_rank.get(cid), l_rank.get(cid), fused[cid])
            for cid in ordered]


def is_bin(heading: str) -> bool:
    """A parent that is a BAG, not a section.

    'Footnotes' holds 81 unrelated notes in Basel CAP. There is no narrative to
    preserve and no heading a reader could look up, so the whole-parent case
    never applies to one — the window around the matched child is strictly more
    useful than the bag. Same defect as the 726 chunks with no usable citation.
    """
    h = re.sub(r"[^a-z0-9 ]+", " ", (heading or "").lower()).strip()
    return h in config.PARENT_BIN_HEADINGS or len(heading or "") > 90


def _window(parent_text: str, child_text: str, cap: int) -> tuple[str, bool]:
    """Trim an oversized parent to a window around the matched child.

    Grows outward a PARAGRAPH at a time, never by character count: a character
    cut lands mid-requirement, and half a requirement reads like a whole one.
    """
    if len(parent_text) <= cap:
        return parent_text, False

    paras = [p for p in parent_text.split("\n\n") if p.strip()]
    key = (child_text or "")[:60]
    centre = next((i for i, p in enumerate(paras) if key and key in p), 0)

    lo = hi = centre
    total = len(paras[centre])
    while True:
        grew = False
        if hi + 1 < len(paras) and total + len(paras[hi + 1]) + 2 <= cap:
            hi += 1; total += len(paras[hi]) + 2; grew = True
        if lo - 1 >= 0 and total + len(paras[lo - 1]) + 2 <= cap:
            lo -= 1; total += len(paras[lo]) + 2; grew = True
        if not grew:
            break

    out = "\n\n".join(paras[lo:hi + 1])
    if lo > 0:
        out = "[...]\n\n" + out
    if hi < len(paras) - 1:
        out = out + "\n\n[...]"
    return out, True


# ---------------------------------------------------------------------------
def retrieve(question: str, *, strategy: str = "parentdoc", quota: int | None = None,
             requirement_only: bool = False, rerank: bool = True,
             rerank_metadata: bool = True) -> Retrieval:
    """Plan, search, merge, and assemble the passages for one question.

    config.RERANK_DROP_BELOW, the relevance floor.
    `rerank_metadata` is ON, because IF reranking is used, locator + parent_heading
    + text beats bare text on the same evidence"""

    p = make_plan(question, quota=quota, requirement_only=requirement_only)

    # A vector search NEVER returns nothing: '' scored 0.820 against this index.
    # Refusing at the door is cheaper and more honest than filtering afterwards.

    if len(re.findall(r"[a-z]", question.lower())) < 3:
        return Retrieval(question, p, [], refused="no searchable text in the question")

    t_start = time.perf_counter()
    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    store = VectorStore(strategy=strategy)
    payloads, tokens, bm25 = _corpus(strategy)
    timings["corpus"] = time.perf_counter() - t0    # ~0 after the first call

    t0 = time.perf_counter()
    vector = embed_query(question)          # ONCE — every facet shares the query
    timings["embed"] = time.perf_counter() - t0
    notes: list[str] = []
    casualties: list[Dropped] = []

    per_facet: list[list[Hit]] = []
    for facet in p.facets:
        if not _facet_can_match(facet.where):
            notes.append(f"facet {facet.label} matches NO document in the registry — skipped")
            continue
        t0 = time.perf_counter()
        hits = _run_facet(facet, vector, payloads, tokens, bm25, store)
        timings["search"] = timings.get("search", 0.0) + (time.perf_counter() - t0)
        if not hits:
            notes.append(f"facet {facet.label} returned nothing")
        per_facet.append(hits)

    # --- rerank: ONE batched pass over EVERY facet's candidates ----------
    # Batching across facets is the point. Sixty pairs in one call amortises the
    # per-call cost once; three calls of twenty pay it three times. The scores
    # are then split back BY FACET, because the quota is per facet — a global pooled
    # cut would let the strongest facet take every slot, which is the loss the
    # quota exists to prevent, arriving one step later than before.
    if rerank and per_facet:
        t0 = time.perf_counter()
        flat = [h for hits in per_facet for h in hits]
        try:
            from regrag.retrieval import reranker

            scores = reranker.score(
                question,
                [reranker.pair_text(h.payload, metadata=rerank_metadata) for h in flat])
        except Exception as exc:                     # noqa: BLE001
            # Reranking is an improvement, not a dependency. A machine that
            # cannot load the model still gets RRF order rather than nothing.
            notes.append(f"rerank unavailable ({type(exc).__name__}) — RRF order kept")
            scores = []
        if scores:
            by_id = {}
            for h, sc in zip(flat, scores):
                by_id[h.chunk_id] = replace(h, rerank=sc)
           
            per_facet = [[by_id[h.chunk_id] for h in hits] for hits in per_facet]
        timings["rerank"] = time.perf_counter() - t0

    # --- the facet margin, BEFORE the chunk floor ------------------------
    # Drop the dead SLICE first (whole facet), then filter chunks inside what remains. A
    # facet with nothing on topic should not get a second chance to sneak one
    # marginal chunk past the floor. See config.FACET_DROP_MARGIN.

    if (rerank and config.FACET_DROP_MARGIN is not None and p.inferred_fanout
            and len(per_facet) > 1):
        bests = [max((h.rerank for h in hits if h.rerank is not None), default=None)
                 for hits in per_facet]
        if any(b is not None for b in bests):
            top = max(b for b in bests if b is not None)
            keep, cut, cut_hits = [], [], []
            for hits, b in zip(per_facet, bests):
                if b is None or top - b > config.FACET_DROP_MARGIN:
                    if hits:
                        cut.append(f"{hits[0].facet}"
                                   + (f" ({b:+.1f}, {top - b:.1f} below)" if b is not None
                                      else " (no scores)"))
                        cut_hits.append(hits)
                else:
                    keep.append(hits)
            if cut and keep:
                for _lost in cut_hits:
                    _record(casualties, "facet_margin", _lost,
                            list(range(len(_lost))))
                notes.append(f"facet margin {config.FACET_DROP_MARGIN:.1f}: dropped "
                             + ", ".join(cut) + f"; winner {top:+.1f}")
                per_facet = keep

    # --- the relevance floor, BEFORE the quota ---------------------------
   
    if rerank and config.RERANK_DROP_BELOW is not None:
        before = sum(len(h) for h in per_facet)
        kept_facets = []
        for hits in per_facet:
            if not hits:
                continue
            survivors = [h for h in hits
                         if h.rerank is None or h.rerank >= config.RERANK_DROP_BELOW]
            # *** THE FLOOR TRIMS. IT NEVER EMPTIES. ***
            # RULE: if the user named a document or a jurisdiction, retrieval
            # never comes back empty. If the planner recognised nothing at all,
            # it may.
            if not survivors and not p.inferred_fanout:
                # TOP-K (which is 2). See config.FLOOR_RESCUE_KEEP 
                survivors = sorted(
                    hits, key=lambda h: -(h.rerank if h.rerank is not None
                                          else h.rrf))[:config.FLOOR_RESCUE_KEEP]
                kept = ", ".join(f"{h.rerank:+.1f}" if h.rerank is not None
                                 else "no score" for h in survivors)
                notes.append(f"relevance floor: every chunk in {hits[0].facet} scored "
                             f"below {config.RERANK_DROP_BELOW:+.1f}, but the question "
                             f"NAMED its target — keeping the top "
                             f"{len(survivors)} ({kept}) rather than emptying a "
                             f"slice the user asked for")
            _kept_ids = {h.chunk_id for h in survivors}
            _lost = [(i, h) for i, h in enumerate(hits)
                     if h.chunk_id not in _kept_ids]
            _record(casualties, "relevance_floor", [h for _, h in _lost],
                    [i for i, _ in _lost])
            if survivors:
                kept_facets.append(survivors)
        per_facet = kept_facets
        after = sum(len(h) for h in per_facet)
        if after < before:
            notes.append(f"relevance floor {config.RERANK_DROP_BELOW:+.1f}: "
                         f"{before - after} of {before} chunk(s) dropped")
        if after == 0:
            # Only reachable on an inferred fan-out - a named target was rescued
            # above. The planner recognised nothing AND nothing scored above the
            # floor: junk by construction, twice over.
            _resolve_loss(casualties, [])
            return Retrieval(question, p, [], notes,
                             refused="planner recognised nothing and no chunk scored "
                                     "above the relevance floor",
                             timings=timings, dropped=casualties)

    # --- the quota, per facet, AFTER reranking ---------------------------
    _quota = p.facets[0].quota
    for _hits in per_facet:
        _record(casualties, "quota", _hits[_quota:],
                list(range(_quota, len(_hits))))
    per_facet = [hits[:_quota] for hits in per_facet]

    # --- FALLBACK ONLY: the merged pool competes on MERIT -----------------
    # `fallback` means the planner recognised NOTHING — no subject, no
    # jurisdiction, no document. 
    
    if p.inferred_fanout and any(h.rerank is not None
                                    for hits in per_facet for h in hits):
        _all_hits = [h for hits in per_facet for h in hits]
        pooled = sorted(_all_hits,
                        key=lambda h: -(h.rerank or 0.0))[:p.facets[0].quota]
        _pool_ids = {h.chunk_id for h in pooled}
        _lost = [(i, h) for i, h in enumerate(_all_hits)
                 if h.chunk_id not in _pool_ids]
        _record(casualties, "fallback_pool", [h for _, h in _lost],
                [i for i, _ in _lost])
        kept = {h.facet for h in pooled}
        notes.append(
            f"fallback + rerank: the {sum(len(h) for h in per_facet)} facet winners "
            f"competed on cross-encoder score for {p.facets[0].quota} slots — "
            f"{', '.join(sorted(kept))} survived"
            + (f" (dropped {', '.join(sorted({h.facet for hits in per_facet for h in hits} - kept))})"
               if len({h.facet for hits in per_facet for h in hits} - kept) else ""))
        per_facet = [pooled]

    # --- round-robin: one from each facet in turn ------------------------
    ordered: list[Hit] = []
    for i in range(max((len(h) for h in per_facet), default=0)):
        for hits in per_facet:
            if i < len(hits):
                ordered.append(hits[i])

    # --- pass 1: resolve parents, deduplicated, NO capping yet -----------
    # The cap depends on HOW MANY passages there are, which is not known until
    # every hit has been resolved. Capping while building would use a count that
    # is still growing.
    t0 = time.perf_counter()
    passages: list[Passage] = []
    seen: dict[str, Passage] = {}
    for hit in ordered:
        pid = hit.payload.get("parent_id")
        if pid in seen:
            seen[pid].children.append(hit)       # same parent, another child
            continue
        parent = store.get_parent(pid) or {}
        full = parent.get("text") or hit.payload.get("text") or ""
        pas = Passage(pid, str(hit.payload.get("short_name", "?")),
                      str(parent.get("heading") or hit.payload.get("parent_heading") or ""),
                      full, False, len(full), [hit])
        seen[pid] = pas
        passages.append(pas)

    # --- pass 1b: drop a passage wholly contained in another --------------
    # See _contained_in for why this is
    # an exact containment test and not a similarity threshold.
    # Run BEFORE the fair share is computed, so the freed budget goes to the
    # passages that remain rather than going unused.
    if len(passages) > 1:
        swallowed = {id(a) for a in passages
                     for b in passages if _contained_in(a, b)}
        if swallowed:
            for pas in passages:
                if id(pas) in swallowed:
                    notes.append(f"{pas.short_name} {pas.heading[:28]!r} "
                                 f"({len(pas.text):,} chars, {pas.parent_id}) dropped: "
                                 f"its text is already inside another passage "
                                 f"from the same document")
            passages = [p for p in passages if id(p) not in swallowed]

    # --- pass 2: each passage gets its FAIR SHARE of the budget -----------
    share = max(config.PARENT_MAX_CHARS,
                config.CONTEXT_MAX_CHARS // max(1, len(passages)))
    for pas in passages:
        cap = config.PARENT_MAX_CHARS if is_bin(pas.heading) else share
        pas.text, pas.windowed = _window(pas.text, pas.children[0].payload.get("text") or "", cap)
    if share > config.PARENT_MAX_CHARS:
        notes.append(f"{len(passages)} passage(s) -> fair share {share:,} chars each "
                     f"(floor {config.PARENT_MAX_CHARS:,}); bins stay at the floor")

    # --- pass 3: the budget, which drops rather than trims ----------------
    kept, used, dropped = [], 0, []
    for pas in passages:
        if kept and used + len(pas.text) > config.CONTEXT_MAX_CHARS:
            dropped.append(pas)
            continue
        kept.append(pas)
        used += len(pas.text)
    if dropped:
        notes.append(f"{len(dropped)} passage(s) dropped at the "
                     f"{config.CONTEXT_MAX_CHARS:,}-char budget: "
                     f"{', '.join(f'{d.short_name} {d.heading[:24]!r}' for d in dropped)}")
        for _pas in dropped:
            casualties.append(Dropped(
                cut="context_budget", facet=",".join(_pas.facets),
                parent_id=_pas.parent_id, short_name=_pas.short_name,
                heading=_pas.heading, text=_pas.text,
                rerank=max((h.rerank for h in _pas.children
                            if h.rerank is not None), default=None),
                rrf=max((h.rrf for h in _pas.children), default=0.0),
                dense_rank=None, lexical_rank=None, rank_in_facet=None,
                locator=(_pas.children[0].payload.get("locator")
                         if _pas.children else None),
                chars=len(_pas.text)))
    _resolve_loss(casualties, kept)
    timings["assemble"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t_start
    return Retrieval(question, p, kept, notes, timings=timings,
                     dropped=casualties)
