"""THE SEAM between the corpus and the code.

Nothing downstream ever opens data/registry.yaml. Everything asks this module.
That is what makes "a corpus change is a DATA change, never a CODE change" true
rather than aspirational — adding a jurisdiction adds filter VALUES to the
YAML, and no function signature moves.

It is also THE GATE. A row with `verified: false` is REFUSED, not warned about.
The closest analogue is the data-quality control that runs before a scorecard
batch: you do not score records that failed validation and note it in a log,
you fail the batch. A silently-skipped exclusion is indistinguishable from a
clean run, which is the whole problem.

Every check here RAISES. There is no permissive mode. If that feels harsh,
consider what the alternative buys you: a corpus that loads with a wrong
`status` and answers confidently from dead text.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import yaml

from regrag import config
from regrag.domain import (
    Applicability,
    Document,
    DocType,
    Framework,
    Jurisdiction,
    KnownGap,
    Status,
    Subject,
)

SCHEMA_VERSION = 4


class RegistryError(Exception):
    """Raised for any registry defect. Always names the offending row(s)."""


class Registry:
    """The loaded, validated corpus map."""

    def __init__(self, documents: list[Document], known_gaps: list[KnownGap], schema_version: int):
        self.documents = documents
        self.known_gaps = known_gaps
        self.schema_version = schema_version
        self._by_id = {d.doc_id: d for d in documents}

    # ---- the gate -------------------------------------------------------

    def indexable(self) -> list[Document]:
        """The ONLY way to get documents for indexing.

        Never iterate `registry.documents` in ingestion code — that bypasses
        the gate, which is exactly the mistake this method exists to prevent.
        """
        return [d for d in self.documents if d.verified]

    def blocked(self) -> list[Document]:
        return [d for d in self.documents if not d.verified]

    # ---- lookups --------------------------------------------------------

    def by_id(self, doc_id: str) -> Document:
        try:
            return self._by_id[doc_id]
        except KeyError:
            raise RegistryError(
                f"unknown doc_id {doc_id!r}; known ids: {sorted(self._by_id)}"
            ) from None

    def select(self, *, verified_only: bool = True, **criteria) -> list[Document]:
        """Filter by any Document field. `select(subject=Subject.CAPITAL_ADEQUACY)`.

        Defaults to verified_only=True so the gate holds unless explicitly
        opted out of (eval sometimes needs blocked rows for negative tests).
        """
        pool = self.indexable() if verified_only else self.documents
        out = []
        for d in pool:
            if all(getattr(d, k) == v for k, v in criteria.items()):
                out.append(d)
        return out

    def current(
        self,
        framework: Framework,
        *,
        volume: str | None = None,
        applicability: Applicability | None = None,
    ) -> Document | None:
        """Axis 3 in one call: the in-force member of a version family.

        This is the query that was IMPOSSIBLE before framework became a
        version-family key — asking for "the current US model risk guidance"
        used to require already knowing it was SR 26-2:

            reg.current(Framework.US_MRM_GUIDANCE)          -> SR 26-2

        A family is (framework, volume, applicability), not framework alone —
        see Document.version_key. So a Basel chapter or an SR 15 variant needs
        its discriminator:

            reg.current(Framework.BASEL_FRAMEWORK, volume="CRE")
            reg.current(Framework.US_CAPITAL_PLANNING_GUIDANCE,
                        applicability=Applicability.US_CATEGORY_I)

        Omitted arguments mean DON'T CARE, not "must be null". Requiring the
        caller to pass applicability=ALL just to reach a Basel chapter would
        leak a filter detail into every call site for no benefit.

        If what you ask for is ambiguous, this RAISES and names the
        discriminator you need rather than silently returning the first match:

            reg.current(Framework.BASEL_FRAMEWORK)
            -> RegistryError: 6 current documents; disambiguate with
               volume=one of ['CAP','CRE','LEX','RBC','SCO','SRP32']
        """
        hits = [
            d
            for d in self.indexable()
            if d.is_current
            and d.framework is framework
            and (volume is None or d.volume == volume)
            and (applicability is None or d.applicability is applicability)
        ]
        if len(hits) > 1 and not all(d.is_reference_data for d in hits):
            vols = sorted({d.volume for d in hits if d.volume})
            aps = sorted({d.applicability.value for d in hits if d.applicability})
            hint = []
            if len(vols) > 1:
                hint.append(f"volume=one of {vols}")
            if len(aps) > 1:
                hint.append(f"applicability=one of {aps}")
            raise RegistryError(
                f"{framework.value} has {len(hits)} current documents "
                f"({[h.short_name for h in hits]}); disambiguate with "
                + (" or ".join(hint) if hint else "a narrower query")
            )
        return hits[0] if hits else None

    def version_families(self) -> dict[tuple, list[Document]]:
        """Every version family and its members, current or superseded.
        Families with more than one member are the axis-3 test cases.
        """
        fams: dict[tuple, list[Document]] = defaultdict(list)
        for d in self.documents:
            fams[d.version_key].append(d)
        return dict(fams)

    def requirement_sources(self) -> list[Document]:
        """Documents that may answer "what is required?".

        Excludes guidelines ("banks should") and reference_data (rank 0).
        """
        return [d for d in self.indexable() if d.is_requirement_source]

    # ---- the refusal path -----------------------------------------------

    def gap_for(self, instrument: str) -> KnownGap | None:
        """Look up a referenced-but-absent instrument, e.g. "12 CFR 217"."""
        needle = instrument.strip().lower()
        for g in self.known_gaps:
            if needle in (g.instrument.lower(), g.short_name.lower()):
                return g
        return None

    def gap_mentioning(self, text: str) -> list[KnownGap]:
        """Gaps whose instrument name appears in a query or a retrieved chunk.

        Cheap substring match on purpose: this feeds a refusal, not an answer,
        and a false positive costs a caveat while a false negative costs a
        confidently wrong number.
        """
        low = text.lower()
        return [g for g in self.known_gaps if g.instrument.lower() in low or g.short_name.lower() in low]

    def __len__(self) -> int:
        return len(self.documents)

    def __repr__(self) -> str:
        return (
            f"<Registry v{self.schema_version}: {len(self.documents)} documents "
            f"({len(self.indexable())} indexable, {len(self.blocked())} blocked), "
            f"{len(self.known_gaps)} known gaps>"
        )


# =============================================================================
# LOADING
# =============================================================================


def load(path: Path | str = config.REGISTRY_PATH, *, check_files: bool = True) -> Registry:
    """Parse and validate the registry. Raises RegistryError on any defect.

    `check_files=False` exists only for tests that build synthetic rows without
    PDFs on disk. Production and ingestion must never pass it.
    """
    path = Path(path)
    if not path.exists():
        raise RegistryError(f"registry not found at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RegistryError(f"{path}: expected a mapping at the top level")

    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise RegistryError(
            f"{path}: schema_version is {version!r}, this code expects {SCHEMA_VERSION}. "
            "Reconcile before loading — a silent version mismatch means fields may "
            "have moved meaning."
        )

    documents = _parse_rows(raw.get("documents") or [], path)
    known_gaps = _parse_gaps(raw.get("known_gaps") or [], path)

    _validate_corpus(documents, known_gaps, path, check_files=check_files)
    return Registry(documents, known_gaps, version)


@lru_cache(maxsize=1)
def load_cached() -> Registry:
    """Load once per process. Use in notebooks; prefer explicit load() in code."""
    return load()


def _parse_rows(rows: list, path: Path) -> list[Document]:
    if not rows:
        raise RegistryError(f"{path}: no documents")
    out, errors = [], []
    for i, row in enumerate(rows):
        ident = row.get("doc_id", f"<row {i}, no doc_id>") if isinstance(row, dict) else f"<row {i}>"
        try:
            out.append(Document(**row))
        except Exception as e:  # pydantic ValidationError or TypeError
            errors.append(f"  {ident}: {e}")
    if errors:
        raise RegistryError(f"{path}: {len(errors)} invalid document row(s):\n" + "\n".join(errors))
    return out


def _parse_gaps(rows: list, path: Path) -> list[KnownGap]:
    out, errors = [], []
    for i, row in enumerate(rows):
        ident = row.get("instrument", f"<gap {i}>") if isinstance(row, dict) else f"<gap {i}>"
        try:
            out.append(KnownGap(**row))
        except Exception as e:
            errors.append(f"  {ident}: {e}")
    if errors:
        raise RegistryError(f"{path}: {len(errors)} invalid known_gaps row(s):\n" + "\n".join(errors))
    return out


# =============================================================================
# CROSS-ROW VALIDATION
#
# These checks cannot live on the Document model, because no single row can see
# them. This is also where today's throwaway bash checks become permanent.
# =============================================================================


def _validate_corpus(
    documents: list[Document],
    known_gaps: list[KnownGap],
    path: Path,
    *,
    check_files: bool,
) -> None:
    problems: list[str] = []
    ids = {d.doc_id for d in documents}

    # 1. unique doc_ids
    for doc_id, n in Counter(d.doc_id for d in documents).items():
        if n > 1:
            problems.append(f"duplicate doc_id {doc_id!r} appears {n} times")

    # 2. unique short_names — they are the citation key, so a collision means
    #    two different documents cite identically and the audit trail breaks
    for name, n in Counter(d.short_name for d in documents).items():
        if n > 1:
            problems.append(f"duplicate short_name {name!r} appears {n} times")

    # 3. every file resolves on disk
    if check_files:
        root = config.PROJECT_ROOT
        for d in documents:
            if not (root / d.file).exists():
                problems.append(f"{d.doc_id}: file not found: {d.file}")

    # 4. no dangling relationship targets
    for d in documents:
        for field in ("supersedes", "superseded_by", "revises", "revised_by"):
            for target in getattr(d, field):
                if target not in ids:
                    problems.append(f"{d.doc_id}.{field} -> unknown doc_id {target!r}")

    # 5. RECIPROCITY. If A supersedes B then B must record A as its successor.
    #    A one-sided link means the version filter works from one direction
    #    only, and which direction it fails in depends on how the query is
    #    phrased — the worst kind of intermittent bug.
    for d in documents:
        for target in d.supersedes:
            if target in ids and d.doc_id not in documents_by_id(documents)[target].superseded_by:
                problems.append(
                    f"{d.doc_id} supersedes {target}, but {target}.superseded_by "
                    f"does not list {d.doc_id}"
                )
        for target in d.revises:
            if target in ids and d.doc_id not in documents_by_id(documents)[target].revised_by:
                problems.append(
                    f"{d.doc_id} revises {target}, but {target}.revised_by "
                    f"does not list {d.doc_id}"
                )

    # 6. at most one current document per version family.
    #
    #    reference_data is EXEMPT, and the exemption is principled rather than a
    #    carve-out for a row we happen to dislike. This check exists to stop two
    #    documents being indistinguishable when answering "what is required?" —
    #    but authority_rank 0 documents are already excluded from every
    #    requirement answer, so an ambiguity between them cannot reach one.
    #    That is exactly the argument used to accept the DFAST scenarios/results
    #    collision (registry STILL OPEN item 1): both are rank 0, so nothing
    #    downstream can be misled by the fact that no filter separates them.
    #    Semantic retrieval distinguishes "scenarios" from "results" perfectly
    #    well; it is only the FILTER that cannot, and the filter's job here is
    #    requirement-scoping.
    #    Grouped by Document.version_key — the SAME predicate Registry.current()
    #    uses. Using two different definitions of "family" is how the registry
    #    loaded clean and then raised at runtime the first time this ran.
    by_family: dict[tuple, list[Document]] = defaultdict(list)
    for d in documents:
        if d.verified and d.is_current and not d.is_reference_data:
            by_family[d.version_key].append(d)
    for key, group in by_family.items():
        if len(group) > 1:
            problems.append(
                f"version family {key} has {len(group)} current documents: "
                f"{[d.doc_id for d in group]} — nothing distinguishes them at "
                "filter time, so Registry.current() cannot resolve one"
            )

    # 7. known_gaps must reference real documents
    for g in known_gaps:
        for ref in g.referenced_by:
            if ref not in ids:
                problems.append(f"known_gap {g.short_name}: referenced_by -> unknown doc_id {ref!r}")

    if problems:
        raise RegistryError(
            f"{path}: {len(problems)} corpus-level problem(s):\n"
            + "\n".join(f"  {p}" for p in problems)
        )


def documents_by_id(documents: list[Document]) -> dict[str, Document]:
    return {d.doc_id: d for d in documents}


# =============================================================================
# CLI — `python -m regrag.registry` prints the corpus map and exits non-zero
# on any defect. Cheap way to check the registry after hand-editing the YAML.
# =============================================================================

if __name__ == "__main__":
    import sys

    try:
        reg = load()
    except RegistryError as e:
        print(f"REGISTRY INVALID\n{e}", file=sys.stderr)
        sys.exit(1)

    print(reg, "\n")
    print(f"{'doc_id':28} {'short_name':30} {'subject':22} rank  req?  cite")
    print("-" * 132)
    for d in reg.documents:
        flag = "" if d.verified else "  [BLOCKED]"
        print(
            f"{d.doc_id:28} {d.short_name:30} {d.subject.value:22} "
            f"{d.authority_rank:^4} {'Y' if d.is_requirement_source else '-':^5} "
            f"{d.citation()}{flag}"
        )

    print("\n-- version families (>1 member = an axis-3 test case) --")
    for key, members in sorted(reg.version_families().items(), key=lambda kv: str(kv[0])):
        fw, vol, applic = key
        label = fw.value + (f"/{vol}" if vol else "") + (f"/{applic.value}" if applic else "")
        marker = "  <-- axis 3" if len(members) > 1 else ""
        print(f"  {label:52} {', '.join(m.short_name for m in members)}{marker}")

    print("\n-- Registry.current() --")
    print(f"  current US model risk guidance: {reg.current(Framework.US_MRM_GUIDANCE)}")
    print(f"  current Basel CRE chapter:      {reg.current(Framework.BASEL_FRAMEWORK, volume='CRE')}")

    print("\n-- requirement sources (standard|rule) --")
    print("  " + ", ".join(d.short_name for d in reg.requirement_sources()))

    print("\n-- known gaps --")
    for g in reg.known_gaps:
        print(f"  {g}  referenced_by={g.referenced_by}")
