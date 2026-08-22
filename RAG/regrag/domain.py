"""Shared types for regrag.

WHY THIS FILE EXISTS — two jobs.

1. TURN THE CONTROLLED VOCABULARIES INTO TYPES.
   Until now they lived in a YAML comment, and a comment cannot reject
   anything. Write `subject: CAPITAL_ADEQUCY` and a plain loader accepts it —
   you get a subject with one member, no filter ever matches it, and there is
   no error anywhere. A silent zero-recall bug. As enums, that typo raises at
   load time with the offending row named.

2. PUT DERIVED FACTS IN ONE PLACE.
   `is_binding` used to be a stored field. It was removed because it was
   perfectly derivable from authority_rank, and two copies of one fact is a bug
   waiting for someone to edit one of them. It lives here as a property.

IMPORTS NOTHING FROM regrag EXCEPT config. This module sits at the bottom of
the dependency graph, which is what makes circular imports impossible.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from regrag import config

# =============================================================================
# CONTROLLED VOCABULARIES — must stay in lockstep with data/registry.yaml
# =============================================================================


class Subject(StrEnum):
    """WHAT a document is about. Axis 1 filters on this.

    Orthogonal to Framework, not a parent of it: BASEL_FRAMEWORK spans
    CAPITAL_ADEQUACY and STRESS_TESTING, while STRESS_TESTING spans BCBS and
    the Federal Reserve. Many-to-many, so neither can be the other's parent.

    Subjects OVERLAP, which makes this a FAN-OUT axis rather than a filter:
    model validation is both a risk-management discipline (SR 26-2, SS1/23) and
    a capital eligibility condition (CRE36 section 8). Scoping a validation
    query to MODEL_RISK_MANAGEMENT silently deletes half the answer.
    """

    MODEL_RISK_MANAGEMENT = "MODEL_RISK_MANAGEMENT"
    CAPITAL_ADEQUACY = "CAPITAL_ADEQUACY"
    CREDIT_IMPAIRMENT = "CREDIT_IMPAIRMENT"
    DATA_QUALITY = "DATA_QUALITY"
    STRESS_TESTING = "STRESS_TESTING"


class Framework(StrEnum):
    """WHICH INSTRUMENT a document is — the identity that persists across
    versions. Axis 3 filters on this plus `status`.

    Two documents share a key if and only if one could replace the other, or
    they are variants of the same instrument. SR 11-7 and SR 26-2 are two
    versions of one framework, so both are US_MRM_GUIDANCE. That is what makes
    "the current US model risk guidance" expressible as a filter instead of
    something you must already know the answer to.
    """

    US_MRM_GUIDANCE = "US_MRM_GUIDANCE"                            # SR 11-7 -> SR 26-2
    UK_MRM_SS = "UK_MRM_SS"                                        # SS1/23
    BASEL_FRAMEWORK = "BASEL_FRAMEWORK"                            # CAP/CRE/LEX/RBC/SCO/SRP32
    IFRS_9 = "IFRS_9"
    BCBS_239 = "BCBS_239"
    BCBS_STRESS_PRINCIPLES = "BCBS_STRESS_PRINCIPLES"              # d450
    BCBS_PROBLEM_ASSETS = "BCBS_PROBLEM_ASSETS"                    # d403 (PAP)
    REG_Y = "REG_Y"                                                # 12 CFR 225.8
    REG_YY = "REG_YY"                                              # 12 CFR Part 252
    US_CAPITAL_PLANNING_GUIDANCE = "US_CAPITAL_PLANNING_GUIDANCE"  # SR 15-18 / SR 15-19
    DFAST = "DFAST"                                                # scenarios + results


class Jurisdiction(StrEnum):
    """Axis 2. Mutually exclusive, so this is a FILTER axis, not fan-out:
    a clause cannot be both US and UK law.
    """

    US = "US"
    UK = "UK"
    EU = "EU"
    IN = "IN"
    GLOBAL = "GLOBAL"


class Issuer(StrEnum):
    FED = "FED"
    OCC = "OCC"
    FDIC = "FDIC"
    US_INTERAGENCY = "US_INTERAGENCY"
    PRA = "PRA"
    BCBS = "BCBS"
    EBA = "EBA"
    ECB = "ECB"
    IASB = "IASB"
    FASB = "FASB"
    RBI = "RBI"


class DocType(StrEnum):
    """`standard` vs `guideline` is load-bearing, not cosmetic.

    BCBS publishes standards and guidelines as separate sets. The consolidated
    Basel Framework is standards + FAQs only; d403 and d450 sit outside it and
    say "banks should", never "banks must". A guideline retrieved as a
    requirement turns should into must — fluent, cited, and wrong about the one
    thing that matters. See config.REQUIREMENT_DOC_TYPES.
    """

    RULE = "rule"
    SUPERVISORY_GUIDANCE = "supervisory_guidance"
    SUPERVISORY_STATEMENT = "supervisory_statement"
    STANDARD = "standard"
    GUIDELINE = "guideline"
    CONSULTATION = "consultation"
    DRAFT = "draft"
    FAQ = "faq"
    REFERENCE_DATA = "reference_data"


class DocRole(StrEnum):
    """Whether ONE document can settle a question.

    `amending` is the dangerous one: it edits another document and is
    meaningless alone. Worse, PDF extraction destroys strikethrough, so deleted
    and added text merge into nonsense ("both all"). That is semantic
    INVERSION, not noise — which is why the CECL ASUs were dropped.
    """

    CONSOLIDATED = "consolidated"
    AMENDING = "amending"
    STANDALONE = "standalone"


class Status(StrEnum):
    IN_FORCE = "in_force"
    PARTIALLY_REVISED = "partially_revised"
    SUPERSEDED = "superseded"
    DRAFT = "draft"
    CONSULTATION = "consultation"
    WITHDRAWN = "withdrawn"


class Applicability(StrEnum):
    """ONLY legally operative scope limits belong here.

    US_OVER_30BN and US_OVER_1BN were removed: they came from "most relevant
    to" and "not expected to pertain" — editorial phrasing, not scope. Encoded
    as filter values they meant a query about a $20bn firm would EXCLUDE the
    current US model risk guidance outright. `None` means always eligible.
    """

    ALL = "ALL"
    GSIB = "GSIB"
    US_CATEGORY_I = "US_CATEGORY_I"
    US_CATEGORY_II_III = "US_CATEGORY_II_III"
    US_CATEGORY_I_TO_IV = "US_CATEGORY_I_TO_IV"
    UK_INTERNAL_MODEL_FIRMS = "UK_INTERNAL_MODEL_FIRMS"


# =============================================================================
# DOCUMENT
# =============================================================================


class Document(BaseModel):
    """One row of data/registry.yaml, validated.

    `extra="forbid"` is deliberate: an unexpected key raises. That is what
    catches a removed field being quietly reintroduced (snapshot_date, binding,
    pages, version_label) or a field name typo that would otherwise be ignored.

    `frozen=True` means a Document cannot be mutated after load. Registry facts
    are curated inputs, not working state — if something wants to change one,
    that is a bug, and it should fail loudly here rather than downstream.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)

    doc_id: str = Field(min_length=1)
    short_name: str = Field(min_length=1)
    file: str = Field(min_length=1)

    subject: Subject
    framework: Framework
    volume: str | None  # populated ONLY when the doc is a subdivision of the framework

    jurisdiction: Jurisdiction
    issuer: Issuer
    doc_type: DocType
    doc_role: DocRole

    effective_from: date | None  # null = chapter-level (Basel) or living text (eCFR)
    effective_to: date | None
    status: Status

    supersedes: list[str] = Field(default_factory=list)
    superseded_by: list[str] = Field(default_factory=list)
    revises: list[str] = Field(default_factory=list)
    revised_by: list[str] = Field(default_factory=list)

    applicability: Applicability | None  # None = no legally operative scope limit
    authority_rank: int = Field(ge=0, le=5)

    source_url: str | None
    notes: str
    verified: bool

    # ---- derived facts: one source of truth, never stored ----

    @property
    def is_binding(self) -> bool:
        """Was the removed `binding` field. Perfectly derivable, so derived.

        Note WHY it once looked independent: BCBS text is globally
        authoritative and binds nobody. That asymmetry is carried by rank 2.
        """
        return self.authority_rank >= config.BINDING_RANK_FLOOR

    @property
    def is_requirement_source(self) -> bool:
        """May this document be cited in answer to "what is required?"."""
        return self.doc_type.value in config.REQUIREMENT_DOC_TYPES

    @property
    def is_current(self) -> bool:
        return self.status in (Status.IN_FORCE, Status.PARTIALLY_REVISED)

    @property
    def version_key(self) -> tuple:
        """Identifies the VERSION FAMILY this document belongs to.

        Two documents are versions of each other only if all three match. That
        `framework` alone is not enough was a real bug: BASEL_FRAMEWORK has six
        concurrently-in-force chapters separated by `volume`, and
        US_CAPITAL_PLANNING_GUIDANCE has two separated by `applicability`.
        Treating framework as the family key made "the current member" a
        nonsense question for both.

          SR 11-7 / SR 26-2      same framework, volume None, applicability None
                                 -> SAME family, so one supersedes the other
          Basel CAP / Basel CRE  differ on volume        -> different families
          SR 15-18 / SR 15-19    differ on applicability -> different families

        IMPORTANT: registry validation and Registry.current() must both use
        this property. They originally used different definitions, so the file
        loaded clean and then raised at runtime — the invariant checked at load
        time has to be the same predicate the accessor depends on.
        """
        return (self.framework, self.volume, self.applicability)

    @property
    def is_reference_data(self) -> bool:
        """authority_rank 0 — states no requirement at all. The 2026 DFAST
        results contain the numeral 4.5, the same number as the Basel CET1
        minimum. Neither is US law. This flag is the guard.
        """
        return self.authority_rank == 0

    # ---- citation ----

    def citation(self, locator: str | None = None) -> str:
        """The canonical citation string.

        There is no free-text citation field in the registry (version_label was
        removed), so this is the single place a citation is composed. If callers
        format their own, they will drift — and a wrong title attached to a
        CORRECT quotation is unfalsifiable from the output.

        `locator` is the chunk-level reference the parser lifts from the text,
        e.g. "CRE36.122". Chapter codes are stable across regenerations of a
        living text; page numbers are not, which is why locator is preferred.

        Superseded status is stated in CAPITALS on purpose — it is the last
        line of defence against version bleed reaching the reader.
        """
        head = self.short_name
        if locator:
            head = f"{head} {locator}"
        elif self.volume:
            head = f"{head} ({self.volume})"

        if self.status is Status.SUPERSEDED:
            tail = "SUPERSEDED"
            if self.effective_to:
                tail = f"{tail} {self.effective_to.isoformat()}"
        else:
            tail = self.status.value
            if self.effective_from:
                tail = f"{tail}, effective {self.effective_from.isoformat()}"

        return f"{head} [{tail}]"

    # ---- projection onto a chunk ----

    def payload(self) -> dict:
        """The subset of this row denormalised onto every chunk of it.

        A PROJECTION, not the whole row. Deliberately excluded:
          notes     — mentor commentary; would balloon every vector's payload
          file      — a local filesystem path is useless in a payload and leaks
                      the machine layout into the index
          verified  — by construction EVERY indexed chunk came from a verified
                      row, so the field would be a constant. Carrying it would
                      invite a filter that can never do anything.

        Derived booleans ARE included, so filters need not recompute them.

        Basel note: effective_from is None here for the five multi-chapter
        files. The chunker overlays the CHAPTER-level date on top of this dict.
        """
        return {
            "doc_id": self.doc_id,
            "short_name": self.short_name,
            "subject": self.subject.value,
            "framework": self.framework.value,
            "volume": self.volume,
            "jurisdiction": self.jurisdiction.value,
            "issuer": self.issuer.value,
            "doc_type": self.doc_type.value,
            "doc_role": self.doc_role.value,
            "status": self.status.value,
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
            "applicability": self.applicability.value if self.applicability else None,
            "authority_rank": self.authority_rank,
            "is_binding": self.is_binding,
            "is_requirement_source": self.is_requirement_source,
            "source_url": self.source_url,
        }

    # ---- within-row consistency ----

    @model_validator(mode="after")
    def _check_row(self) -> "Document":
        if self.status is Status.SUPERSEDED and not self.superseded_by:
            raise ValueError(
                f"{self.doc_id}: status is 'superseded' but superseded_by is empty — "
                "a superseded document with no successor can never be excluded by a "
                "version filter, so it will leak into in-force answers"
            )
        if self.superseded_by and self.status is Status.IN_FORCE:
            raise ValueError(
                f"{self.doc_id}: has superseded_by but status is 'in_force'"
            )
        if self.is_reference_data and self.is_requirement_source:
            raise ValueError(
                f"{self.doc_id}: authority_rank 0 (reference data) but doc_type "
                f"'{self.doc_type.value}' is a requirement source — a results table "
                "must never be answerable as a requirement"
            )
        if self.effective_to and self.effective_from and self.effective_to < self.effective_from:
            raise ValueError(f"{self.doc_id}: effective_to precedes effective_from")
        return self

    def __str__(self) -> str:  # readable in tracebacks and logs
        return f"<{self.doc_id} {self.short_name!r} {self.status.value}>"


# =============================================================================
# KNOWN GAP — an instrument the corpus REFERENCES but does not HOLD
# =============================================================================


class KnownGap(BaseModel):
    """Not a document. Zero vectors, zero embedding cost.

    Its only job is to let a refusal be SPECIFIC: "that provision lives in an
    instrument we do not hold" rather than a bare "I don't know". A refusal
    that names the gap is auditable; one that does not is indistinguishable
    from a retrieval failure — and the reader cannot tell which happened.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    instrument: str
    short_name: str
    title: str
    jurisdiction: Jurisdiction
    issuer: Issuer
    referenced_by: list[str]
    sections_referenced: list[str] = Field(default_factory=list)
    sections_basis: str | None = None
    reason_absent: str
    consequence: str
    refusal: str

    def __str__(self) -> str:
        return f"<GAP {self.short_name} ({len(self.sections_referenced)} sections)>"
