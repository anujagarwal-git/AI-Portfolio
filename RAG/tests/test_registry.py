"""Tests for regrag.registry — the seam and the gate.

WHY THIS FILE MATTERS MORE THAN A NORMAL TEST SUITE.

The registry is 19 rows of hand-curated regulatory facts. A wrong `status` or a
dangling `superseded_by` does not crash anything — it makes the system answer
confidently from dead text. There is no stack trace for that failure mode, so
the only defence is a validator that refuses to load a defective file, plus
tests proving the validator actually refuses.

MOST OF THESE TESTS DELIBERATELY BREAK THE REGISTRY. Each one takes the real
YAML, corrupts exactly one thing, and asserts that loading raises. A validator
nobody has ever seen fail is not a validator; it is a comment.

Run:  pytest tests/test_registry.py -q
  or: python tests/test_registry.py     (standalone, no pytest needed)
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import yaml

from regrag import config
from regrag.domain import Applicability, Document, DocType, Framework, Status, Subject
from regrag.registry import Registry, RegistryError, load

# =============================================================================
# HELPERS — build a deliberately-corrupted copy of the real registry
# =============================================================================


def _raw() -> dict:
    return yaml.safe_load(config.REGISTRY_PATH.read_text(encoding="utf-8"))


def _load_variant(mutate, *, check_files: bool = True) -> Registry:
    """Apply `mutate` to a copy of the real registry, write it, load it.

    Variants are written to a temp dir but `file:` paths stay relative to
    PROJECT_ROOT, so the real PDFs still resolve — only the mutation differs.
    """
    data = copy.deepcopy(_raw())
    mutate(data)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "registry.yaml"
        p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return load(p, check_files=check_files)


def _expect_error(mutate, *, contains: str, check_files: bool = True) -> None:
    try:
        _load_variant(mutate, check_files=check_files)
    except RegistryError as e:
        assert contains.lower() in str(e).lower(), (
            f"raised, but the message did not mention {contains!r}.\nGot: {e}"
        )
        return
    raise AssertionError(f"expected RegistryError mentioning {contains!r}, but load() succeeded")


def _row(data: dict, doc_id: str) -> dict:
    for r in data["documents"]:
        if r["doc_id"] == doc_id:
            return r
    raise KeyError(doc_id)


# =============================================================================
# 1. THE REAL FILE LOADS AND HAS THE SHAPE WE THINK IT HAS
# =============================================================================


def test_real_registry_loads():
    reg = load()
    assert reg.schema_version == 4
    assert len(reg.documents) == 19, f"expected 19 documents, got {len(reg.documents)}"
    assert len(reg.known_gaps) == 1
    assert len(reg.indexable()) == 19
    assert reg.blocked() == []


def test_every_file_resolves():
    # load() already enforces this; asserting it here makes the failure legible
    for d in load().documents:
        assert (config.PROJECT_ROOT / d.file).exists(), f"{d.doc_id}: {d.file} missing"


# =============================================================================
# 2. THE GATE — this had ZERO coverage, because all 19 rows are verified
# =============================================================================


def test_gate_excludes_unverified_row():
    """indexable() must DROP a verified:false row while documents keeps it.

    This is the whole point of the gate. Until this test existed the refusal
    path had never once executed.
    """
    def unverify(data):
        _row(data, "bcbs-cre-consolidated")["verified"] = False

    reg = _load_variant(unverify)
    assert len(reg.documents) == 19
    assert len(reg.indexable()) == 18, "gate did not drop the unverified row"
    assert [d.doc_id for d in reg.blocked()] == ["bcbs-cre-consolidated"]
    assert "bcbs-cre-consolidated" not in {d.doc_id for d in reg.indexable()}


def test_select_respects_the_gate_by_default():
    def unverify(data):
        _row(data, "bcbs-cre-consolidated")["verified"] = False

    reg = _load_variant(unverify)
    assert reg.select(subject=Subject.CAPITAL_ADEQUACY) != reg.select(
        subject=Subject.CAPITAL_ADEQUACY, verified_only=False
    ), "select() must exclude blocked rows unless verified_only=False"


# =============================================================================
# 3. VALIDATION MUST RAISE — one test per defect class
# =============================================================================


def test_unknown_vocabulary_value_raises():
    """A typo in a controlled vocabulary is the quietest possible bug: the row
    loads, no filter ever matches it, and nothing reports zero recall."""
    _expect_error(
        lambda d: _row(d, "bcbs-cre-consolidated").__setitem__("subject", "CAPITAL_ADEQUCY"),
        contains="subject",
    )


def test_removed_field_reintroduced_raises():
    """extra='forbid'. snapshot_date, binding, pages and version_label were all
    removed by decision; silently accepting one back would resurrect a field
    nothing reads."""
    _expect_error(
        lambda d: _row(d, "sr-26-2-2026").__setitem__("snapshot_date", "2026-08-04"),
        contains="snapshot_date",
    )


def test_duplicate_doc_id_raises():
    def dup(data):
        data["documents"].append(copy.deepcopy(_row(data, "sr-26-2-2026")))

    _expect_error(dup, contains="duplicate doc_id")


def test_duplicate_short_name_raises():
    """short_name is the citation key. Two documents citing identically means
    a reader cannot tell which source an answer came from."""
    def dup(data):
        _row(data, "bcbs-lex-consolidated")["short_name"] = "Basel CRE"

    _expect_error(dup, contains="duplicate short_name")


def test_dangling_link_raises():
    _expect_error(
        lambda d: _row(d, "sr-26-2-2026").__setitem__("supersedes", ["sr-99-9-1999"]),
        contains="unknown doc_id",
    )


def test_one_sided_supersession_raises():
    """If A supersedes B but B does not record A, the version filter works from
    one direction only — and which direction fails depends on how the question
    is phrased. The worst kind of intermittent bug."""
    _expect_error(
        lambda d: _row(d, "sr-11-7-2011").__setitem__("superseded_by", []),
        contains="superseded",
    )


def test_missing_file_raises():
    _expect_error(
        lambda d: _row(d, "sr-26-2-2026").__setitem__("file", "data/MRM/does_not_exist.pdf"),
        contains="file not found",
    )


def test_wrong_schema_version_raises():
    """A silent version mismatch means fields may have changed meaning."""
    _expect_error(lambda d: d.__setitem__("schema_version", 3), contains="schema_version")


def test_reference_data_cannot_be_a_requirement_source():
    """The 2026 DFAST results table contains the numeral 4.5 — the same number
    as the Basel CET1 minimum. Neither is US law. If a results table were typed
    'standard' it could be cited as a requirement."""
    _expect_error(
        lambda d: _row(d, "fed-2026-dfast-results").__setitem__("doc_type", "standard"),
        contains="requirement source",
    )


def test_two_current_documents_in_one_version_family_raises():
    """Two in-force members of the same family means current() cannot resolve
    one, so 'the current guidance' becomes unanswerable."""
    def revive(data):
        r = _row(data, "sr-11-7-2011")
        r["status"] = "in_force"
        r["superseded_by"] = []
        _row(data, "sr-26-2-2026")["supersedes"] = []

    _expect_error(revive, contains="current documents")


def test_known_gap_referencing_unknown_doc_raises():
    _expect_error(
        lambda d: d["known_gaps"][0].__setitem__("referenced_by", ["cfr-12-999"]),
        contains="unknown doc_id",
    )


# =============================================================================
# 4. DERIVED FACTS — the fields that were deleted because they were derivable
# =============================================================================


def test_is_binding_is_derived_not_stored():
    reg = load()
    binding = [d.short_name for d in reg.documents if d.is_binding]
    assert binding == ["12 CFR 225.8", "12 CFR Part 252"], (
        f"only the two CFR rules bind; got {binding}"
    )
    for d in reg.documents:
        assert d.is_binding == (d.authority_rank >= 4)


def test_requirement_sources_exclude_guidelines_and_reference_data():
    """BCBS guidelines say 'banks should'. Retrieved as a requirement, should
    becomes must — wrong about the only thing that matters."""
    reg = load()
    excluded = {d.short_name for d in reg.indexable() if not d.is_requirement_source}
    assert "BCBS d403 (PAP)" in excluded
    assert "BCBS d450" in excluded
    assert "2026 DFAST Results" in excluded
    for d in reg.requirement_sources():
        assert d.doc_type in (DocType.STANDARD, DocType.RULE)


def test_payload_is_a_projection_not_the_whole_row():
    p = load().by_id("sr-26-2-2026").payload()
    for leaked in ("notes", "file", "verified"):
        assert leaked not in p, f"payload leaked {leaked!r}"
    assert p["is_binding"] is False
    assert p["is_requirement_source"] is False
    assert set(config.PAYLOAD_INDEX_FIELDS) <= set(p), (
        "every field Qdrant indexes must be present in the payload"
    )


# =============================================================================
# 5. CITATION — there is no free-text citation field, so this is the only source
# =============================================================================


def test_superseded_citation_shouts():
    c = load().by_id("sr-11-7-2011").citation()
    assert "SUPERSEDED" in c, f"version bleed guard missing from citation: {c}"
    assert "2026-04-17" in c


def test_citation_uses_chunk_locator_over_volume():
    """Chapter codes survive regeneration of a living text; page numbers do not."""
    cre = load().by_id("bcbs-cre-consolidated")
    assert cre.citation("CRE36.122").startswith("Basel CRE CRE36.122")
    assert "(CRE)" in cre.citation()  # falls back to volume with no locator


# =============================================================================
# 6. AXIS QUERIES — what the schema v4 rework was for
# =============================================================================


def test_axis3_resolves_current_version():
    reg = load()
    assert reg.current(Framework.US_MRM_GUIDANCE).short_name == "SR 26-2"
    assert reg.by_id("sr-11-7-2011").status is Status.SUPERSEDED


def test_version_family_needs_its_discriminator():
    """framework alone is not a family key: BASEL_FRAMEWORK has six current
    chapters. The error must NAME the discriminator, not just complain."""
    reg = load()
    try:
        reg.current(Framework.BASEL_FRAMEWORK)
    except RegistryError as e:
        assert "volume" in str(e), f"error should name the discriminator: {e}"
    else:
        raise AssertionError("ambiguous current() should have raised")

    assert reg.current(Framework.BASEL_FRAMEWORK, volume="CRE").short_name == "Basel CRE"
    assert (
        reg.current(
            Framework.US_CAPITAL_PLANNING_GUIDANCE,
            applicability=Applicability.US_CATEGORY_I,
        ).short_name
        == "SR 15-18"
    )


def test_sr15_pair_separated_only_by_applicability():
    """The cleanest applicability test in the corpus: same framework, same
    status, same (null) volume. Retrieving the wrong one is undetectable from
    the text, which reads perfectly plausibly either way."""
    reg = load()
    a, b = reg.by_id("sr-15-18-2015r2021"), reg.by_id("sr-15-19-2015r2021")
    assert a.framework is b.framework and a.volume == b.volume and a.status is b.status
    assert a.applicability is not b.applicability
    assert a.version_key != b.version_key


# =============================================================================
# 7. THE REFUSAL PATH
# =============================================================================


def test_known_gap_supports_a_specific_refusal():
    """A refusal that names the gap is auditable. 'I don't know' is
    indistinguishable from a retrieval failure."""
    reg = load()
    gap = reg.gap_for("12 CFR 217")
    assert gap is not None
    assert "217.20" in gap.sections_referenced
    assert set(gap.referenced_by) <= {d.doc_id for d in reg.documents}
    assert "not held in this corpus" in " ".join(gap.refusal.split())

    hits = reg.gap_mentioning("what is the minimum CET1 requirement under 12 CFR 217?")
    assert [g.short_name for g in hits] == ["12 CFR 217"]
    assert reg.gap_mentioning("what does SS1/23 say about validation?") == []


# =============================================================================
# standalone runner — works without pytest installed
# =============================================================================

if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
