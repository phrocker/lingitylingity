from __future__ import annotations

from typing import cast

import pytest

from lingity.analyzer import analyze_text
from lingity.invariants import compare_protected
from lingity.models import JsonValue


def _protected(analysis: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], analysis["protected"])


def _comparison(source: str, candidate: str) -> dict[str, JsonValue]:
    return compare_protected(_protected(analyze_text(source)), _protected(analyze_text(candidate)))


def test_clearer_rewrite_preserves_protected_meaning(
    recommendation_fixture: dict[str, str],
) -> None:
    source = analyze_text(recommendation_fixture["original"])
    candidate = analyze_text(recommendation_fixture["rewrite"])
    comparison = compare_protected(_protected(source), _protected(candidate))
    assert comparison["equivalent"] is True
    signature = cast(list[str], _protected(source)["semantic_signature"])
    assert "identifier:identifier:V2" in signature
    assert "quantity:count:2" in signature
    assert "negation:governance_polarity:architecture_approval_deferred" in signature
    assert "governance:authority:target_architecture_requires_human_approval" in signature


def test_changed_quantity_is_rejected(
    recommendation_fixture: dict[str, str],
) -> None:
    source = analyze_text(recommendation_fixture["original"])
    changed = recommendation_fixture["rewrite"].replace("either messaging path", "three messaging paths")
    candidate = analyze_text(changed)
    comparison = compare_protected(_protected(source), _protected(candidate))
    assert comparison["equivalent"] is False
    assert "quantity:count:2" in cast(list[str], comparison["missing"])
    assert "quantity:count:3" in cast(list[str], comparison["added"])


def test_dropped_duplicate_protected_value_is_rejected() -> None:
    source = analyze_text("ADR-42 must retain two paths and two owners.")
    candidate = analyze_text("ADR-42 must retain two paths and owners.")
    comparison = compare_protected(_protected(source), _protected(candidate))
    assert comparison["equivalent"] is False
    assert cast(list[str], comparison["missing"]) == ["quantity:count:2"]


def test_added_duplicate_protected_value_is_rejected() -> None:
    comparison = _comparison("ADR-42 must retain two paths.", "ADR-42 must retain two paths and two owners.")
    assert comparison["equivalent"] is False
    assert cast(list[str], comparison["added"]) == ["quantity:count:2"]


def test_reordering_only_rewrite_preserves_claims() -> None:
    comparison = _comparison(
        "Team A must deploy; Team B should review; ADR-42 is approved.",
        "ADR-42 is approved; Team B should review; Team A must deploy.",
    )
    assert comparison["equivalent"] is True


@pytest.mark.parametrize(
    ("case_name", "source", "candidate"),
    [
        (
            "dropped_duplicate_concept",
            "Fix the confirmed authorization issues and verify the confirmed authorization issues.",
            "Fix the confirmed authorization issues and verify them.",
        ),
        (
            "polarity_swapped_between_clauses",
            "Do not approve the design; deploy the service.",
            "Approve the design; do not deploy the service.",
        ),
        (
            "modality_swapped_between_actors",
            "Team A must deploy; Team B should review.",
            "Team A should deploy; Team B must review.",
        ),
        ("cannot_to_can", "ADR-42 cannot be approved.", "ADR-42 can be approved."),
        ("approved_to_rejected", "ADR-42 is approved.", "ADR-42 is rejected."),
        ("granted_to_denied", "waiver granted", "waiver denied"),
        ("either_to_two", "Choose either path.", "Choose two paths."),
        ("at_least_to_at_most", "Wait at least 5 seconds.", "Wait at most 5 seconds."),
        ("kg_to_lb", "Use 5 kg.", "Use 5 lb."),
        (
            "obsolete_trigger_with_contradiction",
            "Do not approve the architecture.",
            "Do not approve the architecture. Ratify it now.",
        ),
        (
            "obligation_removed_but_phrase_retained",
            "fix the confirmed authorization issues",
            "note the confirmed authorization issues",
        ),
    ],
)
def test_audited_semantic_drift_cases_are_rejected(case_name: str, source: str, candidate: str) -> None:
    comparison = _comparison(source, candidate)
    assert comparison["equivalent"] is False, case_name
    assert comparison["disposition"] in {"changed", "unresolved"}
    assert comparison["missing"] or comparison["added"] or comparison["unresolved"]


@pytest.mark.parametrize(
    ("case_name", "source", "candidate"),
    [
        ("direct_approval_flip", "Do not approve the architecture.", "Approve the architecture."),
        ("must_to_should", "Team A must deploy.", "Team A should deploy."),
        ("must_not_to_should_not", "Team A must not deploy.", "Team A should not deploy."),
        (
            "dropped_human_approval_sentence",
            "The target architecture returns for human decision.",
            "The target architecture is ready.",
        ),
        ("deleted_authorization_phrase", "Fix the confirmed authorization issues.", "Fix the issues."),
        ("v2_to_v3", "Do not approve irreversible V2 cutover yet.", "Do not approve irreversible V3 cutover yet."),
    ],
)
def test_existing_adversarial_cases_are_rejected(case_name: str, source: str, candidate: str) -> None:
    comparison = _comparison(source, candidate)
    assert comparison["equivalent"] is False, case_name


def test_extracts_modal_citation_and_governance() -> None:
    text = "ADR-42 must not change by more than 15% [E-7]; approval requires a waiver."
    manifest = _protected(analyze_text(text))
    signature = set(cast(list[str], manifest["semantic_signature"]))
    assert "identifier:identifier:ADR-42" in signature
    assert "modal:modal_strength:must not" in signature
    assert "quantity:threshold:operator=gt;unit=percent;value=15" in signature
    assert "citation:reference:[E-7]" in signature
    assert "negation:lexical_negation:not" in signature
    assert "governance:term:approval" in signature
    assert "governance:term:waiver" in signature
