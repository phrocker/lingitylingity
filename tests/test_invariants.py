from __future__ import annotations

from typing import cast

import pytest

from lingity.invariants import compare_protected, extract_protected
from lingity.models import JsonValue
from lingity.nlp import LinguisticModelError
from lingity.profiles import load_profile


def _protected(text: str) -> dict[str, JsonValue]:
    return extract_protected(text, load_profile())


def _comparison(source: str, candidate: str) -> dict[str, JsonValue]:
    return compare_protected(_protected(source), _protected(candidate))


def _signature(text: str) -> list[str]:
    return cast(list[str], _protected(text)["semantic_signature"])


def test_clearer_rewrite_preserves_protected_meaning(
    recommendation_fixture: dict[str, str],
) -> None:
    source = _protected(recommendation_fixture["original"])
    candidate = _protected(recommendation_fixture["rewrite"])
    comparison = compare_protected(source, candidate)
    assert comparison["equivalent"] is True
    signature = cast(list[str], source["semantic_signature"])
    assert "identifier:identifier:V2" in signature
    assert "quantity:count:2" in signature
    assert "negation:governance_polarity:architecture_approval_deferred" in signature
    assert "governance:authority:target_architecture_requires_human_approval" in signature


def test_changed_quantity_is_rejected(
    recommendation_fixture: dict[str, str],
) -> None:
    changed = recommendation_fixture["rewrite"].replace("either messaging path", "three messaging paths")
    comparison = _comparison(recommendation_fixture["original"], changed)
    assert comparison["equivalent"] is False
    assert "quantity:count:2" in cast(list[str], comparison["missing"])
    assert "quantity:count:3" in cast(list[str], comparison["added"])


def test_dropped_duplicate_protected_value_is_rejected() -> None:
    comparison = _comparison("ADR-42 must retain two paths and two owners.", "ADR-42 must retain two paths and owners.")
    assert comparison["equivalent"] is False
    assert cast(list[str], comparison["missing"]) == ["quantity:count:2"]


def test_added_duplicate_protected_value_is_rejected() -> None:
    comparison = _comparison("ADR-42 must retain two paths.", "ADR-42 must retain two paths and two owners.")
    assert comparison["equivalent"] is False
    assert "quantity:count:2" in cast(list[str], comparison["added"])


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
    manifest = _protected(text)
    signature = set(cast(list[str], manifest["semantic_signature"]))
    assert "identifier:identifier:ADR-42" in signature
    assert "claim:action=change;actor=adr-42;modality=must;polarity=negative;status=asserted;target=%" in signature
    assert "quantity:threshold:operator=gt;unit=percent;value=15" in signature
    assert "citation:reference:[E-7]" in signature
    assert "governance:term:approval" in signature
    assert "governance:term:waiver" in signature


@pytest.mark.parametrize(
    "text",
    [
        "Do not approve the architecture or begin an irreversible V2 cutover yet.",
        "Do not approve the architecture yet. Do not begin any irreversible V2 cutover.",
        "Do not approve the architecture, and do not begin an irreversible V2 cutover.",
        "Do not approve the architecture yet, and do not begin any irreversible V2 cutover.",
    ],
)
def test_negative_coordination_preserves_each_claim(text: str) -> None:
    signature = _signature(text)
    assert "claim:action=approve;actor=unspecified;modality=must;polarity=negative;status=deferred;target=architecture" in signature
    assert "claim:action=begin;actor=unspecified;modality=must;polarity=negative;status=deferred;target=v2 cutover" in signature


def test_explicit_and_do_not_cutover_rewrite_matches_canonical_original(
    recommendation_fixture: dict[str, str],
) -> None:
    rewrite = (
        "Recommendation: Do not approve the architecture yet, and do not begin any irreversible V2 cutover. "
        "Treat the hybrid topology found in the repository as a provisional current-state baseline only. "
        "The platform team must immediately address the confirmed authorization concerns. "
        "The messaging team must verify the two critical messaging loss hypotheses. "
        "Require closure evidence for the governed recommendations before a target architecture returns for human decision."
    )
    comparison = _comparison(recommendation_fixture["original"], rewrite)
    assert comparison["equivalent"] is True


def test_approving_and_beginning_cutover_remains_rejected(
    recommendation_fixture: dict[str, str],
) -> None:
    changed = (
        "Recommendation: Approve the architecture and begin the V2 cutover. "
        "The platform team must immediately address the confirmed authorization concerns. "
        "The messaging team must verify the two critical messaging loss hypotheses."
    )
    comparison = _comparison(recommendation_fixture["original"], changed)
    assert comparison["equivalent"] is False
    assert comparison["disposition"] == "changed"


def test_synonym_substitution_is_not_deterministically_equivalent() -> None:
    comparison = _comparison("Do not approve the architecture.", "Do not ratify the architecture.")
    assert comparison["equivalent"] is False
    assert comparison["disposition"] == "changed"


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Do not retire the legacy broker.", "Do not decommission the shared mesh."),
        ("The team must not expand the pilot.", "The team must not cancel the pilot."),
        ("Never delete the audit log.", "Never publish the audit log."),
        ("Do not escalate the incident.", "Do not close the incident."),
        ("The team must retire the broker.", "The team must retire the gateway."),
    ],
)
def test_unseen_different_structural_claims_are_rejected(source: str, candidate: str) -> None:
    forward = _comparison(source, candidate)
    reverse = _comparison(candidate, source)
    assert forward["disposition"] == "changed"
    assert reverse["disposition"] == "changed"


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Do not archive the telemetry buffer.", "Never archive the telemetry buffer."),
        ("The operator must quarantine the cache.", "The operator shall quarantine the cache."),
        ("Archive the telemetry buffer.", "Please archive the telemetry buffer."),
    ],
)
def test_unseen_equivalent_structural_claims_are_preserved(source: str, candidate: str) -> None:
    assert _comparison(source, candidate)["disposition"] == "equivalent"


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("Do not archive the telemetry buffer.", "Do not expose the telemetry buffer."),
        ("The operator must quarantine the cache.", "The operator must quarantine the queue."),
        ("Rotate the credential.", "Do not rotate the credential."),
    ],
)
def test_own_unseen_different_structural_claims_are_rejected(source: str, candidate: str) -> None:
    forward = _comparison(source, candidate)
    reverse = _comparison(candidate, source)
    assert forward["disposition"] == "changed"
    assert reverse["disposition"] == "changed"


def test_protected_extraction_is_deterministic() -> None:
    text = "Team A must deploy; Team B should review; ADR-42 is approved."
    first = _protected(text)
    second = _protected(text)
    assert first == second
    assert first["semantic_signature"] == sorted(cast(list[str], first["semantic_signature"]))


def test_linguistic_model_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_parse(text: str) -> None:
        raise LinguisticModelError("model unavailable")

    monkeypatch.setattr("lingity.invariants.parse", broken_parse)
    with pytest.raises(LinguisticModelError):
        _protected("ADR-42 must remain governed.")
