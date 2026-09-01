from __future__ import annotations

from typing import cast

import pytest

from lingity.invariants import compare_protected, extract_protected
from lingity.models import JsonValue
from lingity.morphology import canonical_action
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
    # Derived from the dependency parse, not from a stored phrasing: deferring
    # "architecture ratification" is an obligation not to ratify, yet.
    assert (
        "claim:action=ratify;actor=unspecified;modality=must;polarity=negative;"
        "status=deferred;target=architecture" in signature
    )
    assert (
        "claim:action=begin;actor=unspecified;modality=must;polarity=negative;"
        "status=deferred;target=irreversible v2 cutover" in signature
    )


def test_higher_scoring_rewrite_that_drops_content_is_rejected(
    recommendation_fixture: dict[str, str],
) -> None:
    """A better score never buys a meaning change.

    The unfaithful rewrite reads more clearly and scores higher than the
    faithful one, but it drops the count "two" and weakens the closure-evidence
    requirement. Acceptance must depend on preserved meaning, not on score.
    """

    comparison = _comparison(
        recommendation_fixture["original"],
        recommendation_fixture["unfaithful_rewrite"],
    )
    assert comparison["equivalent"] is False
    assert comparison["disposition"] != "equivalent"


def test_changed_quantity_is_rejected(
    recommendation_fixture: dict[str, str],
) -> None:
    changed = recommendation_fixture["rewrite"].replace(
        "two critical messaging loss hypotheses",
        "three critical messaging loss hypotheses",
    )
    assert changed != recommendation_fixture["rewrite"], "mutation did not apply"
    comparison = _comparison(recommendation_fixture["original"], changed)
    assert comparison["equivalent"] is False
    assert "quantity:count:2" in cast(list[str], comparison["missing"])
    assert "quantity:count:3" in cast(list[str], comparison["added"])


def test_dropped_duplicate_protected_value_is_rejected() -> None:
    comparison = _comparison("ADR-42 must retain two paths and two owners.", "ADR-42 must retain two paths and owners.")
    assert comparison["equivalent"] is False
    assert "quantity:count:2" in cast(list[str], comparison["missing"])


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
    # The bound travels with the claim. Reducing this target to "%" would let a
    # rewrite change "more than 15%" to any other threshold undetected.
    assert (
        "claim:action=change;actor=adr-42;modality=must;polarity=negative;"
        "status=asserted;target=more than 15 %" in signature
    )
    assert "quantity:threshold:operator=gt;unit=percent;value=15" in signature
    assert "citation:reference:[E-7]" in signature
    # Governance vocabulary is keyed by the action it names, so that the
    # nominal and verbal forms of the same term agree. The key is WordNet's
    # representative for the derivational family, not a display string.
    assert f"governance:term:{canonical_action('approval')}" in signature
    assert f"governance:term:{canonical_action('waiver')}" in signature
    assert canonical_action("approval") == canonical_action("approve")
    assert canonical_action("waiver") == canonical_action("waive")


@pytest.mark.parametrize(
    ("text", "approve_status", "begin_status"),
    [
        (
            "Do not approve the architecture or begin an irreversible V2 cutover yet.",
            "deferred",
            "deferred",
        ),
        (
            "Do not approve the architecture yet, and do not begin any irreversible V2 cutover.",
            "deferred",
            "deferred",
        ),
        (
            "Do not approve the architecture, and do not begin an irreversible V2 cutover.",
            "asserted",
            "asserted",
        ),
        (
            "Do not approve the architecture yet. Do not begin any irreversible V2 cutover.",
            "deferred",
            "asserted",
        ),
    ],
)
def test_negative_coordination_preserves_each_claim(
    text: str,
    approve_status: str,
    begin_status: str,
) -> None:
    """Each conjunct keeps its own negated claim, and deferral keeps its scope.

    A sentence-final "yet" scopes over a coordination, so the first two forms
    defer both directives. The third defers neither and is a permanent
    prohibition; the fourth defers only its first sentence. These are different
    instructions and the signature must distinguish them -- collapsing them
    would let a rewrite convert a postponement into a ban.
    """

    signature = _signature(text)
    assert (
        f"claim:action=approve;actor=unspecified;modality=must;polarity=negative;"
        f"status={approve_status};target=architecture" in signature
    )
    assert (
        f"claim:action=begin;actor=unspecified;modality=must;polarity=negative;"
        f"status={begin_status};target=irreversible v2 cutover" in signature
    )


def test_explicit_and_do_not_cutover_rewrite_matches_canonical_original(
    recommendation_fixture: dict[str, str],
) -> None:
    rewrite = (
        "Recommendation: Do not ratify the architecture yet, and do not begin any irreversible V2 cutover. "
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


CODE_REVIEW_CONCLUSION = (
    "The exact-policy-pin fix is complete and fail-closed, and the analogous "
    "index pin lookup remains exact and bounded. Static inspection and all "
    "requested validation found no actionable defects."
)


def test_linking_verb_states_are_part_of_the_signature() -> None:
    """A text is committed to what it says a thing is, not only what to do."""

    signature = _signature(CODE_REVIEW_CONCLUSION)
    states = [item for item in signature if "action=be;" in item or "action=remain;" in item]
    assert len(states) == 2, signature
    assert any("target=complete,fail close" in item for item in states), signature
    assert any("target=bound,exact" in item for item in states), signature


@pytest.mark.parametrize(
    ("mutation", "replacement"),
    [
        ("is complete and fail-closed", "is incomplete and fail-open"),
        ("complete and fail-closed", "complete and fail-open"),
        ("remains exact and bounded", "remains approximate and bounded"),
        ("exact and bounded", "exact and unbounded"),
        ("fix is complete", "fix is not complete"),
        ("complete and fail-closed", "complete"),
    ],
)
def test_flipping_a_predicate_adjective_is_never_certified_equivalent(
    mutation: str, replacement: str
) -> None:
    """The defect this guards against shipped once and was found on live text.

    Every one of these reverses what the conclusion asserts about the fix. A
    reviewer reading an accepted rewrite would be told the meaning was
    preserved.
    """

    candidate = CODE_REVIEW_CONCLUSION.replace(mutation, replacement)
    assert candidate != CODE_REVIEW_CONCLUSION
    comparison = _comparison(CODE_REVIEW_CONCLUSION, candidate)
    assert comparison["disposition"] == "changed", comparison


def test_unchanged_conclusion_still_compares_equivalent() -> None:
    """State claims must not make the gate reject text it should accept."""

    comparison = _comparison(CODE_REVIEW_CONCLUSION, CODE_REVIEW_CONCLUSION)
    assert comparison["disposition"] == "equivalent", comparison


def test_a_second_finite_predicate_is_not_absorbed_as_a_state() -> None:
    """"is complete and passes review" coordinates two predicates, not two states."""

    signature = _signature("The fix is complete and passes review.")
    states = [item for item in signature if "action=be;" in item]
    assert states == [
        "claim:action=be;actor=fix;modality=assertive;polarity=positive;"
        "status=asserted;target=complete"
    ], signature


def _status_subject(text: str) -> str:
    """Return the entity the governance status names, or "none" if none is emitted."""
    statuses = [item for item in _signature(text) if item.startswith("governance:status")]
    if not statuses:
        return "none"
    return statuses[0].split("subject=")[1]


@pytest.mark.parametrize(
    ("text", "subject"),
    [
        ("The board approved the migration.", "migration"),
        ("The board rejected the proposal.", "proposal"),
        ("The migration was approved by the board.", "migration"),
        ("The migration was approved.", "migration"),
        ("The migration is approved.", "migration"),
        ("The risks remain blocked.", "risks"),
        ("The waiver stays granted.", "waiver"),
        ("The migration is complete and approved.", "migration"),
    ],
)
def test_a_governance_status_names_the_entity_in_the_state(text: str, subject: str) -> None:
    """The status subject is the patient, whatever voice the sentence is written in.

    Every state this tracks is a transitive participle, so the entity in the
    state is never the one who acted. The subject came from the actor finder,
    which is the opposite role: "the board rejected the proposal" recorded the
    board as rejected, and the passive forms recorded either the by-phrase agent
    or nothing at all.
    """
    assert _status_subject(text) == subject


def test_an_active_verb_with_no_object_names_nobody_as_the_state() -> None:
    """"The board approved" does not say what was approved, so nothing is named.

    This is the case that made the old behaviour visible: the subject of an
    active past-tense verb is the actor, and recording it as the patient claimed
    the board had been approved. Naming nobody is the honest answer, and it
    matches how the actor finder already refuses to invent an absent agent.
    """
    assert _status_subject("The board approved.") == "unknown"


@pytest.mark.parametrize(
    ("active", "passive"),
    [
        ("The board approved the migration.", "The migration was approved by the board."),
        ("The board rejected the proposal.", "The proposal was rejected by the board."),
        ("The board granted the waiver.", "The waiver was granted by the board."),
    ],
)
def test_voice_alone_does_not_change_protected_meaning(active: str, passive: str) -> None:
    """Recasting a sentence between voices is the rewrite this tool most wants to allow.

    The pair agreed before this fix only because both sides named the actor and
    were wrong together. They must still agree now that both name the patient,
    or the gate would reject the clearest rewrite it exists to encourage.
    """
    assert _status_subject(active) == _status_subject(passive)
    assert _comparison(active, passive)["equivalent"] is True
