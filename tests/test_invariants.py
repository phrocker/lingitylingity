from __future__ import annotations

from typing import cast

import pytest

from lingity.invariants import (
    _ordering_relations,
    compare_protected,
    extract_protected,
)
from lingity.models import JsonValue
from lingity.morphology import canonical_action
from lingity.nlp import LinguisticModelError
from lingity.profiles import _available_profile_paths, load_profile


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


_TRAILING_GATE = (
    "Require closure evidence for the governed recommendations "
    "before a target architecture returns for human decision."
)
_FRONTED_GATE = (
    "Before a target architecture returns for human decision, "
    "require closure evidence for the governed recommendations."
)


def _ordering(text: str) -> list[str]:
    return [item for item in _signature(text) if item.startswith("order:")]


def test_moving_an_ordering_clause_to_the_front_preserves_meaning() -> None:
    """Fronting a "before" clause reorders words, not instructions.

    The parser attaches a trailing "before" to the nearest noun and a fronted
    one to the main verb, so the same gate used to produce two different
    signatures and the meaning check rejected a faithful reordering.
    """

    assert _comparison(_TRAILING_GATE, _FRONTED_GATE)["equivalent"] is True


def test_an_ordering_clause_does_not_leak_into_the_claim_target() -> None:
    """The gated action is the claim; the sequencing is its own element."""

    targets = [
        item for item in _signature(_TRAILING_GATE) if item.startswith("claim:")
    ]
    assert targets == [
        "claim:action=require;actor=unspecified;modality=must;polarity=positive;"
        "status=asserted;target=closure evidence govern recommendation"
    ], targets


@pytest.mark.parametrize("text", [_TRAILING_GATE, _FRONTED_GATE])
def test_the_sequence_is_reported_the_same_way_from_either_order(text: str) -> None:
    assert _ordering(text) == [
        "order:sequence:earlier=require closure evidence govern recommendation;"
        "later=target architecture return human decision"
    ], _ordering(text)


@pytest.mark.parametrize(
    ("marker", "fronted_marker"),
    [
        ("before", "Before"),
        ("after", "After"),
        ("until", "Until"),
        ("once", "Once"),
        ("following", "Following"),
    ],
)
def test_a_subordinate_clause_states_its_sequence_in_either_position(
    marker: str, fronted_marker: str
) -> None:
    """A conjunction hangs below its clause, so it has no children to read.

    Only the prepositional shape was handled, so "close the findings before the
    design returns" reported no sequence at all and compared equal to the same
    sentence with "after" -- a silently inverted gate.
    """

    trailing = f"Close the findings {marker} the design returns to the board."
    fronted = f"{fronted_marker} the design returns to the board, close the findings."
    assert _ordering(trailing), trailing
    assert _ordering(trailing) == _ordering(fronted), (trailing, fronted)


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        ("before", "after"),
        ("until", "once"),
        ("before", "following"),
    ],
)
def test_swapping_the_ordering_marker_is_still_a_meaning_change(
    earlier: str, later: str
) -> None:
    """Closing the reordering hole must not also let an inversion through."""

    first = f"Close the findings {earlier} the design returns to the board."
    second = f"Close the findings {later} the design returns to the board."
    assert _comparison(first, second)["equivalent"] is False


def test_dropping_the_ordering_clause_is_a_meaning_change() -> None:
    assert (
        _comparison(_TRAILING_GATE, "Require closure evidence for the governed recommendations.")[
            "equivalent"
        ]
        is False
    )


def test_exchanging_the_two_sequenced_clauses_is_a_meaning_change() -> None:
    """Reordering the words is safe; reordering the operands is not."""

    swapped = (
        "Before closure evidence is required for the governed recommendations, "
        "a target architecture returns for human decision."
    )
    assert _comparison(_TRAILING_GATE, swapped)["equivalent"] is False


@pytest.mark.parametrize(
    "clause",
    [
        "before a target architecture returns for human decision",
        "after a target architecture returns for human decision",
        "until a target architecture returns for human decision",
        "once a target architecture returns for human decision",
        "following a target architecture return for human decision",
        "prior to a target architecture returning for human decision",
    ],
)
def test_dropping_a_gate_is_detected_however_the_gate_is_worded(
    clause: str,
) -> None:
    """Excluding an ordering clause from a target must not lose the gate.

    A target sheds the clause so that word order stops changing the claim, but
    the sentence still says something the shortened version does not. Whichever
    element carries it, deleting the gate has to read as a meaning change --
    losing it silently is the one failure this check exists to prevent.
    """

    gated = f"Require closure evidence for the governed recommendations {clause}."
    ungated = "Require closure evidence for the governed recommendations."
    assert _comparison(gated, ungated)["equivalent"] is False, gated


@pytest.mark.parametrize(
    "text",
    [
        "Close the findings before the design returns to the board.",
        "Before the design returns to the board, close the findings.",
    ],
)
def test_a_sequence_is_located_over_both_clauses_it_relates(text: str) -> None:
    """The location has to cover what the finding is about.

    A conjunction has no children, so measuring the marker's own subtree ended
    the span at the marker word and pointed at "Close the findings before" --
    half of a relation whose other half is the clause that follows it.
    """

    profile = load_profile("architecture-review")
    spans = [
        item["text"]
        for item in cast(
            list[dict[str, JsonValue]], extract_protected(text, profile)["items"]
        )
        if item["category"] == "order"
    ]
    assert spans == [text], spans


def test_ordering_relations_are_computed_once_per_document() -> None:
    """Targets and sequences read one shared result rather than recomputing it.

    `_target_tokens` runs once per claim while `_ordering_relations` walks every
    token calling `children` and `subtree`, both of which scan the document.
    Recomputing per claim made extraction quadratic: forty sentences went from
    0.66s to 25s. Short fixtures hide that, so this asserts reuse directly.
    """

    profile = load_profile("architecture-review")
    sentence = (
        "Require closure evidence for the governed recommendations before a "
        "target architecture returns for human decision, and ensure the board "
        "approves the migration until the risks are closed. "
    )
    _ordering_relations.cache_clear()
    extract_protected(sentence * 4, profile)
    info = _ordering_relations.cache_info()

    assert info.hits > info.misses, info
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


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("The waiver is complete and granted.", "waiver"),
        ("The plan is final and approved.", "plan"),
        ("The design is complete, reviewed and accepted.", "design"),
    ],
)
def test_a_coordinated_state_still_names_what_is_in_it(
    sentence: str, expected: str
) -> None:
    """Coordination can put the subject more than one link above the participle.

    "The plan is final and approved" coordinates against the copula, so the
    subject is on the immediate head. "The waiver is complete and granted"
    coordinates against the *adjective* instead, so the subject sits one level
    further up -- and reporting `unknown` there says something was granted
    without saying what.
    """

    assert _status_subject(sentence) == expected


def test_an_active_past_tense_verb_still_names_nobody() -> None:
    """Following the head chain must not reach past the VBD guard.

    "The board approved" says the board did the approving, not that the board
    was approved, so no state subject is available at all.
    """

    assert _status_subject("The board approved.") == "unknown"


def test_no_shipped_profile_protects_meaning_with_memorised_phrases() -> None:
    """The gate must certify a rewrite from the parse, not from listed strings.

    `architecture-review` used to carry "recommended decision" and
    "recommendation:" so the fixture's two headings would cancel out. Those
    strings only ever matched the fixture, so they certified nothing about any
    other document.

    The profile list comes from the loader rather than being written out here,
    so a profile added later is covered without anyone remembering to add it.
    """
    shipped = sorted(_available_profile_paths())
    assert shipped, "no profiles ship, so this test would assert nothing"

    for name in shipped:
        assert load_profile(name).rules["protected_concepts"] == [], name


def test_a_governance_term_counts_once_however_often_it_appears() -> None:
    once = _signature("The board must ratify the migration.")
    twice = _signature("The board must ratify the migration and ratify the rollback.")

    assert once.count("governance:term:ratify") == 1
    assert twice.count("governance:term:ratify") == 1


def test_dropping_a_governance_term_entirely_is_still_detected() -> None:
    comparison = _comparison(
        "The board must ratify the migration.",
        "The board must complete the migration.",
    )

    assert comparison["equivalent"] is False
    assert "governance:term:ratify" in cast(list[str], comparison["missing"])


def test_losing_one_of_two_governed_items_survives_presence_comparison() -> None:
    """Counting terms is not what catches a dropped requirement.

    Both texts say "ratify", so the term is present either way and cannot be
    what fails. The claim carries the target, so the lost rollback still has
    to surface.
    """
    comparison = _comparison(
        "The board must ratify the migration. The board must ratify the rollback.",
        "The board must ratify the migration.",
    )
    missing = cast(list[str], comparison["missing"])

    assert comparison["equivalent"] is False
    assert "governance:term:ratify" not in missing
    assert any("target=rollback" in item for item in missing), missing


def test_only_governance_terms_collapse_to_presence() -> None:
    """A repeated quantity is still counted, so dropping one is still a loss."""
    comparison = _comparison(
        "Verify the two messaging hypotheses. Close the two authorization gaps.",
        "Verify the two messaging hypotheses.",
    )
    missing = cast(list[str], comparison["missing"])

    assert comparison["equivalent"] is False
    assert missing.count("quantity:count:2") == 1


def _statuses(text: str) -> list[str]:
    return [item for item in _signature(text) if item.startswith("governance:status")]


def test_a_reached_state_is_still_reported_as_a_status() -> None:
    assert _statuses("The migration was ratified by the board.") == [
        "governance:status:domain=ratification;polarity=positive;"
        "state=ratified;subject=migration"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "The migration must be ratified by the board.",
        "The migration should be ratified.",
        "The migration will be ratified.",
    ],
)
def test_a_state_under_a_modal_is_not_reported_as_reached(text: str) -> None:
    """Requiring or forecasting a state says it has not been reached."""
    assert _statuses(text) == []


def test_a_denied_state_is_not_reported_as_reached() -> None:
    """Status polarity describes the state word, never the sentence.

    "ratified" is a positive outcome and "rejected" a negative one, so the
    polarity field cannot carry the sentence's own negation. Reporting a
    status here therefore said the migration was ratified.
    """
    assert _statuses("The migration has not been ratified.") == []


def test_the_passive_remediation_keeps_protected_meaning() -> None:
    """LING-PASSIVE-001 asks for exactly this rewrite, so the gate must allow it."""
    comparison = _comparison(
        "The migration must be ratified by the board.",
        "The board must ratify the migration.",
    )

    assert comparison["equivalent"] is True, comparison


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The migration must be ratified.", "The migration is ratified."),
        ("The migration will be ratified.", "The migration is ratified."),
        ("The migration has not been ratified.", "The migration is ratified."),
    ],
)
def test_turning_an_unreached_state_into_a_fact_is_still_rejected(
    source: str, candidate: str
) -> None:
    """Dropping the false status must not drop the protection.

    The claim carries modality and polarity, so an obligation, a forecast, and
    a denial each still differ from the assertion that the state was reached.
    """
    assert _comparison(source, candidate)["equivalent"] is False


@pytest.mark.parametrize(
    "text",
    [
        "The migration must be reviewed and ratified.",
        "The migration has not been reviewed or ratified.",
    ],
)
def test_a_coordinated_state_inherits_the_modality_of_its_conjunct(text: str) -> None:
    """Coordination leaves the auxiliaries on the first conjunct.

    "must be reviewed and ratified" hangs `must` on "reviewed", so reading only
    "ratified"'s own children found no modal and reported the migration as
    ratified -- the inversion this guard exists to stop, surviving a single
    conjunction.
    """
    assert _statuses(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "The migration must be reviewed and ratified.",
        "The migration has not been reviewed or ratified.",
    ],
)
def test_an_unreached_coordinated_state_asserts_no_reached_claim(text: str) -> None:
    """The false status also reached the claims, which is where the gate reads."""
    assert [item for item in _signature(text) if "status=ratified" in item] == []


@pytest.mark.parametrize(
    "text",
    [
        "The migration was reviewed and ratified.",
        "The migration has not been reviewed but has been ratified.",
    ],
)
def test_a_reached_coordinated_state_is_still_reported(text: str) -> None:
    """Inheritance must not swallow states the text does report.

    The second sentence denies only the review. Negation distributes across
    "or" and "nor", not "but", so the ratification stays a fact.
    """
    assert len(_statuses(text)) == 1


def test_a_coordinated_obligation_is_not_equivalent_to_a_fact() -> None:
    """Dropping the false status must not drop the protection."""
    comparison = _comparison(
        "The migration must be reviewed and ratified.",
        "The migration is ratified.",
    )

    assert comparison["equivalent"] is False


def test_modality_inherits_across_a_contrastive_conjunction() -> None:
    """A documented conservative limit, not a silent one.

    "must be reviewed but was ratified last year" reports a real ratification,
    yet `_predicate_modal_tokens` carries `must` to every conjunct regardless
    of the conjunction, so the status is withheld. Negation is already
    restricted to "or"/"nor"; modality is not, and narrowing it would change
    claim construction, which is the gate's safety-critical path.

    The failure is conservative: the gate withholds a fact rather than
    inventing one, so it can only reject a rewrite, never certify a bad one.
    """
    assert _statuses("The migration must be reviewed but was ratified last year.") == []


@pytest.mark.parametrize(
    "text",
    [
        "The migration cannot remain ratified.",
        "The migration must remain ratified.",
        "The migration is not considered ratified.",
        "The board must consider the migration ratified.",
    ],
)
def test_a_bare_complement_inherits_the_modality_of_its_governor(text: str) -> None:
    """A complement leaves the modal and the negation on the verb above it.

    "cannot remain ratified" hangs both on "remain", so reading only
    "ratified" reported a ratification the sentence refuses.
    """
    assert _statuses(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "The migration remains ratified.",
        "The board cannot deny that the migration was ratified.",
    ],
)
def test_a_finite_complement_keeps_the_state_it_asserts(text: str) -> None:
    """Inheritance stops where the complement supplies its own clause.

    "cannot deny that the migration was ratified" asserts the ratification as
    content, and its `was` and `that` say so. Inheriting the matrix negation
    would deny a state the sentence affirms.
    """
    assert len(_statuses(text)) == 1


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The migration cannot remain ratified.", "The migration is ratified."),
        ("The migration is not considered ratified.", "The migration is ratified."),
        ("The board must consider the migration ratified.", "The migration is ratified."),
    ],
)
def test_a_withheld_complement_state_is_not_equivalent_to_a_fact(
    source: str, candidate: str
) -> None:
    """Withholding the false status must not withhold the protection."""
    assert _comparison(source, candidate)["equivalent"] is False


@pytest.mark.parametrize(
    ("source", "candidate"),
    [
        ("The team must have the credentials.", "The team must have the database."),
        ("The team must have the credentials.", "The vendor must have the credentials."),
    ],
)
def test_a_possession_claim_protects_its_actor_and_target(
    source: str, candidate: str
) -> None:
    """Dropping a verb from claim extraction stops comparing its arguments.

    `NON_CLAIM_VERB_LEMMAS` listed "have" to suppress the auxiliary reading,
    but `_is_claim_predicate` already requires pos == "VERB", so the entry only
    silenced the main verb -- and a rewrite could swap either the holder or the
    thing held and still be certified.
    """
    assert _comparison(source, candidate)["equivalent"] is False


def test_a_possession_is_extracted_as_a_claim() -> None:
    claims = [
        item
        for item in _signature("The team must have the credentials.")
        if item.startswith("claim:")
    ]

    assert claims == [
        "claim:action=have;actor=team;modality=must;polarity=positive;"
        "status=asserted;target=credential"
    ], claims


@pytest.mark.xfail(
    strict=True,
    reason=(
        "`use` is still in NON_CLAIM_VERB_LEMMAS, so its actor and target are "
        "never compared and this swap is certified. It cannot be removed until "
        "target normalization equates the canonical fixture's "
        "'repository-evidenced hybrid topology' with 'hybrid topology "
        "evidenced in the repository'; removing it today rejects the canonical "
        "rewrite. Pinned strictly so the fix forces this win into the open."
    ),
)
def test_a_use_claim_protects_its_target() -> None:
    assert (
        _comparison(
            "The board must use the framework.", "The board must use the database."
        )["equivalent"]
        is False
    )


def test_protected_delta_names_signatures_the_manifests_actually_store() -> None:
    """A reported delta must be restorable by name against `semantic_signature`.

    Comparison folds actor and target so a reordered modifier chain does not
    read as a different claim. Reporting that folded form would name a string
    present in neither manifest -- "platform reliability team" when both texts
    say "reliability platform team" -- and a caller matching against
    `semantic_signature` to restore the element would find nothing.
    """

    source = "The reliability platform team must verify the failover runbook."
    candidate = "The reliability platform team must verify the disaster runbook."
    comparison = _comparison(source, candidate)

    missing = cast(list[str], comparison["missing"])
    added = cast(list[str], comparison["added"])
    assert missing and added, "expected a reported delta for a changed target"

    assert set(missing) <= set(_signature(source))
    assert set(added) <= set(_signature(candidate))
