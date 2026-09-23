"""Tests for the critique brief, the verdict loop, and the subagent transport.

These tests exercise the governance properties that matter most: a candidate is
accepted only when it is genuinely better *and* still means the same thing, a
transport can never talk its way to an acceptance, and nothing ever degrades
into a success-shaped default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaValidationError

from lingity.analyzer import analyze_text
from lingity.critique import CritiqueError, build_critique
from lingity.improve import ImprovementError, improve_text, judge_candidate
from lingity.models import JsonValue
from lingity.profiles import SCHEMA_DIR, Profile, load_profile
from lingity.providers import (
    ProviderError,
    available_proposal_providers,
    create_proposal_provider,
)
from lingity.providers.agent import SubagentProvider
from lingity.providers.base import ChallengeResult, ProposalRequest
from lingity.styles import load_style

FIXTURE = Path(__file__).parent / "fixtures" / "recommended-decision.json"


def _fixture() -> tuple[str, str]:
    data = cast(dict[str, str], json.loads(FIXTURE.read_text(encoding="utf-8")))
    return data["original"], data["rewrite"]


def _schema(name: str) -> Draft202012Validator:
    payload = cast(
        dict[str, Any], json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
    )
    return Draft202012Validator(payload)


@pytest.fixture(scope="module")
def profile() -> Profile:
    return load_profile()


class _StubChallenger:
    def __init__(self, disposition: str) -> None:
        self._disposition = disposition

    @property
    def name(self) -> str:
        return "stub"

    def challenge(self, source_text: str, candidate_text: str) -> ChallengeResult:
        claims = ("omitted_claim",) if self._disposition == "material_change" else ()
        return ChallengeResult(
            disposition=self._disposition,
            claims=claims,
            provider="stub",
            model="stub-1",
        )


def test_critique_brief_validates_and_is_deterministic() -> None:
    original, _ = _fixture()
    analysis = analyze_text(original)
    first = build_critique(analysis)
    second = build_critique(analysis)

    _schema("critique.schema.json").validate(first)
    assert first == second
    assert first["critique_sha256"] == second["critique_sha256"]
    assert "style" not in first


def test_critique_brief_carries_defects_and_protected_elements() -> None:
    original, _ = _fixture()
    brief = build_critique(analyze_text(original))

    defects = cast(list[dict[str, JsonValue]], brief["defects"])
    must_preserve = cast(list[dict[str, JsonValue]], brief["must_preserve"])
    assert defects, "a flawed paragraph must yield defects to work on"
    assert must_preserve, "the brief must state what may not change"

    for defect in defects:
        for field in (
            "rule_id",
            "severity",
            "location",
            "observed_value",
            "threshold",
            "remediation",
            "excerpt",
        ):
            assert field in defect, f"defect is missing {field}"
    constraints = cast(dict[str, JsonValue], brief["rewrite_constraints"])
    assert constraints["editing_mode"] == "minimum_necessary_change"
    assert cast(int, constraints["maximum_candidate_readable_words"]) > cast(
        int, constraints["source_readable_words"]
    )


def test_critique_defects_are_ranked_by_severity() -> None:
    original, _ = _fixture()
    brief = build_critique(analyze_text(original))
    order = {"high": 0, "medium": 1, "low": 2}
    ranks = [
        order[cast(str, defect["severity"])]
        for defect in cast(list[dict[str, JsonValue]], brief["defects"])
    ]
    assert ranks == sorted(ranks)


def test_critique_records_prior_rejections() -> None:
    original, _ = _fixture()
    prior: list[dict[str, JsonValue]] = [
        {"index": 1, "rejection_reasons": ["protected meaning is changed"]}
    ]
    brief = build_critique(analyze_text(original), prior_attempts=prior)
    _schema("critique.schema.json").validate(brief)
    assert cast(list[JsonValue], brief["prior_attempts"]) == prior


def test_style_changes_critique_digest_and_is_digest_bound() -> None:
    original, _ = _fixture()
    analysis = analyze_text(original)
    plain = build_critique(analysis)
    style = load_style("architecture-review")
    styled = build_critique(analysis, style=style)

    _schema("critique.schema.json").validate(styled)
    assert styled["critique_sha256"] != plain["critique_sha256"]
    snapshot = cast(dict[str, JsonValue], styled["style"])
    assert snapshot["reference"] == style.reference()
    assert snapshot["contract"] == style.data
    assert snapshot["instructions"] == style.render()
    assert "style_score" not in json.dumps(styled, sort_keys=True)
    assert "style_fit" not in json.dumps(styled, sort_keys=True)


def test_critique_refuses_an_incomplete_artifact() -> None:
    original, _ = _fixture()
    analysis = analyze_text(original)
    del analysis["findings"]
    with pytest.raises(CritiqueError) as error:
        build_critique(analysis)
    assert "findings" in str(error.value)


def test_rewrite_is_accepted(profile: Profile) -> None:
    original, rewrite = _fixture()
    accepted, reasons, evidence = judge_candidate(original, rewrite, profile)
    assert accepted, f"the clearer rewrite should be accepted; got {reasons}"
    assert reasons == ()
    assert evidence["protected_disposition"] == "equivalent"
    assert cast(float, evidence["candidate_score"]) > cast(
        float, evidence["source_score"]
    )


def test_regression_is_rejected(profile: Profile) -> None:
    original, rewrite = _fixture()
    accepted, reasons, _ = judge_candidate(rewrite, original, profile)
    assert not accepted
    assert any("regressed" in reason for reason in reasons)


def test_rejected_candidate_names_the_elements_that_moved(profile: Profile) -> None:
    """A rejection has to be actionable, not just a refusal.

    "Meaning changed" tells a host agent nothing it can fix. The verdict must
    name the protected elements that were dropped so the next attempt can
    restore them, which is the only way an iterative loop can converge.
    """

    original, _ = _fixture()
    dropped = original.replace("two critical messaging loss hypotheses", "hypotheses")
    assert dropped != original, "the fixture text changed; update this mutation"

    accepted, reasons, evidence = judge_candidate(original, dropped, profile)
    assert not accepted
    delta = cast(dict[str, list[str]], evidence["protected_delta"])
    assert set(delta) == {"missing", "added", "unresolved", "specified"}
    moved = delta["missing"] + delta["added"] + delta["unresolved"]
    assert moved, f"a rejected candidate reported no delta; reasons were {reasons}"
    assert any("2" in element for element in moved), (
        f"the dropped count was not named in the delta: {moved}"
    )


def test_identical_text_is_rejected(profile: Profile) -> None:
    original, _ = _fixture()
    accepted, reasons, _ = judge_candidate(original, original, profile)
    assert not accepted, "a tie is not an improvement"
    assert any("did not change" in reason for reason in reasons)


def test_candidate_that_exceeds_the_word_growth_budget_is_rejected(
    profile: Profile,
) -> None:
    original, rewrite = _fixture()
    expanded = rewrite + (
        "\n\nThis additional discussion repeats the recommendation in a more "
        "helpful and comprehensive way for readers who may want more context "
        "before reaching the same decision."
    )
    accepted, reasons, evidence = judge_candidate(original, expanded, profile)
    economy = cast(dict[str, JsonValue], evidence["economy"])
    assert not accepted
    assert economy["passed"] is False
    assert any("readable-word growth budget" in reason for reason in reasons)


def test_shorter_candidate_passes_the_economy_gate(profile: Profile) -> None:
    original, rewrite = _fixture()
    accepted, reasons, evidence = judge_candidate(original, rewrite, profile)
    economy = cast(dict[str, JsonValue], evidence["economy"])
    assert accepted, reasons
    assert economy["passed"] is True
    assert cast(int, economy["candidate_readable_words"]) <= cast(
        int, economy["maximum_candidate_readable_words"]
    )


def test_an_identified_earlier_framing_block_may_be_removed() -> None:
    profile = load_profile("local-service")
    intro = (
        "Permits, inspections, and what tends to be wrong with the heating and "
        "cooling in a Howard County house.\n\n"
    )
    source = (
        "# HVAC in Howard County\n\n"
        + intro
        + "- Howard County permits and inspections\n"
        "- Local prices, dated at the source\n"
        "- No national averages\n\n"
        "## On this page\n\n"
        "1. Do you actually need a permit?\n"
        "2. What the permit costs\n"
        "3. The inspections\n"
        "4. What Howard County houses are, and what goes wrong in them\n\n"
        "Permits, inspections, what the county actually charges, and what tends "
        "to be wrong with the heating and cooling in a Howard County house."
    )
    candidate = source.replace(intro, "", 1)

    accepted, reasons, evidence = judge_candidate(source, candidate, profile)

    assert accepted, reasons
    assert evidence["protected_disposition"] == "equivalent"
    assert cast(dict[str, JsonValue], evidence["economy"])["passed"] is True


def test_keeping_a_removable_framing_block_is_not_a_meaning_change() -> None:
    """Deleting a restated intro is allowed, never required."""
    profile = load_profile("local-service")
    source = (
        "# HVAC in Howard County\n\n"
        "Permits, inspections, and what tends to be wrong with the heating and "
        "cooling in a Howard County house.\n\n"
        "- Howard County permits and inspections\n"
        "- Local prices, dated at the source\n"
        "- No national averages\n\n"
        "## On this page\n\n"
        "1. Do you actually need a permit?\n"
        "2. What the permit costs\n"
        "3. The inspections\n"
        "4. What Howard County houses are, and what goes wrong in them\n\n"
        "Permits, inspections, what the county actually charges, and what tends "
        "to be wrong with the heating and cooling in a Howard County house."
    )
    candidate = source.replace(
        "1. Do you actually need a permit?", "1. Do you need a permit?", 1
    )

    _, _, evidence = judge_candidate(source, candidate, profile)

    assert evidence["protected_disposition"] == "equivalent"
    assert evidence["protected_delta"] == {
        "missing": [],
        "added": [],
        "unresolved": [],
        "specified": [],
    }


def test_earlier_framing_block_with_unique_claims_may_not_be_removed() -> None:
    """Framing detection is partial term overlap, not meaning equivalence.

    An earlier block that carries a claim the later block lacks stays in the
    meaning baseline, so deleting it is reported as a dropped element rather
    than hidden by the duplicated-framing exemption.
    """
    profile = load_profile("local-service")
    intro = (
        "What the county requires, what the work costs here, and what tends to "
        "go wrong in a Howard County house.\n\n"
    )
    source = (
        "# HVAC in Howard County\n\n"
        + intro
        + "- Howard County permits and inspections\n"
        "- Local prices, dated at the source\n"
        "- No national averages\n\n"
        "## On this page\n\n"
        "1. Do you actually need a permit?\n"
        "2. What the permit costs\n"
        "3. The inspections\n"
        "4. What Howard County houses are, and what goes wrong in them\n\n"
        "Permits, inspections, what the county actually charges, and what tends "
        "to be wrong with the heating and cooling in a Howard County house."
    )
    assert "LING-DUPLICATED-FRAMING-001" in {
        cast(str, finding["rule_id"])
        for finding in cast(
            list[dict[str, JsonValue]], analyze_text(source, profile)["findings"]
        )
    }
    candidate = source.replace(intro, "", 1)

    accepted, reasons, evidence = judge_candidate(source, candidate, profile)

    assert not accepted
    assert evidence["protected_disposition"] == "changed"
    delta = cast(dict[str, list[str]], evidence["protected_delta"])
    assert any("action=cost" in element for element in delta["missing"])


def test_later_canonical_framing_block_may_not_be_removed() -> None:
    profile = load_profile("local-service")
    source = (
        "# HVAC in Howard County\n\n"
        "What the county requires, what the work costs here, and what tends to "
        "go wrong in a Howard County house.\n\n"
        "- Howard County permits and inspections\n"
        "- Local prices, dated at the source\n"
        "- No national averages\n\n"
        "## On this page\n\n"
        "1. Do you actually need a permit?\n"
        "2. What the permit costs\n"
        "3. The inspections\n"
        "4. What Howard County houses are, and what goes wrong in them\n\n"
        "Permits, inspections, what the county actually charges, and what tends "
        "to be wrong with the heating and cooling in a Howard County house."
    )
    candidate = source.replace(
        "Permits, inspections, what the county actually charges, and what tends "
        "to be wrong with the heating and cooling in a Howard County house.",
        "",
        1,
    )

    accepted, reasons, evidence = judge_candidate(source, candidate, profile)

    assert not accepted
    assert evidence["protected_disposition"] == "changed"
    assert any("protected meaning is changed" in reason for reason in reasons)


def test_framing_exception_does_not_unprotect_content_outside_the_earlier_block() -> None:
    profile = load_profile("local-service")
    source = (
        "# HVAC in Howard County\n\n"
        "What the county requires, what the work costs here, and what tends to "
        "go wrong in a Howard County house.\n\n"
        "- Howard County permits and inspections\n"
        "- Local prices, dated at the source\n"
        "- No national averages\n\n"
        "## On this page\n\n"
        "1. Do you actually need a permit?\n"
        "2. What the permit costs\n"
        "3. The inspections\n"
        "4. What Howard County houses are, and what goes wrong in them\n\n"
        "Permits, inspections, what the county actually charges, and what tends "
        "to be wrong with the heating and cooling in a Howard County house.\n\n"
        "Howard County requires 2 inspections for permit HVAC-42."
    )
    candidate = source.replace(
        "What the county requires, what the work costs here, and what tends to "
        "go wrong in a Howard County house.\n\n",
        "",
        1,
    ).replace(
        "\n\nHoward County requires 2 inspections for permit HVAC-42.",
        "",
        1,
    )

    accepted, reasons, evidence = judge_candidate(source, candidate, profile)

    assert not accepted
    assert evidence["protected_disposition"] == "changed"
    delta = cast(dict[str, list[str]], evidence["protected_delta"])
    assert any("2" in element or "HVAC-42" in element for element in delta["missing"])
    assert any("protected meaning is changed" in reason for reason in reasons)


@pytest.mark.parametrize(
    "candidate",
    [
        "Recommendation: Approve the architecture and begin the V2 cutover now.",
        "Short and clear.",
        "Recommendation: Do not ratify the architecture or start an irreversible V2 cutover yet.",
    ],
)
def test_meaning_change_is_rejected_even_when_it_scores_higher(
    candidate: str, profile: Profile
) -> None:
    original, _ = _fixture()
    accepted, reasons, evidence = judge_candidate(original, candidate, profile)
    assert cast(float, evidence["candidate_score"]) > cast(
        float, evidence["source_score"]
    ), "this case only proves something if the candidate scores better"
    assert not accepted, "a higher score must never buy a change in meaning"
    assert any("protected meaning" in reason for reason in reasons)


def test_challenger_can_add_doubt_but_never_clear_it(profile: Profile) -> None:
    original, rewrite = _fixture()

    accepted, _, _ = judge_candidate(
        original, rewrite, profile, challenger=_StubChallenger("no_material_change")
    )
    assert accepted

    for disposition in ("material_change", "needs_human"):
        blocked, reasons, _ = judge_candidate(
            original, rewrite, profile, challenger=_StubChallenger(disposition)
        )
        assert not blocked, f"{disposition} must block acceptance"
        assert reasons

    rescued, reasons, _ = judge_candidate(
        rewrite, original, profile, challenger=_StubChallenger("no_material_change")
    )
    assert not rescued, "a challenger must never rescue a deterministic failure"
    assert any("regressed" in reason for reason in reasons)


def test_loop_accepts_a_later_candidate(tmp_path: Path, profile: Profile) -> None:
    original, rewrite = _fixture()
    bad = tmp_path / "c1.txt"
    bad.write_text("Recommendation: Approve everything now.", encoding="utf-8")
    good = tmp_path / "c2.txt"
    good.write_text(rewrite, encoding="utf-8")

    result = improve_text(
        original, profile, SubagentProvider([bad, good]), max_attempts=2
    )
    assert result.accepted
    assert result.selected_text == rewrite
    assert result.selected_score > result.source_score
    assert [attempt.accepted for attempt in result.attempts] == [False, True]
    assert result.attempts[0].rejection_reasons


def test_improvement_loop_passes_style_without_changing_judgement(
    tmp_path: Path, profile: Profile
) -> None:
    original, rewrite = _fixture()
    candidate = tmp_path / "candidate.txt"
    candidate.write_text(rewrite, encoding="utf-8")
    provider = SubagentProvider([candidate])

    result = improve_text(
        original,
        profile,
        provider,
        max_attempts=1,
        style=load_style("architecture-review"),
    )

    assert result.accepted
    assert result.selected_text == rewrite


def test_loop_returns_the_source_when_nothing_is_accepted(
    tmp_path: Path, profile: Profile
) -> None:
    original, _ = _fixture()
    bad = tmp_path / "c1.txt"
    bad.write_text("Recommendation: Approve everything now.", encoding="utf-8")

    result = improve_text(original, profile, SubagentProvider([bad]), max_attempts=1)
    assert not result.accepted
    assert result.selected_text == original, "a failed run must not alter the text"
    assert result.selected_score == result.source_score
    assert "retained unchanged" in result.stop_reason


def test_loop_rejects_a_nonsensical_attempt_budget(profile: Profile) -> None:
    original, _ = _fixture()
    with pytest.raises(ImprovementError):
        improve_text(
            original, profile, SubagentProvider([Path("unused")]), max_attempts=0
        )


def test_subagent_provider_stops_cleanly_when_candidates_run_out(
    tmp_path: Path, profile: Profile
) -> None:
    """Running out of host-written candidates ends the run; it is not a crash."""

    original, _ = _fixture()
    candidate = tmp_path / "c1.txt"
    candidate.write_text("Recommendation: Approve everything now.", encoding="utf-8")

    result = improve_text(
        original, profile, SubagentProvider([candidate]), max_attempts=3
    )
    assert not result.accepted
    assert result.selected_text == original
    assert len(result.attempts) == 1, "only the supplied candidate may be judged"
    assert "ran out of candidates" in result.stop_reason


def test_a_run_that_judges_nothing_is_an_error(tmp_path: Path, profile: Profile) -> None:
    original, _ = _fixture()
    provider = SubagentProvider([tmp_path / "c1.txt"])
    (tmp_path / "c1.txt").write_text("Approve everything now.", encoding="utf-8")
    provider.propose(ProposalRequest(brief=build_critique(analyze_text(original))))

    with pytest.raises(ImprovementError) as error:
        improve_text(original, profile, provider, max_attempts=1)
    assert "nothing to judge" in str(error.value)


def test_subagent_provider_rejects_missing_and_empty_candidates(
    tmp_path: Path,
) -> None:
    original, _ = _fixture()
    brief = build_critique(analyze_text(original))
    request = ProposalRequest(brief=brief)

    missing = tmp_path / "absent.txt"
    with pytest.raises(ProviderError) as absent:
        SubagentProvider([missing]).propose(request)
    assert "does not exist" in str(absent.value)

    blank = tmp_path / "blank.txt"
    blank.write_text("   \n", encoding="utf-8")
    with pytest.raises(ProviderError) as empty:
        SubagentProvider([blank]).propose(request)
    assert "empty" in str(empty.value)


def test_subagent_provider_requires_at_least_one_candidate() -> None:
    with pytest.raises(ProviderError):
        SubagentProvider([])


def test_unknown_provider_names_the_registered_ones() -> None:
    with pytest.raises(ProviderError) as error:
        create_proposal_provider("telepathy")
    message = str(error.value)
    assert "telepathy" in message
    assert "subagent" in message


def test_network_providers_are_registered() -> None:
    registered = available_proposal_providers()
    assert {"subagent", "openai", "anthropic"} <= set(registered)


def test_network_providers_require_an_explicit_model() -> None:
    for name in ("openai", "anthropic"):
        with pytest.raises(ProviderError):
            create_proposal_provider(name, model=None)


def test_verdict_schema_requires_a_reason_for_every_rejection() -> None:
    validator = _schema("verdict.schema.json")
    base: dict[str, JsonValue] = {
        "schema_version": "1.1.0",
        "accepted": False,
        "rejection_reasons": [],
        "source_score": 50.0,
        "candidate_score": 60.0,
        "protected_disposition": "changed",
        "protected_delta": {
            "missing": ["quantity:count:2"],
            "added": [],
            "unresolved": [],
            "specified": [],
        },
        "economy": {
            "source_readable_words": 100,
            "candidate_readable_words": 90,
            "growth": -10,
            "allowed_growth": 12,
            "maximum_candidate_readable_words": 112,
            "prefer_shorter_candidate": True,
            "passed": True,
        },
        "challenge": None,
        "profile": {"name": "architecture-review", "version": "1.4.0", "digest": "0" * 64},
        "linguistic_model": {
            "name": "en_core_web_sm",
            "version": "3.8.0",
            "runtime": "spacy-3.8.13",
            "digest": "0" * 64,
        },
    }
    with pytest.raises(SchemaValidationError):
        validator.validate(base)

    base["rejection_reasons"] = ["protected meaning is changed"]
    validator.validate(base)

    accepted = dict(base)
    accepted["accepted"] = True
    with pytest.raises(SchemaValidationError):
        validator.validate(accepted)
