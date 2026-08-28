from __future__ import annotations

from typing import cast

from lingity.analyzer import analyze_text
from lingity.models import JsonValue


def test_rewrite_scores_better_and_reduces_findings(
    recommendation_fixture: dict[str, str],
) -> None:
    original = analyze_text(recommendation_fixture["original"])
    rewrite = analyze_text(recommendation_fixture["rewrite"])
    original_score = cast(dict[str, JsonValue], original["score"])["value"]
    rewrite_score = cast(dict[str, JsonValue], rewrite["score"])["value"]
    assert isinstance(original_score, float)
    assert isinstance(rewrite_score, float)
    assert rewrite_score > original_score
    assert len(cast(list[JsonValue], rewrite["findings"])) < len(
        cast(list[JsonValue], original["findings"])
    )


def test_required_rule_families_are_detected(
    recommendation_fixture: dict[str, str],
) -> None:
    analysis = analyze_text(recommendation_fixture["original"])
    findings = cast(list[dict[str, JsonValue]], analysis["findings"])
    rule_ids = {cast(str, finding["rule_id"]) for finding in findings}
    assert {
        "LING-SENTENCE-001",
        "LING-ACTION-001",
        "LING-NOMINALIZATION-001",
        "LING-NOUN-STACK-001",
        "LING-AGENCY-001",
        "LING-WEAK-VERB-001",
        "LING-JARGON-001",
        "LING-BUREAUCRACY-001",
    } <= rule_ids


def test_every_finding_is_fully_attributed(
    recommendation_fixture: dict[str, str],
) -> None:
    analysis = analyze_text(recommendation_fixture["original"])
    findings = cast(list[dict[str, JsonValue]], analysis["findings"])
    required = {
        "rule_id", "severity", "location", "observed_value", "threshold", "remediation"
    }
    assert findings
    assert all(required <= finding.keys() for finding in findings)


def test_score_arithmetic_is_attributed(
    recommendation_fixture: dict[str, str],
) -> None:
    analysis = analyze_text(recommendation_fixture["original"])
    score = cast(dict[str, JsonValue], analysis["score"])
    components = cast(list[dict[str, JsonValue]], score["components"])
    contribution = round(
        sum(cast(float, component["weighted_contribution"]) for component in components),
        2,
    )
    assert contribution == score["value"]
    assert sum(cast(int, component["weight"]) for component in components) == 100


def test_passive_voice_is_located() -> None:
    text = "The architecture was approved without a named owner."
    analysis = analyze_text(text)
    findings = cast(list[dict[str, JsonValue]], analysis["findings"])
    passive = next(
        finding for finding in findings if finding["rule_id"] == "LING-PASSIVE-001"
    )
    location = cast(dict[str, JsonValue], passive["location"])
    assert text[cast(int, location["start"]):cast(int, location["end"])] == "was approved"
