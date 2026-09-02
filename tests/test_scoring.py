from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import cast

import pytest
from jsonschema import Draft202012Validator

from lingity.analyzer import analyze_text
from lingity.models import Finding, JsonValue, Location
from lingity.nlp import model_fingerprint
from lingity.profiles import Profile, canonical_json, load_profile
from lingity.scoring import (
    DIMENSIONS,
    HRI_REPORTED_FLOOR,
    HRI_REPORTED_FLOOR_START_POINTS,
    calculate_hri,
)

SCHEMA_PATH = (
    Path(__file__).parents[1] / "lingity" / "schemas" / "v1" / "analysis.schema.json"
)
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "recommended-decision.json"


def _number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value)


def _round_hundredths(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _finding(
    dimension: str,
    points: float,
    *,
    start: int = 0,
    rule_number: int = 1,
) -> Finding:
    return Finding(
        rule_id=f"LING-SYNTHETIC-{rule_number:03d}",
        dimension=dimension,
        severity="high",
        location=Location(start=start, end=start + 1, line=1, column=start + 1),
        observed_value=points,
        threshold=0,
        remediation="Rewrite the text.",
        penalty=points,
    )


def _component(
    score: dict[str, JsonValue],
    dimension: str,
) -> dict[str, JsonValue]:
    components = cast(list[dict[str, JsonValue]], score["components"])
    return next(component for component in components if component["dimension"] == dimension)


def _schema() -> dict[str, object]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert isinstance(schema, dict)
    return cast(dict[str, object], schema)


def _analysis_artifact(
    score: dict[str, JsonValue],
    findings: list[Finding],
    profile: Profile,
) -> dict[str, JsonValue]:
    text = "Synthetic source sentence."
    artifact: dict[str, JsonValue] = {
        "schema_version": "1.1.0",
        "analyzer_version": "test",
        "linguistic_model": cast(dict[str, JsonValue], model_fingerprint()),
        "profile": profile.reference(),
        "source": {
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "length": len(text),
        },
        "sentences": [
            {
                "index": 0,
                "start": 0,
                "end": len(text),
                "text": text,
                "word_count": 3,
                "clause_count": 1,
                "action_count": 0,
            }
        ],
        "protected": {
            "items": [],
            "semantic_signature": [],
            "coverage": {"sentences": 0, "uncovered": []},
            "source_sha256": "0" * 64,
            "sha256": "0" * 64,
        },
        "findings": cast(list[JsonValue], [finding.to_dict() for finding in findings]),
        "score": score,
        "analysis_sha256": "0" * 64,
    }
    artifact["analysis_sha256"] = hashlib.sha256(
        canonical_json(artifact).encode("utf-8")
    ).hexdigest()
    return artifact


def _score_for_dimension_points(points: float, profile: Profile) -> dict[str, JsonValue]:
    findings = [] if points == 0.0 else [_finding("sentence_load", points)]
    return calculate_hri(findings, profile)


def test_dimension_deductions_are_non_increasing_across_dense_sweep() -> None:
    """Worse deductions never raise the reported HRI, even when 2-decimal ties occur."""
    profile = load_profile()
    deduction_points = [step / 2.0 for step in range(0, 2401)]
    values = [
        _number(_score_for_dimension_points(points, profile)["value"])
        for points in deduction_points
    ]

    assert all(later <= earlier for earlier, later in zip(values, values[1:]))


def test_operational_deduction_steps_drop_when_reported_precision_changes() -> None:
    """Ladder rungs are strict until two-decimal reporting reaches its raw-score floor."""
    profile = load_profile()
    ladder = (0.0, 10.0, 25.0, 50.0, 100.0, 150.0, 250.0, 500.0, 750.0, 1000.0)
    scores = {
        points: _score_for_dimension_points(points, profile)
        for points in ladder
    }

    assert _number(scores[100.0]["value"]) > _number(scores[150.0]["value"])

    for lower, higher in zip(ladder, ladder[1:]):
        lower_raw = _number(_component(scores[lower], "sentence_load")["raw_score"])
        higher_raw = _number(_component(scores[higher], "sentence_load")["raw_score"])
        if higher_raw < lower_raw:
            assert _number(scores[higher]["value"]) < _number(scores[lower]["value"])
        else:
            assert higher_raw == HRI_REPORTED_FLOOR
            assert _number(scores[higher]["value"]) == _number(scores[lower]["value"])


def test_resolution_floor_documents_known_two_decimal_limit() -> None:
    """Document the pure exponential floor caused by reporting raw_score to two decimals."""
    profile = load_profile()

    before_floor = calculate_hri(
        [_finding("sentence_load", HRI_REPORTED_FLOOR_START_POINTS - 0.01)],
        profile,
    )
    at_floor = calculate_hri(
        [_finding("sentence_load", HRI_REPORTED_FLOOR_START_POINTS)],
        profile,
    )
    beyond_floor = calculate_hri(
        [_finding("sentence_load", HRI_REPORTED_FLOOR_START_POINTS + 0.01)],
        profile,
    )

    assert _number(_component(before_floor, "sentence_load")["raw_score"]) == 0.01
    assert _number(_component(at_floor, "sentence_load")["raw_score"]) == HRI_REPORTED_FLOOR
    assert _number(_component(beyond_floor, "sentence_load")["raw_score"]) == HRI_REPORTED_FLOOR
    assert _number(at_floor["value"]) == _number(beyond_floor["value"]) == 75.0


def test_zero_findings_score_is_clear() -> None:
    score = calculate_hri([], load_profile())

    assert _number(score["value"]) == 100.0
    assert score["band"] == "clear"
    for dimension in DIMENSIONS:
        component = _component(score, dimension)
        assert _number(component["raw_score"]) == 100.0
        assert _number(component["deducted_points"]) == 0.0


def test_calibration_anchors_preserve_profile_bands() -> None:
    profile = load_profile()

    midpoint = calculate_hri([_finding("sentence_load", 50.0)], profile)
    assert _number(_component(midpoint, "sentence_load")["raw_score"]) == 50.0

    one_bad_dimension = calculate_hri([_finding("sentence_load", 100.0)], profile)
    assert one_bad_dimension["band"] != "clear"

    uniformly_bad = calculate_hri(
        [
            _finding(dimension, 40.0, start=index, rule_number=index + 1)
            for index, dimension in enumerate(DIMENSIONS)
        ],
        profile,
    )
    assert uniformly_bad["band"] == "revision_required"


def test_canonical_decision_fixture_keeps_bands_ordered_with_scores() -> None:
    fixture = cast(dict[str, str], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))

    original_score = cast(dict[str, JsonValue], analyze_text(fixture["original"])["score"])
    rewrite_score = cast(dict[str, JsonValue], analyze_text(fixture["rewrite"])["score"])
    unfaithful_score = cast(
        dict[str, JsonValue], analyze_text(fixture["unfaithful_rewrite"])["score"]
    )

    # The unfaithful rewrite scores best but is rejected by the meaning gate;
    # the faithful one trades score for fidelity. Both facts are asserted in
    # tests/test_invariants.py. Here we only require that the bands stay
    # ordered with the scores. No absolute band name is pinned: the pins this
    # test used to carry held only because three fabricated noun-stack findings
    # depressed the original's score, which made the pin a record of a bug
    # rather than a calibration.
    bands = ["unusable", "revision_required", "usable_but_improvable", "clear"]
    ordered = [original_score, rewrite_score, unfaithful_score]
    values = [_number(score["value"]) for score in ordered]
    indices = [bands.index(cast(str, score["band"])) for score in ordered]

    assert values == sorted(values)
    assert len(set(values)) == len(values)
    assert indices == sorted(indices)
    assert indices[-1] > indices[0]


def test_catastrophic_findings_remain_bounded_and_banded() -> None:
    profile = load_profile()
    findings = [
        _finding(dimension, 100_000.0, start=index, rule_number=index + 1)
        for index, dimension in enumerate(DIMENSIONS)
    ]
    score = calculate_hri(findings, profile)
    value = _number(score["value"])
    bands = cast(list[dict[str, JsonValue]], profile.data["bands"])

    assert 0.0 <= value <= 100.0
    assert any(
        _number(band["minimum"]) <= value <= _number(band["maximum"])
        and score["band"] == band["name"]
        for band in bands
    )


def test_weighted_contributions_are_attributed_and_reproducible() -> None:
    findings = [
        _finding("sentence_load", 50.0, start=0, rule_number=1),
        _finding("sentence_load", 100.0, start=1, rule_number=2),
        _finding("morphology", 25.0, start=2, rule_number=3),
        _finding("agency", 10.0, start=3, rule_number=4),
    ]
    score = calculate_hri(findings, load_profile())
    components = cast(list[dict[str, JsonValue]], score["components"])

    for component in components:
        deductions = cast(list[dict[str, JsonValue]], component["deductions"])
        deducted = round(sum(_number(item["points"]) for item in deductions), 2)
        raw_score = _number(component["raw_score"])
        weight = int(_number(component["weight"]))
        contribution = _round_hundredths(raw_score * weight / 100.0)
        assert _number(component["deducted_points"]) == deducted
        assert _number(component["weighted_contribution"]) == contribution

    total = _round_hundredths(
        sum(_number(component["weighted_contribution"]) for component in components)
    )
    assert total == _number(score["value"])
    assert {component["dimension"]: component["weight"] for component in components} == {
        "sentence_load": 25,
        "morphology": 20,
        "agency": 20,
        "lexical_clarity": 15,
        "structure": 10,
        "redundancy": 10,
    }
    assert sum(int(_number(component["weight"])) for component in components) == 100


def test_score_validates_against_existing_analysis_schema() -> None:
    profile = load_profile()
    findings = [
        _finding("sentence_load", 150.0, start=0, rule_number=1),
        _finding("redundancy", 20.0, start=1, rule_number=2),
    ]
    score = calculate_hri(findings, profile)

    Draft202012Validator(_schema()).validate(_analysis_artifact(score, findings, profile))


def test_strict_subset_of_findings_scores_higher() -> None:
    profile = load_profile()
    subset = [_finding("sentence_load", 50.0, start=0, rule_number=1)]
    superset = [
        *subset,
        _finding("sentence_load", 100.0, start=1, rule_number=2),
    ]

    assert _number(calculate_hri(subset, profile)["value"]) > _number(
        calculate_hri(superset, profile)["value"]
    )


def test_missing_band_raises_readable_error() -> None:
    profile = load_profile()
    data = cast(dict[str, object], deepcopy(profile.data))
    data["bands"] = [
        {"name": "too_low", "minimum": 0, "maximum": 10},
        {"name": "not_high_enough", "minimum": 20, "maximum": 90},
    ]
    broken_profile = Profile(data=data, digest=profile.digest)

    with pytest.raises(ValueError, match=r"HRI value 100\.0 does not fall"):
        calculate_hri([], broken_profile)
