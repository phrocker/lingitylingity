"""Attributed Human Readability Index calculation."""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable, cast

from lingity.models import Finding, JsonValue
from lingity.profiles import Profile

DIMENSIONS = (
    "sentence_load",
    "morphology",
    "agency",
    "lexical_clarity",
    "structure",
    "redundancy",
)

HRI_HALF_LIFE = 50.0
HRI_REPORTED_FLOOR = 0.0
HRI_REPORTED_FLOOR_START_POINTS = 714.39
HRI_FORMULA = (
    "for each component: deducted_points = round_half_up(sum(finding.penalty), 2); "
    "raw_score = round_half_up(100 * 2 ** (-deducted_points / 50.0), 2); "
    "weighted_contribution = round_half_up(raw_score * weight / 100, 2); "
    "value = round_half_up(sum(component.weighted_contribution), 2); "
    "all values are reported to 2 decimals and component raw_score reaches "
    "0.00 at deducted_points >= 714.39"
)


def _round_hundredths(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _raw_score(deducted_points: float) -> float:
    return _round_hundredths(100.0 * 2.0 ** (-deducted_points / HRI_HALF_LIFE))


def _band_for(value: float, profile: Profile) -> str:
    bands = cast(list[dict[str, JsonValue]], profile.data["bands"])
    for entry in bands:
        if _as_float(entry["minimum"]) <= value <= _as_float(entry["maximum"]):
            return str(entry["name"])
    ranges = ", ".join(
        f"{entry['name']}[{entry['minimum']}..{entry['maximum']}]" for entry in bands
    )
    raise ValueError(f"HRI value {value} does not fall within profile bands: {ranges}")


def _as_float(value: JsonValue) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"Expected numeric band boundary, got {value!r}")
    return float(value)


def calculate_hri(findings: Iterable[Finding], profile: Profile) -> dict[str, JsonValue]:
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        grouped[finding.dimension].append(finding)

    components: list[JsonValue] = []
    total = 0.0
    for dimension in DIMENSIONS:
        deductions = sorted(grouped[dimension], key=lambda item: (item.location.start, item.rule_id))
        deducted = _round_hundredths(sum(item.penalty for item in deductions))
        raw_score = _raw_score(deducted)
        weight = profile.weights[dimension]
        contribution = _round_hundredths(raw_score * weight / 100.0)
        total += contribution
        components.append(
            {
                "dimension": dimension,
                "weight": weight,
                "raw_score": raw_score,
                "deducted_points": deducted,
                "weighted_contribution": contribution,
                "deductions": [
                    {"rule_id": item.rule_id, "points": item.penalty}
                    for item in deductions
                ],
            }
        )

    value = _round_hundredths(total)
    band = _band_for(value, profile)
    return {
        "name": "Human Readability Index",
        "value": value,
        "band": band,
        "formula": HRI_FORMULA,
        "components": components,
    }
