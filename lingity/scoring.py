"""Attributed Human Readability Index calculation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

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


def calculate_hri(findings: Iterable[Finding], profile: Profile) -> dict[str, JsonValue]:
    grouped: dict[str, list[Finding]] = defaultdict(list)
    for finding in findings:
        grouped[finding.dimension].append(finding)

    components: list[JsonValue] = []
    total = 0.0
    for dimension in DIMENSIONS:
        deductions = sorted(grouped[dimension], key=lambda item: (item.location.start, item.rule_id))
        deducted = round(sum(item.penalty for item in deductions), 2)
        raw_score = round(max(0.0, 100.0 - deducted), 2)
        weight = profile.weights[dimension]
        contribution = round(raw_score * weight / 100.0, 2)
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

    value = round(total, 2)
    bands = profile.data["bands"]
    band = next(
        entry["name"]
        for entry in bands
        if entry["minimum"] <= value <= entry["maximum"]
    )
    return {
        "name": "Human Readability Index",
        "value": value,
        "band": band,
        "formula": "sum(component.raw_score * component.weight / 100)",
        "components": components,
    }
