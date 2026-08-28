from __future__ import annotations

import json

from lingity.analyzer import analyze_text


def test_analysis_is_byte_deterministic(recommendation_fixture: dict[str, str]) -> None:
    first = json.dumps(analyze_text(recommendation_fixture["original"]), sort_keys=True)
    second = json.dumps(analyze_text(recommendation_fixture["original"]), sort_keys=True)
    assert first == second


def test_empty_input_fails_explicitly() -> None:
    try:
        analyze_text(" \n")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("empty input returned a success-shaped analysis")
