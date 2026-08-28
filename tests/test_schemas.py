from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from lingity.analyzer import analyze_text
from lingity.profiles import PROFILE_DIR, SCHEMA_DIR


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def test_all_schemas_are_valid_draft_2020_12() -> None:
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        Draft202012Validator.check_schema(_json(path))


def test_profile_validates() -> None:
    validator = Draft202012Validator(_json(SCHEMA_DIR / "profile.schema.json"))
    for path in sorted(PROFILE_DIR.glob("*.json")):
        validator.validate(_json(path))


def test_analysis_validates(recommendation_fixture: dict[str, str]) -> None:
    validator = Draft202012Validator(_json(SCHEMA_DIR / "analysis.schema.json"))
    validator.validate(analyze_text(recommendation_fixture["original"]))
