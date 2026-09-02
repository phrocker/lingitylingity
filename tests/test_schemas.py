from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
import pytest

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


def test_analysis_with_uncovered_sentence_validates() -> None:
    """An uncovered sentence carries a dependency fingerprint, and the schema declares it.

    The corpora used by the other schema tests happen to parse cleanly, so
    every ``coverage.uncovered`` list they produce is empty and the shape of an
    uncovered entry goes unchecked. This text does not parse into a claim, so
    it exercises that branch. The non-empty assertion is the point: without it
    the validation below would pass vacuously on an empty list and would not
    notice a field the schema forbids.
    """

    validator = Draft202012Validator(_json(SCHEMA_DIR / "analysis.schema.json"))
    artifact = analyze_text("Defer retirement of the legacy broker.")
    protected = cast(dict[str, Any], artifact["protected"])
    uncovered = cast(list[dict[str, Any]], cast(dict[str, Any], protected["coverage"])["uncovered"])

    assert uncovered, "text no longer produces an uncovered sentence; pick one that does"
    assert all(entry["content"] for entry in uncovered)

    validator.validate(artifact)


def test_analysis_finding_values_reject_null(recommendation_fixture: dict[str, str]) -> None:
    validator = Draft202012Validator(_json(SCHEMA_DIR / "analysis.schema.json"))
    artifact = analyze_text(recommendation_fixture["original"])
    findings = cast(list[dict[str, Any]], artifact["findings"])
    assert findings
    findings[0]["observed_value"] = None

    with pytest.raises(ValidationError):
        validator.validate(artifact)


def test_profile_rules_reject_null_values() -> None:
    validator = Draft202012Validator(_json(SCHEMA_DIR / "profile.schema.json"))
    profile_path = next(PROFILE_DIR.glob("architecture-review.v*.json"))
    profile = _json(profile_path)
    cast(dict[str, Any], profile["rules"])["nominalization_suffixes"] = None

    with pytest.raises(ValidationError):
        validator.validate(profile)


def test_provider_proposal_schema_validates_representative_payload() -> None:
    validator = Draft202012Validator(_json(SCHEMA_DIR / "provider-proposal.schema.json"))
    validator.validate(
        {
            "candidate_text": "Name the owner and verify mitigation before approval.",
            "addressed_rule_ids": ["LING-SENTENCE-001", "LING-WEAK-VERB-001"],
            "claimed_preservations": ["MODALITY", "QUANTITIES"],
            "provider": "example-provider",
            "model": "example-model",
        }
    )


def test_provider_proposal_schema_rejects_invalid_rule_ids() -> None:
    validator = Draft202012Validator(_json(SCHEMA_DIR / "provider-proposal.schema.json"))

    with pytest.raises(ValidationError):
        validator.validate(
            {
                "candidate_text": "Name the owner and verify mitigation before approval.",
                "addressed_rule_ids": ["sentence-001"],
                "claimed_preservations": ["MODALITY"],
                "provider": "example-provider",
                "model": "example-model",
            }
        )


def test_provider_challenge_schema_validates_representative_payload() -> None:
    validator = Draft202012Validator(_json(SCHEMA_DIR / "provider-challenge.schema.json"))
    validator.validate(
        {
            "disposition": "material_change",
            "claims": ["omitted_claim", "changed_modality"],
            "provider": "example-provider",
            "model": "example-model",
        }
    )


def test_provider_challenge_schema_rejects_untyped_claims() -> None:
    validator = Draft202012Validator(_json(SCHEMA_DIR / "provider-challenge.schema.json"))

    with pytest.raises(ValidationError):
        validator.validate(
            {
                "disposition": "material_change",
                "claims": ["looks different"],
                "provider": "example-provider",
                "model": "example-model",
            }
        )
