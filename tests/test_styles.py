from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

import lingity.styles as styles
from lingity.profiles import SCHEMA_DIR
from lingity.styles import (
    STYLE_DIR,
    available_style_names,
    load_style,
)

EXPECTED_STYLES = (
    "architecture-review",
    "conservative-web-editor",
    "local-service-guide",
    "technical-writer",
)


def _contract(name: str) -> dict[str, Any]:
    return load_style(name).data


def test_exactly_four_shipped_contracts_validate_and_load() -> None:
    schema = cast(
        dict[str, Any],
        json.loads(
            (SCHEMA_DIR / "style-contract.schema.json").read_text(encoding="utf-8")
        ),
    )
    validator = Draft202012Validator(schema)

    assert tuple(path.name for path in sorted(STYLE_DIR.glob("*.json"))) == (
        "architecture-review.v1.json",
        "conservative-web-editor.v1.json",
        "local-service-guide.v1.json",
        "technical-writer.v1.json",
    )
    assert available_style_names() == EXPECTED_STYLES
    for name in EXPECTED_STYLES:
        contract = load_style(name)
        validator.validate(contract.data)
        assert contract.name == name
        assert len(contract.digest) == 64


def test_listing_and_rendering_are_deterministic_and_distinct() -> None:
    assert available_style_names() == available_style_names()

    expected_leads = {
        "conservative-web-editor": "Lead with immediate reader impact.",
        "local-service-guide": "Lead with the practical requirement.",
        "architecture-review": "Keep the proposed change first.",
        "technical-writer": (
            "Lead with the task or reader outcome and state when the instructions apply."
        ),
    }
    rendered = {}
    for name, distinction in expected_leads.items():
        first = load_style(name).render()
        second = load_style(name).render()
        assert first == second
        assert distinction in first
        assert "not a deterministic style-fit score" in first
        rendered[name] = first
    assert len(set(rendered.values())) == 4


def test_technical_writer_rendering_preserves_task_structure_and_examples() -> None:
    contract = load_style("technical-writer")
    rendered = contract.render()

    assert "Prerequisite-before-action ordering" in rendered
    assert "Exact interface and command preservation" in rendered
    assert "Expected result after the action sequence" in rendered
    assert "Troubleshooting separated from the main path" in rendered
    assert rendered.index("Place prerequisites") < rendered.index(
        "Present ordered actions"
    )
    assert rendered.index("State the expected result") < rendered.index(
        "Put troubleshooting and recovery last"
    )

    positive_examples = cast(
        list[dict[str, str]], contract.data["positive_examples"]
    )
    negative_examples = cast(
        list[dict[str, str]], contract.data["negative_examples"]
    )
    assert len(positive_examples) >= 2
    assert len(negative_examples) >= 2
    assert "`example config set mode strict`" in rendered
    assert "Settings > Connections" in rendered
    assert "clear the cache and try again" in rendered


def test_filename_name_mismatch_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract("architecture-review")
    contract["name"] = "different-name"
    (tmp_path / "architecture-review.v1.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    monkeypatch.setattr(styles, "STYLE_DIR", tmp_path)

    with pytest.raises(ValueError, match="does not match"):
        load_style("architecture-review")


def test_malformed_contract_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract("architecture-review")
    del contract["negative_examples"]
    (tmp_path / "architecture-review.v1.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    monkeypatch.setattr(styles, "STYLE_DIR", tmp_path)

    with pytest.raises(ValueError, match="negative_examples"):
        load_style("architecture-review")


def test_ambiguous_and_unknown_contract_names_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _contract("architecture-review")
    for filename in ("architecture-review.v1.json", "architecture-review.v1.0.json"):
        (tmp_path / filename).write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(styles, "STYLE_DIR", tmp_path)

    with pytest.raises(ValueError, match="ambiguous"):
        load_style("architecture-review")
    with pytest.raises(ValueError, match="Unknown style"):
        load_style("missing")
