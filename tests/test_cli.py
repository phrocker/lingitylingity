from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from lingity.cli import main
import lingity.profiles as profiles


def test_analyze_and_verify_cli(
    tmp_path: Path,
    recommendation_fixture: dict[str, str],
) -> None:
    source = tmp_path / "source.txt"
    artifact = tmp_path / "analysis.json"
    verification = tmp_path / "verification.json"
    source.write_text(recommendation_fixture["original"], encoding="utf-8")
    assert main(["analyze", str(source), "--output", str(artifact)]) == 0
    assert main(["verify", str(artifact), "--output", str(verification)]) == 0
    result = cast(dict[str, Any], json.loads(verification.read_text(encoding="utf-8")))
    assert result["valid"] is True


def test_verify_rejects_tampering(
    tmp_path: Path,
    recommendation_fixture: dict[str, str],
) -> None:
    source = tmp_path / "source.txt"
    artifact = tmp_path / "analysis.json"
    source.write_text(recommendation_fixture["original"], encoding="utf-8")
    assert main(["analyze", str(source), "--output", str(artifact)]) == 0
    value = cast(dict[str, Any], json.loads(artifact.read_text(encoding="utf-8")))
    value["score"]["value"] = 100
    artifact.write_text(json.dumps(value), encoding="utf-8")
    assert main(["verify", str(artifact)]) == 2


def test_failed_verify_output_removes_stale_success(
    tmp_path: Path,
    recommendation_fixture: dict[str, str],
) -> None:
    source = tmp_path / "source.txt"
    artifact = tmp_path / "analysis.json"
    verification = tmp_path / "verification.json"
    source.write_text(recommendation_fixture["original"], encoding="utf-8")
    assert main(["analyze", str(source), "--output", str(artifact)]) == 0
    assert main(["verify", str(artifact), "--output", str(verification)]) == 0
    assert cast(dict[str, Any], json.loads(verification.read_text(encoding="utf-8")))["valid"] is True

    value = cast(dict[str, Any], json.loads(artifact.read_text(encoding="utf-8")))
    value["score"]["value"] = 0 if value["score"]["value"] != 0 else 100
    artifact.write_text(json.dumps(value), encoding="utf-8")

    assert main(["verify", str(artifact), "--output", str(verification)]) == 2
    assert not verification.exists()


def test_verify_rejects_input_output_alias(
    tmp_path: Path,
    recommendation_fixture: dict[str, str],
) -> None:
    source = tmp_path / "source.txt"
    artifact = tmp_path / "analysis.json"
    source.write_text(recommendation_fixture["original"], encoding="utf-8")
    assert main(["analyze", str(source), "--output", str(artifact)]) == 0

    assert main(["verify", str(artifact), "--output", str(artifact)]) == 2
    assert artifact.exists()


@pytest.mark.parametrize("profile_name", ["*", "", "../architecture-review", "architecture-review*", "missing"])
def test_analyze_rejects_unknown_or_injected_profile_names(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    profile_name: str,
) -> None:
    source = tmp_path / "source.txt"
    artifact = tmp_path / "analysis.json"
    source.write_text("Architects should verify the decision before approval.", encoding="utf-8")

    assert main(["analyze", str(source), "--profile", profile_name, "--output", str(artifact)]) == 2
    captured = capsys.readouterr()
    assert "Valid profiles:" in captured.err
    assert "architecture-review" in captured.err
    assert not artifact.exists()


def test_analyze_rejects_malformed_profile_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("Architects should verify the decision before approval.", encoding="utf-8")
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    source_profile_path = next(profiles.PROFILE_DIR.glob("architecture-review.v*.json"))
    bad_profile = cast(dict[str, Any], json.loads(source_profile_path.read_text(encoding="utf-8")))
    bad_profile["name"] = "broken"
    cast(dict[str, Any], bad_profile["rules"])["nominalization_suffixes"] = None
    (profile_dir / "broken.v1.0.0.json").write_text(json.dumps(bad_profile), encoding="utf-8")
    monkeypatch.setattr(profiles, "PROFILE_DIR", profile_dir)

    assert main(["analyze", str(source), "--profile", "broken"]) == 2
    captured = capsys.readouterr()
    assert "Profile broken.v1.0.0.json is invalid" in captured.err
    assert "nominalization_suffixes" in captured.err
    assert "Traceback" not in captured.err
