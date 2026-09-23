from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from lingity.cli import main
import lingity.profiles as profiles
import lingity.styles as styles


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


def test_styles_and_style_cli_are_deterministic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["styles"]) == 0
    names = json.loads(capsys.readouterr().out)
    assert names == [
        "architecture-review",
        "conservative-web-editor",
        "local-service-guide",
        "technical-writer",
    ]

    assert main(["style", "technical-writer", "--format", "json"]) == 0
    contract = cast(dict[str, Any], json.loads(capsys.readouterr().out))
    assert contract["name"] == "technical-writer"
    assert len(cast(list[object], contract["positive_examples"])) >= 2
    assert len(cast(list[object], contract["negative_examples"])) >= 2

    assert main(["style", "technical-writer", "--format", "prompt"]) == 0
    prompt = capsys.readouterr().out
    assert "Lead with the task or reader outcome" in prompt
    assert "one primary action per step" in prompt
    assert "not a deterministic style-fit score" in prompt


def test_style_cli_rejects_malformed_contract_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = styles.load_style("architecture-review")
    malformed = dict(source.data)
    del malformed["negative_examples"]
    (tmp_path / "architecture-review.v1.json").write_text(
        json.dumps(malformed), encoding="utf-8"
    )
    monkeypatch.setattr(styles, "STYLE_DIR", tmp_path)

    assert main(["style", "architecture-review"]) == 2
    captured = capsys.readouterr()
    assert "negative_examples" in captured.err
    assert "Traceback" not in captured.err


def test_critique_cli_accepts_optional_style(
    tmp_path: Path,
    recommendation_fixture: dict[str, str],
) -> None:
    source = tmp_path / "source.txt"
    plain_path = tmp_path / "plain.json"
    styled_path = tmp_path / "styled.json"
    source.write_text(recommendation_fixture["original"], encoding="utf-8")

    assert main(["critique", str(source), "--output", str(plain_path)]) == 0
    assert (
        main(
            [
                "critique",
                str(source),
                "--style",
                "architecture-review",
                "--output",
                str(styled_path),
            ]
        )
        == 0
    )

    plain = cast(dict[str, Any], json.loads(plain_path.read_text(encoding="utf-8")))
    styled = cast(
        dict[str, Any], json.loads(styled_path.read_text(encoding="utf-8"))
    )
    assert "style" not in plain
    assert styled["style"]["reference"]["name"] == "architecture-review"
    assert plain["critique_sha256"] != styled["critique_sha256"]
