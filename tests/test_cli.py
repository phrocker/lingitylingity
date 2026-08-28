from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from lingity.cli import main


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
