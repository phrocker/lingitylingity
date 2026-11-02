"""Evidence review must never turn missing or invented citations into approval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator

from lingity.cli import main
from lingity.factcheck import EvidenceBundle, EvidenceError, ModelFactChecker
from lingity.improve import improve_text, judge_candidate
from lingity.profiles import SCHEMA_DIR, load_profile
from lingity.providers.agent import SubagentProvider
from lingity.providers.base import ProviderResponseError
from lingity.text import sentences


def _bundle() -> EvidenceBundle:
    return EvidenceBundle.from_payload({"sources": [{
        "id": "source-1",
        "url": "https://example.org/report",
        "retrieved_at": "2026-11-02T12:00:00Z",
        "content": "The cat sat on the mat and then ran home.",
    }]})


def _response(text: str, status: str = "supported", quote: str = "The cat sat on the mat") -> dict[str, Any]:
    return {
        "provider": "stub", "model": "stub-1",
        "sentences": [{
            "index": unit.index, "text": unit.text,
            "claims": ([{
                "text": unit.text, "status": status,
                "citations": ([{"source_id": "source-1", "quote": quote}]
                              if status in ("supported", "contradicted") else []),
                "reason": "checked against the saved report",
            }] if status != "not_factual" else []),
            "reason": "checked against the saved report",
        } for unit in sentences(text)],
    }


def test_bundle_requires_citable_snapshots() -> None:
    with pytest.raises(EvidenceError, match="at least one"):
        EvidenceBundle.from_payload({"sources": []})
    with pytest.raises(EvidenceError, match="duplicate"):
        EvidenceBundle.from_payload({"sources": [
            {"id": "a", "url": "https://example.org/a", "retrieved_at": "2026-11-02T12:00:00Z", "content": "A"},
            {"id": "a", "url": "https://example.org/b", "retrieved_at": "2026-11-02T12:00:00Z", "content": "B"},
        ]})


def test_checker_requires_complete_review_and_real_quotes() -> None:
    bundle = _bundle()
    text = "The cat sat. Then the cat ran home."
    good = _response(text)
    checked = ModelFactChecker("stub", "stub-1", lambda _: good).check(text, bundle)
    assert checked["passed"] is True
    assert checked["evidence_sha256"] == bundle.digest

    missing = _response(text)
    missing["sentences"].pop()
    with pytest.raises(ProviderResponseError, match="every sentence"):
        ModelFactChecker("stub", "stub-1", lambda _: missing).check(text, bundle)

    invented = _response(text, quote="The cat won an election")
    with pytest.raises(ProviderResponseError, match="absent"):
        ModelFactChecker("stub", "stub-1", lambda _: invented).check(text, bundle)

    wrong_text = _response(text)
    wrong_text["sentences"][0]["text"] = "A different assertion."
    with pytest.raises(ProviderResponseError, match="does not match"):
        ModelFactChecker("stub", "stub-1", lambda _: wrong_text).check(text, bundle)


def test_source_and_candidate_both_need_support() -> None:
    source = "The cat sat on the mat, and then the cat ran home."
    candidate = "The cat sat on the mat. Then the cat ran home."
    profile = load_profile()
    bundle = _bundle()

    # The model prompt includes the evidence as well as the sentence list; use
    # explicit call order for these tests rather than parsing prompt content.
    def ordered_checker(statuses: tuple[str, str]) -> ModelFactChecker:
        calls = iter(((source, statuses[0]), (candidate, statuses[1])))
        return ModelFactChecker(
            "stub", "stub-1",
            lambda _: _response(*next(calls)),
        )

    accepted, reasons, evidence = judge_candidate(
        source, candidate, profile, target_grade=0,
        evidence_bundle=bundle, fact_checker=ordered_checker(("supported", "supported")),
    )
    assert accepted and not reasons
    assert cast(dict[str, Any], evidence["fact_check"])["passed"] is True

    rejected, reasons, _ = judge_candidate(
        source, candidate, profile, target_grade=0,
        evidence_bundle=bundle, fact_checker=ordered_checker(("contradicted", "supported")),
    )
    assert not rejected and any("source has contradicted" in reason for reason in reasons)

    rejected, reasons, _ = judge_candidate(
        source, candidate, profile, target_grade=0,
        evidence_bundle=bundle, fact_checker=ordered_checker(("supported", "unresolved")),
    )
    assert not rejected and any("candidate has contradicted or unresolved" in reason for reason in reasons)


def test_cli_fact_check_blocks_unresolved_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "The cat sat on the mat, and then the cat ran home."
    candidate = "The cat sat on the mat. Then the cat ran home."
    source_path = tmp_path / "source.md"
    candidate_path = tmp_path / "candidate.md"
    evidence_path = tmp_path / "evidence.json"
    verdict_path = tmp_path / "verdict.json"
    source_path.write_text(source, encoding="utf-8")
    candidate_path.write_text(candidate, encoding="utf-8")
    evidence_path.write_text(json.dumps({"sources": [{
        "id": "source-1", "url": "https://example.org/report",
        "retrieved_at": "2026-11-02T12:00:00Z",
        "content": "The cat sat on the mat and then ran home.",
    }]}), encoding="utf-8")
    calls = iter(((source, "supported"), (candidate, "unresolved")))
    monkeypatch.setattr(
        "lingity.cli.create_fact_checker",
        lambda provider, model: ModelFactChecker(
            provider, model, lambda _: _response_for_model(*next(calls), provider, model)
        ),
    )
    assert main([
        "judge", str(source_path), "--candidate", str(candidate_path),
        "--target-grade", "0", "--evidence", str(evidence_path),
        "--fact-check-provider", "openai", "--fact-check-model", "test-model",
        "--output", str(verdict_path),
    ]) == 1
    verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    schema = json.loads((SCHEMA_DIR / "verdict.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(verdict)
    assert verdict["fact_check"]["passed"] is False
    assert verdict["fact_check"]["candidate"]["sentences"][0]["status"] == "unresolved"


def test_improve_retains_source_when_fact_check_fails(tmp_path: Path) -> None:
    source = "The cat sat on the mat, and then the cat ran home."
    candidate = "The cat sat on the mat. Then the cat ran home."
    candidate_path = tmp_path / "candidate.md"
    candidate_path.write_text(candidate, encoding="utf-8")
    calls = iter(((source, "supported"), (candidate, "unresolved")))
    checker = ModelFactChecker(
        "stub", "stub-1", lambda _: _response(*next(calls))
    )
    result = improve_text(
        source, load_profile(), SubagentProvider([candidate_path]),
        max_attempts=1, target_grade=0,
        evidence_bundle=_bundle(), fact_checker=checker,
    )
    assert not result.accepted
    assert result.selected_text == source
    assert result.attempts[0].fact_check is not None
    attempts = cast(list[dict[str, Any]], result.to_dict()["attempts"])
    assert attempts[0]["fact_check"]["passed"] is False


def _response_for_model(text: str, status: str, provider: str, model: str) -> dict[str, Any]:
    result = _response(text, status)
    result["provider"] = provider
    result["model"] = model
    return result
