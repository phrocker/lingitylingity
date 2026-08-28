from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import pytest

from lingity.models import JsonValue
from lingity.providers.anthropic_provider import (
    ANTHROPIC_API_KEY_ENV,
    AnthropicDriftChallenger,
    AnthropicProposalProvider,
    AnthropicTransportResponse,
    create_anthropic_proposal_provider,
)
from lingity.providers.base import (
    ProposalRequest,
    ProviderError,
    ProviderResponseError,
)

MODEL = "claude-test-model"
SECRET = "sk-ant-test-secret-never-leak"


@dataclass(frozen=True)
class RecordedCall:
    url: str
    headers: dict[str, str]
    body: bytes
    timeout: float


@dataclass
class FakeTransport:
    response: AnthropicTransportResponse
    calls: list[RecordedCall] = field(default_factory=list)

    def __call__(
        self, url: str, headers: Mapping[str, str], body: bytes, timeout: float
    ) -> AnthropicTransportResponse:
        self.calls.append(
            RecordedCall(
                url=url,
                headers=dict(headers),
                body=body,
                timeout=timeout,
            )
        )
        return self.response


def _proposal_request() -> ProposalRequest:
    return ProposalRequest(
        brief={
            "critique_sha256": "0" * 64,
            "source": {
                "text": "The owner should complete 3 checks before approval.",
                "sha256": "1" * 64,
            },
            "defects": [
                {
                    "rule_id": "LING-SENTENCE-001",
                    "severity": "high",
                    "remediation": "Make the sentence more direct.",
                    "excerpt": "should complete 3 checks",
                }
            ],
            "must_preserve": [
                {
                    "category": "quantity",
                    "kind": "number",
                    "text": "3",
                    "normalized": "3",
                },
                {
                    "category": "modality",
                    "kind": "modal",
                    "text": "should",
                    "normalized": "should",
                },
            ],
        }
    )


def _api_response_with_text(
    model_text: dict[str, JsonValue] | str, *, stop_reason: str = "end_turn"
) -> bytes:
    text = model_text if isinstance(model_text, str) else json.dumps(model_text)
    response: dict[str, JsonValue] = {
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": text}],
    }
    return json.dumps(response).encode("utf-8")


def _api_response_without_text() -> bytes:
    response: dict[str, JsonValue] = {
        "stop_reason": "end_turn",
        "content": [{"type": "tool_use", "name": "not_allowed"}],
    }
    return json.dumps(response).encode("utf-8")


def _valid_proposal_payload() -> dict[str, JsonValue]:
    return {
        "candidate_text": "The owner should complete 3 checks before approval.",
        "addressed_rule_ids": ["LING-SENTENCE-001"],
        "claimed_preservations": ["3", "should"],
        "provider": "anthropic",
        "model": MODEL,
    }


def _assert_secret_absent(value: object) -> None:
    assert SECRET not in str(value)


def test_well_formed_response_produces_correct_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, SECRET)
    body = _api_response_with_text(_valid_proposal_payload())
    transport = FakeTransport(AnthropicTransportResponse(status=200, body=body))
    provider = AnthropicProposalProvider(model=MODEL, transport=transport)

    response = provider.propose(_proposal_request())

    assert response.candidate_text == "The owner should complete 3 checks before approval."
    assert response.addressed_rule_ids == ("LING-SENTENCE-001",)
    assert response.claimed_preservations == ("3", "should")
    assert response.provider == "anthropic"
    assert response.model == MODEL
    assert provider.last_raw_response_sha256 == hashlib.sha256(body).hexdigest()
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call.url == "https://api.anthropic.com/v1/messages"
    assert call.headers["x-api-key"] == SECRET
    request_body = cast(dict[str, JsonValue], json.loads(call.body.decode("utf-8")))
    assert request_body["model"] == MODEL
    _assert_secret_absent(call.body.decode("utf-8"))
    _assert_secret_absent(response.to_dict())


def test_missing_api_key_raises_provider_error_naming_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ANTHROPIC_API_KEY_ENV, raising=False)
    transport = FakeTransport(
        AnthropicTransportResponse(
            status=200, body=_api_response_with_text(_valid_proposal_payload())
        )
    )
    provider = AnthropicProposalProvider(model=MODEL, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        provider.propose(_proposal_request())

    assert ANTHROPIC_API_KEY_ENV in str(exc_info.value)
    assert transport.calls == []


def test_constructing_without_explicit_model_fails_closed() -> None:
    with pytest.raises(ProviderError, match="model"):
        AnthropicProposalProvider(model="")
    with pytest.raises(ProviderError, match="model"):
        AnthropicDriftChallenger(model=cast(str, None))
    with pytest.raises(ProviderError, match="model"):
        create_anthropic_proposal_provider()


def test_response_violating_proposal_schema_raises_provider_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, SECRET)
    invalid_payload: dict[str, JsonValue] = {
        "candidate_text": "",
        "addressed_rule_ids": ["not-a-lingity-rule"],
        "claimed_preservations": [],
        "provider": "anthropic",
        "model": MODEL,
    }
    transport = FakeTransport(
        AnthropicTransportResponse(
            status=200, body=_api_response_with_text(invalid_payload)
        )
    )
    provider = AnthropicProposalProvider(model=MODEL, transport=transport)

    with pytest.raises(ProviderResponseError) as exc_info:
        provider.propose(_proposal_request())

    assert "provider-proposal.schema.json" in str(exc_info.value)
    _assert_secret_absent(exc_info.value)


def test_non_json_body_raises_explicit_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, SECRET)
    transport = FakeTransport(AnthropicTransportResponse(status=200, body=b"not json"))
    provider = AnthropicProposalProvider(model=MODEL, transport=transport)

    with pytest.raises(ProviderResponseError, match="not valid JSON") as exc_info:
        provider.propose(_proposal_request())

    _assert_secret_absent(exc_info.value)


def test_http_error_raises_explicit_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, SECRET)
    transport = FakeTransport(
        AnthropicTransportResponse(status=429, body=SECRET.encode("utf-8"))
    )
    provider = AnthropicProposalProvider(model=MODEL, transport=transport)

    with pytest.raises(ProviderError, match="HTTP status 429") as exc_info:
        provider.propose(_proposal_request())

    _assert_secret_absent(exc_info.value)


def test_truncated_stop_reason_raises_explicit_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, SECRET)
    transport = FakeTransport(
        AnthropicTransportResponse(
            status=200,
            body=_api_response_with_text(
                _valid_proposal_payload(), stop_reason="max_tokens"
            ),
        )
    )
    provider = AnthropicProposalProvider(model=MODEL, transport=transport)

    with pytest.raises(ProviderResponseError, match="truncated") as exc_info:
        provider.propose(_proposal_request())

    _assert_secret_absent(exc_info.value)


def test_response_with_no_text_content_block_raises_explicit_response_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, SECRET)
    transport = FakeTransport(
        AnthropicTransportResponse(status=200, body=_api_response_without_text())
    )
    provider = AnthropicProposalProvider(model=MODEL, transport=transport)

    with pytest.raises(ProviderResponseError, match="no usable text") as exc_info:
        provider.propose(_proposal_request())

    _assert_secret_absent(exc_info.value)


def test_challenger_maps_valid_response_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, SECRET)
    challenge_payload: dict[str, JsonValue] = {
        "disposition": "material_change",
        "claims": ["changed_modality", "omitted_claim"],
        "provider": "anthropic",
        "model": MODEL,
    }
    body = _api_response_with_text(challenge_payload)
    transport = FakeTransport(AnthropicTransportResponse(status=200, body=body))
    challenger = AnthropicDriftChallenger(model=MODEL, transport=transport)

    result = challenger.challenge("The owner must approve.", "The owner may approve.")

    assert result.disposition == "material_change"
    assert result.claims == ("changed_modality", "omitted_claim")
    assert result.provider == "anthropic"
    assert result.model == MODEL
    assert challenger.last_raw_response_sha256 == hashlib.sha256(body).hexdigest()
    _assert_secret_absent(result.to_dict())


def test_unparseable_challenge_response_does_not_become_no_material_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ANTHROPIC_API_KEY_ENV, SECRET)
    transport = FakeTransport(
        AnthropicTransportResponse(
            status=200, body=_api_response_with_text("I think it is probably fine.")
        )
    )
    challenger = AnthropicDriftChallenger(model=MODEL, transport=transport)

    with pytest.raises(ProviderResponseError, match="not valid JSON") as exc_info:
        challenger.challenge("The owner must approve.", "The owner may approve.")

    _assert_secret_absent(exc_info.value)

