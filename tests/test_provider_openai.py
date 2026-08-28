from __future__ import annotations

import email.message
import json
import urllib.error
import urllib.request
from typing import cast

import pytest

from lingity.models import JsonValue
from lingity.providers.base import (
    ChallengeResult,
    ProposalRequest,
    ProposalResponse,
    ProviderError,
    ProviderResponseError,
)
from lingity.providers.openai_provider import (
    OpenAIDriftChallenger,
    OpenAIProposalProvider,
)

SECRET = "sk-test-secret-value"
MODEL = "gpt-test"


class FakeTransport:
    def __init__(self, body: bytes | None = None, error: Exception | None = None) -> None:
        self.body = body
        self.error = error
        self.requests: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> bytes:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        if self.body is None:
            raise AssertionError("fake transport was called without a body or error")
        return self.body


def _chat_response(payload: object, *, finish_reason: str = "stop") -> bytes:
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return json.dumps(
        {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {"content": content},
                }
            ]
        }
    ).encode("utf-8")


def _proposal_payload() -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], {
        "candidate_text": "The release manager must verify 3 controls before approval.",
        "addressed_rule_ids": ["LING-SENTENCE-001"],
        "claimed_preservations": ["release manager", "must", "3 controls"],
        "provider": "openai",
        "model": MODEL,
    })


def _challenge_payload() -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], {
        "disposition": "material_change",
        "claims": ["omitted_claim"],
        "provider": "openai",
        "model": MODEL,
    })


def _brief() -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], {
        "critique_sha256": "a" * 64,
        "source": {
            "text": "The release manager must verify 3 controls before approval.",
            "sha256": "b" * 64,
        },
        "defects": [
            {
                "rule_id": "LING-SENTENCE-001",
                "severity": "high",
                "remediation": "Shorten the sentence.",
                "excerpt": "The release manager must verify 3 controls before approval.",
            }
        ],
        "must_preserve": [
            {
                "category": "governance",
                "kind": "modal",
                "text": "must",
                "normalized": "must",
            },
            {
                "category": "quantity",
                "kind": "number",
                "text": "3 controls",
                "normalized": "3 controls",
            },
        ],
    })


def _provider_response(transport: FakeTransport) -> ProposalResponse:
    provider = OpenAIProposalProvider(model=MODEL, transport=transport)
    return provider.propose(ProposalRequest(brief=_brief()))


def _assert_secret_absent(value: object) -> None:
    assert SECRET not in str(value)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)


def test_well_formed_response_produces_proposal_response() -> None:
    transport = FakeTransport(_chat_response(_proposal_payload()))

    result = _provider_response(transport)

    assert result == ProposalResponse(
        candidate_text="The release manager must verify 3 controls before approval.",
        addressed_rule_ids=("LING-SENTENCE-001",),
        claimed_preservations=("release manager", "must", "3 controls"),
        provider="openai",
        model=MODEL,
    )
    assert transport.requests
    request_data = cast(bytes, transport.requests[0].data)
    request_body = json.loads(request_data.decode("utf-8"))
    assert request_body["model"] == MODEL
    _assert_secret_absent(result.to_dict())


def test_missing_openai_api_key_raises_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    transport = FakeTransport(_chat_response(_proposal_payload()))
    provider = OpenAIProposalProvider(model=MODEL, transport=transport)

    with pytest.raises(ProviderError) as exc_info:
        provider.propose(ProposalRequest(brief=_brief()))

    assert "OPENAI_API_KEY" in str(exc_info.value)
    _assert_secret_absent(exc_info.value)
    assert not transport.requests


@pytest.mark.parametrize("model", [None, ""])
def test_constructing_without_explicit_model_fails_closed(model: str | None) -> None:
    with pytest.raises(ProviderError) as exc_info:
        OpenAIProposalProvider(model=model)

    assert "explicit" in str(exc_info.value)
    assert "model" in str(exc_info.value)
    _assert_secret_absent(exc_info.value)


def test_response_violating_proposal_schema_raises_provider_response_error() -> None:
    payload = _proposal_payload()
    del payload["candidate_text"]
    transport = FakeTransport(_chat_response(payload))

    with pytest.raises(ProviderResponseError) as exc_info:
        _provider_response(transport)

    assert "provider-proposal.schema.json" in str(exc_info.value)
    assert "candidate_text" in str(exc_info.value)
    _assert_secret_absent(exc_info.value)


def test_non_json_body_raises_explicit_provider_response_error() -> None:
    transport = FakeTransport(b"not json")

    with pytest.raises(ProviderResponseError) as exc_info:
        _provider_response(transport)

    assert "response body was not valid JSON" in str(exc_info.value)
    _assert_secret_absent(exc_info.value)


def test_http_error_raises_explicit_provider_error() -> None:
    error = urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=429,
        msg="Too Many Requests",
        hdrs=email.message.Message(),
        fp=None,
    )
    transport = FakeTransport(error=error)

    with pytest.raises(ProviderError) as exc_info:
        _provider_response(transport)

    assert "HTTP error 429" in str(exc_info.value)
    assert "proposal" in str(exc_info.value)
    _assert_secret_absent(exc_info.value)


def test_json_body_that_is_not_object_raises_explicit_error() -> None:
    transport = FakeTransport(b"[1, 2, 3]")

    with pytest.raises(ProviderResponseError) as exc_info:
        _provider_response(transport)

    assert "response body was JSON but not an object" in str(exc_info.value)
    _assert_secret_absent(exc_info.value)


def test_challenger_maps_valid_response_correctly() -> None:
    transport = FakeTransport(_chat_response(_challenge_payload()))
    challenger = OpenAIDriftChallenger(model=MODEL, transport=transport)

    result = challenger.challenge("The owner must approve it.", "The owner may approve it.")

    assert result == ChallengeResult(
        disposition="material_change",
        claims=("omitted_claim",),
        provider="openai",
        model=MODEL,
    )
    _assert_secret_absent(result.to_dict())


def test_unparseable_challenge_response_does_not_become_no_material_change() -> None:
    transport = FakeTransport(_chat_response("not json"))
    challenger = OpenAIDriftChallenger(model=MODEL, transport=transport)

    with pytest.raises(ProviderResponseError) as exc_info:
        challenger.challenge("The owner must approve it.", "The owner may approve it.")

    assert "challenge content was not valid JSON" in str(exc_info.value)
    assert "no_material_change" not in str(exc_info.value)
    _assert_secret_absent(exc_info.value)
