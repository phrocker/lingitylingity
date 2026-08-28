"""OpenAI-backed Lingity proposal and drift-challenge providers."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, Final, cast

from lingity.models import JsonValue
from lingity.providers.base import (
    ChallengeResult,
    ProposalRequest,
    ProposalResponse,
    ProviderError,
    ProviderResponseError,
)

OPENAI_API_KEY_ENV: Final = "OPENAI_API_KEY"
DEFAULT_BASE_URL: Final = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS: Final = 30.0

OpenAITransport = Callable[[urllib.request.Request, float], bytes]


def _urlopen_transport(request: urllib.request.Request, timeout: float) -> bytes:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return cast(bytes, response.read())


def _require_model(model: str | None) -> str:
    if model is None or not isinstance(model, str) or not model.strip():
        raise ProviderError(
            "OpenAI provider requires an explicit non-empty model; no default "
            "model is available"
        )
    return model


def _require_base_url(base_url: str | None) -> str:
    if base_url is None or not isinstance(base_url, str) or not base_url.strip():
        raise ProviderError("OpenAI provider requires a non-empty base_url")
    return base_url.rstrip("/")


def _require_timeout(timeout: float | int | None) -> float:
    if timeout is None:
        return DEFAULT_TIMEOUT_SECONDS
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)):
        raise ProviderError(
            "OpenAI provider timeout must be a number of seconds greater than zero"
        )
    value = float(timeout)
    if not math.isfinite(value) or value <= 0:
        raise ProviderError(
            f"OpenAI provider timeout must be greater than zero seconds; received {value}"
        )
    return value


def _api_key() -> str:
    key = os.environ.get(OPENAI_API_KEY_ENV)
    if key is None or not key.strip():
        raise ProviderError(
            f"OpenAI provider requires the {OPENAI_API_KEY_ENV} environment variable"
        )
    return key


def _json_object_from_text(text: str, context: str) -> dict[str, JsonValue]:
    try:
        decoded: object = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProviderResponseError(
            f"{context} was not valid JSON: {error.msg} at line {error.lineno} "
            f"column {error.colno}"
        ) from error
    if not isinstance(decoded, dict):
        raise ProviderResponseError(
            f"{context} was JSON but not an object; expected a JSON object"
        )
    return cast(dict[str, JsonValue], decoded)


def _require_mapping(value: JsonValue | None, field: str, context: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict):
        raise ProviderError(f"{context} is missing its {field!r} object")
    return value


def _require_list(value: JsonValue | None, field: str, context: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ProviderError(f"{context} is missing its {field!r} list")
    return value


def _require_str(value: JsonValue | None, field: str, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProviderError(f"{context} is missing its non-empty {field!r} string")
    return value


def _proposal_defects(brief: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    defects = _require_list(brief.get("defects"), "defects", "critique brief")
    selected: list[dict[str, JsonValue]] = []
    for index, item in enumerate(defects):
        defect = _require_mapping(item, f"defects[{index}]", "critique brief")
        selected.append(
            {
                "rule_id": _require_str(
                    defect.get("rule_id"), "rule_id", f"critique brief defects[{index}]"
                ),
                "severity": _require_str(
                    defect.get("severity"), "severity", f"critique brief defects[{index}]"
                ),
                "remediation": _require_str(
                    defect.get("remediation"),
                    "remediation",
                    f"critique brief defects[{index}]",
                ),
                "excerpt": _require_str(
                    defect.get("excerpt"), "excerpt", f"critique brief defects[{index}]"
                ),
            }
        )
    return selected


def _must_preserve(brief: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    values = _require_list(brief.get("must_preserve"), "must_preserve", "critique brief")
    selected: list[dict[str, JsonValue]] = []
    for index, item in enumerate(values):
        protected = _require_mapping(
            item, f"must_preserve[{index}]", "critique brief"
        )
        selected.append(
            {
                "category": _require_str(
                    protected.get("category"),
                    "category",
                    f"critique brief must_preserve[{index}]",
                ),
                "kind": _require_str(
                    protected.get("kind"), "kind", f"critique brief must_preserve[{index}]"
                ),
                "text": _require_str(
                    protected.get("text"), "text", f"critique brief must_preserve[{index}]"
                ),
                "normalized": _require_str(
                    protected.get("normalized"),
                    "normalized",
                    f"critique brief must_preserve[{index}]",
                ),
            }
        )
    return selected


def _serialize(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


class _OpenAIClient:
    def __init__(
        self,
        *,
        model: str | None,
        base_url: str | None,
        timeout: float | int | None,
        transport: OpenAITransport | None,
    ) -> None:
        self._model = _require_model(model)
        self._base_url = _require_base_url(base_url)
        self._timeout = _require_timeout(timeout)
        self._transport = transport or _urlopen_transport

    @property
    def model(self) -> str:
        return self._model

    def complete_json(self, prompt: str, purpose: str) -> dict[str, JsonValue]:
        key = _api_key()
        body: dict[str, JsonValue] = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a JSON-only transport for Lingity. Return only "
                        "one JSON object and do not include prose or Markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            raw = self._transport(request, self._timeout)
        except urllib.error.HTTPError as error:
            raise ProviderError(
                f"OpenAI API HTTP error {error.code}: expected successful "
                f"chat completion response for {purpose}; received {error.reason}"
            ) from error
        except urllib.error.URLError as error:
            raise ProviderError(
                f"OpenAI API request failed for {purpose}: expected chat "
                f"completion response; received {error.reason}"
            ) from error

        if not isinstance(raw, bytes):
            raise ProviderResponseError(
                f"OpenAI API transport returned {type(raw).__name__}; expected bytes"
            )
        try:
            raw_text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProviderResponseError(
                "OpenAI API response body was not valid UTF-8; expected a JSON object"
            ) from error

        envelope = _json_object_from_text(raw_text, "OpenAI API response body")
        content = self._extract_content(envelope, purpose)
        return _json_object_from_text(content, f"OpenAI {purpose} content")

    def _extract_content(self, envelope: Mapping[str, JsonValue], purpose: str) -> str:
        choices = envelope.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderResponseError(
                f"OpenAI API response for {purpose} is missing a non-empty choices list"
            )
        first = choices[0]
        if not isinstance(first, dict):
            raise ProviderResponseError(
                f"OpenAI API response for {purpose} has a non-object first choice"
            )
        finish_reason = first.get("finish_reason")
        if finish_reason != "stop":
            raise ProviderResponseError(
                f"OpenAI API response for {purpose} did not complete cleanly: "
                f"expected finish_reason 'stop', received {finish_reason!r}"
            )
        message = first.get("message")
        if not isinstance(message, dict):
            raise ProviderResponseError(
                f"OpenAI API response for {purpose} is missing a message object"
            )
        refusal = message.get("refusal")
        if refusal is not None and (not isinstance(refusal, str) or refusal.strip()):
            raise ProviderResponseError(
                f"OpenAI API response for {purpose} was refused; expected JSON content"
            )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderResponseError(
                f"OpenAI API response for {purpose} is missing non-empty JSON content"
            )
        return content


class OpenAIProposalProvider:
    """OpenAI proposal provider for Lingity."""

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = DEFAULT_BASE_URL,
        timeout: float | int | None = DEFAULT_TIMEOUT_SECONDS,
        transport: OpenAITransport | None = None,
    ) -> None:
        self._client = _OpenAIClient(
            model=model, base_url=base_url, timeout=timeout, transport=transport
        )

    @property
    def name(self) -> str:
        return "openai"

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        payload = self._client.complete_json(
            self._proposal_prompt(request), "proposal"
        )
        self._validate_identity(payload, "proposal")
        return ProposalResponse.from_payload(payload, self.name)

    def _proposal_prompt(self, request: ProposalRequest) -> str:
        source = _require_mapping(request.brief.get("source"), "source", "critique brief")
        source_text = _require_str(source.get("text"), "text", "critique brief source")
        critique_sha256 = request.critique_sha256
        defects = _proposal_defects(request.brief)
        preserve = _must_preserve(request.brief)
        return (
            "Lingity needs a candidate rewrite. The deterministic Lingity loop, "
            "not you, will decide whether to accept it.\n\n"
            "Preserve every protected element exactly as written. Do not change "
            "identifiers, quantities, modal terms, negation, citations, or "
            "governance claims.\n\n"
            "Improve only the ranked clarity defects listed below. Your claims "
            "are non-binding and will be verified deterministically.\n\n"
            "Return only one JSON object matching this schema shape exactly:\n"
            "{\n"
            '  "candidate_text": "non-empty rewritten text",\n'
            '  "addressed_rule_ids": ["LING-SENTENCE-001"],\n'
            '  "claimed_preservations": ["the exact protected text you preserved"],\n'
            f'  "provider": "{self.name}",\n'
            f'  "model": "{self._client.model}"\n'
            "}\n\n"
            f"Source text:\n{source_text}\n\n"
            f"Critique SHA-256:\n{critique_sha256}\n\n"
            f"Ranked defects:\n{_serialize(cast(JsonValue, defects))}\n\n"
            f"Protected elements that must be preserved exactly:\n"
            f"{_serialize(cast(JsonValue, preserve))}"
        )

    def _validate_identity(self, payload: Mapping[str, JsonValue], purpose: str) -> None:
        provider = payload.get("provider")
        model = payload.get("model")
        if provider != self.name:
            raise ProviderResponseError(
                f"OpenAI {purpose} response identified provider {provider!r}; "
                f"expected {self.name!r}"
            )
        if model != self._client.model:
            raise ProviderResponseError(
                f"OpenAI {purpose} response identified model {model!r}; "
                f"expected {self._client.model!r}"
            )


class OpenAIDriftChallenger:
    """OpenAI challenger that can only raise semantic-drift doubt."""

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = DEFAULT_BASE_URL,
        timeout: float | int | None = DEFAULT_TIMEOUT_SECONDS,
        transport: OpenAITransport | None = None,
    ) -> None:
        self._client = _OpenAIClient(
            model=model, base_url=base_url, timeout=timeout, transport=transport
        )

    @property
    def name(self) -> str:
        return "openai"

    def challenge(self, source_text: str, candidate_text: str) -> ChallengeResult:
        payload = self._client.complete_json(
            self._challenge_prompt(source_text, candidate_text), "challenge"
        )
        self._validate_identity(payload, "challenge")
        return ChallengeResult.from_payload(payload, self.name)

    def _challenge_prompt(self, source_text: str, candidate_text: str) -> str:
        if not source_text:
            raise ProviderError("OpenAI challenge requires non-empty source_text")
        if not candidate_text:
            raise ProviderError("OpenAI challenge requires non-empty candidate_text")
        return (
            "Compare the source text and candidate text for material meaning drift. "
            "You may only raise doubt; you do not approve the rewrite, and Lingity "
            "will verify any result deterministically.\n\n"
            "Return disposition 'material_change' when you identify a material "
            "difference. Return 'needs_human' only when you explicitly cannot "
            "resolve equivalence. Return 'no_material_change' only when the "
            "candidate preserves every claim, actor, condition, scope, uncertainty, "
            "recommendation strength, and modality.\n\n"
            "Use only these claim tags: omitted_claim, added_claim, changed_modality, "
            "changed_actor_or_ownership, changed_condition_or_scope, "
            "changed_uncertainty, changed_recommendation_or_decision_strength.\n\n"
            "Return only one JSON object matching this schema shape exactly:\n"
            "{\n"
            '  "disposition": "material_change",\n'
            '  "claims": ["omitted_claim"],\n'
            f'  "provider": "{self.name}",\n'
            f'  "model": "{self._client.model}"\n'
            "}\n\n"
            f"Source text:\n{source_text}\n\n"
            f"Candidate text:\n{candidate_text}"
        )

    def _validate_identity(self, payload: Mapping[str, JsonValue], purpose: str) -> None:
        provider = payload.get("provider")
        model = payload.get("model")
        if provider != self.name:
            raise ProviderResponseError(
                f"OpenAI {purpose} response identified provider {provider!r}; "
                f"expected {self.name!r}"
            )
        if model != self._client.model:
            raise ProviderResponseError(
                f"OpenAI {purpose} response identified model {model!r}; "
                f"expected {self._client.model!r}"
            )


def create_openai_proposal_provider(**options: Any) -> OpenAIProposalProvider:
    return OpenAIProposalProvider(**options)


def create_openai_challenger(**options: Any) -> OpenAIDriftChallenger:
    return OpenAIDriftChallenger(**options)
