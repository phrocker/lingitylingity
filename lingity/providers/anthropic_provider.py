"""Anthropic proposal and drift-challenge providers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, NoReturn, TypeAlias, cast

from lingity.models import JsonValue
from lingity.providers.base import (
    ChallengeResult,
    ProposalRequest,
    ProposalResponse,
    ProviderError,
    ProviderResponseError,
)

ANTHROPIC_API_KEY_ENV: Final = "ANTHROPIC_API_KEY"
DEFAULT_BASE_URL: Final = "https://api.anthropic.com/v1"
DEFAULT_ANTHROPIC_VERSION: Final = "2023-06-01"
DEFAULT_TIMEOUT_SECONDS: Final = 60.0
PROPOSAL_MAX_TOKENS: Final = 4096
CHALLENGE_MAX_TOKENS: Final = 1024
TRUNCATION_STOP_REASONS: Final = frozenset(
    {"max_tokens", "model_context_window_exceeded"}
)


@dataclass(frozen=True)
class AnthropicTransportResponse:
    """HTTP response returned by an injectable Anthropic transport."""

    status: int
    body: bytes


AnthropicTransport: TypeAlias = Callable[
    [str, Mapping[str, str], bytes, float], AnthropicTransportResponse
]


def _require_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderError(
            f"Anthropic provider requires an explicit non-empty {name}; "
            f"received {type(value).__name__}"
        )
    return value


def _require_timeout(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderError(
            "Anthropic provider timeout must be a positive number of seconds; "
            f"received {type(value).__name__}"
        )
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ProviderError(
            "Anthropic provider timeout must be a positive finite number of seconds"
        )
    return timeout


def _urllib_transport(
    url: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> AnthropicTransportResponse:
    request = urllib.request.Request(
        url, data=body, headers=dict(headers), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = int(response.getcode())
            response_body = cast(bytes, response.read())
            return AnthropicTransportResponse(status=status, body=response_body)
    except urllib.error.HTTPError as error:
        return AnthropicTransportResponse(status=error.code, body=b"")
    except urllib.error.URLError as error:
        raise ProviderError(
            "Anthropic Messages API request failed before a response was received"
        ) from error
    except TimeoutError as error:
        raise ProviderError("Anthropic Messages API request timed out") from error


def _to_json_value(value: object, context: str) -> JsonValue:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProviderResponseError(
                f"{context} contains a non-finite number; expected valid JSON"
            )
        return value
    if isinstance(value, list):
        return [_to_json_value(item, context) for item in value]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProviderResponseError(
                    f"{context} contains a non-string object key; expected JSON object keys"
                )
            result[key] = _to_json_value(item, context)
        return result
    raise ProviderResponseError(
        f"{context} contains {type(value).__name__}; expected a JSON-compatible value"
    )


def _parse_json_object(text: str, context: str) -> dict[str, JsonValue]:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProviderResponseError(
            f"{context} was not valid JSON; expected exactly one JSON object"
        ) from error

    value = _to_json_value(raw, context)
    if not isinstance(value, dict):
        raise ProviderResponseError(
            f"{context} was {type(value).__name__}; expected a JSON object"
        )
    return value


def _parse_response_body(body: bytes) -> dict[str, JsonValue]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProviderResponseError(
            "Anthropic Messages API response body was not valid UTF-8"
        ) from error
    return _parse_json_object(text, "Anthropic Messages API response body")


def _extract_text_content(response: Mapping[str, JsonValue]) -> str:
    stop_reason = response.get("stop_reason")
    if not isinstance(stop_reason, str):
        raise ProviderResponseError(
            "Anthropic Messages API response is missing string field 'stop_reason'"
        )
    if stop_reason in TRUNCATION_STOP_REASONS:
        raise ProviderResponseError(
            f"Anthropic Messages API response was truncated with stop_reason "
            f"{stop_reason!r}; expected a complete JSON object"
        )

    content = response.get("content")
    if not isinstance(content, list):
        raise ProviderResponseError(
            "Anthropic Messages API response is missing list field 'content'"
        )

    text_parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            text_parts.append(text)

    if not text_parts:
        raise ProviderResponseError(
            "Anthropic Messages API response contained no usable text content block"
        )
    return "".join(text_parts).strip()


def _json_dump(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _object_list(
    value: JsonValue | None, field_name: str
) -> list[dict[str, JsonValue]]:
    if not isinstance(value, list):
        raise ProviderError(
            f"critique brief is missing list field {field_name!r}; expected "
            "the deterministic critique builder output"
        )
    result: list[dict[str, JsonValue]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ProviderError(
                f"critique brief field {field_name!r} item {index} is "
                f"{type(item).__name__}; expected an object"
            )
        result.append(item)
    return result


def _string_field(
    value: Mapping[str, JsonValue], field_name: str, context: str
) -> str:
    item = value.get(field_name)
    if not isinstance(item, str):
        raise ProviderError(
            f"{context} is missing string field {field_name!r}; expected a "
            "complete critique brief"
        )
    return item


def _proposal_prompt(request: ProposalRequest) -> str:
    source_text = request.source_text
    defects: list[dict[str, JsonValue]] = []
    for index, defect in enumerate(_object_list(request.brief.get("defects"), "defects")):
        context = f"critique brief defect {index}"
        defects.append(
            {
                "rule_id": _string_field(defect, "rule_id", context),
                "severity": _string_field(defect, "severity", context),
                "remediation": _string_field(defect, "remediation", context),
                "excerpt": _string_field(defect, "excerpt", context),
            }
        )

    must_preserve = _object_list(request.brief.get("must_preserve"), "must_preserve")

    return (
        "You are a Lingity proposal provider. You may propose a rewrite, but "
        "your claims are not binding: deterministic Lingity code will verify "
        "readability, protected-element equivalence, and semantic drift.\n\n"
        "Rewrite the source text to address the ranked defects. Preserve every "
        "protected element exactly, including identifiers, quantities, modal "
        "terms, negation, citations, and governance claims. Do not add, omit, "
        "or weaken protected meaning.\n\n"
        "Return only a JSON object matching this contract, with no Markdown and "
        "no surrounding prose: {\n"
        '  "candidate_text": "non-empty rewritten text",\n'
        '  "addressed_rule_ids": ["LING-...-000"],\n'
        '  "claimed_preservations": ["protected element text or tag"],\n'
        '  "provider": "anthropic",\n'
        '  "model": "the exact model name supplied in this request"\n'
        "}\n\n"
        f"Critique SHA-256: {request.critique_sha256}\n\n"
        f"Source text:\n{source_text}\n\n"
        f"Ranked defects:\n{_json_dump(cast(JsonValue, defects))}\n\n"
        f"Must preserve exactly:\n{_json_dump(cast(JsonValue, must_preserve))}"
    )


def _challenge_prompt(source_text: str, candidate_text: str) -> str:
    return (
        "You are a Lingity semantic-drift challenger. You may only raise doubt; "
        "you do not approve rewrites. Compare the source and candidate for "
        "material meaning drift. Return no_material_change only when you are "
        "confident that no material claim changed. Return needs_human only when "
        "you explicitly cannot resolve equivalence.\n\n"
        "Return only a JSON object matching this contract, with no Markdown and "
        "no surrounding prose: {\n"
        '  "disposition": "no_material_change | material_change | needs_human",\n'
        '  "claims": ["omitted_claim | added_claim | changed_modality | '
        'changed_actor_or_ownership | changed_condition_or_scope | '
        'changed_uncertainty | changed_recommendation_or_decision_strength"],\n'
        '  "provider": "anthropic",\n'
        '  "model": "the exact model name supplied in this request"\n'
        "}\n\n"
        "Source text:\n"
        f"{source_text}\n\n"
        "Candidate text:\n"
        f"{candidate_text}"
    )


def _ensure_provider_and_model(
    payload: Mapping[str, JsonValue], expected_provider: str, expected_model: str
) -> None:
    provider = payload.get("provider")
    if isinstance(provider, str) and provider != expected_provider:
        raise ProviderResponseError(
            "provider response identified an unexpected provider; expected "
            f"{expected_provider!r}"
        )
    model = payload.get("model")
    if isinstance(model, str) and model != expected_model:
        raise ProviderResponseError(
            "provider response identified an unexpected model; expected the "
            "explicitly configured Anthropic model"
        )


def _sanitized_response_error(error: ProviderResponseError) -> NoReturn:
    message = str(error)
    api_key = os.environ.get(ANTHROPIC_API_KEY_ENV)
    if api_key:
        message = message.replace(api_key, "<redacted Anthropic API key>")
    raise ProviderResponseError(message) from error


def _challenge_result_from_payload(
    payload: dict[str, JsonValue], provider_name: str
) -> ChallengeResult:
    result = ChallengeResult.from_payload(payload, provider_name)
    if result.disposition == "no_material_change" and result.claims:
        raise ProviderResponseError(
            "Anthropic challenge response is ambiguous: disposition "
            "'no_material_change' cannot include material-change claim tags"
        )
    return result


class _AnthropicMessagesClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout: float,
        anthropic_version: str,
        transport: AnthropicTransport | None,
    ) -> None:
        self.model = _require_non_empty_string(model, "model")
        self._base_url = _require_non_empty_string(base_url, "base_url").rstrip("/")
        self._timeout = _require_timeout(timeout)
        self._anthropic_version = _require_non_empty_string(
            anthropic_version, "anthropic_version"
        )
        if transport is not None and not callable(transport):
            raise ProviderError(
                "Anthropic provider transport option must be callable when provided"
            )
        self._transport = transport or _urllib_transport
        self.last_raw_response_sha256: str | None = None

    def send_prompt(self, prompt: str, max_tokens: int) -> dict[str, JsonValue]:
        api_key = os.environ.get(ANTHROPIC_API_KEY_ENV)
        if not api_key:
            raise ProviderError(
                f"Anthropic provider requires environment variable "
                f"{ANTHROPIC_API_KEY_ENV}; no API key was provided"
            )

        body: dict[str, JsonValue] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }
        encoded_body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "accept": "application/json",
            "x-api-key": api_key,
            "anthropic-version": self._anthropic_version,
        }

        try:
            response = self._transport(
                f"{self._base_url}/messages", headers, encoded_body, self._timeout
            )
        except urllib.error.HTTPError as error:
            response = AnthropicTransportResponse(status=error.code, body=b"")
        except TimeoutError as error:
            raise ProviderError("Anthropic Messages API request timed out") from error
        except OSError as error:
            raise ProviderError(
                "Anthropic Messages API request failed before a response was received"
            ) from error

        if response.status < 200 or response.status >= 300:
            raise ProviderError(
                f"Anthropic Messages API returned HTTP status {response.status}; "
                "expected a 2xx response"
            )

        self.last_raw_response_sha256 = hashlib.sha256(response.body).hexdigest()
        api_payload = _parse_response_body(response.body)
        text = _extract_text_content(api_payload)
        return _parse_json_object(text, "Anthropic text content block")


@dataclass(frozen=True)
class _ResolvedFactoryOptions:
    model: str
    base_url: str
    timeout: float
    anthropic_version: str
    transport: AnthropicTransport | None


class AnthropicProposalProvider:
    """ProposalProvider implementation backed by Anthropic Messages."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        transport: AnthropicTransport | None = None,
    ) -> None:
        self._client = _AnthropicMessagesClient(
            model=model,
            base_url=base_url,
            timeout=timeout,
            anthropic_version=anthropic_version,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def last_raw_response_sha256(self) -> str | None:
        return self._client.last_raw_response_sha256

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        payload = self._client.send_prompt(_proposal_prompt(request), PROPOSAL_MAX_TOKENS)
        _ensure_provider_and_model(payload, self.name, self._client.model)
        try:
            return ProposalResponse.from_payload(payload, self.name)
        except ProviderResponseError as error:
            _sanitized_response_error(error)


class AnthropicDriftChallenger:
    """DriftChallenger implementation backed by Anthropic Messages."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        anthropic_version: str = DEFAULT_ANTHROPIC_VERSION,
        transport: AnthropicTransport | None = None,
    ) -> None:
        self._client = _AnthropicMessagesClient(
            model=model,
            base_url=base_url,
            timeout=timeout,
            anthropic_version=anthropic_version,
            transport=transport,
        )

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def last_raw_response_sha256(self) -> str | None:
        return self._client.last_raw_response_sha256

    def challenge(self, source_text: str, candidate_text: str) -> ChallengeResult:
        payload = self._client.send_prompt(
            _challenge_prompt(source_text, candidate_text), CHALLENGE_MAX_TOKENS
        )
        _ensure_provider_and_model(payload, self.name, self._client.model)
        try:
            return _challenge_result_from_payload(payload, self.name)
        except ProviderResponseError as error:
            _sanitized_response_error(error)


def create_anthropic_proposal_provider(
    **options: object,
) -> AnthropicProposalProvider:
    """Registry factory for the Anthropic proposal provider."""

    resolved = _validated_factory_options(options)
    return AnthropicProposalProvider(
        model=resolved.model,
        base_url=resolved.base_url,
        timeout=resolved.timeout,
        anthropic_version=resolved.anthropic_version,
        transport=resolved.transport,
    )


def create_anthropic_challenger(**options: object) -> AnthropicDriftChallenger:
    """Registry factory for the Anthropic drift challenger."""

    resolved = _validated_factory_options(options)
    return AnthropicDriftChallenger(
        model=resolved.model,
        base_url=resolved.base_url,
        timeout=resolved.timeout,
        anthropic_version=resolved.anthropic_version,
        transport=resolved.transport,
    )


def _validated_factory_options(
    options: Mapping[str, object],
) -> _ResolvedFactoryOptions:
    allowed_options = {
        "model",
        "base_url",
        "timeout",
        "anthropic_version",
        "transport",
    }
    unknown_options = sorted(set(options) - allowed_options)
    if unknown_options:
        raise ProviderError(
            "Anthropic provider factory received unsupported option(s): "
            + ", ".join(unknown_options)
        )
    if "model" not in options:
        raise ProviderError(
            "Anthropic provider factory requires a 'model' option; no default "
            "model is allowed"
        )
    model = _require_non_empty_string(options["model"], "model")
    base_url = DEFAULT_BASE_URL
    anthropic_version = DEFAULT_ANTHROPIC_VERSION
    timeout = DEFAULT_TIMEOUT_SECONDS
    transport: AnthropicTransport | None = None

    if "base_url" in options:
        base_url = _require_non_empty_string(options["base_url"], "base_url")
    if "anthropic_version" in options:
        anthropic_version = _require_non_empty_string(
            options["anthropic_version"], "anthropic_version"
        )
    if "timeout" in options:
        timeout = _require_timeout(options["timeout"])
    if "transport" in options:
        raw_transport = options["transport"]
        if raw_transport is not None and not callable(raw_transport):
            raise ProviderError(
                "Anthropic provider transport option must be callable when provided"
            )
        transport = cast(AnthropicTransport | None, raw_transport)
    return _ResolvedFactoryOptions(
        model=model,
        base_url=base_url,
        timeout=timeout,
        anthropic_version=anthropic_version,
        transport=transport,
    )
