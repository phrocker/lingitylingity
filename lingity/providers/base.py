"""Provider protocol, response validation, and registry.

A provider is a *transport*. It carries a critique brief out to some rewriting
or reviewing capability and carries a response back. It never decides whether a
candidate is acceptable — that authority belongs to the deterministic loop,
which re-runs the analyzer and the meaning gate on whatever comes back.

Every response is validated against the published provider schemas before it
reaches the loop. A provider that answers with a malformed payload fails loudly
rather than degrading into a success-shaped default.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaValidationError

from lingity.models import JsonValue

SCHEMA_DIR: Final = Path(__file__).resolve().parents[1] / "schemas" / "v1"

CHALLENGE_DISPOSITIONS: Final = frozenset(
    {"no_material_change", "material_change", "needs_human"}
)


class ProviderError(RuntimeError):
    """Raised when a provider cannot be resolved, configured, or trusted."""


class ProviderResponseError(ProviderError):
    """Raised when a provider returns a payload that violates its schema."""


class ProviderExhausted(ProviderError):
    """Raised when a transport has no further candidate to offer.

    This is not a failure of the run. A subagent transport serves candidates the
    host agent has already written, so running out simply ends the loop early.
    The loop treats it as a stop condition; every other provider error still
    propagates, because those mean something actually went wrong.
    """


@dataclass(frozen=True)
class ProposalRequest:
    """Everything a provider is allowed to see when drafting a candidate."""

    brief: Mapping[str, JsonValue]

    @property
    def source_text(self) -> str:
        source = self.brief.get("source")
        if not isinstance(source, dict):
            raise ProviderError("critique brief is missing its source block")
        text = source.get("text")
        if not isinstance(text, str) or not text:
            raise ProviderError("critique brief carries no source text")
        return text

    @property
    def critique_sha256(self) -> str:
        digest = self.brief.get("critique_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ProviderError("critique brief is missing a valid critique_sha256")
        return digest


@dataclass(frozen=True)
class ProposalResponse:
    """A candidate rewrite plus the provider's own (non-binding) claims."""

    candidate_text: str
    addressed_rule_ids: tuple[str, ...]
    claimed_preservations: tuple[str, ...]
    provider: str
    model: str

    @staticmethod
    def from_payload(
        payload: dict[str, JsonValue], provider_name: str
    ) -> ProposalResponse:
        validate_proposal(payload, provider_name)
        return ProposalResponse(
            candidate_text=cast(str, payload["candidate_text"]),
            addressed_rule_ids=tuple(cast(list[str], payload["addressed_rule_ids"])),
            claimed_preservations=tuple(
                cast(list[str], payload["claimed_preservations"])
            ),
            provider=cast(str, payload["provider"]),
            model=cast(str, payload["model"]),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "candidate_text": self.candidate_text,
            "addressed_rule_ids": list(self.addressed_rule_ids),
            "claimed_preservations": list(self.claimed_preservations),
            "provider": self.provider,
            "model": self.model,
        }


@dataclass(frozen=True)
class ChallengeResult:
    """Typed doubt about meaning drift. It may raise doubt, never clear it."""

    disposition: str
    claims: tuple[str, ...]
    provider: str
    model: str

    @staticmethod
    def from_payload(
        payload: dict[str, JsonValue], provider_name: str
    ) -> ChallengeResult:
        validate_challenge(payload, provider_name)
        return ChallengeResult(
            disposition=cast(str, payload["disposition"]),
            claims=tuple(cast(list[str], payload["claims"])),
            provider=cast(str, payload["provider"]),
            model=cast(str, payload["model"]),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "disposition": self.disposition,
            "claims": list(self.claims),
            "provider": self.provider,
            "model": self.model,
        }


class ProposalProvider(Protocol):
    """Turns a critique brief into a candidate rewrite."""

    @property
    def name(self) -> str: ...

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        """Return a proposal or raise an explicit provider exception."""


class DriftChallenger(Protocol):
    """Compares two texts and returns typed drift claims."""

    @property
    def name(self) -> str: ...

    def challenge(self, source_text: str, candidate_text: str) -> ChallengeResult:
        """Return a typed challenge; never authoritatively approve a candidate."""


def _load_validator(schema_name: str) -> Draft202012Validator:
    path = SCHEMA_DIR / schema_name
    if not path.is_file():
        raise ProviderError(f"provider schema {schema_name!r} is missing at {path}")
    schema = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return Draft202012Validator(schema)


def _validate(
    payload: dict[str, JsonValue], schema_name: str, provider_name: str
) -> None:
    try:
        _load_validator(schema_name).validate(payload)
    except SchemaValidationError as error:
        path = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise ProviderResponseError(
            f"provider {provider_name!r} returned a response that violates "
            f"{schema_name} at {path}: {error.message}"
        ) from error


def validate_proposal(payload: dict[str, JsonValue], provider_name: str) -> None:
    """Validate a proposal payload, naming the offending provider on failure."""

    _validate(payload, "provider-proposal.schema.json", provider_name)


def validate_challenge(payload: dict[str, JsonValue], provider_name: str) -> None:
    """Validate a challenge payload, naming the offending provider on failure."""

    _validate(payload, "provider-challenge.schema.json", provider_name)


ProposalFactory = Callable[..., ProposalProvider]
ChallengeFactory = Callable[..., DriftChallenger]

_PROPOSAL_FACTORIES: Final[dict[str, ProposalFactory]] = {}
_CHALLENGE_FACTORIES: Final[dict[str, ChallengeFactory]] = {}


def register_proposal_provider(name: str, factory: ProposalFactory) -> None:
    if name in _PROPOSAL_FACTORIES:
        raise ProviderError(f"proposal provider {name!r} is already registered")
    _PROPOSAL_FACTORIES[name] = factory


def register_challenge_provider(name: str, factory: ChallengeFactory) -> None:
    if name in _CHALLENGE_FACTORIES:
        raise ProviderError(f"challenge provider {name!r} is already registered")
    _CHALLENGE_FACTORIES[name] = factory


def available_proposal_providers() -> tuple[str, ...]:
    return tuple(sorted(_PROPOSAL_FACTORIES))


def available_challenge_providers() -> tuple[str, ...]:
    return tuple(sorted(_CHALLENGE_FACTORIES))


def create_proposal_provider(name: str, **options: Any) -> ProposalProvider:
    """Resolve a registered proposal provider, or fail naming what exists."""

    if name not in _PROPOSAL_FACTORIES:
        raise ProviderError(
            f"unknown proposal provider {name!r}; registered providers are "
            f"{list(available_proposal_providers())}"
        )
    return _PROPOSAL_FACTORIES[name](**options)


def create_challenge_provider(name: str, **options: Any) -> DriftChallenger:
    """Resolve a registered challenge provider, or fail naming what exists."""

    if name not in _CHALLENGE_FACTORIES:
        raise ProviderError(
            f"unknown challenge provider {name!r}; registered providers are "
            f"{list(available_challenge_providers())}"
        )
    return _CHALLENGE_FACTORIES[name](**options)
