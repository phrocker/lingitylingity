"""Provider contracts and the transport registry.

Providers are transports, never authorities. Registration lives here so the set
of available transports is explicit and discoverable.

Network-backed providers are registered through thin factories that import
their module on first use. That keeps importing Lingity free of credentials and
network concerns, while still surfacing a genuine import failure as a real
error at the moment a caller asks for that transport — nothing is swallowed.
"""

from __future__ import annotations

from typing import Any

from lingity.providers.agent import SubagentProvider, create_subagent_provider
from lingity.providers.base import (
    ChallengeResult,
    DriftChallenger,
    ProposalProvider,
    ProposalRequest,
    ProposalResponse,
    ProviderError,
    ProviderExhausted,
    ProviderResponseError,
    available_challenge_providers,
    available_proposal_providers,
    create_challenge_provider,
    create_proposal_provider,
    register_challenge_provider,
    register_proposal_provider,
    validate_challenge,
    validate_proposal,
)

__all__ = [
    "ChallengeResult",
    "DriftChallenger",
    "ProposalProvider",
    "ProposalRequest",
    "ProposalResponse",
    "ProviderError",
    "ProviderExhausted",
    "ProviderResponseError",
    "SubagentProvider",
    "available_challenge_providers",
    "available_proposal_providers",
    "create_challenge_provider",
    "create_proposal_provider",
    "register_challenge_provider",
    "register_proposal_provider",
    "validate_challenge",
    "validate_proposal",
]


def _openai_proposal(**options: Any) -> ProposalProvider:
    from lingity.providers.openai_provider import create_openai_proposal_provider

    return create_openai_proposal_provider(**options)


def _openai_challenge(**options: Any) -> DriftChallenger:
    from lingity.providers.openai_provider import create_openai_challenger

    return create_openai_challenger(**options)


def _anthropic_proposal(**options: Any) -> ProposalProvider:
    from lingity.providers.anthropic_provider import (
        create_anthropic_proposal_provider,
    )

    return create_anthropic_proposal_provider(**options)


def _anthropic_challenge(**options: Any) -> DriftChallenger:
    from lingity.providers.anthropic_provider import create_anthropic_challenger

    return create_anthropic_challenger(**options)


register_proposal_provider("subagent", create_subagent_provider)
register_proposal_provider("openai", _openai_proposal)
register_proposal_provider("anthropic", _anthropic_proposal)
register_challenge_provider("openai", _openai_challenge)
register_challenge_provider("anthropic", _anthropic_challenge)
