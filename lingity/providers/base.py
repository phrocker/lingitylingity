"""Non-authoritative provider interfaces for future milestones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ProposalRequest:
    source_text: str
    analysis_sha256: str
    profile_digest: str
    prior_rejection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProposalResponse:
    candidate_text: str
    addressed_rule_ids: tuple[str, ...]
    claimed_preservations: tuple[str, ...]
    provider: str
    model: str


@dataclass(frozen=True)
class ChallengeResult:
    disposition: str
    claims: tuple[str, ...]
    provider: str
    model: str


class ProposalProvider(Protocol):
    def propose(self, request: ProposalRequest) -> ProposalResponse:
        """Return a proposal or raise an explicit provider exception."""


class DriftChallenger(Protocol):
    def challenge(self, source_text: str, candidate_text: str) -> ChallengeResult:
        """Return a typed challenge; never authoritatively approve a candidate."""
