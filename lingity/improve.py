"""The deterministic improvement loop.

This module holds the acceptance authority. A provider proposes; this loop
re-analyzes the candidate from scratch, re-extracts its protected elements, and
decides. A candidate is accepted only when *every* condition holds:

* protected meaning is equivalent to the source,
* the Human Readability Index strictly improves,
* no new high-severity finding is introduced,
* and no semantic-drift challenge raised material doubt.

A regression is never accepted, a tie is never accepted, and an unresolved
meaning comparison is never accepted. When the loop runs out of attempts it
returns the source text unchanged with the reasons every candidate failed. It
never returns a success-shaped result it cannot justify.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, cast

from lingity.analyzer import analyze_text
from lingity.critique import build_critique, response_digest
from lingity.invariants import compare_protected, extract_protected
from lingity.models import JsonValue
from lingity.profiles import Profile
from lingity.providers.base import (
    ChallengeResult,
    DriftChallenger,
    ProposalProvider,
    ProposalRequest,
    ProviderExhausted,
)

DEFAULT_MAX_ATTEMPTS: Final = 3
HIGH_SEVERITY: Final = "high"


class ImprovementError(RuntimeError):
    """Raised when the loop cannot be run as configured."""


@dataclass(frozen=True)
class AttemptRecord:
    """One provider attempt and the deterministic verdict on it."""

    index: int
    provider: str
    model: str
    candidate_sha256: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    score_before: float
    score_after: float
    protected_disposition: str
    challenge: ChallengeResult | None = None
    addressed_rule_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, JsonValue]:
        record: dict[str, JsonValue] = {
            "index": self.index,
            "provider": self.provider,
            "model": self.model,
            "candidate_sha256": self.candidate_sha256,
            "accepted": self.accepted,
            "rejection_reasons": list(self.rejection_reasons),
            "score_before": self.score_before,
            "score_after": self.score_after,
            "protected_disposition": self.protected_disposition,
            "addressed_rule_ids": list(self.addressed_rule_ids),
        }
        record["challenge"] = (
            self.challenge.to_dict() if self.challenge is not None else None
        )
        return record


@dataclass(frozen=True)
class ImprovementResult:
    """The outcome of a bounded improvement run."""

    accepted: bool
    selected_text: str
    source_score: float
    selected_score: float
    attempts: tuple[AttemptRecord, ...] = field(default=())
    stop_reason: str = ""

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "accepted": self.accepted,
            "selected_text": self.selected_text,
            "source_score": self.source_score,
            "selected_score": self.selected_score,
            "stop_reason": self.stop_reason,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


def _score_of(analysis: dict[str, JsonValue]) -> float:
    score = analysis.get("score")
    if not isinstance(score, dict):
        raise ImprovementError("analysis artifact is missing its score block")
    value = score.get("value")
    if not isinstance(value, (int, float)):
        raise ImprovementError("analysis score is missing a numeric value")
    return float(value)


def _high_severity_rules(analysis: dict[str, JsonValue]) -> set[str]:
    findings = analysis.get("findings")
    if not isinstance(findings, list):
        raise ImprovementError("analysis artifact is missing its findings list")
    rules: set[str] = set()
    for finding in findings:
        item = cast(dict[str, JsonValue], finding)
        if item.get("severity") == HIGH_SEVERITY:
            rules.add(cast(str, item["rule_id"]))
    return rules


def judge_candidate(
    source_text: str,
    candidate_text: str,
    profile: Profile,
    *,
    challenger: DriftChallenger | None = None,
) -> tuple[bool, tuple[str, ...], dict[str, JsonValue]]:
    """Decide a single candidate deterministically.

    Returns ``(accepted, rejection_reasons, evidence)``. This function performs
    no provider call of its own except an optional drift challenge, and the
    challenge may only *add* doubt — it can never clear a deterministic failure.
    """

    source_analysis = analyze_text(source_text, profile=profile)
    candidate_analysis = analyze_text(candidate_text, profile=profile)

    source_score = _score_of(source_analysis)
    candidate_score = _score_of(candidate_analysis)

    comparison = compare_protected(
        extract_protected(source_text, profile),
        extract_protected(candidate_text, profile),
    )
    disposition = cast(str, comparison["disposition"])

    reasons: list[str] = []

    if disposition != "equivalent":
        missing = cast(list[JsonValue], comparison.get("missing") or [])
        added = cast(list[JsonValue], comparison.get("added") or [])
        unresolved = cast(list[JsonValue], comparison.get("unresolved") or [])
        detail: list[str] = []
        if missing:
            detail.append(f"{len(missing)} protected element(s) dropped")
        if added:
            detail.append(f"{len(added)} protected element(s) introduced")
        if unresolved:
            detail.append(f"{len(unresolved)} protected element(s) unresolved")
        reasons.append(
            f"protected meaning is {disposition}: " + "; ".join(detail)
            if detail
            else f"protected meaning is {disposition}"
        )

    if candidate_score <= source_score:
        verb = "did not change" if candidate_score == source_score else "regressed"
        reasons.append(
            f"readability {verb}: candidate scores {candidate_score:.2f} against "
            f"source {source_score:.2f}; a rewrite must strictly improve the score"
        )

    new_high = _high_severity_rules(candidate_analysis) - _high_severity_rules(
        source_analysis
    )
    if new_high:
        reasons.append(
            "candidate introduces new high-severity finding(s): "
            + ", ".join(sorted(new_high))
        )

    challenge: ChallengeResult | None = None
    if challenger is not None:
        challenge = challenger.challenge(source_text, candidate_text)
        if challenge.disposition == "material_change":
            reasons.append(
                f"semantic-drift challenger {challenge.provider!r} reported a "
                f"material change: {', '.join(challenge.claims) or 'unspecified'}"
            )
        elif challenge.disposition == "needs_human":
            reasons.append(
                f"semantic-drift challenger {challenge.provider!r} could not "
                "resolve equivalence and requires human review"
            )

    evidence: dict[str, JsonValue] = {
        "source_score": source_score,
        "candidate_score": candidate_score,
        "protected_disposition": disposition,
        "challenge": challenge.to_dict() if challenge is not None else None,
    }
    return (not reasons, tuple(reasons), evidence)


def improve_text(
    source_text: str,
    profile: Profile,
    provider: ProposalProvider,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    challenger: DriftChallenger | None = None,
) -> ImprovementResult:
    """Run the bounded improvement loop and return an attributed outcome."""

    if max_attempts < 1:
        raise ImprovementError(
            f"max_attempts must be at least 1; received {max_attempts}"
        )

    source_analysis = analyze_text(source_text, profile=profile)
    source_score = _score_of(source_analysis)

    attempts: list[AttemptRecord] = []
    prior: list[dict[str, JsonValue]] = []
    exhausted_after: int | None = None

    for index in range(1, max_attempts + 1):
        brief = build_critique(source_analysis, prior_attempts=prior)
        try:
            proposal = provider.propose(ProposalRequest(brief=brief))
        except ProviderExhausted:
            exhausted_after = index - 1
            break

        accepted, reasons, evidence = judge_candidate(
            source_text, proposal.candidate_text, profile, challenger=challenger
        )
        record = AttemptRecord(
            index=index,
            provider=proposal.provider,
            model=proposal.model,
            candidate_sha256=response_digest(proposal.candidate_text),
            accepted=accepted,
            rejection_reasons=reasons,
            score_before=source_score,
            score_after=cast(float, evidence["candidate_score"]),
            protected_disposition=cast(str, evidence["protected_disposition"]),
            addressed_rule_ids=proposal.addressed_rule_ids,
        )
        attempts.append(record)

        if accepted:
            return ImprovementResult(
                accepted=True,
                selected_text=proposal.candidate_text,
                source_score=source_score,
                selected_score=record.score_after,
                attempts=tuple(attempts),
                stop_reason=f"candidate accepted on attempt {index}",
            )

        prior.append(
            {
                "index": index,
                "candidate_sha256": record.candidate_sha256,
                "rejection_reasons": list(reasons),
                "score_after": record.score_after,
                "protected_disposition": record.protected_disposition,
            }
        )

    if exhausted_after == 0:
        raise ImprovementError(
            "the provider offered no candidate at all, so there was nothing to "
            "judge; a run must evaluate at least one candidate"
        )

    stop_reason = (
        f"no candidate satisfied the acceptance rules within {max_attempts} "
        "attempt(s); the source text is retained unchanged"
        if exhausted_after is None
        else (
            f"the provider ran out of candidates after {exhausted_after} "
            f"attempt(s), none of which satisfied the acceptance rules; the "
            "source text is retained unchanged"
        )
    )

    return ImprovementResult(
        accepted=False,
        selected_text=source_text,
        source_score=source_score,
        selected_score=source_score,
        attempts=tuple(attempts),
        stop_reason=stop_reason,
    )


def rejection_summary(attempts: Sequence[AttemptRecord]) -> tuple[str, ...]:
    """Flatten every rejection reason across attempts, in order."""

    return tuple(
        f"attempt {attempt.index}: {reason}"
        for attempt in attempts
        for reason in attempt.rejection_reasons
    )
