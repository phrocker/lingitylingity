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

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final, cast

from lingity.analyzer import analyze_text
from lingity.critique import allowed_word_growth, build_critique, response_digest
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
from lingity.styles import StyleContract

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


def _readable_word_count(analysis: dict[str, JsonValue]) -> int:
    sentences = analysis.get("sentences")
    if not isinstance(sentences, list):
        raise ImprovementError("analysis artifact is missing its sentences list")
    total = 0
    for sentence in sentences:
        if not isinstance(sentence, dict):
            raise ImprovementError("analysis artifact carries an invalid sentence record")
        count = sentence.get("word_count")
        if isinstance(count, bool) or not isinstance(count, int):
            raise ImprovementError("analysis sentence is missing an integer word_count")
        total += count
    return total


def _economy_evidence(
    source_analysis: dict[str, JsonValue],
    candidate_analysis: dict[str, JsonValue],
    profile: Profile,
) -> dict[str, JsonValue]:
    source_words = _readable_word_count(source_analysis)
    candidate_words = _readable_word_count(candidate_analysis)
    policy = profile.rewrite_policy
    allowed_growth = allowed_word_growth(
        source_words,
        policy["max_readable_word_growth_percent"],
        int(policy["max_readable_word_growth_absolute"]),
    )
    maximum = source_words + allowed_growth
    return {
        "source_readable_words": source_words,
        "candidate_readable_words": candidate_words,
        "growth": candidate_words - source_words,
        "allowed_growth": allowed_growth,
        "maximum_candidate_readable_words": maximum,
        "prefer_shorter_candidate": bool(policy["prefer_shorter_candidate"]),
        "passed": candidate_words <= maximum,
    }


def _span(value: object, source_text: str, label: str) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise ImprovementError(f"duplicated-framing finding is missing its {label}")
    start = value.get("start")
    end = value.get("end")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end > len(source_text)
        or start >= end
    ):
        raise ImprovementError(
            f"duplicated-framing finding carries an invalid {label}"
        )
    return start, end


def _restated_by(
    source_text: str,
    earlier: tuple[int, int],
    retained: tuple[int, int],
    profile: Profile,
) -> bool:
    """Whether every protected element of the earlier block survives in the later one.

    The framing detector only establishes partial term overlap, so an earlier
    block can still carry an identifier, quantity, condition, or claim the
    later block lacks. Such a block stays in the meaning baseline.
    """
    comparison = compare_protected(
        extract_protected(source_text[earlier[0] : earlier[1]], profile),
        extract_protected(source_text[retained[0] : retained[1]], profile),
    )
    unresolved = cast(list[str], comparison.get("unresolved") or [])
    return not comparison.get("missing") and not any(
        reason.startswith("source:") for reason in unresolved
    )


def _removable_framing_spans(
    source_text: str, source_analysis: dict[str, JsonValue], profile: Profile
) -> list[tuple[int, int]]:
    """Earlier framing blocks that the later block fully restates.

    The later block remains the canonical statement of the page's scope. An
    earlier block is only listed when its protected elements are all present
    in the retained block, so removing it cannot hide lost content.
    """
    findings = source_analysis.get("findings")
    if not isinstance(findings, list):
        raise ImprovementError("analysis artifact is missing its findings list")
    spans: set[tuple[int, int]] = set()
    for raw in findings:
        if not isinstance(raw, dict) or raw.get("rule_id") != "LING-DUPLICATED-FRAMING-001":
            continue
        observed = raw.get("observed_value")
        if not isinstance(observed, dict):
            raise ImprovementError(
                "duplicated-framing finding is missing its observed value"
            )
        earlier = _span(observed.get("first_location"), source_text, "first location")
        retained = _span(raw.get("location"), source_text, "location")
        if _restated_by(source_text, earlier, retained, profile):
            spans.add(earlier)
    return sorted(spans)


def _without(source_text: str, spans: Iterable[tuple[int, int]]) -> str:
    """Delete blocks together with the blank lines that follow them.

    A block span ends at its last character, so deleting only the span leaves
    an extra blank line where an author's deletion would leave none, and the
    parse of the neighbouring block can differ on that whitespace alone.
    """
    text = source_text
    for start, end in sorted(spans, reverse=True):
        while end < len(text) and text[end].isspace():
            end += 1
        text = text[:start] + text[end:]
    return text


def _flattened(text: str) -> str:
    return " ".join(text.split())


def _delta_size(comparison: dict[str, JsonValue]) -> int:
    return sum(
        len(cast(list[JsonValue], comparison.get(key) or []))
        for key in ("missing", "added", "unresolved")
    )


def _compare_meaning(
    source_text: str,
    candidate_text: str,
    source_analysis: dict[str, JsonValue],
    profile: Profile,
) -> dict[str, JsonValue]:
    """Compare against the full source, then exempt removable framing blocks.

    The full source is authoritative, so a candidate that keeps a removable
    framing block is judged exactly as it would be without the exemption:
    deletion is an allowed remediation, never a required one.

    When the full comparison fails, each removable block the candidate no
    longer contains is considered once, in source order, and stays exempted
    only if dropping it from the baseline shrinks the protected delta. A block
    the candidate deleted stops counting as missing; a block the candidate kept
    would start counting as added, so it is not exempted. This settles any subset of deleted blocks with one
    extraction per block. Every baseline tried omits only blocks whose
    protected elements the later block restates, so the search can miss an
    equivalence but never manufacture one. When no baseline is equivalent, the
    full comparison is reported so the delta names what moved relative to the
    text as written.
    """
    candidate = extract_protected(candidate_text, profile)
    full = compare_protected(extract_protected(source_text, profile), candidate)
    if full["disposition"] == "equivalent":
        return full
    best = full
    exempted: list[tuple[int, int]] = []
    # Protected comparison is blind to position, so it cannot tell which of two
    # blocks with the same elements the candidate kept. The exemption is tied to
    # the earlier block itself: it applies only once that block's text is gone
    # from the candidate, so keeping it and deleting the canonical later block
    # never qualifies. An earlier block whose text also occurs elsewhere is
    # therefore never exempted, which fails closed.
    flattened = _flattened(candidate_text)
    for span in _removable_framing_spans(source_text, source_analysis, profile):
        if _flattened(source_text[span[0] : span[1]]) in flattened:
            continue
        trial = compare_protected(
            extract_protected(_without(source_text, [*exempted, span]), profile),
            candidate,
        )
        if trial["disposition"] == "equivalent":
            return trial
        if _delta_size(trial) < _delta_size(best):
            exempted.append(span)
            best = trial
    return full


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
    economy = _economy_evidence(source_analysis, candidate_analysis, profile)

    comparison = _compare_meaning(
        source_text, candidate_text, source_analysis, profile
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

    if not cast(bool, economy["passed"]):
        reasons.append(
            "candidate exceeds the readable-word growth budget: "
            f"{economy['candidate_readable_words']} words against a maximum of "
            f"{economy['maximum_candidate_readable_words']} from a "
            f"{economy['source_readable_words']}-word source"
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
        "economy": economy,
        "protected_disposition": disposition,
        # A rejection that only says "meaning changed" cannot be acted on. The
        # exact elements that moved are carried through so a host agent can
        # restore them by name on the next attempt.
        "protected_delta": {
            "missing": list(cast(list[JsonValue], comparison.get("missing") or [])),
            "added": list(cast(list[JsonValue], comparison.get("added") or [])),
            "unresolved": list(
                cast(list[JsonValue], comparison.get("unresolved") or [])
            ),
            "specified": list(
                cast(list[JsonValue], comparison.get("specified") or [])
            ),
        },
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
    style: StyleContract | None = None,
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
        brief = build_critique(
            source_analysis, prior_attempts=prior, style=style
        )
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
