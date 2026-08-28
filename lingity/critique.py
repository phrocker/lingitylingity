"""Deterministic improvement briefs.

A critique brief is the complete, provider-agnostic statement of what is wrong
with a text and what a rewrite is forbidden to change. It is built entirely
from deterministic analysis: no network call, no model judgement, no guessing.

The brief is the only thing a proposal provider ever sees. Keeping it a plain
data structure is what lets the same critique drive an in-process model call,
an out-of-process subagent handoff, or a human editor without any of them
gaining authority over the verdict.
"""

from __future__ import annotations

import hashlib
from typing import Final, cast

from lingity.models import JsonValue
from lingity.profiles import sha256_json

CRITIQUE_SCHEMA_VERSION: Final = "1.0.0"
CRITIQUE_KIND: Final = "lingity.critique.v1"

SEVERITY_ORDER: Final[dict[str, int]] = {"high": 0, "medium": 1, "low": 2}

MAX_EXCERPT_CHARS: Final = 240


class CritiqueError(RuntimeError):
    """Raised when a brief cannot be built from the supplied artifact."""


def _require(container: dict[str, JsonValue], key: str, context: str) -> JsonValue:
    if key not in container:
        raise CritiqueError(
            f"cannot build a critique brief: {context} is missing the required "
            f"{key!r} field; refusing to continue with an incomplete brief"
        )
    return container[key]


def _excerpt(text: str, start: int, end: int) -> str:
    if start < 0 or end > len(text) or start >= end:
        raise CritiqueError(
            f"finding location [{start}, {end}) does not fall inside the "
            f"analyzed text of length {len(text)}"
        )
    span = text[start:end]
    if len(span) <= MAX_EXCERPT_CHARS:
        return span
    return span[: MAX_EXCERPT_CHARS - 1] + "\u2026"


def _severity_rank(severity: str) -> int:
    if severity not in SEVERITY_ORDER:
        raise CritiqueError(
            f"unknown finding severity {severity!r}; expected one of "
            f"{sorted(SEVERITY_ORDER)}"
        )
    return SEVERITY_ORDER[severity]


def build_critique(
    analysis: dict[str, JsonValue],
    *,
    prior_attempts: list[dict[str, JsonValue]] | None = None,
) -> dict[str, JsonValue]:
    """Build a deterministic improvement brief from an analysis artifact.

    ``prior_attempts`` carries the rejection record of earlier candidates so a
    provider is told exactly why its last attempt failed instead of rediscovering
    the same rejection.
    """

    source = cast(dict[str, JsonValue], _require(analysis, "source", "the analysis artifact"))
    text = cast(str, _require(source, "text", "the analysis source"))
    score = cast(dict[str, JsonValue], _require(analysis, "score", "the analysis artifact"))
    protected = cast(
        dict[str, JsonValue], _require(analysis, "protected", "the analysis artifact")
    )

    raw_findings = cast(
        list[dict[str, JsonValue]], _require(analysis, "findings", "the analysis artifact")
    )
    defects: list[dict[str, JsonValue]] = []
    for finding in raw_findings:
        location = cast(dict[str, JsonValue], _require(finding, "location", "a finding"))
        start = cast(int, _require(location, "start", "a finding location"))
        end = cast(int, _require(location, "end", "a finding location"))
        severity = cast(str, _require(finding, "severity", "a finding"))
        _severity_rank(severity)
        defects.append(
            {
                "rule_id": _require(finding, "rule_id", "a finding"),
                "dimension": _require(finding, "dimension", "a finding"),
                "severity": severity,
                "location": location,
                "observed_value": _require(finding, "observed_value", "a finding"),
                "threshold": _require(finding, "threshold", "a finding"),
                "remediation": _require(finding, "remediation", "a finding"),
                "excerpt": _excerpt(text, start, end),
            }
        )

    def _defect_order(item: dict[str, JsonValue]) -> tuple[int, int, str]:
        location = cast(dict[str, JsonValue], item["location"])
        return (
            _severity_rank(cast(str, item["severity"])),
            cast(int, location["start"]),
            cast(str, item["rule_id"]),
        )

    defects.sort(key=_defect_order)

    must_preserve: list[dict[str, JsonValue]] = []
    for item in cast(list[dict[str, JsonValue]], _require(protected, "items", "the protected manifest")):
        must_preserve.append(
            {
                "category": _require(item, "category", "a protected element"),
                "kind": _require(item, "kind", "a protected element"),
                "text": _require(item, "text", "a protected element"),
                "normalized": _require(item, "normalized", "a protected element"),
            }
        )

    brief: dict[str, JsonValue] = {
        "schema_version": CRITIQUE_SCHEMA_VERSION,
        "kind": CRITIQUE_KIND,
        "analyzer_version": _require(analysis, "analyzer_version", "the analysis artifact"),
        "linguistic_model": _require(analysis, "linguistic_model", "the analysis artifact"),
        "profile": _require(analysis, "profile", "the analysis artifact"),
        "source": {
            "text": text,
            "sha256": _require(source, "sha256", "the analysis source"),
        },
        "current_score": {
            "value": _require(score, "value", "the analysis score"),
            "band": _require(score, "band", "the analysis score"),
        },
        "defects": cast(JsonValue, defects),
        "must_preserve": cast(JsonValue, must_preserve),
        "protected_sha256": _require(protected, "sha256", "the protected manifest"),
        "acceptance": {
            "requires_higher_score": True,
            "requires_protected_equivalence": True,
            "forbids_new_high_severity_findings": True,
        },
        "prior_attempts": cast(JsonValue, list(prior_attempts or [])),
    }
    brief["critique_sha256"] = sha256_json(brief)
    return brief


def critique_digest(brief: dict[str, JsonValue]) -> str:
    """Return the stable digest recorded in provenance for a brief."""

    digest = brief.get("critique_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise CritiqueError("critique brief is missing a valid critique_sha256")
    return digest


def response_digest(payload: str) -> str:
    """Hash a raw provider response so provenance can cite it without storing it."""

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
