"""Protected-element extraction and deterministic comparison."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Iterable, cast

from lingity.models import JsonValue
from lingity.profiles import Profile, sha256_json
from lingity.text import line_column

URL_RE = re.compile(r"https?://[^\s<>()]+")
CODE_RE = re.compile(r"`[^`\n]+`")
CITATION_RE = re.compile(r"\[[A-Za-z0-9_.:-]+\]|\([A-Za-z][A-Za-z .'-]+,\s*\d{4}\)")
IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Z]{2,}(?:-[A-Z0-9]+)*|[A-Za-z]+-\d+(?:-[A-Za-z0-9]+)*|[A-Z]\d+(?:\.\d+)*)\b"
)
QUANTITY_RE = re.compile(
    r"(?:\b\d{4}-\d{2}-\d{2}\b|\b\d+(?:\.\d+)?%|(?:>=|<=|>|<)\s*\d+(?:\.\d+)?|"
    r"\b\d+(?:\.\d+)?\s+(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b|"
    r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|either)\b)",
    re.IGNORECASE,
)
MODAL_RE = re.compile(r"\b(?:must not|shall not|should not|may not|must|shall|should|may|required to)\b", re.IGNORECASE)
NEGATION_RE = re.compile(r"\b(?:not|never|no|without|neither|nor)\b", re.IGNORECASE)
GOVERNANCE_RE = re.compile(
    r"\b(?:approve|approval|ratification|waiver|risk|applicability|recommendation|"
    r"decision|cutover|provisional|target architecture)\b",
    re.IGNORECASE,
)

NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "either": "2",
}


def _item(
    text: str,
    category: str,
    kind: str,
    value: str,
    start: int,
    end: int,
    source: str,
) -> dict[str, JsonValue]:
    line, column = line_column(source, start)
    return {
        "category": category,
        "kind": kind,
        "text": text[start:end],
        "normalized": value,
        "location": {
            "start": start,
            "end": end,
            "line": line,
            "column": column,
        },
    }


def _regex_items(
    source: str,
    pattern: re.Pattern[str],
    category: str,
    kind: str,
    normalizer: Any,
) -> Iterable[dict[str, JsonValue]]:
    for match in pattern.finditer(source):
        yield _item(
            source,
            category,
            kind,
            cast(str, normalizer(match.group(0))),
            match.start(),
            match.end(),
            source,
        )


def extract_protected(text: str, profile: Profile) -> dict[str, JsonValue]:
    items: list[dict[str, JsonValue]] = []
    items.extend(_regex_items(text, IDENTIFIER_RE, "identifier", "identifier", lambda value: value.upper()))
    for match in QUANTITY_RE.finditer(text):
        surface = match.group(0)
        lowered = surface.lower()
        normalized = NUMBER_WORDS.get(lowered, re.sub(r"\s+", " ", lowered))
        if re.fullmatch(r"\d+(?:\.\d+)?%", surface):
            kind = "percentage"
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", surface):
            kind = "date"
        elif re.search(r"\b(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", surface, re.IGNORECASE):
            kind = "duration"
        elif re.match(r"(?:>=|<=|>|<)", surface):
            kind = "threshold"
        else:
            kind = "count"
        items.append(
            _item(text, "quantity", kind, normalized, match.start(), match.end(), text)
        )
    items.extend(_regex_items(text, MODAL_RE, "modal", "modal_strength", lambda value: value.lower()))
    items.extend(_regex_items(text, URL_RE, "citation", "url", lambda value: value))
    items.extend(_regex_items(text, CODE_RE, "citation", "code", lambda value: value))
    items.extend(_regex_items(text, CITATION_RE, "citation", "reference", lambda value: value))

    concepts = cast(list[dict[str, Any]], profile.rules["protected_concepts"])
    covered_ranges: list[tuple[int, int]] = []
    for concept in concepts:
        for expression in cast(list[str], concept["patterns"]):
            concept_match = re.search(expression, text, re.IGNORECASE)
            if concept_match is not None:
                covered_ranges.append((concept_match.start(), concept_match.end()))
                items.append(
                    _item(
                        text,
                        cast(str, concept["category"]),
                        cast(str, concept["kind"]),
                        cast(str, concept["value"]),
                        concept_match.start(),
                        concept_match.end(),
                        text,
                    )
                )
                break
    for match in NEGATION_RE.finditer(text):
        if not any(start <= match.start() and match.end() <= end for start, end in covered_ranges):
            items.append(
                _item(
                    text, "negation", "lexical_negation", match.group(0).lower(),
                    match.start(), match.end(), text,
                )
            )
    for match in GOVERNANCE_RE.finditer(text):
        if not any(start <= match.start() and match.end() <= end for start, end in covered_ranges):
            items.append(
                _item(
                    text, "governance", "term", match.group(0).lower(),
                    match.start(), match.end(), text,
                )
            )

    unique: dict[tuple[str, str, str, int, int], dict[str, JsonValue]] = {}
    for item in items:
        location = cast(dict[str, JsonValue], item["location"])
        key = (
            cast(str, item["category"]),
            cast(str, item["kind"]),
            cast(str, item["normalized"]),
            cast(int, location["start"]),
            cast(int, location["end"]),
        )
        unique[key] = item
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            cast(dict[str, int], item["location"])["start"],
            cast(str, item["category"]),
            cast(str, item["normalized"]),
        ),
    )
    signature = sorted(
        f"{item['category']}:{item['kind']}:{item['normalized']}"
        for item in ordered
    )
    manifest: dict[str, JsonValue] = {
        "items": cast(list[JsonValue], ordered),
        "semantic_signature": cast(list[JsonValue], signature),
    }
    manifest["sha256"] = sha256_json(manifest)
    return manifest


def compare_protected(
    source_manifest: dict[str, JsonValue],
    candidate_manifest: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    source = Counter(cast(list[str], source_manifest["semantic_signature"]))
    candidate = Counter(cast(list[str], candidate_manifest["semantic_signature"]))
    missing = sorted((source - candidate).elements())
    added = sorted((candidate - source).elements())
    return {
        "equivalent": not missing and not added,
        "source_manifest_sha256": cast(str, source_manifest["sha256"]),
        "candidate_manifest_sha256": cast(str, candidate_manifest["sha256"]),
        "missing": cast(list[JsonValue], missing),
        "added": cast(list[JsonValue], added),
    }


def source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
