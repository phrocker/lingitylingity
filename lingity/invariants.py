"""Protected-element extraction and deterministic comparison."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, cast

from lingity.models import JsonValue
from lingity.profiles import Profile, sha256_json
from lingity.text import line_column, sentences

URL_RE = re.compile(r"https?://[^\s<>()]+")
CODE_RE = re.compile(r"`[^`\n]+`")
CITATION_RE = re.compile(r"\[[A-Za-z0-9_.:-]+\]|\([A-Za-z][A-Za-z .'-]+,\s*\d{4}\)")
IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z]{2,}-\d+(?:-[A-Za-z0-9]+)*|[A-Z]\d+(?:\.\d+)*|[A-Z]{1,8}\d[A-Z0-9]*(?:-[A-Z0-9]+)*)\b"
)
_NUMBER_WORD_PATTERN = r"one|two|three|four|five|six|seven|eight|nine|ten|either"
_NUMBER_PATTERN = rf"\d+(?:\.\d+)?|{_NUMBER_WORD_PATTERN}"
_UNIT_PATTERN = (
    r"%|seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"kg|kgs|kilograms?|lb|lbs|pounds?"
)
QUANTITY_RE = re.compile(
    rf"\b\d{{4}}-\d{{2}}-\d{{2}}\b|"
    rf"\b(?:at\s+least|at\s+most|no\s+more\s+than|no\s+less\s+than|more\s+than|less\s+than|fewer\s+than|over|under)\s+(?:{_NUMBER_PATTERN})(?:\s*(?:{_UNIT_PATTERN}))?|"
    rf"(?:>=|<=|>|<)\s*(?:\d+(?:\.\d+)?)(?:\s*(?:{_UNIT_PATTERN}))?|"
    rf"\b\d+(?:\.\d+)?%|"
    rf"\b(?:{_NUMBER_PATTERN})\s+(?:{_UNIT_PATTERN})\b|"
    rf"\b(?:{_NUMBER_PATTERN})\b",
    re.IGNORECASE,
)
MODAL_RE = re.compile(
    r"\b(?:must\s+not|shall\s+not|should\s+not|may\s+not|can\s+not|cannot|can't|mustn't|"
    r"shouldn't|won't|wouldn't|couldn't|must|shall|should|may|can|required\s+to|needs?\s+to|needed\s+to)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(?:cannot|can't|can\s+not|do\s+not|don't|does\s+not|doesn't|did\s+not|didn't|"
    r"will\s+not|won't|must\s+not|mustn't|shall\s+not|should\s+not|shouldn't|may\s+not|"
    r"could\s+not|couldn't|would\s+not|wouldn't|is\s+not|isn't|are\s+not|aren't|"
    r"was\s+not|wasn't|were\s+not|weren't|not|never|no|without|neither|nor)\b",
    re.IGNORECASE,
)
GOVERNANCE_RE = re.compile(
    r"\b(?:approve|approval|ratification|waiver|risk|applicability|recommendation|"
    r"decision|cutover|provisional|target architecture)\b",
    re.IGNORECASE,
)
GOVERNANCE_STATUS_PATTERNS = (
    re.compile(
        r"\b(?P<subject>[A-Za-z][A-Za-z0-9-]*(?:\s+[A-Za-z][A-Za-z0-9-]*){0,4})\s+"
        r"(?:is|are|was|were|be|been|being)\s+"
        r"(?P<state>approved|rejected|granted|denied|ratified|waived|accepted|blocked)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<subject>waiver|approval|ratification|recommendation|decision|architecture|design|ADR-\d+(?:-[A-Za-z0-9]+)?)\s+"
        r"(?P<state>approved|rejected|granted|denied|ratified|waived|accepted|blocked)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?P<state>approved|rejected|granted|denied|ratified|waived|accepted|blocked)\b", re.IGNORECASE),
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
}

STATE_NORMALIZATION = {
    "approved": "approved",
    "rejected": "rejected",
    "granted": "granted",
    "denied": "denied",
    "ratified": "ratified",
    "waived": "waived",
    "accepted": "accepted",
    "blocked": "blocked",
}
STATE_POLARITY = {
    "approved": "positive",
    "granted": "positive",
    "ratified": "positive",
    "waived": "positive",
    "accepted": "positive",
    "rejected": "negative",
    "denied": "negative",
    "blocked": "negative",
}
STATE_DOMAIN = {
    "approved": "approval",
    "rejected": "approval",
    "granted": "waiver",
    "denied": "waiver",
    "ratified": "ratification",
    "waived": "waiver",
    "accepted": "acceptance",
    "blocked": "blocking",
}

ACTION_NORMALIZATION = {
    "approve": "approve",
    "approved": "approve",
    "ratify": "approve",
    "ratified": "approve",
    "accept": "approve",
    "accepted": "approve",
    "deploy": "deploy",
    "review": "review",
    "begin": "begin",
    "start": "begin",
    "fix": "resolve",
    "address": "resolve",
    "resolve": "resolve",
    "remediate": "resolve",
    "note": "note",
    "observe": "note",
    "mention": "note",
}
CLAIM_VERB_RE = r"approve|deploy|ratify|review|begin|start|fix|address|resolve|remediate|note|observe|mention|accept"
AUTHORIZATION_RE = re.compile(r"\bconfirmed\s+authorization\s+(?:concerns|issues)\b", re.IGNORECASE)
ACTOR_MODAL_RE = re.compile(
    rf"\b(?P<actor>[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){{0,3}})\s+"
    rf"(?P<modal>must\s+not|shall\s+not|should\s+not|may\s+not|can\s+not|cannot|can't|mustn't|shouldn't|must|shall|should|may|can|required\s+to|needs?\s+to)\s+"
    rf"(?P<verb>{CLAIM_VERB_RE}|be)\b(?P<object>[^.;]*)",
    re.IGNORECASE,
)
IMPERATIVE_RE = re.compile(
    rf"^\s*(?:first,?\s+|then\s+|now\s+|immediately\s+)?(?P<neg>do\s+not|don't|never)?\s*"
    rf"(?P<verb>{CLAIM_VERB_RE})\b(?P<object>[^.;]*)",
    re.IGNORECASE,
)
PASSIVE_APPROVAL_RE = re.compile(r"\bbe\s+(?P<state>approved|rejected|ratified|accepted|blocked)\b", re.IGNORECASE)
PRONOUN_TARGETS = {"it", "this", "that", "them", "these", "those"}
STOP_TARGET_WORDS = {
    "a", "an", "and", "any", "as", "before", "by", "for", "from", "in", "into", "now", "of", "on", "or",
    "our", "path", "paths", "please", "the", "to", "with", "yet",
}
CONCEPT_CLAIMS = {
    "architecture_approval_deferred": (
        ("unspecified", "approve", "architecture", "must", "negative", "deferred"),
    ),
    "v2_cutover_deferred": (
        ("unspecified", "begin", "v2 cutover", "must", "negative", "deferred"),
    ),
    "messaging_loss_requires_resolution": (
        ("unspecified", "resolve", "messaging loss", "must", "positive", "required"),
    ),
    "recommendation_closure_evidence_required": (
        ("unspecified", "provide_evidence", "governed recommendations", "must", "positive", "required"),
    ),
    "target_architecture_requires_human_approval": (
        ("human", "approve", "target architecture", "must", "positive", "required"),
    ),
}


@dataclass(frozen=True)
class ConceptSpan:
    start: int
    end: int
    category: str
    kind: str
    value: str


@dataclass(frozen=True)
class QuantityParts:
    kind: str
    normalized: str


@dataclass(frozen=True)
class ClauseSpan:
    start: int
    end: int
    text: str


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


def _contains(spans: Iterable[tuple[int, int]], start: int, end: int) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def _concept_for_span(spans: Iterable[ConceptSpan], start: int, end: int) -> ConceptSpan | None:
    for span in spans:
        if span.start <= start and end <= span.end:
            return span
    return None


def _normalize_modal(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.lower().replace("’", "'")).strip()
    equivalents = {
        "shall": "must",
        "required to": "must",
        "need to": "must",
        "needs to": "must",
        "needed to": "must",
        "shall not": "must not",
        "mustn't": "must not",
        "shouldn't": "should not",
        "can't": "cannot",
        "can not": "cannot",
    }
    return equivalents.get(lowered, lowered)


def _normalize_negation(value: str) -> str:
    lowered = re.sub(r"\s+", " ", value.lower().replace("’", "'")).strip()
    if lowered in {"no", "without", "neither", "nor"}:
        return lowered
    if lowered == "never":
        return "never"
    return "not"


def _normalize_number(value: str) -> str:
    lowered = value.lower()
    return NUMBER_WORDS.get(lowered, lowered)


def _normalize_unit(value: str | None) -> str:
    if value is None or not value:
        return "count"
    lowered = value.lower().strip()
    if lowered == "%":
        return "percent"
    if lowered.startswith("second"):
        return "second"
    if lowered.startswith("minute"):
        return "minute"
    if lowered.startswith("hour"):
        return "hour"
    if lowered.startswith("day"):
        return "day"
    if lowered.startswith("week"):
        return "week"
    if lowered.startswith("month"):
        return "month"
    if lowered.startswith("year"):
        return "year"
    if lowered in {"kg", "kgs", "kilogram", "kilograms"}:
        return "kg"
    if lowered in {"lb", "lbs", "pound", "pounds"}:
        return "lb"
    return lowered


def _quantity_operator(surface: str) -> str:
    lowered = surface.lower().strip()
    if lowered.startswith((">=", "at least", "no less than")):
        return "gte"
    if lowered.startswith(("<=", "at most", "no more than")):
        return "lte"
    if lowered.startswith((">", "more than", "over")):
        return "gt"
    if lowered.startswith(("<", "less than", "fewer than", "under")):
        return "lt"
    return "eq"


def _quantity_parts(surface: str, concept: ConceptSpan | None) -> QuantityParts:
    lowered = re.sub(r"\s+", " ", surface.lower()).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", lowered):
        return QuantityParts("date", lowered)
    if lowered == "either":
        if concept is not None and concept.value == "messaging_loss_requires_resolution":
            return QuantityParts("count", "2")
        return QuantityParts("quantifier", "either")

    number_match = re.search(rf"\b(?:\d+(?:\.\d+)?|{_NUMBER_WORD_PATTERN})\b", lowered, re.IGNORECASE)
    number = _normalize_number(number_match.group(0)) if number_match is not None else lowered
    unit_match = re.search(rf"(?:{_UNIT_PATTERN})\b|%", lowered, re.IGNORECASE)
    unit = _normalize_unit(unit_match.group(0)) if unit_match is not None else "count"
    operator = _quantity_operator(lowered)

    if unit == "percent" and operator == "eq":
        return QuantityParts("percentage", f"{number}%")
    if unit in {"second", "minute", "hour", "day", "week", "month", "year"}:
        return QuantityParts("duration", f"operator={operator};unit={unit};value={number}")
    if unit in {"kg", "lb"}:
        return QuantityParts("measurement", f"operator={operator};unit={unit};value={number}")
    if operator != "eq":
        return QuantityParts("threshold", f"operator={operator};unit={unit};value={number}")
    return QuantityParts("count", number)


def _normalize_subject(value: str | None) -> str:
    if value is None:
        return "unknown"
    words = [word.lower() for word in re.findall(r"[A-Za-z0-9-]+", value)]
    if words[:1] == ["team"]:
        return " ".join(words) if len(words) > 1 else "team"
    filtered = [word for word in words if word not in {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being"}]
    filtered = [word for word in filtered if word not in {"can", "cannot", "must", "shall", "should", "may", "will", "would", "could", "not"}]
    return " ".join(filtered) if filtered else "unknown"


def _status_items(text: str) -> Iterable[dict[str, JsonValue]]:
    seen: set[tuple[int, int]] = set()
    for pattern in GOVERNANCE_STATUS_PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start("state"), match.end("state"))
            if span in seen:
                continue
            seen.add(span)
            state = STATE_NORMALIZATION[match.group("state").lower()]
            subject = _normalize_subject(match.groupdict().get("subject"))
            normalized = (
                f"domain={STATE_DOMAIN[state]};polarity={STATE_POLARITY[state]};"
                f"state={state};subject={subject}"
            )
            yield _item(text, "governance", "status", normalized, span[0], span[1], text)


def _clause_spans(text: str) -> Iterable[ClauseSpan]:
    for sentence in sentences(text):
        offset = sentence.start
        for part in re.finditer(r"[^;]+", sentence.text):
            raw_start = offset + part.start()
            raw_end = offset + part.end()
            fragment = text[raw_start:raw_end]
            leading = len(fragment) - len(fragment.lstrip())
            trailing = len(fragment.rstrip())
            start = raw_start + leading
            end = raw_start + trailing
            if start < end:
                yield ClauseSpan(start, end, text[start:end])


def _normalize_target(value: str) -> str:
    cleaned = re.sub(r"\b(?:now|yet|first|then)\b", " ", value.lower())
    words = re.findall(r"[A-Za-z0-9-]+", cleaned)
    filtered = [word for word in words if word not in STOP_TARGET_WORDS]
    if not filtered:
        return "unspecified"
    if len(filtered) == 1 and filtered[0] in PRONOUN_TARGETS:
        return cast(str, filtered[0])
    return " ".join(filtered)


def _claim_signature(actor: str, action: str, target: str, modality: str, polarity: str, status: str) -> str:
    return (
        f"claim:action={action};actor={actor};modality={modality};"
        f"polarity={polarity};status={status};target={target}"
    )


def _add_claim(
    claims: list[tuple[int, int, str]],
    start: int,
    end: int,
    actor: str,
    action: str,
    target: str,
    modality: str,
    polarity: str,
    status: str = "asserted",
) -> None:
    claims.append((start, end, _claim_signature(actor, action, target, modality, polarity, status)))


def _modal_polarity(modal: str) -> tuple[str, str]:
    normalized = _normalize_modal(modal)
    if normalized in {"must not", "should not", "may not", "cannot"}:
        return normalized.replace(" not", ""), "negative"
    return normalized, "positive"


def _verb_action(verb: str, obj: str) -> tuple[str, str, str]:
    lowered = verb.lower()
    passive = PASSIVE_APPROVAL_RE.search(f"be {obj}" if lowered == "be" else obj)
    if lowered == "be" and passive is not None:
        state = STATE_NORMALIZATION[passive.group("state").lower()]
        return ACTION_NORMALIZATION.get(state, state), "self", state
    action = ACTION_NORMALIZATION.get(lowered, lowered)
    target = _normalize_target(obj)
    return action, target, "asserted"


def _authorization_claims(text: str, claims: list[tuple[int, int, str]]) -> None:
    for match in re.finditer(rf"\b(?P<verb>fix|address|resolve|remediate|note|observe|mention)\b[^.;]*{AUTHORIZATION_RE.pattern}", text, re.IGNORECASE):
        verb = match.group("verb").lower()
        action = ACTION_NORMALIZATION.get(verb, verb)
        _add_claim(claims, match.start(), match.end(), "unspecified", action, "authorization issues", "imperative", "positive")


def _concept_claims(concepts: Iterable[ConceptSpan], claims: list[tuple[int, int, str]]) -> None:
    for concept in concepts:
        for actor, action, target, modality, polarity, status in CONCEPT_CLAIMS.get(concept.value, ()):
            _add_claim(claims, concept.start, concept.end, actor, action, target, modality, polarity, status)


def _generic_claims(text: str, concepts: Iterable[ConceptSpan], claims: list[tuple[int, int, str]]) -> None:
    concept_ranges = [(concept.start, concept.end) for concept in concepts if concept.value in CONCEPT_CLAIMS]
    for clause in _clause_spans(text):
        if _contains(concept_ranges, clause.start, clause.end):
            continue
        for match in ACTOR_MODAL_RE.finditer(clause.text):
            start = clause.start + match.start()
            end = clause.start + match.end()
            if _contains(concept_ranges, start, end):
                continue
            modal, polarity = _modal_polarity(match.group("modal"))
            action, target, status = _verb_action(match.group("verb"), match.group("object"))
            actor = _normalize_subject(match.group("actor"))
            if target == "self":
                target = actor
            _add_claim(claims, start, end, actor, action, target, modal, polarity, status)

        imperative = IMPERATIVE_RE.search(clause.text)
        if imperative is None:
            continue
        if AUTHORIZATION_RE.search(clause.text) is not None:
            continue
        start = clause.start + imperative.start()
        end = clause.start + imperative.end()
        if _contains(concept_ranges, start, end):
            continue
        neg = imperative.group("neg")
        polarity = "negative" if neg is not None else "positive"
        action, target, status = _verb_action(imperative.group("verb"), imperative.group("object"))
        _add_claim(claims, start, end, "unspecified", action, target, "imperative", polarity, status)


def _status_claims(text: str, claims: list[tuple[int, int, str]]) -> None:
    seen: set[tuple[int, int]] = set()
    for pattern in GOVERNANCE_STATUS_PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start("state"), match.end("state"))
            if span in seen:
                continue
            seen.add(span)
            state = STATE_NORMALIZATION[match.group("state").lower()]
            subject = _normalize_subject(match.groupdict().get("subject"))
            _add_claim(
                claims,
                match.start("state"),
                match.end("state"),
                "unspecified",
                STATE_DOMAIN[state],
                subject,
                "state",
                STATE_POLARITY[state],
                state,
            )


def _semantic_claim_signatures(text: str, concepts: Iterable[ConceptSpan]) -> list[str]:
    concept_list = list(concepts)
    claims: list[tuple[int, int, str]] = []
    _concept_claims(concept_list, claims)
    _authorization_claims(text, claims)
    _generic_claims(text, concept_list, claims)
    _status_claims(text, claims)
    unique = sorted(set(claims), key=lambda item: (item[0], item[1], item[2]))
    return [signature for _, _, signature in unique]


def _extract_concepts(text: str, profile: Profile) -> tuple[list[dict[str, JsonValue]], list[ConceptSpan]]:
    items: list[dict[str, JsonValue]] = []
    spans: list[ConceptSpan] = []
    concepts = cast(list[dict[str, Any]], profile.rules["protected_concepts"])
    seen: set[tuple[str, str, str, int, int]] = set()
    for concept in concepts:
        category = cast(str, concept["category"])
        kind = cast(str, concept["kind"])
        value = cast(str, concept["value"])
        for expression in cast(list[str], concept["patterns"]):
            for concept_match in re.finditer(expression, text, re.IGNORECASE):
                key = (category, kind, value, concept_match.start(), concept_match.end())
                if key in seen:
                    continue
                seen.add(key)
                spans.append(ConceptSpan(concept_match.start(), concept_match.end(), category, kind, value))
                items.append(_item(text, category, kind, value, concept_match.start(), concept_match.end(), text))
    return items, spans


def extract_protected(text: str, profile: Profile) -> dict[str, JsonValue]:
    items: list[dict[str, JsonValue]] = []
    concept_items, concept_spans = _extract_concepts(text, profile)
    items.extend(concept_items)
    covered_ranges = [(span.start, span.end) for span in concept_spans]

    citation_items = (
        list(_regex_items(text, URL_RE, "citation", "url", lambda value: value))
        + list(_regex_items(text, CODE_RE, "citation", "code", lambda value: value))
        + list(_regex_items(text, CITATION_RE, "citation", "reference", lambda value: value))
    )
    citation_ranges = []
    for item in citation_items:
        location = cast(dict[str, JsonValue], item["location"])
        citation_ranges.append((cast(int, location["start"]), cast(int, location["end"])))
    items.extend(citation_items)

    identifier_items = []
    for match in IDENTIFIER_RE.finditer(text):
        if not _contains(citation_ranges, match.start(), match.end()):
            identifier_items.append(_item(text, "identifier", "identifier", match.group(0).upper(), match.start(), match.end(), text))
    identifier_ranges = []
    for item in identifier_items:
        location = cast(dict[str, JsonValue], item["location"])
        identifier_ranges.append((cast(int, location["start"]), cast(int, location["end"])))
    items.extend(identifier_items)

    non_quantity_ranges = citation_ranges + identifier_ranges
    for match in QUANTITY_RE.finditer(text):
        if _contains(non_quantity_ranges, match.start(), match.end()):
            continue
        surface = match.group(0)
        parts = _quantity_parts(surface, _concept_for_span(concept_spans, match.start(), match.end()))
        items.append(_item(text, "quantity", parts.kind, parts.normalized, match.start(), match.end(), text))
    for match in MODAL_RE.finditer(text):
        if not _contains(covered_ranges, match.start(), match.end()):
            items.append(
                _item(text, "modal", "modal_strength", _normalize_modal(match.group(0)), match.start(), match.end(), text)
            )
    items.extend(_status_items(text))

    for match in NEGATION_RE.finditer(text):
        if not _contains(covered_ranges, match.start(), match.end()):
            items.append(_item(text, "negation", "lexical_negation", _normalize_negation(match.group(0)), match.start(), match.end(), text))
    for match in GOVERNANCE_RE.finditer(text):
        if not _contains(covered_ranges, match.start(), match.end()):
            items.append(_item(text, "governance", "term", match.group(0).lower(), match.start(), match.end(), text))

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
            cast(str, item["kind"]),
            cast(str, item["normalized"]),
        ),
    )
    signature = sorted(
        [f"{item['category']}:{item['kind']}:{item['normalized']}" for item in ordered]
        + _semantic_claim_signatures(text, concept_spans)
    )
    manifest: dict[str, JsonValue] = {
        "items": cast(list[JsonValue], ordered),
        "semantic_signature": cast(list[JsonValue], signature),
    }
    manifest["sha256"] = sha256_json(manifest)
    return manifest


def _parse_claim(signature: str) -> dict[str, str] | None:
    if not signature.startswith("claim:"):
        return None
    result: dict[str, str] = {}
    for part in signature[len("claim:"):].split(";"):
        key, separator, value = part.partition("=")
        if separator:
            result[key] = value
    return result


def _same_or_ambiguous_target(left: str, right: str) -> bool:
    return left == right or left in PRONOUN_TARGETS or right in PRONOUN_TARGETS or left == "unknown" or right == "unknown"


def _unresolved_claim_reasons(signatures: Iterable[str], label: str) -> list[str]:
    claims = [claim for claim in (_parse_claim(signature) for signature in signatures) if claim is not None]
    reasons: list[str] = []
    for index, left in enumerate(claims):
        if left.get("status") not in {"asserted", "required", "deferred"} and left.get("target") == "unknown":
            reasons.append(f"{label}: governance status target is unresolved for {left.get('status', 'unknown')}")
        for right in claims[index + 1:]:
            if left.get("action") != right.get("action"):
                continue
            if left.get("polarity") == right.get("polarity"):
                continue
            if _same_or_ambiguous_target(left.get("target", "unknown"), right.get("target", "unknown")):
                reasons.append(
                    f"{label}: conflicting polarity for action={left.get('action', 'unknown')} "
                    f"target={left.get('target', 'unknown')}/{right.get('target', 'unknown')}"
                )
    return sorted(set(reasons))


def compare_protected(
    source_manifest: dict[str, JsonValue],
    candidate_manifest: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    source_signatures = cast(list[str], source_manifest["semantic_signature"])
    candidate_signatures = cast(list[str], candidate_manifest["semantic_signature"])
    source = Counter(source_signatures)
    candidate = Counter(candidate_signatures)
    missing = sorted((source - candidate).elements())
    added = sorted((candidate - source).elements())
    unresolved = _unresolved_claim_reasons(source_signatures, "source") + _unresolved_claim_reasons(candidate_signatures, "candidate")
    equivalent = not missing and not added and not unresolved
    disposition = "equivalent" if equivalent else ("unresolved" if unresolved else "changed")
    return {
        "equivalent": equivalent,
        "source_manifest_sha256": cast(str, source_manifest["sha256"]),
        "candidate_manifest_sha256": cast(str, candidate_manifest["sha256"]),
        "missing": cast(list[JsonValue], missing),
        "added": cast(list[JsonValue], added),
        "disposition": disposition,
        "unresolved": cast(list[JsonValue], sorted(unresolved)),
    }


def source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
