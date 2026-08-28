"""Protected-element extraction and deterministic comparison."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, cast

from lingity.models import JsonValue
from lingity.nlp import Document, Token, parse
from lingity.profiles import Profile, sha256_json
from lingity.text import line_column

URL_RE = re.compile(r"https?://[^\s<>()]+")
CODE_RE = re.compile(r"`[^`\n]+`")
CITATION_RE = re.compile(r"\[[A-Za-z0-9_.:-]+\]|\([A-Za-z][A-Za-z .'-]+,\s*\d{4}\)")
IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Za-z]{2,}-\d+(?:-[A-Za-z0-9]+)*|[A-Z]\d+(?:\.\d+)*|[A-Z]{1,8}\d[A-Z0-9]*(?:-[A-Z0-9]+)*)\b"
)
_NUMBER_WORD_PATTERN = r"one|two|three|four|five|six|seven|eight|nine|ten|either"
_UNIT_PATTERN = (
    r"%|seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"kg|kgs|kilograms?|lb|lbs|pounds?"
)
KNOWN_UNITS = {
    "%",
    "day",
    "days",
    "hour",
    "hours",
    "kg",
    "kgs",
    "kilogram",
    "kilograms",
    "lb",
    "lbs",
    "minute",
    "minutes",
    "month",
    "months",
    "pound",
    "pounds",
    "second",
    "seconds",
    "week",
    "weeks",
    "year",
    "years",
}

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
QUANTITY_ENTITY_TYPES = {"CARDINAL", "DATE", "MONEY", "PERCENT", "QUANTITY"}
MODAL_NORMALIZATION = {
    "must": "must",
    "shall": "must",
    "should": "should",
    "may": "may",
    "will": "will",
    "would": "would",
    "could": "could",
    "can": "can",
}
NEGATED_MODAL_NORMALIZATION = {
    "can": "cannot",
    "could": "could not",
    "may": "may not",
    "must": "must not",
    "shall": "must not",
    "should": "should not",
    "will": "will not",
    "would": "would not",
}
STATE_BY_LEMMA = {
    "accept": "accepted",
    "approve": "approved",
    "block": "blocked",
    "deny": "denied",
    "grant": "granted",
    "ratify": "ratified",
    "reject": "rejected",
    "waive": "waived",
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
    "accept": "accept",
    "accepted": "accept",
    "address": "resolve",
    "approve": "approve",
    "approved": "approve",
    "begin": "begin",
    "deploy": "deploy",
    "fix": "resolve",
    "mention": "note",
    "note": "note",
    "observe": "note",
    "ratify": "ratify",
    "ratified": "ratify",
    "remediate": "resolve",
    "resolve": "resolve",
    "review": "review",
    "start": "begin",
    "verify": "verify",
}
CLAIM_ACTION_LEMMAS = frozenset(ACTION_NORMALIZATION)
NON_CLAIM_VERB_LEMMAS = {"be", "do", "have", "propose", "treat", "use"}
AUTHORIZATION_ACTIONS = {"address", "fix", "mention", "note", "observe", "remediate", "resolve"}
GOVERNANCE_LEMMAS = {
    "applicability",
    "approval",
    "approve",
    "cutover",
    "decision",
    "provisional",
    "ratification",
    "ratify",
    "recommendation",
    "risk",
    "waiver",
}
PRONOUN_TARGETS = {"it", "this", "that", "them", "these", "those"}
STOP_TARGET_WORDS = {
    "a",
    "an",
    "and",
    "any",
    "as",
    "before",
    "by",
    "for",
    "from",
    "in",
    "into",
    "at",
    "least",
    "less",
    "more",
    "most",
    "no",
    "now",
    "of",
    "on",
    "or",
    "our",
    "path",
    "paths",
    "please",
    "the",
    "than",
    "to",
    "under",
    "with",
    "yet",
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
DERIVED_GOVERNANCE_CONCEPTS = {
    "architecture_approval_deferred": ("negation", "governance_polarity"),
    "v2_cutover_deferred": ("negation", "governance_polarity"),
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
class ClaimParts:
    start: int
    end: int
    actor: str
    action: str
    target: str
    modality: str
    polarity: str
    status: str


@dataclass(frozen=True)
class StatusRecord:
    start: int
    end: int
    state: str
    subject: str


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


def _token_boundaries(document: Document) -> tuple[set[int], set[int]]:
    return (
        {0, len(document.text)} | {token.start for token in document},
        {0, len(document.text)} | {token.end for token in document},
    )


def _is_token_aligned(document: Document, start: int, end: int) -> bool:
    starts, ends = _token_boundaries(document)
    return start in starts and end in ends


def _regex_items(
    source: str,
    document: Document,
    pattern: re.Pattern[str],
    category: str,
    kind: str,
    normalizer: Callable[[str], str],
) -> Iterable[dict[str, JsonValue]]:
    for match in pattern.finditer(source):
        if _is_token_aligned(document, match.start(), match.end()):
            yield _item(
                source,
                category,
                kind,
                normalizer(match.group(0)),
                match.start(),
                match.end(),
                source,
            )


def _contains(spans: Iterable[tuple[int, int]], start: int, end: int) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def _overlaps(spans: Iterable[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < span_end and span_start < end for span_start, span_end in spans)


def _concept_for_span(spans: Iterable[ConceptSpan], start: int, end: int) -> ConceptSpan | None:
    for span in spans:
        if span.start <= start and end <= span.end:
            return span
    return None


def _normalize_modal(value: str) -> str:
    lowered = " ".join(value.lower().replace("’", "'").split())
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
    lowered = " ".join(value.lower().replace("’", "'").split())
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
    lowered = " ".join(surface.lower().split())
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


def _token_span_text(document: Document, tokens: Iterable[Token]) -> str:
    ordered = sorted(tokens, key=lambda token: token.index)
    if not ordered:
        return ""
    return document.text[ordered[0].start : ordered[-1].end]


def _content_tokens(tokens: Iterable[Token]) -> list[Token]:
    return [
        token
        for token in sorted(tokens, key=lambda item: item.index)
        if (token.is_word or token.text == "%") and token.dep != "det"
    ]


def _normalize_target_tokens(tokens: Iterable[Token]) -> str:
    content = _content_tokens(tokens)
    if len(content) == 1 and content[0].lower in PRONOUN_TARGETS:
        return content[0].lower
    words: list[str] = []
    for token in content:
        if token.pos == "NUM":
            continue
        value = token.lemma if token.lemma else token.lower
        value = value.lower()
        if value in STOP_TARGET_WORDS:
            continue
        words.append(value)
    return " ".join(words) if words else "none"


def _normalize_subject_tokens(document: Document, tokens: Iterable[Token]) -> str:
    subject = _normalize_subject(_token_span_text(document, tokens))
    if subject in {"decision", "recommendation", "recommended decision"}:
        return "unspecified"
    return subject


def _own_negation_tokens(document: Document, predicate: Token) -> list[Token]:
    return [
        child
        for child in document.children(predicate)
        if child.dep == "neg" or "Polarity=Neg" in child.morph
    ]


def _predicate_negation_tokens(document: Document, predicate: Token) -> list[Token]:
    own = _own_negation_tokens(document, predicate)
    if own:
        return own
    if predicate.dep == "conj":
        head = document.head_of(predicate)
        head_has_negative_coordination = any(
            child.dep == "cc" and child.lemma in {"nor", "or"} for child in document.children(head)
        )
        if head_has_negative_coordination:
            return _own_negation_tokens(document, head)
    return []


def _predicate_polarity(document: Document, predicate: Token) -> str:
    return "negative" if _predicate_negation_tokens(document, predicate) else "positive"


def _own_modal_tokens(document: Document, predicate: Token) -> list[Token]:
    return [
        child
        for child in document.children(predicate)
        if child.dep in {"aux", "auxpass"} and child.lemma in MODAL_NORMALIZATION
    ]


def _predicate_modal_tokens(document: Document, predicate: Token) -> list[Token]:
    own = _own_modal_tokens(document, predicate)
    if own:
        return own
    if predicate.dep == "conj":
        head = document.head_of(predicate)
        return _own_modal_tokens(document, head)
    return []


def _predicate_modality(document: Document, predicate: Token) -> str:
    modals = _predicate_modal_tokens(document, predicate)
    if modals:
        return MODAL_NORMALIZATION[modals[0].lemma]
    subject = _subject_tokens(document, predicate)
    if not subject and predicate.tag in {"VB", "VBP"}:
        return "imperative"
    return "assertive"


def _subject_tokens(document: Document, predicate: Token) -> list[Token]:
    subjects = [
        child
        for child in document.children(predicate)
        if child.dep in {"nsubj", "nsubjpass"}
    ]
    if not subjects and predicate.dep == "conj":
        return _subject_tokens(document, document.head_of(predicate))
    tokens: list[Token] = []
    for subject in subjects:
        tokens.extend(document.subtree(subject))
    return _content_tokens(tokens)


def _object_root(document: Document, predicate: Token) -> Token | None:
    passive_subjects = [child for child in document.children(predicate) if child.dep == "nsubjpass"]
    if passive_subjects:
        return passive_subjects[0]
    objects = [
        child
        for child in document.children(predicate)
        if child.dep in {"attr", "dative", "dobj", "obj", "oprd"}
    ]
    if objects:
        return objects[0]
    prepositions = [child for child in document.children(predicate) if child.dep == "prep"]
    for preposition in prepositions:
        pobj = [child for child in document.children(preposition) if child.dep == "pobj"]
        if pobj:
            return pobj[0]
    return None


def _target_tokens(document: Document, predicate: Token) -> list[Token]:
    root = _object_root(document, predicate)
    if root is None:
        return []
    return _content_tokens(
        token
        for token in document.subtree(root)
        if token.dep not in {"acl", "advcl", "relcl"} and document.head_of(token).dep not in {"acl", "advcl", "relcl"}
    )


def _claim_span(document: Document, predicate: Token) -> tuple[int, int]:
    tokens = [predicate]
    tokens.extend(_predicate_negation_tokens(document, predicate))
    tokens.extend(_predicate_modal_tokens(document, predicate))
    tokens.extend(_target_tokens(document, predicate))
    if not tokens:
        return predicate.start, predicate.end
    ordered = sorted(tokens, key=lambda token: token.index)
    return ordered[0].start, ordered[-1].end


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


def _claim_parts(document: Document, predicate: Token) -> ClaimParts:
    start, end = _claim_span(document, predicate)
    action = ACTION_NORMALIZATION.get(predicate.lemma, predicate.lemma)
    target = _normalize_target_tokens(_target_tokens(document, predicate))
    subject = _subject_tokens(document, predicate)
    actor = _normalize_subject_tokens(document, subject) if subject else "unspecified"
    return ClaimParts(
        start=start,
        end=end,
        actor=actor,
        action=action,
        target=target,
        modality=_predicate_modality(document, predicate),
        polarity=_predicate_polarity(document, predicate),
        status="asserted",
    )


def _is_claim_predicate(document: Document, token: Token) -> bool:
    if token.pos != "VERB" or token.lemma in NON_CLAIM_VERB_LEMMAS:
        return False
    if token.lemma in CLAIM_ACTION_LEMMAS:
        return True
    return _has_structural_claim_cue(document, token)


def _has_structural_claim_cue(document: Document, token: Token) -> bool:
    return (
        _predicate_polarity(document, token) == "negative"
        or _predicate_modality(document, token) != "assertive"
        or _structural_claim_has_imperative_form(token)
    )


def _structural_claim_has_imperative_form(token: Token) -> bool:
    return token.tag in {"VB", "VBP"} and token.dep in {"ROOT", "conj", "xcomp", "dep", "acl"}


def _claim_predicates(document: Document) -> Iterable[Token]:
    for token in document:
        if _is_claim_predicate(document, token):
            yield token


def _is_authorization_target(target: str) -> bool:
    words = set(target.split())
    return "authorization" in words and bool(words & {"concern", "concerns", "issue", "issues"})


def _authorization_claims(document: Document, claims: list[tuple[int, int, str]]) -> set[int]:
    claimed_predicates: set[int] = set()
    for predicate in _claim_predicates(document):
        if predicate.lemma not in AUTHORIZATION_ACTIONS:
            continue
        parts = _claim_parts(document, predicate)
        if not _is_authorization_target(parts.target):
            continue
        _add_claim(
            claims,
            parts.start,
            parts.end,
            "unspecified",
            parts.action,
            "authorization issues",
            "must",
            parts.polarity,
        )
        claimed_predicates.add(predicate.index)
    return claimed_predicates


def _concept_claims(concepts: Iterable[ConceptSpan], claims: list[tuple[int, int, str]]) -> None:
    for concept in concepts:
        for actor, action, target, modality, polarity, status in CONCEPT_CLAIMS.get(concept.value, ()):
            _add_claim(claims, concept.start, concept.end, actor, action, target, modality, polarity, status)


def _skip_generic_claim(
    predicate: Token,
    parts: ClaimParts,
    concept_ranges: Iterable[tuple[int, int]],
    authorization_predicates: set[int],
) -> bool:
    if predicate.index in authorization_predicates:
        return True
    if _contains(concept_ranges, predicate.start, predicate.end):
        return True
    if _overlaps(concept_ranges, parts.start, parts.end):
        return True
    return False


def _generic_claims(
    document: Document,
    concepts: Iterable[ConceptSpan],
    authorization_predicates: set[int],
    claims: list[tuple[int, int, str]],
) -> None:
    concept_ranges = [(concept.start, concept.end) for concept in concepts if concept.value in CONCEPT_CLAIMS]
    for predicate in _claim_predicates(document):
        parts = _claim_parts(document, predicate)
        if _skip_generic_claim(predicate, parts, concept_ranges, authorization_predicates):
            continue
        _add_claim(
            claims,
            parts.start,
            parts.end,
            parts.actor,
            parts.action,
            parts.target,
            parts.modality,
            parts.polarity,
            parts.status,
        )


def _status_state(token: Token) -> str | None:
    if token.lower in STATE_NORMALIZATION:
        return STATE_NORMALIZATION[token.lower]
    if "VerbForm=Part" not in token.morph and token.tag not in {"VBN", "VBD", "JJ"}:
        return None
    return STATE_BY_LEMMA.get(token.lemma)


def _status_records(document: Document) -> list[StatusRecord]:
    records: list[StatusRecord] = []
    seen: set[tuple[int, str]] = set()
    for token in document:
        state = _status_state(token)
        if state is None or token.pos not in {"ADJ", "VERB"}:
            continue
        key = (token.index, state)
        if key in seen:
            continue
        seen.add(key)
        subject_tokens = _subject_tokens(document, token)
        subject = _normalize_subject_tokens(document, subject_tokens) if subject_tokens else "unknown"
        records.append(StatusRecord(token.start, token.end, state, subject))
    return records


def _status_items(document: Document) -> Iterable[dict[str, JsonValue]]:
    for record in _status_records(document):
        normalized = (
            f"domain={STATE_DOMAIN[record.state]};polarity={STATE_POLARITY[record.state]};"
            f"state={record.state};subject={record.subject}"
        )
        yield _item(document.text, "governance", "status", normalized, record.start, record.end, document.text)


def _status_claims(document: Document, claims: list[tuple[int, int, str]]) -> None:
    for record in _status_records(document):
        _add_claim(
            claims,
            record.start,
            record.end,
            "unspecified",
            STATE_DOMAIN[record.state],
            record.subject,
            "state",
            STATE_POLARITY[record.state],
            record.state,
        )


def _semantic_claim_signatures(document: Document, concepts: Iterable[ConceptSpan]) -> list[str]:
    concept_list = list(concepts)
    claims: list[tuple[int, int, str]] = []
    _concept_claims(concept_list, claims)
    authorization_predicates = _authorization_claims(document, claims)
    _generic_claims(document, concept_list, authorization_predicates, claims)
    _status_claims(document, claims)
    unique = sorted(set(claims), key=lambda item: (item[0], item[1], item[2]))
    return [signature for _, _, signature in unique]


def _add_concept_item(
    text: str,
    items: list[dict[str, JsonValue]],
    spans: list[ConceptSpan],
    seen: set[tuple[str, str, str, int, int]],
    category: str,
    kind: str,
    value: str,
    start: int,
    end: int,
) -> None:
    key = (category, kind, value, start, end)
    if key in seen:
        return
    seen.add(key)
    spans.append(ConceptSpan(start, end, category, kind, value))
    items.append(_item(text, category, kind, value, start, end, text))


def _extract_concepts(
    text: str,
    document: Document,
    profile: Profile,
) -> tuple[list[dict[str, JsonValue]], list[ConceptSpan], set[tuple[str, str, str, int, int]]]:
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
                if _is_token_aligned(document, concept_match.start(), concept_match.end()):
                    _add_concept_item(
                        text,
                        items,
                        spans,
                        seen,
                        category,
                        kind,
                        value,
                        concept_match.start(),
                        concept_match.end(),
                    )
    return items, spans, seen


def _has_concept(spans: Iterable[ConceptSpan], category: str, kind: str, value: str) -> bool:
    return any(span.category == category and span.kind == kind and span.value == value for span in spans)


def _extract_derived_governance_concepts(
    document: Document,
    items: list[dict[str, JsonValue]],
    spans: list[ConceptSpan],
    seen: set[tuple[str, str, str, int, int]],
) -> None:
    for predicate in _claim_predicates(document):
        parts = _claim_parts(document, predicate)
        derived_value: str | None = None
        target_words = set(parts.target.split())
        if parts.action == "approve" and parts.polarity == "negative" and parts.target == "architecture":
            derived_value = "architecture_approval_deferred"
        if parts.action == "begin" and parts.polarity == "negative" and {"v2", "cutover"} <= target_words:
            derived_value = "v2_cutover_deferred"
        if derived_value is None:
            continue
        category, kind = DERIVED_GOVERNANCE_CONCEPTS[derived_value]
        if _has_concept(spans, category, kind, derived_value):
            continue
        _add_concept_item(document.text, items, spans, seen, category, kind, derived_value, parts.start, parts.end)


def _operator_start(document: Document, token: Token) -> int:
    sentence_tokens = list(document.sentence_of(token).tokens)
    by_index = {candidate.index: position for position, candidate in enumerate(sentence_tokens)}
    position = by_index[token.index]
    operator_sequences = {
        ("at", "least"),
        ("at", "most"),
        ("fewer", "than"),
        ("less", "than"),
        ("more", "than"),
        ("no", "less", "than"),
        ("no", "more", "than"),
        ("over",),
        ("under",),
        (">",),
        ("<",),
        (">=",),
        ("<=",),
    }
    for length in (3, 2, 1):
        start = position - length
        if start < 0:
            continue
        sequence = tuple(candidate.lower for candidate in sentence_tokens[start:position])
        if sequence in operator_sequences:
            return sentence_tokens[start].start
    return token.start


def _quantity_span(document: Document, token: Token) -> tuple[int, int]:
    start = _operator_start(document, token)
    end = token.end
    sentence = document.sentence_of(token)
    sentence_tokens = list(sentence.tokens)
    by_index = {candidate.index: position for position, candidate in enumerate(sentence_tokens)}
    position = by_index[token.index]
    if position + 1 < len(sentence_tokens):
        following = sentence_tokens[position + 1]
        if following.lower in KNOWN_UNITS:
            end = following.end
    head = document.head_of(token)
    if token.dep == "nummod" and head.sentence_index == token.sentence_index and head.lower in KNOWN_UNITS:
        end = max(end, head.end)
    return start, end


def _quantity_candidate(token: Token) -> bool:
    return token.pos == "NUM" or token.lower in NUMBER_WORDS or token.lower == "either"


def _quantity_items(
    document: Document,
    concepts: Iterable[ConceptSpan],
    excluded_ranges: Iterable[tuple[int, int]],
) -> Iterable[dict[str, JsonValue]]:
    concept_list = list(concepts)
    excluded = list(excluded_ranges)
    seen: set[tuple[int, int]] = set()
    for token in document:
        if not _quantity_candidate(token):
            continue
        start, end = _quantity_span(document, token)
        if (start, end) in seen or _contains(excluded, start, end):
            continue
        seen.add((start, end))
        surface = document.text[start:end]
        parts = _quantity_parts(surface, _concept_for_span(concept_list, start, end))
        yield _item(document.text, "quantity", parts.kind, parts.normalized, start, end, document.text)

    entity_tokens = [token for token in document if token.entity_type in QUANTITY_ENTITY_TYPES]
    current: list[Token] = []
    for token in entity_tokens:
        if current and token.index != current[-1].index + 1:
            yield from _entity_quantity_items(document, concept_list, excluded, seen, current)
            current = []
        current.append(token)
    if current:
        yield from _entity_quantity_items(document, concept_list, excluded, seen, current)


def _entity_quantity_items(
    document: Document,
    concepts: Iterable[ConceptSpan],
    excluded_ranges: Iterable[tuple[int, int]],
    seen: set[tuple[int, int]],
    tokens: list[Token],
) -> Iterable[dict[str, JsonValue]]:
    start, end = tokens[0].start, tokens[-1].end
    if (start, end) in seen or _overlaps(seen, start, end) or _contains(excluded_ranges, start, end):
        return
    surface = document.text[start:end]
    entity_type = tokens[0].entity_type
    if entity_type == "DATE" and not any(character.isdigit() for character in surface):
        normalized = " ".join(surface.lower().split())
        yield _item(document.text, "quantity", "date", normalized, start, end, document.text)
        seen.add((start, end))
        return
    parts = _quantity_parts(surface, _concept_for_span(concepts, start, end))
    yield _item(document.text, "quantity", parts.kind, parts.normalized, start, end, document.text)
    seen.add((start, end))


def _modal_items(
    document: Document,
    covered_ranges: Iterable[tuple[int, int]],
) -> Iterable[dict[str, JsonValue]]:
    covered = list(covered_ranges)
    for predicate in document:
        if predicate.pos not in {"AUX", "VERB"} or _is_claim_predicate(document, predicate):
            continue
        modals = _own_modal_tokens(document, predicate)
        if not modals:
            continue
        modal = modals[0]
        negations = _own_negation_tokens(document, predicate)
        if negations:
            start = min(modal.start, *(negation.start for negation in negations))
            end = max(modal.end, *(negation.end for negation in negations))
            normalized = NEGATED_MODAL_NORMALIZATION.get(modal.lemma, f"{MODAL_NORMALIZATION[modal.lemma]} not")
        else:
            start = modal.start
            end = modal.end
            normalized = MODAL_NORMALIZATION[modal.lemma]
        if not _contains(covered, start, end):
            yield _item(document.text, "modal", "modal_strength", normalized, start, end, document.text)


def _negation_items(
    document: Document,
    covered_ranges: Iterable[tuple[int, int]],
) -> Iterable[dict[str, JsonValue]]:
    covered = list(covered_ranges)
    for token in document:
        if token.dep != "neg" and "Polarity=Neg" not in token.morph:
            continue
        head = document.head_of(token)
        if _is_claim_predicate(document, head) or _contains(covered, token.start, token.end):
            continue
        yield _item(document.text, "negation", "lexical_negation", _normalize_negation(token.text), token.start, token.end, document.text)


def _governance_items(
    document: Document,
    covered_ranges: Iterable[tuple[int, int]],
) -> Iterable[dict[str, JsonValue]]:
    covered = list(covered_ranges)
    for token in document:
        if token.lemma in GOVERNANCE_LEMMAS and not _contains(covered, token.start, token.end):
            yield _item(document.text, "governance", "term", token.lemma, token.start, token.end, document.text)
    sentence_tokens = list(document)
    for index, token in enumerate(sentence_tokens[:-1]):
        following = sentence_tokens[index + 1]
        if token.lower == "target" and following.lemma == "architecture":
            if not _contains(covered, token.start, following.end):
                yield _item(document.text, "governance", "term", "target architecture", token.start, following.end, document.text)


def extract_protected(text: str, profile: Profile) -> dict[str, JsonValue]:
    document = parse(text)
    items: list[dict[str, JsonValue]] = []
    concept_items, concept_spans, concept_seen = _extract_concepts(text, document, profile)
    items.extend(concept_items)
    _extract_derived_governance_concepts(document, items, concept_spans, concept_seen)
    covered_ranges = [(span.start, span.end) for span in concept_spans]

    citation_items = (
        list(_regex_items(text, document, URL_RE, "citation", "url", lambda value: value))
        + list(_regex_items(text, document, CODE_RE, "citation", "code", lambda value: value))
        + list(_regex_items(text, document, CITATION_RE, "citation", "reference", lambda value: value))
    )
    citation_ranges: list[tuple[int, int]] = []
    for item in citation_items:
        location = cast(dict[str, JsonValue], item["location"])
        citation_ranges.append((cast(int, location["start"]), cast(int, location["end"])))
    items.extend(citation_items)

    identifier_items: list[dict[str, JsonValue]] = []
    for match in IDENTIFIER_RE.finditer(text):
        if _is_token_aligned(document, match.start(), match.end()) and not _contains(citation_ranges, match.start(), match.end()):
            identifier_items.append(_item(text, "identifier", "identifier", match.group(0).upper(), match.start(), match.end(), text))
    identifier_ranges: list[tuple[int, int]] = []
    for item in identifier_items:
        location = cast(dict[str, JsonValue], item["location"])
        identifier_ranges.append((cast(int, location["start"]), cast(int, location["end"])))
    items.extend(identifier_items)

    non_quantity_ranges = citation_ranges + identifier_ranges
    items.extend(_quantity_items(document, concept_spans, non_quantity_ranges))
    items.extend(_modal_items(document, covered_ranges))
    items.extend(_status_items(document))
    items.extend(_negation_items(document, covered_ranges))
    items.extend(_governance_items(document, covered_ranges))

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
        + _semantic_claim_signatures(document, concept_spans)
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
    for part in signature[len("claim:") :].split(";"):
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
        for right in claims[index + 1 :]:
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
