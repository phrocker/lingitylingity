"""Protected-element extraction and deterministic comparison."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, cast

from lingity.models import JsonValue
from lingity.morphology import canonical_action, canonical_action_info
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

# Words removable from a target without altering what the target denotes.
#
# Every entry must be semantically inert. An earlier revision of this set also
# held "no", "yet", "before", "any", "least", "less", "more" and "most", which
# made the gate report these pairs as equivalent:
#
#   "Grant access to no users."            ~ "Grant access to users."
#   "Do not deploy the gateway yet."       ~ "Do not deploy the gateway."
#   "Close the findings before the review." ~ "... after the review."
#
# That is, it erased a negation, collapsed a deferral into a permanent
# prohibition, and let an ordering constraint be inverted. Filtering by word
# identity cannot distinguish a semantically inert determiner from a negative
# one, so those roles are now handled structurally: negative determiners feed
# claim polarity, deferral markers feed claim status, and ordering prepositions
# are captured as explicit relations. Do not add a word here unless removing it
# provably cannot change what the phrase denotes.
STOP_TARGET_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "our",
    "please",
    "the",
    "to",
    "with",
}

# Determiners that negate what they introduce ("no users", "neither path").
# These must never be silently dropped from a target.
NEGATIVE_DETERMINERS = {"no", "neither", "none"}

# Markers that make an obligation a deferral rather than a prohibition.
DEFERRAL_MARKERS = {"yet"}

# Verbs that postpone their complement rather than asserting it. These are
# operators, not actions: the governance content lives in what they scope over.
DEFERRAL_OPERATORS = {
    "defer",
    "postpone",
    "delay",
    "suspend",
    "pause",
    "shelve",
    "deferral",
    "postponement",
    "suspension",
}

# Deferral verbs that only defer when carrying a particle, so that "hold the
# line" is not read as deferring a line.
DEFERRAL_PARTICLE_OPERATORS = {"hold", "put", "push"}

# Prepositions and subordinators that impose an ordering between events. The
# distinction between them is propositional content, not phrasing.
ORDERING_MARKERS = {
    "before": "before",
    "prior": "before",
    "until": "before",
    "after": "after",
    "following": "after",
    "once": "after",
}

# Dependency labels that open a new clause. Crossing one means leaving the
# current predicate's argument structure.
_CLAUSE_BOUNDARY_DEPS = {"acl", "advcl", "ccomp", "csubj", "pcomp", "relcl", "xcomp"}

# Guard against pathological or cyclic head chains while walking upwards.
_MAX_HEAD_WALK = 12

# Parts of speech that make a sentence carry assertable content. A sentence
# with none of these (a heading fragment, a bare list marker) is not something
# the gate needs to have understood.
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "NUM"}


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
    if lowered == "both":
        return "2"
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
        # "either" selects one of two; it is a quantifier, not a count.
        return QuantityParts("quantifier", "either")
    if lowered == "both":
        # "both X" and "the two X" name the same cardinality.
        return QuantityParts("count", "2")

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
        if (token.is_word or token.text == "%")
        and (token.dep != "det" or _is_numeral_determiner(token))
    ]


def _is_numeral_determiner(token: Token) -> bool:
    """A determiner that carries a count is propositional, not syntactic sugar.

    "both findings" states a cardinality the way "two findings" does, so it must
    survive into the target even though the parser labels it a determiner.
    """

    return token.lower == "both" or token.lower in NUMBER_WORDS


def _normalize_target_tokens(tokens: Iterable[Token]) -> str:
    content = _content_tokens(tokens)
    if len(content) == 1 and content[0].lower in PRONOUN_TARGETS:
        return content[0].lower
    words: list[str] = []
    for token in content:
        if token.pos == "NUM" or token.lower == "both":
            # Numerals are propositional content: "two hypotheses" and "three
            # hypotheses" are different claims. Canonicalise the surface form so
            # that "two", "both" and "2" agree, but never drop the value.
            words.append(_normalize_number(token.lower))
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
    negations = [
        child
        for child in document.children(predicate)
        if child.dep == "neg" or "Polarity=Neg" in child.morph
    ]
    negations.extend(_negative_determiner_tokens(document, predicate))
    return negations


def _negative_determiner_tokens(document: Document, predicate: Token) -> list[Token]:
    """Negation carried by a determiner inside the predicate's own arguments.

    "Grant access to no users" negates the claim as surely as "do not grant
    access to users" does, but spaCy attaches ``no`` as a determiner rather
    than as ``neg``, so it is invisible to the dependency-label check above.
    Only the predicate's own arguments are inspected; a negative determiner
    inside a subordinate clause belongs to that clause's claim, not this one.
    """

    root = _object_root(document, predicate)
    if root is None:
        return []
    return [
        token
        for token in document.subtree(root)
        if token.dep == "det"
        and token.lower in NEGATIVE_DETERMINERS
        and _governing_predicate(document, token) is predicate
    ]


def _governing_predicate(document: Document, token: Token) -> Token | None:
    """Walk up to the nearest verbal head, stopping at a clause boundary."""

    current = token
    for _ in range(_MAX_HEAD_WALK):
        head = document.head_of(current)
        if head is current:
            return None
        if head.pos in {"VERB", "AUX"}:
            return head
        if head.dep in _CLAUSE_BOUNDARY_DEPS:
            return None
        current = head
    return None


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
        # An imperative is an obligation on the reader. Recording it as its own
        # modality would make "Do not ratify X" disagree with "X must not be
        # ratified", which are the same instruction.
        return "must"
    return "assertive"


def _subject_tokens(document: Document, predicate: Token) -> list[Token]:
    passive_subjects = [
        child for child in document.children(predicate) if child.dep == "nsubjpass"
    ]
    if passive_subjects:
        # In a passive clause the grammatical subject is the patient, not the
        # actor. Reading it as the actor inverts the sentence: "the gaps must
        # be resolved by the platform team" would claim the gaps do the
        # resolving. The actor is the by-phrase, and when there is none the
        # actor is genuinely absent -- that absence is precisely what
        # LING-AGENCY-001 reports, so it must not be papered over.
        return _passive_agent_tokens(document, predicate)
    subjects = [child for child in document.children(predicate) if child.dep == "nsubj"]
    subjects = [
        subject for subject in subjects if not _is_discourse_label(document, subject)
    ]
    if not subjects and predicate.dep == "conj":
        return _subject_tokens(document, document.head_of(predicate))
    tokens: list[Token] = []
    for subject in subjects:
        tokens.extend(document.subtree(subject))
    return _content_tokens(tokens)


def _is_discourse_label(document: Document, subject: Token) -> bool:
    """Is this nominal a heading such as "Recommendation:" rather than a subject?

    A colon-terminated nominal opening a sentence labels the statement; it does
    not perform the action. spaCy attaches it as nsubj, which both invents an
    actor and hides the imperative behind a false subject -- so "Recommendation:
    Do not approve X" would be read as the recommendation doing the approving.
    """

    sentence = document.sentence_of(subject)
    subtree = document.subtree(subject)
    first = min(token.index for token in subtree)
    if first != min(token.index for token in sentence.tokens):
        return False
    if any(token.text == ":" for token in subtree):
        return True
    following = subject.index + 1
    try:
        separator = document.token(following)
    except (IndexError, KeyError):
        return False
    return separator.text == ":"


def _passive_agent_tokens(document: Document, predicate: Token) -> list[Token]:
    tokens: list[Token] = []
    for child in document.children(predicate):
        if child.dep != "agent":
            continue
        for grandchild in document.children(child):
            if grandchild.dep == "pobj":
                tokens.extend(document.subtree(grandchild))
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
    # An ordering clause is dropped only when _ordering_relations reports it, so
    # the content is represented as an order:sequence rather than lost. Trailing
    # markers attach low -- "before" lands on the nearest noun inside the object
    # -- so without this the same sentence produced a target that swallowed the
    # clause when trailing and one that did not when fronted.
    ordered_away = {
        index for relation in _ordering_relations(document) for index in relation.owned
    }
    # Material belonging to a nested clause is excluded so that targets do not
    # swallow the rest of the sentence. The predicate's own object is never
    # nested material, though: when a discourse label makes the predicate
    # itself an "acl", excluding its object would erase what the directive is
    # about.
    return _content_tokens(
        token
        for token in document.subtree(root)
        if token.dep not in {"acl", "advcl", "relcl"}
        and token.index not in ordered_away
        and (
            document.head_of(token).index == predicate.index
            or document.head_of(token).dep not in {"acl", "advcl", "relcl"}
        )
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
    deferral = _deferral_parts(document, predicate)
    if deferral is not None:
        return deferral
    start, end = _claim_span(document, predicate)
    action = _resolve_action(predicate)
    target = _normalize_target_tokens(_target_tokens(document, predicate))
    subject = _subject_tokens(document, predicate)
    actor = _normalize_subject_tokens(document, subject) if subject else "unspecified"
    polarity = _predicate_polarity(document, predicate)
    return ClaimParts(
        start=start,
        end=end,
        actor=actor,
        action=action,
        target=target,
        modality=_predicate_modality(document, predicate),
        polarity=polarity,
        status=_predicate_status(document, predicate, polarity),
    )


def _resolve_action(token: Token) -> str:
    """Canonical action key for a predicate token.

    Declared synonyms win so that curated governance distinctions survive;
    otherwise the shared verb/nominalization key from WordNet is used, so that
    "ratify" and "architecture ratification" agree.
    """

    lemma = token.lemma.lower() if token.lemma else token.lower
    if lemma in ACTION_NORMALIZATION:
        return ACTION_NORMALIZATION[lemma]
    return canonical_action(lemma)


def _nominal_action(token: Token) -> str | None:
    """Action named by a nominalization, or None if the noun names no action.

    "ratification" names ratifying; "architecture" names no action. WordNet's
    derivational links decide, so this generalises to vocabulary the profile has
    never seen rather than to a list we maintain.
    """

    if token.pos not in {"NOUN", "PROPN"}:
        return None
    lemma = token.lemma.lower() if token.lemma else token.lower
    if lemma in ACTION_NORMALIZATION:
        return ACTION_NORMALIZATION[lemma]
    info = canonical_action_info(lemma)
    if not info.wordnet_component or lemma in info.wordnet_component:
        return None
    return info.key


def _is_deferral_operator(document: Document, token: Token) -> bool:
    if token.pos not in {"VERB", "NOUN"}:
        return False
    lemma = token.lemma.lower() if token.lemma else token.lower
    if lemma in DEFERRAL_PARTICLE_OPERATORS:
        return any(
            child.lower in {"off", "back"} and child.dep in {"prt", "advmod"}
            for child in document.children(token)
        )
    return lemma in DEFERRAL_OPERATORS


def _deferral_parts(document: Document, predicate: Token) -> ClaimParts | None:
    parts = _deferral_parts_list(document, predicate)
    return parts[0] if parts else None


def _deferral_parts_list(document: Document, predicate: Token) -> list[ClaimParts]:
    """Rewrite a deferral into a negated, deferred claim about its complement.

    "Defer architecture ratification", "propose deferring architecture
    ratification" and "do not ratify the architecture yet" state the same
    governance position. Representing the deferral operator as an action in its
    own right would make those three disagree, so the operator is unwound onto
    the thing being deferred. Deferring an event that has no verbal reading
    ("defer the V2 cutover") is deferring its commencement.
    """

    if not _is_deferral_operator(document, predicate):
        return []
    complements = _deferral_complements(document, predicate)
    if not complements:
        return []
    start, end = _claim_span(document, predicate)
    subject = _subject_tokens(document, predicate)
    actor = _normalize_subject_tokens(document, subject) if subject else "unspecified"

    parts: list[ClaimParts] = []
    for root in complements:
        if root.pos in {"VERB", "AUX"}:
            action = _resolve_action(root)
            target = _normalize_target_tokens(_target_tokens(document, root))
        else:
            nominal = _nominal_action(root)
            modifiers = [
                token
                for token in document.subtree(root)
                if token.index != root.index
                and token.index != predicate.index
                and token.dep not in _CLAUSE_BOUNDARY_DEPS
            ]
            if nominal is not None:
                action = nominal
                target = _normalize_target_tokens(modifiers)
            else:
                action = ACTION_NORMALIZATION.get("begin", "begin")
                target = _normalize_target_tokens(document.subtree(root))
        parts.append(
            ClaimParts(
                start=start,
                end=end,
                actor=actor,
                action=action,
                target=target,
                modality="must",
                polarity="negative",
                status="deferred",
            )
        )
    return parts


def _deferral_complements(document: Document, predicate: Token) -> list[Token]:
    """What is being deferred.

    Coordination is read from both the complement and the operator's head:
    spaCy attaches the second conjunct of "propose deferring X and Y" to
    "propose" rather than to "X", so reading only the complement's own
    conjuncts silently loses Y — and losing a governance directive is exactly
    the failure this gate exists to prevent.
    """

    roots: list[Token] = []
    for child in document.children(predicate):
        if child.dep in {"dobj", "obj", "xcomp", "ccomp", "pobj"}:
            roots.append(child)
        elif child.dep == "prep":
            roots.extend(
                grandchild
                for grandchild in document.children(child)
                if grandchild.dep == "pobj"
            )
    if not roots:
        # Sentence-initial imperatives are frequently mis-tagged as nominal
        # modifiers of their own object ("Defer ratification" parses with
        # "Defer" as a compound under "ratification"). The complement is then
        # the operator's head. Ignoring this would drop the directive entirely,
        # so the inversion is repaired rather than tolerated.
        head = document.head_of(predicate)
        if head is not predicate and head.pos in {"NOUN", "PROPN"}:
            roots = [head]
    if not roots:
        return []
    conjuncts: list[Token] = []
    for root in roots:
        conjuncts.extend(
            child for child in document.children(root) if child.dep == "conj"
        )
    head = document.head_of(predicate)
    if head is not predicate:
        conjuncts.extend(
            child
            for child in document.children(head)
            if child.dep == "conj"
            and child.pos in {"NOUN", "PROPN"}
            and child.index > predicate.index
        )
    return roots + conjuncts


def _predicate_status(document: Document, predicate: Token, polarity: str) -> str:
    """Distinguish "not yet" from "not ever".

    A deferral marker turns a prohibition into a postponement. Treating them
    alike would let a rewrite convert "do not deploy yet" into "do not deploy"
    and pass the gate.
    """

    if polarity == "negative" and _has_deferral_marker(document, predicate):
        return "deferred"
    return "asserted"


def _has_deferral_marker(document: Document, predicate: Token) -> bool:
    if _own_deferral_marker(document, predicate):
        return True
    # A sentence-final "yet" scopes over an entire negated coordination:
    # "do not approve the architecture or begin the cutover yet" defers both.
    # Reading it as marking only the conjunct it attaches to would treat the
    # first directive as a permanent prohibition.
    root = predicate
    guard = 0
    while root.dep == "conj" and guard < _MAX_HEAD_WALK:
        head = document.head_of(root)
        if head is root:
            break
        root = head
        guard += 1
    coordination = [root] + [
        child for child in document.children(root) if child.dep == "conj"
    ]
    if predicate.index not in {member.index for member in coordination}:
        return False
    return any(_own_deferral_marker(document, member) for member in coordination)


def _own_deferral_marker(document: Document, predicate: Token) -> bool:
    return any(
        child.lower in DEFERRAL_MARKERS and child.dep in {"advmod", "npadvmod"}
        for child in document.children(predicate)
    )


def _is_claim_predicate(document: Document, token: Token) -> bool:
    if token.pos != "VERB" or token.lemma in NON_CLAIM_VERB_LEMMAS:
        return False
    if token.lemma in CLAIM_ACTION_LEMMAS:
        return True
    if _is_deferral_operator(document, token):
        # Deferring something is always a directive about it, whatever form the
        # operator takes. "Propose deferring ratification" carries no modal, no
        # negation and no imperative inflection, so without this the entire
        # directive is silently dropped.
        return True
    return _has_structural_claim_cue(document, token)


def _has_structural_claim_cue(document: Document, token: Token) -> bool:
    return (
        _predicate_polarity(document, token) == "negative"
        or _predicate_modality(document, token) != "assertive"
        or _structural_claim_has_imperative_form(token)
        or _is_declarative_main_predicate(document, token)
    )


def _is_declarative_main_predicate(document: Document, token: Token) -> bool:
    """A finite main-clause verb states something the text is committed to.

    "The replication path loses data." asserts a fact about the system. A
    rewrite that drops it drops governed content, so the assertion has to enter
    the signature even though it carries no modal, no negation and no
    imperative inflection.

    Finiteness can sit on an auxiliary instead of the main verb: "the critical
    path is completing security evidence" is exactly as assertive as "the
    critical path completes security evidence". Reading only the main verb's
    tag would leave every progressive and perfect sentence unextracted, and an
    unextracted sentence is one a rewrite can silently change.
    """

    if token.dep not in {"ROOT", "conj", "ccomp", "advcl"}:
        return False
    if not any(child.dep in {"nsubj", "nsubjpass"} for child in document.children(token)):
        return False
    if token.tag in {"VBZ", "VBD", "VBP"}:
        return True
    if token.tag not in {"VBG", "VBN"} or not _has_finite_auxiliary(document, token):
        return False
    return not _is_modality_carrier(document, token)


def _is_modality_carrier(document: Document, token: Token) -> bool:
    """"is required to verify" states one obligation, not two events.

    A passive participle whose complement is an infinitival clause supplies the
    modality of that clause; the modal reader already folds it into the inner
    predicate. Recording it again as its own assertion would make "is required
    to verify X" disagree with "must verify X".
    """

    children = document.children(token)
    return any(child.dep == "auxpass" for child in children) and any(
        child.dep == "xcomp" for child in children
    )


def _has_finite_auxiliary(document: Document, token: Token) -> bool:
    return any(
        child.dep in {"aux", "auxpass"} and child.tag in {"VBZ", "VBD", "VBP"}
        for child in document.children(token)
    )


def _structural_claim_has_imperative_form(token: Token) -> bool:
    return token.tag in {"VB", "VBP"} and token.dep in {"ROOT", "conj", "xcomp", "dep", "acl"}


def _claim_predicates(document: Document) -> Iterable[Token]:
    for token in document:
        if _is_claim_predicate(document, token):
            yield token


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
    # Concepts no longer suppress parsed claims. Suppression existed so that a
    # memorised concept claim would not be duplicated by the parse; with the
    # memorised claims gone, suppressing here would delete real content.
    concept_ranges: list[tuple[int, int]] = []
    for predicate in _claim_predicates(document):
        deferrals = _deferral_parts_list(document, predicate)
        if deferrals:
            # A coordinated deferral defers each conjunct separately; emitting
            # only the first would drop a directive.
            for deferred in deferrals:
                _add_claim(
                    claims,
                    deferred.start,
                    deferred.end,
                    deferred.actor,
                    deferred.action,
                    deferred.target,
                    deferred.modality,
                    deferred.polarity,
                    deferred.status,
                )
            continue
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


def _has_own_subject(document: Document, token: Token) -> bool:
    return any(child.dep in {"nsubj", "nsubjpass"} for child in document.children(token))


def _is_adjectival(token: Token) -> bool:
    """A predicate complement, as opposed to a second finite predicate.

    "remains exact and bounded" coordinates two states; "is complete and passes
    review" coordinates two predicates, and the second belongs to the ordinary
    claim extractor. Finite tense marking is what separates them.
    """

    return token.pos == "ADJ" or token.tag in {"VBN", "VBG"}


def _predicate_complements(document: Document, predicate: Token) -> list[Token]:
    """Collect the predicate complements of a linking verb.

    "is complete and fail-closed" and "remains exact and bounded" attach their
    conjuncts inconsistently -- sometimes under the first complement, sometimes
    directly under the verb -- so both attachment points are collected. A
    conjunct carrying its own subject is a separate clause, not a shared
    complement, and is left for that clause's own extraction.
    """

    complements: list[Token] = []
    for child in document.children(predicate):
        if child.dep in {"acomp", "attr", "oprd"}:
            complements.append(child)
            complements.extend(
                grandchild
                for grandchild in document.children(child)
                if grandchild.dep == "conj" and not _has_own_subject(document, grandchild)
            )
        elif (
            child.dep == "conj"
            and complements
            and _is_adjectival(child)
            and not _has_own_subject(document, child)
        ):
            complements.append(child)
    return complements


def _complement_tokens(document: Document, complement: Token, siblings: Iterable[Token]) -> list[Token]:
    """The complement's own words, excluding any coordinated complement.

    A conjunct is normalized as its own state, so leaving it inside the head's
    subtree would record the pair twice under two different keys. Only nested
    conjuncts are removed -- a conjunct that hangs off the verb instead never
    overlaps its sibling.
    """

    excluded = {
        token.index
        for sibling in siblings
        if sibling.index != complement.index and _is_descendant(document, sibling, complement)
        for token in document.subtree(sibling)
    }
    return [token for token in document.subtree(complement) if token.index not in excluded]


def _is_descendant(document: Document, token: Token, ancestor: Token) -> bool:
    return any(
        member.index == token.index
        for member in document.subtree(ancestor)
        if member.index != ancestor.index
    )


def _state_claim_predicates(document: Document) -> Iterable[tuple[Token, list[Token], str]]:
    """Linking predicates that assert a state, with their normalized states."""

    for predicate in document:
        if predicate.pos not in {"VERB", "AUX"}:
            continue
        if not _subject_tokens(document, predicate):
            continue
        complements = _predicate_complements(document, predicate)
        if not complements:
            continue
        states = sorted(
            {
                _normalize_target_tokens(_complement_tokens(document, complement, complements))
                for complement in complements
            }
        )
        target = ",".join(state for state in states if state and state != "none")
        if not target:
            continue
        yield predicate, complements, target


def _state_claims(document: Document, claims: list[tuple[int, int, str]]) -> set[int]:
    """Extract what a text asserts a thing *is*, not only what it does.

    "The fix is complete and fail-closed" commits the author to a state. Without
    this the copula yields no claim at all, and rewriting it to "is incomplete
    and fail-open" compares equivalent -- a false certification of meaning that
    the coverage guard cannot catch whenever any other sentence parses cleanly.
    """

    claimed: set[int] = set()
    for predicate, complements, target in _state_claim_predicates(document):
        start = min([predicate.start, *(token.start for token in complements)])
        end = max([predicate.end, *(token.end for token in complements)])
        polarity = _predicate_polarity(document, predicate)
        _add_claim(
            claims,
            start,
            end,
            _normalize_subject_tokens(document, _subject_tokens(document, predicate)),
            _resolve_action(predicate),
            target,
            _predicate_modality(document, predicate),
            polarity,
            _predicate_status(document, predicate, polarity),
        )
        claimed.add(predicate.index)
    return claimed


def _semantic_claim_signatures(document: Document, concepts: Iterable[ConceptSpan]) -> list[str]:
    concept_list = list(concepts)
    claims: list[tuple[int, int, str]] = []
    state_predicates = _state_claims(document, claims)
    _generic_claims(document, concept_list, state_predicates, claims)
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
    return token.pos == "NUM" or token.lower in NUMBER_WORDS or token.lower in {"either", "both"}


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
            # Compare governance vocabulary by the action it names, not by its
            # surface form. LING-NOMINAL-001 asks writers to turn
            # "ratification" into "ratify"; the gate must not then reject the
            # remediation it recommended.
            yield _item(
                document.text,
                "governance",
                "term",
                canonical_action(token.lemma),
                token.start,
                token.end,
                document.text,
            )
    sentence_tokens = list(document)
    for index, token in enumerate(sentence_tokens[:-1]):
        following = sentence_tokens[index + 1]
        if token.lower == "target" and following.lemma == "architecture":
            if not _contains(covered, token.start, following.end):
                yield _item(document.text, "governance", "term", "target architecture", token.start, following.end, document.text)


_SUBORDINATING_DEPS = {"mark", "advmod"}


@dataclass(frozen=True)
class OrderingRelation:
    """One temporal relation.

    `start`/`end` span both sequenced sides and the marker between them, since
    that is what the relation describes. `owned` is narrower: just the indices
    of the clause the marker introduces, which is what a claim target may drop
    on the grounds that this relation reports it.
    """

    marker: Token
    anchor: Token
    earlier: str
    later: str
    start: int
    end: int
    owned: frozenset[int]


def _ordering_anchor(document: Document, start: Token) -> Token:
    """Return the predicate that governs `start`.

    An ordering marker relates two clauses, but the parser attaches it wherever
    it sits: fronted, "before" attaches to the main verb; trailing, it attaches
    to the nearest preceding noun. Taking the attachment point as the anchor
    made the same sentence produce two different relations depending on clause
    order, so a rewrite that only moved the clause looked like a meaning change.

    Walking up to the governing predicate makes the anchor the clause, which is
    what the relation is actually about.
    """
    current = start
    for _ in range(_MAX_HEAD_WALK):
        if current.pos in {"VERB", "AUX"}:
            return current
        head = document.head_of(current)
        if head.index == current.index:
            return current
        current = head
    return current


def _ordering_sides(
    document: Document, marker: Token, *, subordinating: bool
) -> tuple[Token, list[Token]] | None:
    """Return the anchor predicate and the tokens the marker introduces.

    The two shapes are opposites. As a preposition the sequenced clause hangs
    below the marker, so it is found among its children. As a subordinating
    conjunction the marker hangs below the clause instead, so the clause is its
    head and it has no children at all -- which is why only the prepositional
    shape used to yield a relation, and "close the findings before the design
    returns" compared equal to "...after the design returns".
    """
    head = document.head_of(marker)
    if head.index == marker.index:
        return None
    if subordinating:
        governor = document.head_of(head)
        if governor.index == head.index:
            # A fronted subordinate clause is sometimes parsed as the root, with
            # the main clause demoted to the unlabelled "dep" fallback beneath
            # it. The relation is still stated, so recover the main clause from
            # the inverted parse rather than dropping the sequencing entirely.
            demoted = [
                child
                for child in document.children(head)
                if child.pos == "VERB" and child.dep in {"dep", "conj", "parataxis"}
            ]
            if not demoted:
                return None
            return _ordering_anchor(document, demoted[0]), [head]
        return _ordering_anchor(document, governor), [head]
    related = [
        child for child in document.children(marker) if child.dep in {"pobj", "advcl"}
    ]
    if not related:
        related = [
            child
            for child in document.children(marker)
            if child.pos in {"NOUN", "PROPN", "VERB"}
        ]
    if not related:
        return None
    return _ordering_anchor(document, head), related


@lru_cache(maxsize=32)
def _ordering_relations(document: Document) -> tuple[OrderingRelation, ...]:
    """Return every temporal relation the document states.

    Shared with `_target_tokens` so that a claim target drops an ordering clause
    only when that clause is provably represented here. Excluding it on the
    marker alone would silently lose governed content whenever this function
    declined to emit a relation.

    Memoized because `_target_tokens` runs once per claim while this walks every
    token calling `children` and `subtree`, both of which scan the document.
    Recomputing it per claim made extraction quadratic in document length: a
    forty-sentence document went from 0.66s to 25s, which the test suite did not
    show because its documents are a sentence or two long.
    """
    relations: list[OrderingRelation] = []
    for token in document:
        if token.lower not in ORDERING_MARKERS:
            continue
        # A subordinating marker is labelled "mark" when the parser reads its
        # clause as a clause, and "advmod" when it does not -- "once a target
        # architecture returns" parses the verb as a noun and demotes the marker
        # to an adverb. The relation is stated either way, so both labels are
        # read, but only from a conjunction: an ordinary adverb sequences
        # nothing.
        subordinating = token.dep in _SUBORDINATING_DEPS and token.pos == "SCONJ"
        if token.dep != "prep" and not subordinating:
            continue
        sides = _ordering_sides(document, token, subordinating=subordinating)
        if sides is None:
            continue
        anchor, related = sides
        if anchor.index == token.index:
            continue
        anchor_indices = {member.index for member in document.subtree(anchor)}
        owned = {
            member.index for relative in related for member in document.subtree(relative)
        }
        if anchor.index in owned:
            # The recovered main clause sits inside the marked clause's subtree
            # in an inverted parse, so remove it before measuring the two sides.
            owned -= anchor_indices
        anchor_tokens = [
            candidate
            for candidate in document.subtree(anchor)
            if candidate.index != token.index and candidate.index not in owned
        ]
        other_tokens = [
            member
            for relative in related
            for member in document.subtree(relative)
            if member.index != token.index and member.index in owned
        ]
        anchor_side = _normalize_target_tokens(anchor_tokens)
        other_side = _normalize_target_tokens(other_tokens)
        if not anchor_side or not other_side:
            continue
        if token.lower in {"before", "until", "prior"}:
            earlier, later = anchor_side, other_side
        else:
            earlier, later = other_side, anchor_side
        # The span covers both sequenced sides and the marker between them. A
        # conjunction has no children, so measuring the marker's own subtree
        # ended the span at the marker word and reported "close the findings
        # before" as the location of a relation whose other half is the clause
        # that follows it.
        span = anchor_tokens + other_tokens + [token]
        relations.append(
            OrderingRelation(
                marker=token,
                anchor=anchor,
                earlier=earlier,
                later=later,
                start=min(member.start for member in span),
                end=max(member.end for member in span),
                owned=frozenset(owned | {token.index}),
            )
        )
    return tuple(relations)


def _order_items(document: Document) -> Iterable[dict[str, JsonValue]]:
    """Emit explicit sequencing relations for temporal prepositions.

    "Close the findings before the design returns" and "...after the design
    returns" are opposite instructions built from identical words. Sequencing
    lives only in the preposition, so unless the relation is represented the
    two are indistinguishable and a rewrite could silently invert a gate.
    Both directions are normalized to an earlier/later pair so that "A before
    B" and "B after A" agree.
    """

    for relation in _ordering_relations(document):
        yield _item(
            document.text,
            "order",
            "sequence",
            f"earlier={relation.earlier};later={relation.later}",
            relation.start,
            relation.end,
            document.text,
        )


def extract_protected(text: str, profile: Profile) -> dict[str, JsonValue]:
    document = parse(text)
    items: list[dict[str, JsonValue]] = []
    concept_items, concept_spans, concept_seen = _extract_concepts(text, document, profile)
    items.extend(concept_items)
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
    items.extend(_order_items(document))

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
        "coverage": _coverage(document, ordered),
        "source_sha256": source_sha256(text),
    }
    manifest["sha256"] = sha256_json(manifest)
    return manifest


def _coverage(document: Document, items: list[dict[str, JsonValue]]) -> JsonValue:
    """Record which sentences contributed nothing to the manifest.

    A sentence that yields neither a claim nor a protected item has not been
    understood, and a comparison that silently treats it as matching would be a
    success-shaped fallback on the safety-critical path. Two texts that both
    fail to parse would otherwise compare 'equivalent' purely because both
    manifests are empty. Callers must treat an uncovered sentence as grounds for
    an unresolved verdict, never as assurance.
    """

    covered: set[int] = set()
    for item in items:
        location = cast(dict[str, JsonValue], item["location"])
        start = cast(int, location["start"])
        for sentence in document.sentences:
            if sentence.start <= start < sentence.end:
                covered.add(sentence.index)
                break
    for predicate in _claim_predicates(document):
        covered.add(document.sentence_of(predicate).index)
    for predicate, _, _ in _state_claim_predicates(document):
        covered.add(document.sentence_of(predicate).index)

    uncovered: list[JsonValue] = []
    for sentence in document.sentences:
        if sentence.index in covered:
            continue
        if not any(token.pos in _CONTENT_POS for token in sentence.tokens):
            continue
        uncovered.append(
            cast(
                JsonValue,
                {
                    "sentence_index": sentence.index,
                    "text": sentence.text.strip(),
                },
            )
        )
    return cast(
        JsonValue,
        {"sentences": len(document.sentences), "uncovered": uncovered},
    )


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


def _uncovered_reasons(manifest: dict[str, JsonValue], label: str) -> list[str]:
    """Sentences that produced no extractable meaning block a verdict.

    Without this, two texts that both defeat the parser produce two empty
    manifests and compare 'equivalent' — the gate would certify meaning it
    never read.
    """

    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict):
        return [
            f"{label}: manifest carries no coverage record, so equivalence "
            "cannot be established"
        ]
    uncovered = coverage.get("uncovered")
    if not isinstance(uncovered, list):
        return []
    reasons: list[str] = []
    for entry in uncovered:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text", "")).strip()
        reasons.append(
            f"{label}: no proposition or protected element could be extracted "
            f"from {text!r}; equivalence cannot be established for it"
        )
    return reasons


def _parse_claim_signature(signature: str) -> dict[str, str] | None:
    if not signature.startswith("claim:"):
        return None
    fields: dict[str, str] = {}
    for part in signature[len("claim:") :].split(";"):
        key, separator, value = part.partition("=")
        if not separator:
            return None
        fields[key] = value
    return fields


def _reconcile_actor_specification(
    missing: list[str],
    added: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Allow naming a previously unnamed actor, but never un-naming one.

    LING-AGENCY-001 asks writers to replace hidden agency with a named actor.
    If the gate treated that as a meaning change it would reject its own
    remediation and no candidate could ever satisfy both. Supplying an actor
    where the source had none adds information without contradicting the
    source, so it is permitted and reported. The reverse -- dropping or
    swapping a named actor -- removes accountability and stays a violation.
    """

    remaining_missing = list(missing)
    remaining_added = list(added)
    specified: list[str] = []
    for source_signature in list(remaining_missing):
        source_claim = _parse_claim_signature(source_signature)
        if source_claim is None or source_claim.get("actor") != "unspecified":
            continue
        for candidate_signature in list(remaining_added):
            candidate_claim = _parse_claim_signature(candidate_signature)
            if candidate_claim is None:
                continue
            if candidate_claim.get("actor") == "unspecified":
                continue
            if {k: v for k, v in source_claim.items() if k != "actor"} != {
                k: v for k, v in candidate_claim.items() if k != "actor"
            }:
                continue
            remaining_missing.remove(source_signature)
            remaining_added.remove(candidate_signature)
            specified.append(
                f"actor specified: {source_claim.get('action', '?')} "
                f"assigned to '{candidate_claim['actor']}'"
            )
            break
    return remaining_missing, remaining_added, sorted(specified)


def compare_protected(
    source_manifest: dict[str, JsonValue],
    candidate_manifest: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    source_signatures = cast(list[str], source_manifest["semantic_signature"])
    candidate_signatures = cast(list[str], candidate_manifest["semantic_signature"])
    if source_manifest.get("source_sha256") == candidate_manifest.get("source_sha256"):
        # Unchanged text preserves its own meaning by construction. Incomplete
        # extraction is a reason to doubt a *rewrite*, not a reason to doubt
        # that a text says what it says.
        return {
            "equivalent": True,
            "source_manifest_sha256": cast(str, source_manifest["sha256"]),
            "candidate_manifest_sha256": cast(str, candidate_manifest["sha256"]),
            "missing": cast(list[JsonValue], []),
            "added": cast(list[JsonValue], []),
            "specified": cast(list[JsonValue], []),
            "disposition": "equivalent",
            "unresolved": cast(list[JsonValue], []),
        }
    source = Counter(source_signatures)
    candidate = Counter(candidate_signatures)
    missing = sorted((source - candidate).elements())
    added = sorted((candidate - source).elements())
    missing, added, specified = _reconcile_actor_specification(missing, added)
    unresolved = (
        _unresolved_claim_reasons(source_signatures, "source")
        + _unresolved_claim_reasons(candidate_signatures, "candidate")
        + _uncovered_reasons(source_manifest, "source")
        + _uncovered_reasons(candidate_manifest, "candidate")
    )
    equivalent = not missing and not added and not unresolved
    if not equivalent and not missing and not added and unresolved:
        # Doubt exists to stop us certifying a change we cannot see. When
        # nothing moved and both texts raise exactly the same doubts about
        # exactly the same content, there is no change being hidden -- unless
        # nothing was extracted at all, which is the vacuous case that must
        # stay unresolved.
        source_reasons = sorted(
            reason.split(":", 1)[1].strip()
            for reason in unresolved
            if reason.startswith("source:")
        )
        candidate_reasons = sorted(
            reason.split(":", 1)[1].strip()
            for reason in unresolved
            if reason.startswith("candidate:")
        )
        extracted = bool(source_signatures) and bool(candidate_signatures)
        if extracted and source_reasons and source_reasons == candidate_reasons:
            unresolved = []
            equivalent = True
    disposition = "equivalent" if equivalent else ("unresolved" if unresolved else "changed")
    return {
        "equivalent": equivalent,
        "source_manifest_sha256": cast(str, source_manifest["sha256"]),
        "candidate_manifest_sha256": cast(str, candidate_manifest["sha256"]),
        "missing": cast(list[JsonValue], missing),
        "added": cast(list[JsonValue], added),
        "specified": cast(list[JsonValue], specified),
        "disposition": disposition,
        "unresolved": cast(list[JsonValue], sorted(unresolved)),
    }


def source_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
