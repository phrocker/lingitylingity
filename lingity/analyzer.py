"""Deterministic linguistic analysis."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import cast

from lingity.invariants import extract_protected, source_sha256
from lingity.lexicon import (
    ACTION_VERBS,
    AUXILIARIES,
    COMMON_ADJECTIVES,
    COMMON_SUBJECT_NOUNS,
    CONJUNCTIONS,
    DETERMINERS,
    FINITE_ACTION_FORMS,
    IRREGULAR_PAST_PARTICIPLES,
    MODALS,
    PASSIVE_PARTICIPLES,
    PAST_ACTION_FORMS,
    PREPOSITIONS,
    SUBJECT_PRONOUNS,
)
from lingity.models import Finding, JsonValue, Location
from lingity.profiles import Profile, canonical_json, load_profile
from lingity.scoring import calculate_hri
from lingity.text import Sentence, Token, line_column, sentences

ANALYZER_VERSION = "1.1.2"
MODEL_NAME = "lingity-regex-en"
MODEL_VERSION = "1.1.2"

WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z0-9]+)*")
PUNCTUATION_BREAK_RE = re.compile(r"[,;:()\[\]{}.!?—–]")
SUBORDINATE_MARKERS = frozenset(
    {"after", "although", "because", "before", "that", "unless", "which", "while"}
)
LEADING_MODIFIERS = frozenset({"finally", "first", "next", "then"})

STOPWORDS = {
    "a", "an", "and", "any", "as", "at", "before", "but", "by", "do", "for",
    "from", "in", "into", "is", "it", "not", "of", "on", "only", "or", "our",
    "out", "that", "the", "then", "this", "to", "with", "yet",
}

RULE_DIMENSIONS = {
    "LING-SENTENCE-001": "sentence_load",
    "LING-CLAUSE-001": "sentence_load",
    "LING-ACTION-001": "sentence_load",
    "LING-PUNCTUATION-001": "sentence_load",
    "LING-NOMINALIZATION-001": "morphology",
    "LING-NOUN-STACK-001": "morphology",
    "LING-COMPOUND-DEPTH-001": "morphology",
    "LING-WEAK-VERB-001": "morphology",
    "LING-AGENCY-001": "agency",
    "LING-PASSIVE-001": "agency",
    "LING-ACTOR-001": "agency",
    "LING-INDIRECT-PREDICATE-001": "agency",
    "LING-JARGON-001": "lexical_clarity",
    "LING-BUREAUCRACY-001": "lexical_clarity",
    "LING-COMPOUND-001": "lexical_clarity",
    "LING-ABBREVIATION-001": "lexical_clarity",
    "LING-STRUCTURE-001": "structure",
    "LING-LIST-001": "structure",
    "LING-MIXED-PURPOSE-001": "structure",
    "LING-REDUNDANCY-001": "redundancy",
    "LING-FILLER-001": "redundancy",
    "LING-DUPLICATED-RECOMMENDATION-001": "redundancy",
    "LING-QUALIFIER-001": "redundancy",
}

DESIGN_SIGNAL_RULES = {
    "sentence_load.punctuation_depth": "LING-PUNCTUATION-001",
    "noun_stacking.consecutive_noun_modifiers": "LING-NOUN-STACK-001",
    "noun_stacking.compound_depth": "LING-COMPOUND-DEPTH-001",
    "agency.explicit_actor_action_pairs": "LING-ACTOR-001",
    "voice.indirect_predicates": "LING-INDIRECT-PREDICATE-001",
    "lexical_clarity.uncommon_compounds": "LING-COMPOUND-001",
    "lexical_clarity.abbreviation_density": "LING-ABBREVIATION-001",
    "structure.list_suitability": "LING-LIST-001",
    "structure.mixed_purpose_sentences": "LING-MIXED-PURPOSE-001",
    "redundancy.filler_phrases": "LING-FILLER-001",
    "redundancy.duplicated_recommendations": "LING-DUPLICATED-RECOMMENDATION-001",
    "redundancy.repeated_qualifiers": "LING-QUALIFIER-001",
}

DESIGN_TABLE_RULES = {
    "sentence_load": (
        "LING-SENTENCE-001",
        "LING-CLAUSE-001",
        "LING-PUNCTUATION-001",
        "LING-ACTION-001",
    ),
    "morphology": (
        "LING-NOMINALIZATION-001",
        "LING-WEAK-VERB-001",
    ),
    "noun_stacking": (
        "LING-NOUN-STACK-001",
        "LING-COMPOUND-DEPTH-001",
    ),
    "agency": (
        "LING-ACTOR-001",
        "LING-AGENCY-001",
    ),
    "voice": (
        "LING-PASSIVE-001",
        "LING-INDIRECT-PREDICATE-001",
    ),
    "lexical_clarity": (
        "LING-JARGON-001",
        "LING-COMPOUND-001",
        "LING-ABBREVIATION-001",
    ),
    "structure": (
        "LING-STRUCTURE-001",
        "LING-LIST-001",
        "LING-MIXED-PURPOSE-001",
    ),
    "redundancy": (
        "LING-QUALIFIER-001",
        "LING-DUPLICATED-RECOMMENDATION-001",
        "LING-FILLER-001",
    ),
}

OBLIGATION_WORDS = frozenset({"must", "shall", "should"})
IMPERSONAL_SUBJECTS = frozenset({"it", "there"})
PASSIVE_BE_AUXILIARIES = frozenset({"am", "are", "be", "been", "being", "is", "was", "were"})
GET_PASSIVE_AUXILIARIES = frozenset({"get", "gets", "got", "getting"})
GET_PASSIVE_PARTICIPLES = frozenset({"deprecated"})
ADVERB_BOUNDARIES = frozenset(
    {
        "again",
        "always",
        "annually",
        "daily",
        "later",
        "monthly",
        "never",
        "often",
        "once",
        "quarterly",
        "soon",
        "twice",
        "weekly",
    }
)
ABSTRACT_SUBJECTS = frozenset(
    {
        "approval",
        "assessment",
        "closure",
        "decision",
        "evidence",
        "implementation",
        "migration",
        "ratification",
        "recommendation",
        "remediation",
        "resolution",
        "validation",
        "verification",
    }
)


def _location(text: str, start: int, end: int, sentence_index: int | None) -> Location:
    line, column = line_column(text, start)
    return Location(start, end, line, column, sentence_index)


def _severity(observed: float, threshold: float) -> str:
    ratio = observed / threshold if threshold else observed
    if ratio >= 2:
        return "high"
    if ratio >= 1.35:
        return "medium"
    return "low"


def _word(token: Token) -> str:
    return token.text.lower().strip("-.")


def _gap_before(sentence: Sentence, index: int) -> str:
    if index == 0:
        return sentence.text[: sentence.tokens[index].start - sentence.start]
    previous = sentence.tokens[index - 1]
    current = sentence.tokens[index]
    return sentence.text[previous.end - sentence.start: current.start - sentence.start]


def _gap_after(sentence: Sentence, index: int) -> str:
    current = sentence.tokens[index]
    if index + 1 >= len(sentence.tokens):
        return sentence.text[current.end - sentence.start:]
    following = sentence.tokens[index + 1]
    return sentence.text[current.end - sentence.start: following.start - sentence.start]


def _starts_segment(sentence: Sentence, index: int) -> bool:
    if index == 0:
        return True
    return PUNCTUATION_BREAK_RE.search(_gap_before(sentence, index)) is not None


def _previous_significant_word(sentence: Sentence, index: int) -> str | None:
    cursor = index - 1
    while cursor >= 0:
        word = _word(sentence.tokens[cursor])
        if word not in {"not", "never"} and not word.endswith("ly"):
            return word
        cursor -= 1
    return None


def _next_word(sentence: Sentence, index: int) -> str | None:
    if index + 1 >= len(sentence.tokens):
        return None
    return _word(sentence.tokens[index + 1])


def _is_lexical_predicate(word: str) -> bool:
    return (
        word in ACTION_VERBS
        or word in FINITE_ACTION_FORMS
        or word in PAST_ACTION_FORMS
        or word in IRREGULAR_PAST_PARTICIPLES
        or (word.endswith("ed") and len(word) > 4)
    )


def _looks_like_subject(token: Token) -> bool:
    word = _word(token)
    return (
        word in SUBJECT_PRONOUNS
        or word in COMMON_SUBJECT_NOUNS
        or (token.text[:1].isupper() and len(word) > 1)
        or (word.endswith("s") and len(word) > 3 and word not in COMMON_ADJECTIVES)
    )


def _has_subject_before(sentence: Sentence, index: int) -> bool:
    subject_words = 0
    cursor = index - 1
    while cursor >= 0:
        gap = _gap_before(sentence, cursor + 1) if cursor + 1 < len(sentence.tokens) else ""
        if PUNCTUATION_BREAK_RE.search(gap) is not None:
            break
        word = _word(sentence.tokens[cursor])
        if word in DETERMINERS or word in COMMON_ADJECTIVES:
            cursor -= 1
            continue
        if word in PREPOSITIONS or word in CONJUNCTIONS or word in MODALS or word in AUXILIARIES:
            break
        if _is_action_token(sentence, cursor):
            break
        subject_words += 1
        cursor -= 1
    return subject_words > 0


def _looks_like_finite_s_predicate(sentence: Sentence, index: int) -> bool:
    word = _word(sentence.tokens[index])
    if (
        len(word) < 4
        or not word.endswith("s")
        or word.endswith("ss")
        or word in DETERMINERS
        or word in COMMON_ADJECTIVES
        or word in PREPOSITIONS
        or word in CONJUNCTIONS
    ):
        return False
    if word in {"needs", "requires"}:
        return _has_subject_before(sentence, index)
    next_word = _next_word(sentence, index)
    if next_word is not None and _is_lexical_predicate(next_word):
        return False
    return _has_subject_before(sentence, index)


def _starts_predicate_after_modifiers(sentence: Sentence, index: int) -> bool:
    cursor = index - 1
    while cursor >= 0:
        word = _word(sentence.tokens[cursor])
        if word not in LEADING_MODIFIERS and not word.endswith("ly"):
            return False
        if _starts_segment(sentence, cursor):
            return True
        cursor -= 1
    return True


def _is_action_context(sentence: Sentence, index: int) -> bool:
    word = _word(sentence.tokens[index])
    previous = _previous_significant_word(sentence, index)
    if previous in MODALS or previous in AUXILIARIES or previous == "to":
        return True
    if (
        index > 0
        and _looks_like_subject(sentence.tokens[index - 1])
        and PUNCTUATION_BREAK_RE.search(_gap_before(sentence, index)) is None
    ):
        return True
    if word in ACTION_VERBS and (_starts_segment(sentence, index) or _starts_predicate_after_modifiers(sentence, index)):
        return True
    if index > 0 and _word(sentence.tokens[index - 1]) in CONJUNCTIONS:
        return index == 1 or _starts_segment(sentence, index - 1)
    return False


def _is_action_token(sentence: Sentence, index: int) -> bool:
    word = _word(sentence.tokens[index])
    if word in MODALS or word in AUXILIARIES:
        return False
    if word in ACTION_VERBS:
        return _is_action_context(sentence, index)
    if word in FINITE_ACTION_FORMS:
        return _is_action_context(sentence, index) or _looks_like_finite_s_predicate(sentence, index)
    if word in PAST_ACTION_FORMS:
        previous = _previous_significant_word(sentence, index)
        return previous in AUXILIARIES or (
            index > 0
            and _looks_like_subject(sentence.tokens[index - 1])
            and PUNCTUATION_BREAK_RE.search(_gap_before(sentence, index)) is None
        )
    if _looks_like_finite_s_predicate(sentence, index):
        return True
    return False


def _action_count(sentence: Sentence) -> int:
    return sum(1 for index, _token in enumerate(sentence.tokens) if _is_action_token(sentence, index))


def _nominalizations(sentence: Sentence, profile: Profile) -> list[str]:
    suffixes = tuple(cast(list[str], profile.rules["nominalization_suffixes"]))
    exclusions = {word.lower() for word in cast(list[str], profile.rules["nominalization_exclusions"])}
    explicit = {word.lower() for word in cast(list[str], profile.rules["nominalizations"])}
    result: list[str] = []
    for token in sentence.tokens:
        word = _word(token)
        singular = word[:-1] if word.endswith("s") and len(word) > 4 else word
        if (
            word in exclusions
            or singular in exclusions
            or word in ACTION_VERBS
            or word in COMMON_ADJECTIVES
            or word in DETERMINERS
        ):
            continue
        if word in explicit or singular in explicit or word.endswith(suffixes):
            result.append(token.text)
    return result


def _is_predicate_or_boundary(sentence: Sentence, index: int) -> bool:
    word = _word(sentence.tokens[index])
    if word in AUXILIARIES or word in MODALS or word in PREPOSITIONS or word in CONJUNCTIONS:
        return True
    if word in IRREGULAR_PAST_PARTICIPLES:
        return True
    return _is_action_token(sentence, index) or _looks_like_finite_s_predicate(sentence, index)


def _is_noun_like(sentence: Sentence, index: int) -> bool:
    token = sentence.tokens[index]
    word = _word(token)
    if (
        not word
        or len(word) == 1
        or word in STOPWORDS
        or word in DETERMINERS
        or word in COMMON_ADJECTIVES
        or word in ADVERB_BOUNDARIES
        or "." in token.text
        or word.endswith("ly")
        or word.endswith("ing")
        or _is_predicate_or_boundary(sentence, index)
    ):
        return False
    if word.endswith("ed") and "-" not in token.text:
        return False
    return True


def _noun_stacks(sentence: Sentence, profile: Profile) -> list[tuple[int, int, str]]:
    minimum = int(profile.thresholds["noun_stack_words"])
    result: list[tuple[int, int, str]] = []
    run: list[Token] = []
    for index, token in enumerate(sentence.tokens):
        if PUNCTUATION_BREAK_RE.search(_gap_before(sentence, index)) is not None:
            if len(run) >= minimum:
                result.append((run[0].start, run[-1].end, " ".join(item.text for item in run)))
            run = []
        if _is_noun_like(sentence, index):
            run.append(token)
        else:
            if len(run) >= minimum:
                result.append((run[0].start, run[-1].end, " ".join(item.text for item in run)))
            run = []
    if len(run) >= minimum:
        result.append((run[0].start, run[-1].end, " ".join(item.text for item in run)))
    return result


def _is_passive_participle(word: str) -> bool:
    return word in PASSIVE_PARTICIPLES or (word.endswith("ed") and len(word) > 4)


def _is_get_passive_participle(word: str) -> bool:
    return word in PASSIVE_PARTICIPLES or word in GET_PASSIVE_PARTICIPLES


def _next_passive_candidate(tokens: tuple[Token, ...], start: int) -> int | None:
    cursor = start
    while cursor < len(tokens):
        candidate = _word(tokens[cursor])
        if candidate in {"not", "never"} or candidate in ADVERB_BOUNDARIES or candidate.endswith("ly"):
            cursor += 1
            continue
        return cursor
    return None


def _passive_spans(sentence: Sentence) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    tokens = sentence.tokens
    for index, token in enumerate(tokens):
        auxiliary = _word(token)
        if auxiliary not in PASSIVE_BE_AUXILIARIES and auxiliary not in GET_PASSIVE_AUXILIARIES:
            continue
        cursor = _next_passive_candidate(tokens, index + 1)
        if cursor is None:
            continue
        participle = _word(tokens[cursor])
        if auxiliary in PASSIVE_BE_AUXILIARIES and _is_passive_participle(participle):
            spans.append((token.start, tokens[cursor].end, sentence.text[token.start - sentence.start: tokens[cursor].end - sentence.start]))
        if auxiliary in GET_PASSIVE_AUXILIARIES and _is_get_passive_participle(participle):
            spans.append((token.start, tokens[cursor].end, sentence.text[token.start - sentence.start: tokens[cursor].end - sentence.start]))
    return spans


def _has_verb_between(sentence: Sentence, start: int, end: int) -> bool:
    return any(_is_action_token(sentence, index) for index in range(start, end))


def _clause_count(sentence: Sentence) -> int:
    count = 1
    tokens = sentence.tokens
    for index, token in enumerate(tokens):
        word = _word(token)
        if word in SUBORDINATE_MARKERS and _has_verb_between(sentence, index + 1, len(tokens)):
            count += 1
        elif word in {"and", "but"} and _has_verb_between(sentence, 0, index):
            if index + 1 < len(tokens) and _looks_like_subject(tokens[index + 1]):
                count += 1
    count += sentence.text.count(";")
    return count


def _phrase_findings(
    text: str,
    phrase_map: dict[str, list[str]],
    rule_id: str,
    dimension: str,
    remediation: str,
    penalty: float,
) -> list[Finding]:
    findings: list[Finding] = []
    for label, expressions in phrase_map.items():
        for expression in expressions:
            for match in re.finditer(expression, text, re.IGNORECASE):
                findings.append(
                    Finding(
                        rule_id=rule_id,
                        dimension=dimension,
                        severity="medium",
                        location=_location(text, match.start(), match.end(), None),
                        observed_value={"phrase": match.group(0), "classification": label},
                        threshold={"allowed_occurrences": 0},
                        remediation=remediation,
                        penalty=penalty,
                    )
                )
    return findings


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _punctuation_depth(sentence: Sentence) -> dict[str, int]:
    parenthetical_marks = sentence.text.count("(") + sentence.text.count(")") + sentence.text.count("[") + sentence.text.count("]")
    parentheticals = parenthetical_marks // 2
    dashes = sentence.text.count("—") + sentence.text.count("–")
    semicolons = sentence.text.count(";")
    commas = sentence.text.count(",")
    score = parentheticals * 2 + semicolons * 2 + dashes + max(0, commas - 2)
    return {
        "score": score,
        "parentheticals": parentheticals,
        "dashes": dashes,
        "semicolons": semicolons,
        "commas": commas,
    }


def _punctuation_finding(text: str, sentence: Sentence, profile: Profile) -> Finding | None:
    threshold = int(profile.thresholds["max_punctuation_depth"])
    observed = _punctuation_depth(sentence)
    score = observed["score"]
    if score <= threshold:
        return None
    return Finding(
        "LING-PUNCTUATION-001",
        "sentence_load",
        _severity(score, threshold),
        _location(text, sentence.start, sentence.end, sentence.index),
        cast(dict[str, JsonValue], observed),
        threshold,
        "Split parenthetical, semicolon, dash, or deeply comma-nested interruptions into simpler sentences or bullets.",
        min(14.0, (score - threshold) * 2.0),
    )


def _profile_words(profile: Profile, key: str) -> frozenset[str]:
    return frozenset(word.lower() for word in cast(list[str], profile.rules[key]))


def _is_responsible_actor_word(word: str, actor_terms: frozenset[str]) -> bool:
    return word in actor_terms or word in SUBJECT_PRONOUNS


def _has_responsible_actor(sentence: Sentence, profile: Profile) -> bool:
    actor_terms = _profile_words(profile, "actor_terms")
    for token in sentence.tokens:
        word = _word(token)
        if word in IMPERSONAL_SUBJECTS:
            continue
        if _is_responsible_actor_word(word, actor_terms):
            return True
    for token in sentence.tokens[1:]:
        word = _word(token)
        if token.text[:1].isupper() and word not in ABSTRACT_SUBJECTS:
            return True
    return False


def _directive_marker(sentence: Sentence) -> tuple[str, int, int] | None:
    tokens = sentence.tokens
    if not tokens:
        return None
    first_word = _word(tokens[0])
    if first_word in ACTION_VERBS or first_word in FINITE_ACTION_FORMS:
        return ("imperative", tokens[0].start, tokens[0].end)
    for index, token in enumerate(tokens):
        word = _word(token)
        if word in OBLIGATION_WORDS:
            return (word, token.start, token.end)
        if word in {"recommend", "recommends", "recommended", "require", "requires", "required"}:
            return (word, token.start, token.end)
        if word in {"need", "needs"} and index + 1 < len(tokens) and _word(tokens[index + 1]) == "to":
            return ("needs to", token.start, tokens[index + 1].end)
    return None


def _actor_action_findings(
    text: str,
    profile: Profile,
    agency_spans: list[tuple[int, int]],
) -> list[Finding]:
    findings: list[Finding] = []
    for sentence in sentences(text):
        marker = _directive_marker(sentence)
        if marker is None or _has_responsible_actor(sentence, profile):
            continue
        if _overlaps(sentence.start, sentence.end, agency_spans):
            continue
        findings.append(
            Finding(
                "LING-ACTOR-001",
                "agency",
                "medium",
                _location(text, sentence.start, sentence.end, sentence.index),
                {
                    "directive": marker[0],
                    "text": sentence.text,
                },
                {"responsible_actor_required": 1},
                "Name the responsible team, role, owner, or system that will perform the action.",
                7.0,
            )
        )
    return findings


def _compound_depth_findings(text: str, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    threshold = int(profile.thresholds["max_compound_parts"])
    allowed = _profile_words(profile, "allowed_compounds")
    for sentence in sentences(text):
        for token in sentence.tokens:
            word = _word(token)
            if "-" not in word or word in allowed:
                continue
            parts = [part for part in word.split("-") if part]
            depth = len(parts)
            if depth <= threshold:
                continue
            findings.append(
                Finding(
                    "LING-COMPOUND-DEPTH-001",
                    "morphology",
                    _severity(depth, threshold),
                    _location(text, token.start, token.end, sentence.index),
                    {
                        "compound": token.text,
                        "parts": depth,
                        "hyphens": word.count("-"),
                    },
                    threshold,
                    "Break the compound chain into a short phrase that names the core noun and modifiers separately.",
                    min(10.0, (depth - threshold) * 3.0),
                )
            )
    return findings


def _compound_findings(
    text: str,
    profile: Profile,
    lexical_spans: list[tuple[int, int]],
) -> list[Finding]:
    findings: list[Finding] = []
    allowed = _profile_words(profile, "allowed_compounds")
    suffixes = tuple(cast(list[str], profile.rules["compound_modifier_suffixes"]))
    for sentence in sentences(text):
        for index, token in enumerate(sentence.tokens):
            word = _word(token)
            if "-" not in word or word in allowed or _overlaps(token.start, token.end, lexical_spans):
                continue
            hyphens = word.count("-")
            parts = [part for part in word.split("-") if part]
            is_uncommon = hyphens >= 2 or (len(parts) == 2 and parts[1] in suffixes and index + 1 < len(sentence.tokens))
            if not is_uncommon:
                continue
            findings.append(
                Finding(
                    "LING-COMPOUND-001",
                    "lexical_clarity",
                    "low" if hyphens < 2 else "medium",
                    _location(text, token.start, token.end, sentence.index),
                    {"compound": token.text, "hyphens": hyphens},
                    {"allowed_hyphens_without_profile_approval": 1},
                    "Replace machine-made compound modifiers with a short phrase or a profile-approved term.",
                    4.0 if hyphens < 2 else 6.0,
                )
            )
    return findings


def _defined_acronyms(text: str) -> set[str]:
    defined: set[str] = set()
    expansion_before = re.compile(r"\b([A-Za-z][A-Za-z -]{2,80})\s+\(([A-Z]{2,8})\)")
    expansion_after = re.compile(r"\b([A-Z]{2,8})\s+\(([A-Za-z][A-Za-z -]{2,80})\)")
    for match in expansion_before.finditer(text):
        defined.add(match.group(2))
    for match in expansion_after.finditer(text):
        defined.add(match.group(1))
    return defined


def _acronym_matches(text: str, allowed: frozenset[str], defined: set[str]) -> list[re.Match[str]]:
    matches: list[re.Match[str]] = []
    for match in re.finditer(r"\b[A-Z]{2,8}s?\b", text):
        acronym = match.group(0).removesuffix("s")
        if acronym in allowed or acronym in defined:
            continue
        matches.append(match)
    return matches


def _abbreviation_findings(text: str, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    allowed = frozenset(cast(list[str], profile.rules["known_acronyms"]))
    defined = _defined_acronyms(text)
    max_sentence = int(profile.thresholds["max_undefined_acronyms_per_sentence"])
    max_paragraph = int(profile.thresholds["max_undefined_acronyms_per_paragraph"])
    sentence_spans: list[tuple[int, int]] = []
    for sentence in sentences(text):
        matches = _acronym_matches(sentence.text, allowed, defined)
        if len(matches) <= max_sentence:
            continue
        first = matches[0]
        start = sentence.start + first.start()
        end = sentence.start + matches[-1].end()
        acronyms = sorted({match.group(0).removesuffix("s") for match in matches})
        sentence_spans.append((sentence.start, sentence.end))
        findings.append(
            Finding(
                "LING-ABBREVIATION-001",
                "lexical_clarity",
                _severity(len(matches), max_sentence),
                _location(text, start, end, sentence.index),
                {"undefined_acronyms": cast(list[JsonValue], acronyms), "count": len(matches)},
                max_sentence,
                "Expand each acronym on first use or add genuinely common terms to the profile allow-list.",
                min(12.0, (len(matches) - max_sentence) * 3.0),
            )
        )
    offset = 0
    for paragraph in re.split(r"\n\s*\n", text):
        if not paragraph:
            offset += 2
            continue
        paragraph_start = text.find(paragraph, offset)
        paragraph_end = paragraph_start + len(paragraph)
        offset = paragraph_end
        if _overlaps(paragraph_start, paragraph_end, sentence_spans):
            continue
        matches = _acronym_matches(paragraph, allowed, defined)
        if len(matches) <= max_paragraph:
            continue
        acronyms = sorted({match.group(0).removesuffix("s") for match in matches})
        findings.append(
            Finding(
                "LING-ABBREVIATION-001",
                "lexical_clarity",
                _severity(len(matches), max_paragraph),
                _location(text, paragraph_start + matches[0].start(), paragraph_start + matches[-1].end(), None),
                {"undefined_acronyms": cast(list[JsonValue], acronyms), "count": len(matches)},
                max_paragraph,
                "Expand each acronym on first use or add genuinely common terms to the profile allow-list.",
                min(12.0, (len(matches) - max_paragraph) * 2.0),
            )
        )
    return findings


def _inline_list_items(sentence: Sentence) -> list[str]:
    if "," not in sentence.text:
        return []
    pieces = [piece.strip(" .;:") for piece in re.split(r",|\band\b|\bor\b", sentence.text, flags=re.IGNORECASE)]
    return [piece for piece in pieces if len(WORD_RE.findall(piece)) >= 1]


def _list_suitability_findings(text: str, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    threshold = int(profile.thresholds["max_inline_list_items"])
    for sentence in sentences(text):
        items = _inline_list_items(sentence)
        if len(items) <= threshold:
            continue
        findings.append(
            Finding(
                "LING-LIST-001",
                "structure",
                _severity(len(items), threshold),
                _location(text, sentence.start, sentence.end, sentence.index),
                {"items": len(items), "text": sentence.text},
                threshold,
                "Convert the long inline enumeration into a bulleted list with one item per line.",
                min(12.0, (len(items) - threshold) * 2.0),
            )
        )
    return findings


def _mixed_purpose_findings(text: str, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    purpose_markers = cast(dict[str, list[str]], profile.rules["purpose_markers"])
    threshold = int(profile.thresholds["max_sentence_purposes"])
    for sentence in sentences(text):
        purposes: list[str] = []
        for label, expressions in sorted(purpose_markers.items()):
            if any(re.search(expression, sentence.text, re.IGNORECASE) for expression in expressions):
                purposes.append(label)
        if len(purposes) <= threshold:
            continue
        findings.append(
            Finding(
                "LING-MIXED-PURPOSE-001",
                "structure",
                _severity(len(purposes), threshold),
                _location(text, sentence.start, sentence.end, sentence.index),
                {"purposes": cast(list[JsonValue], purposes), "count": len(purposes)},
                threshold,
                "Separate the decision, remediation work, evidence, and exit criteria into distinct sentences or bullets.",
                min(14.0, (len(purposes) - threshold) * 4.0),
            )
        )
    return findings


def _recommendation_group(word: str, groups: dict[str, list[str]]) -> str | None:
    for label, words in sorted(groups.items()):
        if word in words:
            return label
    return None


def _recommendation_signature(
    sentence: Sentence,
    profile: Profile,
) -> tuple[str, list[str]] | None:
    if _directive_marker(sentence) is None:
        return None
    groups = cast(dict[str, list[str]], profile.rules["recommendation_action_groups"])
    actor_terms = _profile_words(profile, "actor_terms")
    group: str | None = None
    for index, _token in enumerate(sentence.tokens):
        word = _word(sentence.tokens[index])
        candidate = _recommendation_group(word, groups)
        if candidate is not None and (_is_action_token(sentence, index) or word in ACTION_VERBS or word in FINITE_ACTION_FORMS):
            group = candidate
            break
    if group is None:
        return None
    terms = sorted(
        {
            _word(token)
            for token in sentence.tokens
            if len(_word(token)) >= 5
            and _word(token) not in STOPWORDS
            and _word(token) not in actor_terms
            and _word(token) not in ACTION_VERBS
            and _word(token) not in FINITE_ACTION_FORMS
            and _word(token) not in OBLIGATION_WORDS
        }
    )
    if not terms:
        return None
    return (group, terms)


def _duplicated_recommendation_findings(text: str, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    seen: dict[str, list[tuple[Sentence, set[str]]]] = defaultdict(list)
    for sentence in sentences(text):
        signature = _recommendation_signature(sentence, profile)
        if signature is None:
            continue
        group, terms = signature
        term_set = set(terms)
        for previous, previous_terms in seen[group]:
            shared_terms = sorted(term_set & previous_terms)
            if not shared_terms:
                continue
            findings.append(
                Finding(
                    "LING-DUPLICATED-RECOMMENDATION-001",
                    "redundancy",
                    "medium",
                    _location(text, sentence.start, sentence.end, sentence.index),
                    {
                        "action_group": group,
                        "shared_terms": cast(list[JsonValue], shared_terms),
                        "first_sentence": previous.text,
                        "duplicate_sentence": sentence.text,
                    },
                    {"allowed_restatements": 0},
                    "Consolidate repeated recommendations into one obligation with one owner and one acceptance condition.",
                    8.0,
                )
            )
            break
        seen[group].append((sentence, term_set))
    return findings


def _qualifier_findings(text: str, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    qualifiers = _profile_words(profile, "qualifiers")
    threshold = int(profile.thresholds["max_qualifiers_per_sentence"])
    for sentence in sentences(text):
        indexes = [index for index, token in enumerate(sentence.tokens) if _word(token) in qualifiers]
        if len(indexes) <= 1:
            continue
        adjacent: tuple[int, int] | None = None
        for left, right in zip(indexes, indexes[1:]):
            if right == left + 1:
                adjacent = (left, right)
                break
        if adjacent is not None:
            left_token = sentence.tokens[adjacent[0]]
            right_token = sentence.tokens[adjacent[1]]
            findings.append(
                Finding(
                    "LING-QUALIFIER-001",
                    "redundancy",
                    "medium",
                    _location(text, left_token.start, right_token.end, sentence.index),
                    {"qualifiers": [left_token.text, right_token.text], "pattern": "stacked"},
                    {"max_adjacent_qualifiers": 1},
                    "Keep only the qualifier that changes the governed meaning, or remove both if neither does.",
                    5.0,
                )
            )
            continue
        if len(indexes) <= threshold:
            continue
        terms = [sentence.tokens[index].text for index in indexes]
        findings.append(
            Finding(
                "LING-QUALIFIER-001",
                "redundancy",
                _severity(len(indexes), threshold),
                _location(text, sentence.tokens[indexes[0]].start, sentence.tokens[indexes[-1]].end, sentence.index),
                {"qualifiers": cast(list[JsonValue], terms), "count": len(indexes)},
                threshold,
                "Remove repeated hedges or intensifiers unless each one changes the obligation.",
                min(10.0, (len(indexes) - threshold) * 3.0),
            )
        )
    return findings


def _analyze_sentences(text: str, profile: Profile) -> tuple[list[dict[str, JsonValue]], list[Finding]]:
    records: list[dict[str, JsonValue]] = []
    findings: list[Finding] = []
    max_words = int(profile.thresholds["max_sentence_words"])
    max_actions = int(profile.thresholds["max_actions_per_sentence"])
    max_clauses = int(profile.thresholds["max_clauses_per_sentence"])
    max_nominalizations = int(profile.thresholds["max_nominalizations_per_sentence"])

    for sentence in sentences(text):
        word_count = len(sentence.tokens)
        action_count = _action_count(sentence)
        clause_count = _clause_count(sentence)
        records.append(
            {
                "index": sentence.index,
                "start": sentence.start,
                "end": sentence.end,
                "text": sentence.text,
                "word_count": word_count,
                "clause_count": clause_count,
                "action_count": action_count,
            }
        )
        location = _location(text, sentence.start, sentence.end, sentence.index)
        if word_count > max_words:
            findings.append(
                Finding(
                    "LING-SENTENCE-001", "sentence_load", _severity(word_count, max_words),
                    location, word_count, max_words,
                    "Split the sentence so each sentence carries one main purpose.",
                    min(30.0, (word_count - max_words) * 1.5),
                )
            )
        if action_count > max_actions:
            findings.append(
                Finding(
                    "LING-ACTION-001", "sentence_load", _severity(action_count, max_actions),
                    location, action_count, max_actions,
                    "Separate the decision, actions, and exit criteria into distinct sentences.",
                    min(24.0, (action_count - max_actions) * 6.0),
                )
            )
        if clause_count > max_clauses:
            findings.append(
                Finding(
                    "LING-CLAUSE-001", "sentence_load", _severity(clause_count, max_clauses),
                    location, clause_count, max_clauses,
                    "Reduce subordinate clauses or split the sentence.",
                    min(18.0, (clause_count - max_clauses) * 4.0),
                )
            )
        punctuation = _punctuation_finding(text, sentence, profile)
        if punctuation is not None:
            findings.append(punctuation)
        nominalizations = _nominalizations(sentence, profile)
        if len(nominalizations) > max_nominalizations:
            findings.append(
                Finding(
                    "LING-NOMINALIZATION-001", "morphology",
                    _severity(len(nominalizations), max_nominalizations),
                    location, {
                        "count": len(nominalizations),
                        "terms": cast(list[JsonValue], nominalizations),
                    },
                    max_nominalizations,
                    "Replace abstract action nouns with direct verbs where meaning permits.",
                    min(25.0, (len(nominalizations) - max_nominalizations) * 5.0),
                )
            )
        for start, end, stack in _noun_stacks(sentence, profile):
            size = len(stack.split())
            findings.append(
                Finding(
                    "LING-NOUN-STACK-001", "morphology", _severity(size, int(profile.thresholds["noun_stack_words"])),
                    _location(text, start, end, sentence.index),
                    {"words": size, "text": stack},
                    int(profile.thresholds["noun_stack_words"]) - 1,
                    "Unpack the noun stack with a verb or preposition.",
                    min(12.0, 3.0 + (size - 3) * 2.0),
                )
            )
        for start, end, phrase in _passive_spans(sentence):
            findings.append(
                Finding(
                    "LING-PASSIVE-001", "agency", "medium",
                    _location(text, start, end, sentence.index),
                    phrase, {"allowed_occurrences": 0},
                    "Name the actor and use an active verb.",
                    8.0,
                )
            )
    return records, findings


def _paragraph_findings(text: str, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    threshold = int(profile.thresholds["max_paragraph_words"])
    offset = 0
    for paragraph in re.split(r"\n\s*\n", text):
        if not paragraph:
            offset += 2
            continue
        start = text.find(paragraph, offset)
        end = start + len(paragraph)
        offset = end
        count = len(WORD_RE.findall(paragraph))
        if count > threshold:
            findings.append(
                Finding(
                    "LING-STRUCTURE-001", "structure", _severity(count, threshold),
                    _location(text, start, end, None), count, threshold,
                    "Break the paragraph into a decision, rationale, and required actions.",
                    min(20.0, (count - threshold) * 0.75),
                )
            )
    return findings


def _redundancy_findings(text: str, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    lowered = [token.lower() for token in WORD_RE.findall(text)]
    counts = Counter(lowered)
    threshold = int(profile.thresholds["max_repeated_content_word"])
    for word, count in sorted(counts.items()):
        if len(word) < 7 or word in STOPWORDS or count <= threshold:
            continue
        match = re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE)
        if match is None:
            continue
        findings.append(
            Finding(
                "LING-REDUNDANCY-001", "redundancy", "low",
                _location(text, match.start(), match.end(), None),
                {"term": word, "occurrences": count}, threshold,
                "Remove repeated qualifiers or consolidate repeated statements.",
                min(10.0, (count - threshold) * 2.0),
            )
        )
    return findings


def analyze_text(text: str, profile: Profile | None = None) -> dict[str, JsonValue]:
    if not text.strip():
        raise ValueError("Input text must not be empty")
    selected_profile = profile or load_profile()
    sentence_records, findings = _analyze_sentences(text, selected_profile)
    hidden_agency_findings = _phrase_findings(
        text,
        cast(dict[str, list[str]], selected_profile.rules["hidden_agency"]),
        "LING-AGENCY-001",
        "agency",
        "Name the responsible actor and state the action directly.",
        10.0,
    )
    findings.extend(hidden_agency_findings)
    agency_spans = [(finding.location.start, finding.location.end) for finding in hidden_agency_findings]
    findings.extend(_actor_action_findings(text, selected_profile, agency_spans))
    indirect_predicates = _phrase_findings(
        text,
        cast(dict[str, list[str]], selected_profile.rules["indirect_predicates"]),
        "LING-INDIRECT-PREDICATE-001",
        "agency",
        "Replace the expletive or purpose wrapper with a direct subject and predicate.",
        6.0,
    )
    findings.extend(
        finding
        for finding in indirect_predicates
        if not _overlaps(finding.location.start, finding.location.end, agency_spans)
    )
    findings.extend(
        _phrase_findings(
            text,
            cast(dict[str, list[str]], selected_profile.rules["weak_verbs"]),
            "LING-WEAK-VERB-001",
            "morphology",
            "Replace the weak construction with one precise verb.",
            7.0,
        )
    )
    jargon_findings = _phrase_findings(
        text,
        cast(dict[str, list[str]], selected_profile.rules["jargon"]),
        "LING-JARGON-001",
        "lexical_clarity",
        "Use the profile's plain-language alternative.",
        6.0,
    )
    findings.extend(jargon_findings)
    lexical_spans = [(finding.location.start, finding.location.end) for finding in jargon_findings]
    compound_depth_findings = _compound_depth_findings(text, selected_profile)
    findings.extend(compound_depth_findings)
    compound_depth_spans = [(finding.location.start, finding.location.end) for finding in compound_depth_findings]
    findings.extend(_compound_findings(text, selected_profile, lexical_spans + compound_depth_spans))
    findings.extend(_abbreviation_findings(text, selected_profile))
    findings.extend(
        _phrase_findings(
            text,
            cast(dict[str, list[str]], selected_profile.rules["bureaucratic_phrases"]),
            "LING-BUREAUCRACY-001",
            "lexical_clarity",
            "State the recommendation or action directly.",
            8.0,
        )
    )
    findings.extend(
        _phrase_findings(
            text,
            cast(dict[str, list[str]], selected_profile.rules["filler_phrases"]),
            "LING-FILLER-001",
            "redundancy",
            "Remove the filler phrase and keep the governed condition or action.",
            4.0,
        )
    )
    findings.extend(_paragraph_findings(text, selected_profile))
    findings.extend(_list_suitability_findings(text, selected_profile))
    findings.extend(_mixed_purpose_findings(text, selected_profile))
    findings.extend(_redundancy_findings(text, selected_profile))
    findings.extend(_duplicated_recommendation_findings(text, selected_profile))
    findings.extend(_qualifier_findings(text, selected_profile))
    findings.sort(key=lambda item: (item.location.start, item.rule_id, str(item.observed_value)))

    model_contract = {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "rules_digest": hashlib.sha256(
            canonical_json(selected_profile.rules).encode("utf-8")
        ).hexdigest(),
    }
    result: dict[str, JsonValue] = {
        "schema_version": "1.0.0",
        "analyzer_version": ANALYZER_VERSION,
        "linguistic_model": cast(dict[str, JsonValue], model_contract),
        "profile": selected_profile.reference(),
        "source": {
            "text": text,
            "sha256": source_sha256(text),
            "length": len(text),
        },
        "sentences": cast(list[JsonValue], sentence_records),
        "protected": extract_protected(text, selected_profile),
        "findings": [finding.to_dict() for finding in findings],
        "score": calculate_hri(findings, selected_profile),
    }
    result["analysis_sha256"] = hashlib.sha256(canonical_json(result).encode("utf-8")).hexdigest()
    return result
