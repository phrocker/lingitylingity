"""Deterministic linguistic analysis backed by dependency parses."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Iterable, cast

from lingity.invariants import extract_protected, source_sha256
from lingity.models import Finding, JsonValue, Location
from lingity.nlp import Document, Sentence, Token, model_fingerprint, parse
from lingity.profiles import Profile, canonical_json, load_profile
from lingity.scoring import calculate_hri
from lingity.text import line_column

ANALYZER_VERSION = "1.3.0"

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

CLAUSAL_DEPS = frozenset({"advcl", "ccomp", "relcl", "xcomp"})
SUBJECT_DEPS = frozenset({"nsubj", "nsubjpass", "csubj", "csubjpass"})
PASSIVE_DEPS = frozenset({"auxpass", "nsubjpass", "csubjpass"})
LIGHT_VERBS = frozenset({"carry", "conduct", "make", "perform"})
OBLIGATION_AUXILIARIES = frozenset({"must", "shall", "should"})
IMPERSONAL_SUBJECTS = frozenset({"it", "there"})
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
STOP_LEMMAS = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "as",
        "at",
        "be",
        "before",
        "but",
        "by",
        "do",
        "for",
        "from",
        "have",
        "in",
        "into",
        "it",
        "not",
        "of",
        "on",
        "only",
        "or",
        "our",
        "out",
        "that",
        "the",
        "then",
        "this",
        "to",
        "with",
        "yet",
    }
)
PURPOSE_WRAPPERS = frozenset({"goal", "intent", "objective", "purpose"})
INDIRECT_COMPLEMENTS = frozenset({"clear", "critical", "expected", "important", "necessary", "possible", "unclear"})


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


def _lemma(token: Token) -> str:
    return token.lemma or token.lower


def _profile_words(profile: Profile, key: str) -> frozenset[str]:
    return frozenset(word.lower() for word in cast(list[str], profile.rules[key]))


def _children(document: Document, token: Token, *deps: str) -> tuple[Token, ...]:
    wanted = set(deps)
    return tuple(child for child in document.children(token) if not wanted or child.dep in wanted)


def _is_verb_head(document: Document, sentence: Sentence, token: Token) -> bool:
    if token.pos != "VERB" or token.dep in {"aux", "auxpass", "amod", "acomp"}:
        return False
    subject = _subject_of(document, token)
    if subject is not None and "," in sentence.text[subject.end - sentence.start : token.start - sentence.start]:
        return subject.pos in {"PRON", "PROPN"} or _lemma(subject) in _profile_actor_like_subjects()
    return True


def _profile_actor_like_subjects() -> frozenset[str]:
    return frozenset({"architect", "component", "engineer", "gateway", "owner", "platform", "reviewer", "service", "system", "team"})


def _action_count(document: Document, sentence: Sentence) -> int:
    return sum(1 for token in sentence.tokens if _is_verb_head(document, sentence, token))


def _clause_count(document: Document, sentence: Sentence) -> int:
    count = 1
    for token in sentence.tokens:
        if token.pos != "VERB":
            continue
        if token.dep in CLAUSAL_DEPS:
            count += 1
        elif token.dep == "conj" and document.token(token.head).pos == "VERB":
            count += 1
    return count


def _is_nominalization(token: Token, profile: Profile) -> bool:
    if token.pos != "NOUN":
        return False
    suffixes = tuple(cast(list[str], profile.rules["nominalization_suffixes"]))
    exclusions = {word.lower() for word in cast(list[str], profile.rules["nominalization_exclusions"])}
    explicit = {word.lower() for word in cast(list[str], profile.rules["nominalizations"])}
    lemma = _lemma(token)
    singular = lemma[:-1] if lemma.endswith("s") and len(lemma) > 4 else lemma
    if lemma in exclusions or singular in exclusions:
        return False
    return lemma in explicit or singular in explicit or lemma.endswith(suffixes)


def _nominalizations(sentence: Sentence, profile: Profile) -> list[str]:
    return [token.text for token in sentence.tokens if _is_nominalization(token, profile)]


def _passive_verbs(document: Document, sentence: Sentence) -> list[Token]:
    verbs: list[Token] = []
    for token in sentence.tokens:
        if token.pos != "VERB":
            continue
        children = document.children(token)
        if token.dep in {"amod", "acomp"}:
            continue
        if any(child.dep in PASSIVE_DEPS for child in children):
            verbs.append(token)
    return verbs


def _has_agent(document: Document, verb: Token) -> bool:
    return any(child.dep == "agent" for child in document.children(verb))


def _passive_span(document: Document, verb: Token) -> tuple[int, int, str]:
    auxiliaries = [
        child
        for child in document.children(verb)
        if child.dep in {"aux", "auxpass", "neg", "advmod"} and child.start <= verb.start
    ]
    start = min([verb.start, *(token.start for token in auxiliaries)])
    end = verb.end
    return start, end, document.text[start:end]


def _passive_findings(document: Document) -> list[Finding]:
    findings: list[Finding] = []
    for sentence in document.sentences:
        for verb in _passive_verbs(document, sentence):
            start, end, phrase = _passive_span(document, verb)
            findings.append(
                Finding(
                    "LING-PASSIVE-001",
                    "agency",
                    "medium",
                    _location(document.text, start, end, sentence.index),
                    phrase,
                    {"allowed_occurrences": 0},
                    "Name the actor and use an active verb.",
                    8.0,
                )
            )
    return findings


def _noun_stack_tokens(document: Document, head: Token) -> tuple[Token, ...]:
    if head.pos not in {"NOUN", "PROPN"} or head.dep == "compound":
        return ()
    compounds = [
        token
        for token in document.subtree(head)
        if token.dep == "compound" and token.is_word and token.pos in {"NOUN", "PROPN"}
    ]
    if not compounds:
        return ()
    return tuple(sorted([*compounds, head], key=lambda token: token.index))


def _noun_stack_findings(document: Document, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    minimum = int(profile.thresholds["noun_stack_words"])
    emitted: set[tuple[int, int]] = set()
    for sentence in document.sentences:
        for head in sentence.tokens:
            stack = _noun_stack_tokens(document, head)
            if len(stack) < minimum:
                continue
            start = stack[0].start
            end = stack[-1].end
            key = (start, end)
            if key in emitted:
                continue
            emitted.add(key)
            text = document.text[start:end]
            findings.append(
                Finding(
                    "LING-NOUN-STACK-001",
                    "morphology",
                    _severity(len(stack), minimum),
                    _location(document.text, start, end, sentence.index),
                    {"words": len(stack), "text": text},
                    minimum - 1,
                    "Unpack the noun stack with a verb or preposition.",
                    min(12.0, 3.0 + (len(stack) - 3) * 2.0),
                )
            )
    return findings


def _hyphenated_terms(sentence: Sentence) -> list[tuple[int, int, str, int]]:
    result: list[tuple[int, int, str, int]] = []
    tokens = sentence.tokens
    index = 0
    while index < len(tokens):
        if not tokens[index].is_word:
            index += 1
            continue
        cursor = index
        parts = [tokens[index]]
        while cursor + 2 < len(tokens) and tokens[cursor + 1].text == "-" and tokens[cursor + 2].is_word:
            parts.append(tokens[cursor + 2])
            cursor += 2
        if len(parts) > 1:
            start = tokens[index].start
            end = tokens[cursor].end
            text = sentence.text[start - sentence.start : end - sentence.start].lower()
            result.append((start, end, text, len(parts)))
            index = cursor + 1
        else:
            index += 1
    return result


def _compound_depth_findings(document: Document, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    threshold = int(profile.thresholds["max_compound_parts"])
    allowed = _profile_words(profile, "allowed_compounds")
    for sentence in document.sentences:
        for start, end, term, parts in _hyphenated_terms(sentence):
            if term in allowed or parts <= threshold:
                continue
            findings.append(
                Finding(
                    "LING-COMPOUND-DEPTH-001",
                    "morphology",
                    _severity(parts, threshold),
                    _location(document.text, start, end, sentence.index),
                    {"compound": document.text[start:end], "parts": parts, "hyphens": parts - 1},
                    threshold,
                    "Break the compound chain into a short phrase that names the core noun and modifiers separately.",
                    min(10.0, (parts - threshold) * 3.0),
                )
            )
    return findings


def _overlaps(start: int, end: int, spans: Iterable[tuple[int, int]]) -> bool:
    return any(start < span_end and end > span_start for span_start, span_end in spans)


def _compound_findings(document: Document, profile: Profile, lexical_spans: list[tuple[int, int]]) -> list[Finding]:
    findings: list[Finding] = []
    allowed = _profile_words(profile, "allowed_compounds")
    suffixes = tuple(cast(list[str], profile.rules["compound_modifier_suffixes"]))
    for sentence in document.sentences:
        for start, end, term, parts in _hyphenated_terms(sentence):
            if term in allowed or _overlaps(start, end, lexical_spans):
                continue
            final_part = term.split("-")[-1]
            uncommon = parts >= 3 or (parts == 2 and final_part in suffixes)
            if not uncommon:
                continue
            findings.append(
                Finding(
                    "LING-COMPOUND-001",
                    "lexical_clarity",
                    "low" if parts == 2 else "medium",
                    _location(document.text, start, end, sentence.index),
                    {"compound": document.text[start:end], "hyphens": parts - 1},
                    {"allowed_hyphens_without_profile_approval": 1},
                    "Replace machine-made compound modifiers with a short phrase or a profile-approved term.",
                    4.0 if parts == 2 else 6.0,
                )
            )
    return findings


def _phrase_terms(phrase: str) -> tuple[str, ...]:
    stripped = phrase.lower().strip(" .;:()[]{}")
    return tuple(part.strip(" .;:()[]{}") for part in stripped.replace("-", " ").split() if part.strip(" .;:()[]{}"))


def _word_tokens(sentence: Sentence) -> tuple[Token, ...]:
    return tuple(token for token in sentence.tokens if token.is_word)


def _phrase_matches(sentence: Sentence, phrase: str) -> list[tuple[int, int, str]]:
    terms = _phrase_terms(phrase)
    if not terms:
        return []
    words = _word_tokens(sentence)
    lemmas = tuple(_lemma(token) for token in words)
    matches: list[tuple[int, int, str]] = []
    width = len(terms)
    for index in range(0, len(lemmas) - width + 1):
        if lemmas[index : index + width] != terms:
            continue
        start = words[index].start
        end = words[index + width - 1].end
        matches.append((start, end, sentence.text[start - sentence.start : end - sentence.start]))
    return matches


def _phrase_findings(
    document: Document,
    phrase_map: dict[str, list[str]],
    rule_id: str,
    dimension: str,
    remediation: str,
    penalty: float,
) -> list[Finding]:
    findings: list[Finding] = []
    for sentence in document.sentences:
        for label, phrases in sorted(phrase_map.items()):
            for phrase in sorted(phrases):
                for start, end, observed in _phrase_matches(sentence, phrase):
                    findings.append(
                        Finding(
                            rule_id=rule_id,
                            dimension=dimension,
                            severity="medium",
                            location=_location(document.text, start, end, sentence.index),
                            observed_value={"phrase": observed, "classification": label},
                            threshold={"allowed_occurrences": 0},
                            remediation=remediation,
                            penalty=penalty,
                        )
                    )
    return findings


def _has_overt_subject(document: Document, verb: Token) -> bool:
    return any(child.dep in SUBJECT_DEPS and _lemma(child) not in IMPERSONAL_SUBJECTS for child in document.children(verb))


def _subject_of(document: Document, verb: Token) -> Token | None:
    subjects = sorted((child for child in document.children(verb) if child.dep in SUBJECT_DEPS), key=lambda child: child.index)
    return subjects[0] if subjects else None


def _has_obligation_aux(document: Document, verb: Token) -> bool:
    return any(child.dep == "aux" and _lemma(child) in OBLIGATION_AUXILIARIES for child in document.children(verb))


def _is_imperative(document: Document, sentence: Sentence, verb: Token) -> bool:
    first_word = next((token for token in sentence.tokens if token.is_word), None)
    return (
        first_word is not None
        and first_word.index == verb.index
        and verb.pos == "VERB"
        and verb.tag == "VB"
        and not _has_overt_subject(document, verb)
    )


def _directive_marker(document: Document, sentence: Sentence) -> tuple[str, int, int, Token] | None:
    for token in sentence.tokens:
        if token.pos == "VERB" and _has_obligation_aux(document, token):
            auxiliaries = sorted(
                (child for child in document.children(token) if child.dep == "aux" and _lemma(child) in OBLIGATION_AUXILIARIES),
                key=lambda child: child.index,
            )
            marker = auxiliaries[0]
            return (_lemma(marker), marker.start, marker.end, token)
        if token.pos == "VERB" and _is_imperative(document, sentence, token):
            return ("imperative", token.start, token.end, token)
    return None


def _has_responsible_actor(document: Document, sentence: Sentence, profile: Profile) -> bool:
    actor_terms = _profile_words(profile, "actor_terms")
    for token in sentence.tokens:
        if token.dep in SUBJECT_DEPS and _lemma(token) in actor_terms:
            return True
        if token.dep in SUBJECT_DEPS and token.pos == "PRON" and _lemma(token) not in IMPERSONAL_SUBJECTS:
            return True
        if token.dep in SUBJECT_DEPS and token.pos == "PROPN":
            return True
    return any(token.pos == "PROPN" and token.entity_type in {"ORG", "PERSON"} for token in sentence.tokens)


def _actor_action_findings(document: Document, profile: Profile, agency_spans: list[tuple[int, int]]) -> list[Finding]:
    findings: list[Finding] = []
    for sentence in document.sentences:
        marker = _directive_marker(document, sentence)
        if marker is None:
            continue
        label, _start, _end, verb = marker
        if _has_responsible_actor(document, sentence, profile):
            continue
        # A profile may require that the subject of a directive be an actor the
        # profile recognises. Without it, any overt noun satisfies the rule, so
        # "the market should prioritise retention" reports nothing and a
        # profile's choice of actor terms decides nothing.
        strict = bool(profile.thresholds.get("require_responsible_actor", 0))
        if not strict and _has_overt_subject(document, verb):
            continue
        if _overlaps(sentence.start, sentence.end, agency_spans):
            continue
        findings.append(
            Finding(
                "LING-ACTOR-001",
                "agency",
                "medium",
                _location(document.text, sentence.start, sentence.end, sentence.index),
                {"directive": label, "text": sentence.text},
                {"responsible_actor_required": 1},
                "Name the responsible team, role, owner, or system that will perform the action.",
                7.0,
            )
        )
    return findings


def _hidden_agency_findings(document: Document, profile: Profile) -> list[Finding]:
    findings = _phrase_findings(
        document,
        cast(dict[str, list[str]], profile.rules["hidden_agency"]),
        "LING-AGENCY-001",
        "agency",
        "Name the responsible actor and state the action directly.",
        10.0,
    )
    emitted = {(finding.location.start, finding.location.end) for finding in findings}
    for sentence in document.sentences:
        for verb in _passive_verbs(document, sentence):
            if _has_agent(document, verb):
                continue
            subject = _subject_of(document, verb)
            subject_lemma = _lemma(subject) if subject is not None else ""
            if subject_lemma not in IMPERSONAL_SUBJECTS | ABSTRACT_SUBJECTS and not _has_obligation_aux(document, verb):
                continue
            start, end, phrase = _passive_span(document, verb)
            if (start, end) in emitted:
                continue
            emitted.add((start, end))
            findings.append(
                Finding(
                    "LING-AGENCY-001",
                    "agency",
                    "medium",
                    _location(document.text, start, end, sentence.index),
                    {"passive_predicate": phrase, "subject": subject.text if subject is not None else None},
                    {"explicit_actor_required": 1},
                    "Name the responsible actor and state the action directly.",
                    10.0,
                )
            )
    return findings


def _indirect_predicate_findings(document: Document, profile: Profile) -> list[Finding]:
    findings = _phrase_findings(
        document,
        cast(dict[str, list[str]], profile.rules["indirect_predicates"]),
        "LING-INDIRECT-PREDICATE-001",
        "agency",
        "Replace the expletive or purpose wrapper with a direct subject and predicate.",
        6.0,
    )
    emitted = {(finding.location.start, finding.location.end) for finding in findings}
    for sentence in document.sentences:
        for token in sentence.tokens:
            if token.dep == "expl":
                start, end = sentence.start, sentence.end
            elif _lemma(token) == "it" and token.dep in SUBJECT_DEPS:
                head = document.token(token.head)
                complement = any(_lemma(child) in INDIRECT_COMPLEMENTS for child in document.children(head))
                has_that_clause = any(child.dep == "ccomp" for child in document.children(head))
                if not (complement and has_that_clause):
                    continue
                start, end = sentence.start, sentence.end
            elif _lemma(token) in PURPOSE_WRAPPERS and token.dep in SUBJECT_DEPS:
                head = document.token(token.head)
                if not any(child.dep == "prep" and _lemma(child) == "of" for child in document.children(token)):
                    continue
                if not any(child.dep == "xcomp" and child.pos == "VERB" for child in document.children(head)):
                    continue
                start, end = sentence.start, sentence.end
            else:
                continue
            if (start, end) in emitted:
                continue
            emitted.add((start, end))
            findings.append(
                Finding(
                    "LING-INDIRECT-PREDICATE-001",
                    "agency",
                    "medium",
                    _location(document.text, start, end, sentence.index),
                    {"text": document.text[start:end]},
                    {"direct_predicate_required": 1},
                    "Replace the expletive or purpose wrapper with a direct subject and predicate.",
                    6.0,
                )
            )
    return findings


def _has_particle(document: Document, verb: Token, particle: str) -> bool:
    return any(child.dep == "prt" and _lemma(child) == particle for child in document.children(verb))


def _direct_objects(document: Document, verb: Token) -> tuple[Token, ...]:
    return tuple(child for child in document.children(verb) if child.dep in {"dobj", "obj", "attr"})


def _weak_verb_findings(document: Document, profile: Profile) -> list[Finding]:
    findings = _phrase_findings(
        document,
        cast(dict[str, list[str]], profile.rules["weak_verbs"]),
        "LING-WEAK-VERB-001",
        "morphology",
        "Replace the weak construction with one precise verb.",
        7.0,
    )
    spans = {(finding.location.start, finding.location.end) for finding in findings}
    for sentence in document.sentences:
        for verb in sentence.tokens:
            if verb.pos != "VERB":
                continue
            lemma = _lemma(verb)
            objects = _direct_objects(document, verb)
            is_light_verb = lemma in LIGHT_VERBS and (lemma != "carry" or _has_particle(document, verb, "out"))
            if is_light_verb:
                nominal_objects = [obj for obj in objects if _is_nominalization(obj, profile)]
                if nominal_objects:
                    obj = nominal_objects[0]
                    start, end = verb.start, obj.end
                    if (start, end) not in spans:
                        spans.add((start, end))
                        findings.append(
                            Finding(
                                "LING-WEAK-VERB-001",
                                "morphology",
                                "medium",
                                _location(document.text, start, end, sentence.index),
                                {"verb": verb.text, "object": obj.text},
                                {"light_verb_nominalizations": 0},
                                "Replace the weak construction with one precise verb.",
                                7.0,
                            )
                        )
            for xcomp in _children(document, verb, "xcomp"):
                xcomp_objects = _direct_objects(document, xcomp)
                if any(_is_nominalization(obj, profile) for obj in xcomp_objects):
                    start, end = verb.start, max(obj.end for obj in xcomp_objects)
                    if (start, end) in spans:
                        continue
                    spans.add((start, end))
                    findings.append(
                        Finding(
                            "LING-WEAK-VERB-001",
                            "morphology",
                            "medium",
                            _location(document.text, start, end, sentence.index),
                            {"verb": verb.text, "xcomp": xcomp.text},
                            {"light_verb_nominalizations": 0},
                            "Replace the weak construction with one precise verb.",
                            7.0,
                        )
                    )
    return findings


def _normalized_acronym(text: str) -> str | None:
    if 2 <= len(text) <= 8 and text.isalpha() and text.upper() == text:
        return text
    if 3 <= len(text) <= 9 and text.endswith("s") and text[:-1].isalpha() and text[:-1].upper() == text[:-1]:
        return text[:-1]
    return None


def _defined_acronyms(document: Document) -> set[str]:
    defined: set[str] = set()
    tokens = document.tokens
    for index, token in enumerate(tokens):
        if token.text != "(" or index + 2 >= len(tokens) or tokens[index + 2].text != ")":
            continue
        acronym = _normalized_acronym(tokens[index + 1].text)
        if acronym is None:
            continue
        before_words = [prior for prior in tokens[max(0, index - 8) : index] if prior.is_alpha]
        after_words = [later for later in tokens[index + 3 : min(len(tokens), index + 11)] if later.is_alpha]
        if before_words or after_words:
            defined.add(acronym)
    return defined


def _acronym_tokens(tokens: Iterable[Token], allowed: frozenset[str], defined: set[str]) -> list[Token]:
    matches: list[Token] = []
    for token in tokens:
        acronym = _normalized_acronym(token.text)
        if acronym is None or acronym in allowed or acronym in defined:
            continue
        matches.append(token)
    return matches


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    position = 0
    for line in text.splitlines(keepends=True):
        line_start = position
        line_end = position + len(line)
        content = line.rstrip("\r\n")
        if content.strip():
            if start is None:
                start = line_start
        elif start is not None:
            spans.append((start, line_start))
            start = None
        position = line_end
    if start is not None:
        spans.append((start, len(text)))
    if not spans and text.strip():
        spans.append((len(text) - len(text.lstrip()), len(text.rstrip())))
    return [(start, end) for start, end in spans if start < end]


def _abbreviation_findings(document: Document, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    allowed = frozenset(cast(list[str], profile.rules["known_acronyms"]))
    defined = _defined_acronyms(document)
    max_sentence = int(profile.thresholds["max_undefined_acronyms_per_sentence"])
    max_paragraph = int(profile.thresholds["max_undefined_acronyms_per_paragraph"])
    sentence_spans: list[tuple[int, int]] = []
    for sentence in document.sentences:
        matches = _acronym_tokens(sentence.tokens, allowed, defined)
        if len(matches) <= max_sentence:
            continue
        acronyms = sorted({cast(str, _normalized_acronym(match.text)) for match in matches})
        sentence_spans.append((sentence.start, sentence.end))
        findings.append(
            Finding(
                "LING-ABBREVIATION-001",
                "lexical_clarity",
                _severity(len(matches), max_sentence),
                _location(document.text, matches[0].start, matches[-1].end, sentence.index),
                {"undefined_acronyms": cast(list[JsonValue], acronyms), "count": len(matches)},
                max_sentence,
                "Expand each acronym on first use or add genuinely common terms to the profile allow-list.",
                min(12.0, (len(matches) - max_sentence) * 3.0),
            )
        )
    for paragraph_start, paragraph_end in _paragraph_spans(document.text):
        if _overlaps(paragraph_start, paragraph_end, sentence_spans):
            continue
        matches = _acronym_tokens(
            (token for token in document.tokens if paragraph_start <= token.start and token.end <= paragraph_end),
            allowed,
            defined,
        )
        if len(matches) <= max_paragraph:
            continue
        acronyms = sorted({cast(str, _normalized_acronym(match.text)) for match in matches})
        findings.append(
            Finding(
                "LING-ABBREVIATION-001",
                "lexical_clarity",
                _severity(len(matches), max_paragraph),
                _location(document.text, matches[0].start, matches[-1].end, None),
                {"undefined_acronyms": cast(list[JsonValue], acronyms), "count": len(matches)},
                max_paragraph,
                "Expand each acronym on first use or add genuinely common terms to the profile allow-list.",
                min(12.0, (len(matches) - max_paragraph) * 2.0),
            )
        )
    return findings


def _paragraph_findings(document: Document, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    threshold = int(profile.thresholds["max_paragraph_words"])
    for start, end in _paragraph_spans(document.text):
        count = sum(1 for token in document.tokens if start <= token.start and token.end <= end and token.is_word)
        if count <= threshold:
            continue
        findings.append(
            Finding(
                "LING-STRUCTURE-001",
                "structure",
                _severity(count, threshold),
                _location(document.text, start, end, None),
                count,
                threshold,
                "Break the paragraph into a decision, rationale, and required actions.",
                min(20.0, (count - threshold) * 0.75),
            )
        )
    return findings


def _punctuation_depth(sentence: Sentence) -> dict[str, int]:
    parenthetical_marks = sentence.text.count("(") + sentence.text.count(")") + sentence.text.count("[") + sentence.text.count("]")
    parentheticals = parenthetical_marks // 2
    dashes = sentence.text.count("—") + sentence.text.count("–")
    semicolons = sentence.text.count(";")
    commas = sentence.text.count(",")
    score = parentheticals * 2 + semicolons * 2 + dashes + max(0, commas - 2)
    return {"score": score, "parentheticals": parentheticals, "dashes": dashes, "semicolons": semicolons, "commas": commas}


def _punctuation_finding(document: Document, sentence: Sentence, profile: Profile) -> Finding | None:
    threshold = int(profile.thresholds["max_punctuation_depth"])
    observed = _punctuation_depth(sentence)
    score = observed["score"]
    if score <= threshold:
        return None
    return Finding(
        "LING-PUNCTUATION-001",
        "sentence_load",
        _severity(score, threshold),
        _location(document.text, sentence.start, sentence.end, sentence.index),
        cast(dict[str, JsonValue], observed),
        threshold,
        "Split parenthetical, semicolon, dash, or deeply comma-nested interruptions into simpler sentences or bullets.",
        min(14.0, (score - threshold) * 2.0),
    )


def _inline_list_items(sentence: Sentence) -> list[tuple[int, int]]:
    if sentence.text.count(",") == 0:
        return []
    items: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end: int | None = None
    for token in sentence.tokens:
        lemma = _lemma(token)
        is_separator = token.text == "," or (lemma in {"and", "or"} and sentence.text.count(",") > 0)
        if is_separator:
            if current_start is not None and current_end is not None:
                items.append((current_start, current_end))
            current_start = None
            current_end = None
            continue
        if token.is_word:
            current_start = token.start if current_start is None else current_start
            current_end = token.end
    if current_start is not None and current_end is not None:
        items.append((current_start, current_end))
    return [item for item in items if item[0] < item[1]]


def _list_suitability_findings(document: Document, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    threshold = int(profile.thresholds["max_inline_list_items"])
    for sentence in document.sentences:
        items = _inline_list_items(sentence)
        if len(items) <= threshold:
            continue
        findings.append(
            Finding(
                "LING-LIST-001",
                "structure",
                _severity(len(items), threshold),
                _location(document.text, sentence.start, sentence.end, sentence.index),
                {"items": len(items), "text": sentence.text},
                threshold,
                "Convert the long inline enumeration into a bulleted list with one item per line.",
                min(12.0, (len(items) - threshold) * 2.0),
            )
        )
    return findings


def _mixed_purpose_findings(document: Document, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    purpose_markers = cast(dict[str, list[str]], profile.rules["purpose_markers"])
    threshold = int(profile.thresholds["max_sentence_purposes"])
    for sentence in document.sentences:
        purposes: list[str] = []
        for label, phrases in sorted(purpose_markers.items()):
            if any(_phrase_matches(sentence, phrase) for phrase in phrases):
                purposes.append(label)
        if len(purposes) <= threshold:
            continue
        findings.append(
            Finding(
                "LING-MIXED-PURPOSE-001",
                "structure",
                _severity(len(purposes), threshold),
                _location(document.text, sentence.start, sentence.end, sentence.index),
                {"purposes": cast(list[JsonValue], purposes), "count": len(purposes)},
                threshold,
                "Separate the decision, remediation work, evidence, and exit criteria into distinct sentences or bullets.",
                min(14.0, (len(purposes) - threshold) * 4.0),
            )
        )
    return findings


def _recommendation_group(lemma: str, groups: dict[str, list[str]]) -> str | None:
    for label, words in sorted(groups.items()):
        if lemma in {word.lower() for word in words}:
            return label
    return None


def _recommendation_signature(document: Document, sentence: Sentence, profile: Profile) -> tuple[str, list[str]] | None:
    if _directive_marker(document, sentence) is None:
        return None
    groups = cast(dict[str, list[str]], profile.rules["recommendation_action_groups"])
    actor_terms = _profile_words(profile, "actor_terms")
    group: str | None = None
    for token in sentence.tokens:
        candidate = _recommendation_group(_lemma(token), groups)
        if candidate is not None and token.pos == "VERB":
            group = candidate
            break
    if group is None:
        return None
    terms = sorted(
        {
            _lemma(token)
            for token in sentence.tokens
            if len(_lemma(token)) >= 5
            and token.is_word
            and _lemma(token) not in STOP_LEMMAS
            and _lemma(token) not in actor_terms
            and _lemma(token) not in OBLIGATION_AUXILIARIES
            and token.pos not in {"AUX", "DET", "PUNCT"}
        }
    )
    if not terms:
        return None
    return (group, terms)


def _duplicated_recommendation_findings(document: Document, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    seen: dict[str, list[tuple[Sentence, set[str]]]] = defaultdict(list)
    for sentence in document.sentences:
        signature = _recommendation_signature(document, sentence, profile)
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
                    _location(document.text, sentence.start, sentence.end, sentence.index),
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


def _qualifier_findings(document: Document, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    qualifiers = _profile_words(profile, "qualifiers")
    threshold = int(profile.thresholds["max_qualifiers_per_sentence"])
    for sentence in document.sentences:
        indexes = [index for index, token in enumerate(sentence.tokens) if _lemma(token) in qualifiers]
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
                    _location(document.text, left_token.start, right_token.end, sentence.index),
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
                _location(document.text, sentence.tokens[indexes[0]].start, sentence.tokens[indexes[-1]].end, sentence.index),
                {"qualifiers": cast(list[JsonValue], terms), "count": len(indexes)},
                threshold,
                "Remove repeated hedges or intensifiers unless each one changes the obligation.",
                min(10.0, (len(indexes) - threshold) * 3.0),
            )
        )
    return findings


def _redundancy_findings(document: Document, profile: Profile) -> list[Finding]:
    findings: list[Finding] = []
    content_tokens = [
        token
        for token in document.tokens
        if token.is_word and len(_lemma(token)) >= 7 and _lemma(token) not in STOP_LEMMAS
    ]
    counts = Counter(_lemma(token) for token in content_tokens)
    threshold = int(profile.thresholds["max_repeated_content_word"])
    for lemma, count in sorted(counts.items()):
        if count <= threshold:
            continue
        first = next(token for token in content_tokens if _lemma(token) == lemma)
        findings.append(
            Finding(
                "LING-REDUNDANCY-001",
                "redundancy",
                "low",
                _location(document.text, first.start, first.end, None),
                {"term": lemma, "occurrences": count},
                threshold,
                "Remove repeated qualifiers or consolidate repeated statements.",
                min(10.0, (count - threshold) * 2.0),
            )
        )
    return findings


def _analyze_sentences(document: Document, profile: Profile) -> tuple[list[dict[str, JsonValue]], list[Finding]]:
    records: list[dict[str, JsonValue]] = []
    findings: list[Finding] = []
    max_words = int(profile.thresholds["max_sentence_words"])
    max_actions = int(profile.thresholds["max_actions_per_sentence"])
    max_clauses = int(profile.thresholds["max_clauses_per_sentence"])
    max_nominalizations = int(profile.thresholds["max_nominalizations_per_sentence"])

    for sentence in document.sentences:
        word_count = len(sentence.words)
        action_count = _action_count(document, sentence)
        clause_count = _clause_count(document, sentence)
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
        location = _location(document.text, sentence.start, sentence.end, sentence.index)
        if word_count > max_words:
            findings.append(
                Finding(
                    "LING-SENTENCE-001",
                    "sentence_load",
                    _severity(word_count, max_words),
                    location,
                    word_count,
                    max_words,
                    "Split the sentence so each sentence carries one main purpose.",
                    min(30.0, (word_count - max_words) * 1.5),
                )
            )
        if action_count > max_actions:
            findings.append(
                Finding(
                    "LING-ACTION-001",
                    "sentence_load",
                    _severity(action_count, max_actions),
                    location,
                    action_count,
                    max_actions,
                    "Separate the decision, actions, and exit criteria into distinct sentences.",
                    min(24.0, (action_count - max_actions) * 6.0),
                )
            )
        if clause_count > max_clauses:
            findings.append(
                Finding(
                    "LING-CLAUSE-001",
                    "sentence_load",
                    _severity(clause_count, max_clauses),
                    location,
                    clause_count,
                    max_clauses,
                    "Reduce subordinate clauses or split the sentence.",
                    min(18.0, (clause_count - max_clauses) * 4.0),
                )
            )
        punctuation = _punctuation_finding(document, sentence, profile)
        if punctuation is not None:
            findings.append(punctuation)
        nominalizations = _nominalizations(sentence, profile)
        if len(nominalizations) > max_nominalizations:
            findings.append(
                Finding(
                    "LING-NOMINALIZATION-001",
                    "morphology",
                    _severity(len(nominalizations), max_nominalizations),
                    location,
                    {"count": len(nominalizations), "terms": cast(list[JsonValue], nominalizations)},
                    max_nominalizations,
                    "Replace abstract action nouns with direct verbs where meaning permits.",
                    min(25.0, (len(nominalizations) - max_nominalizations) * 5.0),
                )
            )
    findings.extend(_noun_stack_findings(document, profile))
    findings.extend(_passive_findings(document))
    return records, findings


def _abstract_evidence_findings(document: Document, profile: Profile) -> list[Finding]:
    """Flag "<nominalization> evidence" compounds structurally.

    "closure evidence", "attestation evidence" and "remediation evidence" are
    the same construction: an abstract noun standing in for the act that would
    actually produce the evidence. Listing the phrases seen in one fixture
    detects that fixture, not the construction, so the modifier is tested for
    being a nominalization instead.
    """

    heads = {
        word.lower()
        for word in cast(list[str], profile.rules["abstract_evidence_heads"])
    }
    findings: list[Finding] = []
    for token in document:
        if token.lower not in heads and _lemma(token) not in heads:
            continue
        for child in document.children(token):
            if child.dep != "compound" or child.index >= token.index:
                continue
            if not _is_nominalization(child, profile):
                continue
            phrase = f"{child.text} {token.text}"
            findings.append(
                Finding(
                    rule_id="LING-JARGON-001",
                    dimension="lexical_clarity",
                    severity="medium",
                    location=_location(
                        document.text,
                        child.start,
                        token.end,
                        document.sentence_of(token).index,
                    ),
                    observed_value={
                        "phrase": phrase,
                        "classification": "abstract evidence phrase",
                        "nominalization": child.text,
                    },
                    threshold={
                        "rule": "a nominalization must not stand in as the modifier of an evidence noun"
                    },
                    remediation=(
                        f"Name the act and who performs it, rather than {phrase!r}: "
                        f"say who must show that the work was done."
                    ),
                    penalty=6.0,
                )
            )
    return findings


def _dedupe_findings(findings: list[Finding]) -> list[Finding]:
    result: list[Finding] = []
    seen: set[tuple[str, int, int, str]] = set()
    for finding in sorted(findings, key=lambda item: (item.location.start, item.location.end, item.rule_id, canonical_json(item.observed_value))):
        key = (finding.rule_id, finding.location.start, finding.location.end, canonical_json(finding.observed_value))
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result


def analyze_text(text: str, profile: Profile | None = None) -> dict[str, JsonValue]:
    if not text.strip():
        raise ValueError("Input text must not be empty")
    selected_profile = profile or load_profile()
    document = parse(text)
    sentence_records, findings = _analyze_sentences(document, selected_profile)

    hidden_agency_findings = _hidden_agency_findings(document, selected_profile)
    findings.extend(hidden_agency_findings)
    agency_spans = [(finding.location.start, finding.location.end) for finding in hidden_agency_findings]
    findings.extend(_actor_action_findings(document, selected_profile, agency_spans))
    findings.extend(
        finding
        for finding in _indirect_predicate_findings(document, selected_profile)
        if not _overlaps(finding.location.start, finding.location.end, agency_spans)
    )
    findings.extend(_weak_verb_findings(document, selected_profile))
    jargon_findings = _phrase_findings(
        document,
        cast(dict[str, list[str]], selected_profile.rules["jargon"]),
        "LING-JARGON-001",
        "lexical_clarity",
        "Use the profile's plain-language alternative.",
        6.0,
    )
    findings.extend(jargon_findings)
    findings.extend(_abstract_evidence_findings(document, selected_profile))
    lexical_spans = [(finding.location.start, finding.location.end) for finding in jargon_findings]
    compound_depth_findings = _compound_depth_findings(document, selected_profile)
    findings.extend(compound_depth_findings)
    compound_depth_spans = [(finding.location.start, finding.location.end) for finding in compound_depth_findings]
    findings.extend(_compound_findings(document, selected_profile, lexical_spans + compound_depth_spans))
    findings.extend(_abbreviation_findings(document, selected_profile))
    findings.extend(
        _phrase_findings(
            document,
            cast(dict[str, list[str]], selected_profile.rules["bureaucratic_phrases"]),
            "LING-BUREAUCRACY-001",
            "lexical_clarity",
            "State the recommendation or action directly.",
            8.0,
        )
    )
    findings.extend(
        _phrase_findings(
            document,
            cast(dict[str, list[str]], selected_profile.rules["filler_phrases"]),
            "LING-FILLER-001",
            "redundancy",
            "Remove the filler phrase and keep the governed condition or action.",
            4.0,
        )
    )
    findings.extend(_paragraph_findings(document, selected_profile))
    findings.extend(_list_suitability_findings(document, selected_profile))
    findings.extend(_mixed_purpose_findings(document, selected_profile))
    findings.extend(_redundancy_findings(document, selected_profile))
    findings.extend(_duplicated_recommendation_findings(document, selected_profile))
    findings.extend(_qualifier_findings(document, selected_profile))
    findings = _dedupe_findings(findings)

    result: dict[str, JsonValue] = {
        "schema_version": "1.0.0",
        "analyzer_version": ANALYZER_VERSION,
        "linguistic_model": cast(dict[str, JsonValue], model_fingerprint()),
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
