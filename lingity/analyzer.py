"""Deterministic linguistic analysis."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, cast

from lingity.invariants import extract_protected, source_sha256
from lingity.models import Finding, JsonValue, Location
from lingity.profiles import Profile, canonical_json, load_profile
from lingity.scoring import calculate_hri
from lingity.text import Sentence, line_column, sentences

ANALYZER_VERSION = "1.0.0"
MODEL_NAME = "lingity-regex-en"
MODEL_VERSION = "1.0.0"

CLAUSE_RE = re.compile(r"[,;:]|\b(?:and|but|because|while|although|before|after|unless|which|that)\b", re.IGNORECASE)
PASSIVE_RE = re.compile(
    r"\b(?:am|is|are|was|were|be|been|being)\s+(?:\w+\s+){0,2}\w+(?:ed|en)\b",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z0-9]+)*")
VERB_SUFFIX_RE = re.compile(r"(?:ate|ify|ise|ize|ing|ed)$", re.IGNORECASE)

STOPWORDS = {
    "a", "an", "and", "any", "as", "at", "before", "but", "by", "do", "for",
    "from", "in", "into", "is", "it", "not", "of", "on", "only", "or", "our",
    "that", "the", "then", "this", "to", "with", "yet",
}

COMMON_VERBS = {
    "address", "approve", "begin", "bring", "can", "collect", "defer", "determine",
    "fix", "found", "have", "keep", "lose", "move", "provide", "propose", "require",
    "return", "review", "run", "should", "treat", "use", "verify",
}


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


def _action_count(sentence: Sentence) -> int:
    count = 0
    for token in sentence.tokens:
        word = token.text.lower().strip("-")
        if word in COMMON_VERBS or VERB_SUFFIX_RE.search(word):
            count += 1
    return count


def _nominalizations(sentence: Sentence, profile: Profile) -> list[str]:
    suffixes = tuple(cast(list[str], profile.rules["nominalization_suffixes"]))
    exclusions = set(cast(list[str], profile.rules["nominalization_exclusions"]))
    explicit = set(cast(list[str], profile.rules["nominalizations"]))
    result: list[str] = []
    for token in sentence.tokens:
        word = token.text.lower()
        if word in explicit or (word not in exclusions and word.endswith(suffixes)):
            result.append(token.text)
    return result


def _noun_stacks(sentence: Sentence, profile: Profile) -> list[tuple[int, int, str]]:
    minimum = int(profile.thresholds["noun_stack_words"])
    result: list[tuple[int, int, str]] = []
    run: list[Any] = []
    for token in sentence.tokens:
        word = token.text.lower()
        noun_like = (
            word not in STOPWORDS
            and word not in COMMON_VERBS
            and not word.endswith("ly")
            and not word.endswith("ing")
            and not word.endswith("ed")
        )
        if noun_like:
            run.append(token)
        else:
            if len(run) >= minimum:
                result.append((run[0].start, run[-1].end, " ".join(item.text for item in run)))
            run = []
    if len(run) >= minimum:
        result.append((run[0].start, run[-1].end, " ".join(item.text for item in run)))
    return result


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
        clause_count = 1 + len(CLAUSE_RE.findall(sentence.text))
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
        for match in PASSIVE_RE.finditer(sentence.text):
            start = sentence.start + match.start()
            end = sentence.start + match.end()
            findings.append(
                Finding(
                    "LING-PASSIVE-001", "agency", "medium",
                    _location(text, start, end, sentence.index),
                    match.group(0), {"allowed_occurrences": 0},
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
    findings.extend(
        _phrase_findings(
            text,
            cast(dict[str, list[str]], selected_profile.rules["hidden_agency"]),
            "LING-AGENCY-001",
            "agency",
            "Name the responsible actor and state the action directly.",
            10.0,
        )
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
    findings.extend(
        _phrase_findings(
            text,
            cast(dict[str, list[str]], selected_profile.rules["jargon"]),
            "LING-JARGON-001",
            "lexical_clarity",
            "Use the profile's plain-language alternative.",
            6.0,
        )
    )
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
    findings.extend(_paragraph_findings(text, selected_profile))
    findings.extend(_redundancy_findings(text, selected_profile))
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
