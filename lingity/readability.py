"""Deterministic grade-level readability estimates.

The classic readability formulas -- Flesch-Kincaid, Flesch Reading Ease,
Coleman-Liau, SMOG, Gunning Fog and ARI -- are functions of five surface counts:
words, sentences, syllables, characters and polysyllables. They are estimates,
not measurements: they disagree with each other by several grades on dense
text, and none of them reads meaning. Lingity reports several side by side, with
the exact formula and the exact counts each one used, so a reader can see why
two of them disagree instead of trusting either.

Everything here is deterministic and offline:

* Sentences are the analyzer's own sentences, so a grade line and a finding
  point at the same span.
* Syllables come from a vendored table derived from a pinned CMUdict release
  (``lingity/data/cmudict-syllables.tsv``). The table's hash is pinned in this
  module and checked when it loads; a different table fails closed. Words the
  dictionary does not list fall back to a documented spelling rule,
  ``FALLBACK_RULE``.
* Figures, codes, citations, URLs and inline code are normalized by a
  documented rule, ``NORMALIZATION_RULE``, because a figure such as
  ``$11,898,367,000`` is one token to a reader but eleven characters to
  Coleman-Liau and ARI.

Nothing in this module changes the Human Readability Index. The only path from
these estimates into a score is ``LING-GRADE-001``, which fires only when a
profile sets ``max_grade``.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from pathlib import Path
from typing import Iterable, cast

from lingity.invariants import CITATION_RE, URL_RE
from lingity.markdown import (
    inline_code_spans,
    is_markdown_link,
    normalize_reference_label,
    reference_labels,
)
from lingity.models import JsonValue
from lingity.nlp import Document

READABILITY_VERSION = "1.0.0"

DATA_DIR = Path(__file__).resolve().parent / "data"
SYLLABLE_TABLE = DATA_DIR / "cmudict-syllables.tsv"
SYLLABLE_SOURCE: dict[str, JsonValue] = {
    "name": "CMUdict",
    "repository": "https://github.com/cmusphinx/cmudict",
    "commit": "74790861f652b15e4ac49015a90074ad62a27690",
    "source_file": "cmudict.dict",
    "source_sha256": "81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22",
    "license": "BSD-2-Clause-style; Copyright (C) 1993-2015 Carnegie Mellon University",
    "table": "lingity/data/cmudict-syllables.tsv",
    "table_sha256": "9f291ef6e5e22b6a24e134e9f0adee4ef7d477cfd3aed08245219c5612703b73",
    "entries": 126052,
    "derivation": (
        "syllables = number of phones carrying a stress digit (0, 1 or 2) in the "
        "first pronunciation CMUdict lists for the headword; built by "
        "scripts/build_syllable_table.py"
    ),
}

FALLBACK_RULE = (
    "for a word the table does not list: lowercase it and keep only a-z; count "
    "maximal runs of the vowels a, e, i, o, u and y, where a word-initial y is a "
    "consonant; subtract 1 for a final 'e' unless the word ends in a consonant "
    "followed by 'le'; subtract 1 for a final 'ed' unless 't' or 'd' precedes "
    "it; the result is never below 1. A word joined by hyphens, slashes or "
    "periods is looked up whole first, then as the sum of its parts. A word with "
    "an apostrophe is looked up whole first, then by the part before the "
    "apostrophe."
)

FIGURE_SYLLABLES = 1
FIGURE_CHARACTERS = 3
NORMALIZATION_RULE = (
    "a counted word that contains a digit -- an amount, count, percentage, year, "
    "page number or code such as W00A01 -- counts as one word of "
    f"{FIGURE_SYLLABLES} syllable and {FIGURE_CHARACTERS} characters, never a "
    "polysyllable; words inside inline code, a URL, a bracketed reference such "
    "as [ADR-12] that is not a Markdown link, an author-year citation such as "
    "(Smith, 2019) or a page citation such as (p. P.1) are not counted at all, "
    "and a sentence left with no counted word is not counted as a sentence"
)

WORD_DEFINITION = (
    "a maximal run of letters and digits, joined across a single apostrophe, "
    "hyphen, period, comma, colon or slash that has a letter or digit on both "
    "sides; so 11,898,367,000, P.1, don't and well-known are one word each"
)
SENTENCE_DEFINITION = (
    "a sentence of the analyzer's parse (the same sentence_index findings use) "
    "that holds at least one counted word"
)
CHARACTER_DEFINITION = "letters and digits in counted words; punctuation and spaces are not characters"
POLYSYLLABLE_DEFINITION = (
    "a counted word of three or more syllables; Gunning Fog's proper-noun, "
    "compound and suffix exclusions are not applied"
)
ROUNDING = "every value is computed in IEEE-754 double precision, then rounded half up to 2 decimals"

FORMULAS: dict[str, str] = {
    "flesch_kincaid_grade": "0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59",
    "flesch_reading_ease": "206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)",
    "coleman_liau_index": "0.0588 * (100 * characters / words) - 0.296 * (100 * sentences / words) - 15.8",
    "smog_grade": "1.0430 * sqrt(polysyllables * 30 / sentences) + 3.1291",
    "gunning_fog": "0.4 * ((words / sentences) + 100 * (polysyllables / words))",
    "automated_readability_index": "4.71 * (characters / words) + 0.5 * (words / sentences) - 21.43",
}
FORMULA_COUNTS: dict[str, tuple[str, ...]] = {
    "flesch_kincaid_grade": ("words", "sentences", "syllables"),
    "flesch_reading_ease": ("words", "sentences", "syllables"),
    "coleman_liau_index": ("words", "sentences", "characters"),
    "smog_grade": ("sentences", "polysyllables"),
    "gunning_fog": ("words", "sentences", "polysyllables"),
    "automated_readability_index": ("words", "sentences", "characters"),
}
GRADE_FORMULA = "flesch_kincaid_grade"
GRADE_RULE_ID = "LING-GRADE-001"
DEFAULT_GRADE_MIN_WORDS = 5


def validate_target_grade(value: float | None) -> None:
    """Reject targets that cannot be represented in a JSON rewrite contract."""
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("target_grade must be a finite non-negative number")

_WORD_RE = re.compile(r"[^\W_]+(?:['’\-.,:/][^\W_]+)*")
_PAGE_CITATION_RE = re.compile(r"\((?:p|pp)\.\s*[^()\n]*\)")
_VOWEL_RUN_RE = re.compile(r"[aeiouy]+")
_PART_SPLIT_RE = re.compile(r"[-./,:]")


class SyllableTableError(RuntimeError):
    """Raised when the vendored syllable table is missing or is not the pinned one."""


@lru_cache(maxsize=1)
def syllable_table() -> dict[str, int]:
    """Load the vendored table, refusing any bytes other than the pinned ones."""
    try:
        raw = SYLLABLE_TABLE.read_bytes()
    except OSError as error:
        raise SyllableTableError(f"syllable table {SYLLABLE_TABLE} is not readable: {error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SYLLABLE_SOURCE["table_sha256"]:
        raise SyllableTableError(
            f"syllable table {SYLLABLE_TABLE} has sha256 {digest}, not the pinned "
            f"{SYLLABLE_SOURCE['table_sha256']}; readability estimates are only "
            "reproducible against the pinned table"
        )
    table: dict[str, int] = {}
    for line in raw.decode("utf-8").splitlines():
        word, count = line.split("\t")
        table[word] = int(count)
    return table


def fallback_syllables(word: str) -> int:
    """Estimate syllables from spelling alone; see ``FALLBACK_RULE``."""
    letters = "".join(character for character in word.lower() if "a" <= character <= "z")
    if not letters:
        return 1
    body = letters[1:] if letters[0] == "y" else letters
    count = len(_VOWEL_RUN_RE.findall(body))
    if letters[0] == "y" and len(letters) > 1 and letters[1] not in "aeiouy":
        # "y" before a consonant opens a syllable ("ypsilanti"); before a vowel
        # it is a glide and its vowel run is already counted.
        count += 1
    syllabic_le = letters.endswith("le") and len(letters) > 2 and letters[-3] not in "aeiouy"
    if letters.endswith("e") and count > 1 and not syllabic_le:
        count -= 1
    if letters.endswith("ed") and count > 1 and len(letters) > 2 and letters[-3] not in "td":
        count -= 1
    return max(1, count)


def syllables(word: str) -> tuple[int, bool]:
    """Return a word's syllable count and whether the dictionary supplied it."""
    table = syllable_table()
    key = word.lower().replace("’", "'")
    if key in table:
        return table[key], True
    if "'" in key:
        base = key.split("'", 1)[0]
        if base in table:
            return table[base], True
        return fallback_syllables(base or key), False
    parts = [part for part in _PART_SPLIT_RE.split(key) if part]
    if len(parts) > 1 and any(character.isalpha() for character in key):
        total = 0
        known = True
        for part in parts:
            count, found = syllables(part)
            total += count
            known = known and found
        return total, known
    return fallback_syllables(key), False


@dataclass(frozen=True)
class Counts:
    words: int = 0
    sentences: int = 0
    syllables: int = 0
    characters: int = 0
    polysyllables: int = 0
    figures: int = 0
    excluded_words: int = 0
    dictionary_words: int = 0
    fallback_words: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(
            *(getattr(self, name) + getattr(other, name) for name in self.__dataclass_fields__)
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {name: cast(int, getattr(self, name)) for name in self.__dataclass_fields__}


def _closing_bracket(text: str, start: int, opener: str, closer: str) -> int:
    """Find a matching Markdown delimiter, allowing nesting and escapes."""
    depth = 0
    for position in range(start, len(text)):
        backslashes = 0
        previous = position - 1
        while previous >= 0 and text[previous] == "\\":
            backslashes += 1
            previous -= 1
        if backslashes % 2:
            continue
        if text[position] == opener:
            depth += 1
        elif text[position] == closer:
            depth -= 1
            if depth == 0:
                return position
    return -1


def _links(
    text: str, references: frozenset[str], code: tuple[tuple[int, int], ...]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Locate link destinations and visible labels in source Markdown."""
    excluded: list[tuple[int, int]] = []
    labels: list[tuple[int, int]] = []
    position = 0
    while position < len(text):
        if text[position] != "[" or _inside(code, position, position + 1):
            position += 1
            continue
        close = _closing_bracket(text, position, "[", "]")
        if close < 0:
            position += 1
            continue
        label = text[position + 1 : close]
        next_position = close + 1
        destination_end = -1
        if next_position < len(text) and text[next_position] == "(":
            destination_end = _closing_bracket(text, next_position, "(", ")")
        elif next_position < len(text) and text[next_position] == "[":
            reference_end = _closing_bracket(text, next_position, "[", "]")
            if reference_end >= 0:
                reference = text[next_position + 1 : reference_end] or label
                if normalize_reference_label(reference) in references:
                    destination_end = reference_end
        elif normalize_reference_label(label) in references:
            destination_end = close
        if destination_end >= 0 and is_markdown_link(
            text[position : destination_end + 1], references
        ):
            labels.append((position, close + 1))
            if destination_end > close:
                excluded.append((close + 1, destination_end + 1))
            position = destination_end + 1
        else:
            position = close + 1
    return excluded, labels


def _excluded_spans(text: str, references: frozenset[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    code = inline_code_spans(text)
    spans.extend(code)
    link_destinations, link_labels = _links(text, references, code)
    spans.extend(link_destinations)
    for pattern in (URL_RE, _PAGE_CITATION_RE):
        spans.extend(match.span() for match in pattern.finditer(text))
    for match in CITATION_RE.finditer(text):
        # Markdown link labels are prose, including reference and shortcut links.
        if match.group(0).startswith("[") and _inside(link_labels, *match.span()):
            continue
        spans.append(match.span())
    return spans


def _inside(spans: Iterable[tuple[int, int]], start: int, end: int) -> bool:
    return any(span_start <= start and end <= span_end for span_start, span_end in spans)


def count_text(
    text: str,
    normalize: bool = True,
    references: frozenset[str] = frozenset(),
    precomputed_exclusions: tuple[tuple[int, int], ...] | None = None,
) -> Counts:
    """Count one sentence's text under the normalization rule, or without it."""
    excluded = (
        list(precomputed_exclusions)
        if normalize and precomputed_exclusions is not None
        else _excluded_spans(text, references) if normalize else []
    )
    words = syllable_total = characters = polysyllables = figures = 0
    excluded_words = dictionary_words = fallback_words = 0
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        if _inside(excluded, match.start(), match.end()):
            excluded_words += 1
            continue
        words += 1
        if normalize and any(character.isdigit() for character in word):
            figures += 1
            syllable_total += FIGURE_SYLLABLES
            characters += FIGURE_CHARACTERS
            continue
        count, known = syllables(word)
        if known:
            dictionary_words += 1
        else:
            fallback_words += 1
        syllable_total += count
        characters += sum(1 for character in word if character.isalnum())
        if count >= 3:
            polysyllables += 1
    return Counts(
        words=words,
        sentences=1 if words else 0,
        syllables=syllable_total,
        characters=characters,
        polysyllables=polysyllables,
        figures=figures,
        excluded_words=excluded_words,
        dictionary_words=dictionary_words,
        fallback_words=fallback_words,
    )


def _round(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def formula_values(counts: Counts) -> dict[str, JsonValue]:
    """Evaluate every formula in ``FORMULAS`` over one set of counts."""
    if counts.words == 0 or counts.sentences == 0:
        return {name: None for name in FORMULAS}
    words = float(counts.words)
    sentences = float(counts.sentences)
    per_sentence = words / sentences
    per_word = counts.syllables / words
    letters = counts.characters / words
    values = {
        "flesch_kincaid_grade": 0.39 * per_sentence + 11.8 * per_word - 15.59,
        "flesch_reading_ease": 206.835 - 1.015 * per_sentence - 84.6 * per_word,
        "coleman_liau_index": 0.0588 * (100 * letters) - 0.296 * (100 * sentences / words) - 15.8,
        "smog_grade": 1.0430 * math.sqrt(counts.polysyllables * 30 / sentences) + 3.1291,
        "gunning_fog": 0.4 * (per_sentence + 100 * (counts.polysyllables / words)),
        "automated_readability_index": 4.71 * letters + 0.5 * per_sentence - 21.43,
    }
    return {name: _round(values[name]) for name in FORMULAS}


def _estimates(counts: Counts) -> dict[str, JsonValue]:
    values = formula_values(counts)
    shared = counts.to_dict()
    return {
        name: {
            "value": values[name],
            "formula": FORMULAS[name],
            "counts": {key: shared[key] for key in FORMULA_COUNTS[name]},
        }
        for name in FORMULAS
    }


def grade_threshold(thresholds: dict[str, int | float]) -> dict[str, JsonValue] | None:
    """The opt-in grade policy a profile sets, or ``None`` when it sets none."""
    if "max_grade" not in thresholds:
        return None
    return {
        "rule_id": GRADE_RULE_ID,
        "formula": GRADE_FORMULA,
        "max_grade": thresholds["max_grade"],
        "min_words": int(thresholds.get("grade_min_words", DEFAULT_GRADE_MIN_WORDS)),
    }


def readability_estimates(document: Document, thresholds: dict[str, int | float]) -> dict[str, JsonValue]:
    """Build the artifact's ``readability_estimates`` block for a parsed document."""
    sentence_entries: list[JsonValue] = []
    total = Counts()
    raw_total = Counts()
    references = reference_labels(document.text)
    # The linguistic parser joins each group's source spans with one space and
    # then splits that chunk into sentences. Locate markup on that same chunk:
    # a code span or link may cross a sentence or a wrapped source line.
    groups: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
    for group in document.groups:
        if group:
            chunk = " ".join(document.text[start:end] for start, end in group)
            groups.append((group[0][0], group[-1][1], tuple(_excluded_spans(chunk, references))))
    group_index = 0
    for sentence in document.sentences:
        while group_index < len(groups) and sentence.start >= groups[group_index][1]:
            group_index += 1
        if (
            group_index >= len(groups)
            or sentence.start < groups[group_index][0]
            or sentence.end > groups[group_index][1]
        ):
            raise ValueError("readability sentence does not map to a prose block")
        chunk_start = sentence.chunk_start
        chunk_end = chunk_start + len(sentence.text)
        local_exclusions = tuple(
            (max(0, start - chunk_start), min(len(sentence.text), end - chunk_start))
            for start, end in groups[group_index][2]
            if start < chunk_end and end > chunk_start
        )
        counts = count_text(sentence.text, precomputed_exclusions=local_exclusions)
        raw_total = raw_total + count_text(sentence.text, normalize=False)
        total = total + counts
        if counts.words == 0:
            continue
        sentence_entries.append(
            {
                "sentence_index": sentence.index,
                "start": sentence.start,
                "end": sentence.end,
                "counts": counts.to_dict(),
                "values": formula_values(counts),
            }
        )
    return {
        "version": READABILITY_VERSION,
        "method": {
            "word": WORD_DEFINITION,
            "sentence": SENTENCE_DEFINITION,
            "character": CHARACTER_DEFINITION,
            "polysyllable": POLYSYLLABLE_DEFINITION,
            "rounding": ROUNDING,
            "formulas": cast(dict[str, JsonValue], dict(FORMULAS)),
            "syllables": {
                "source": dict(SYLLABLE_SOURCE),
                "pronunciation": "first listed",
                "fallback": FALLBACK_RULE,
            },
            "normalization": {
                "applied": True,
                "rule": NORMALIZATION_RULE,
                "figure_syllables": FIGURE_SYLLABLES,
                "figure_characters": FIGURE_CHARACTERS,
            },
            "grade_threshold": grade_threshold(thresholds),
        },
        "document": {
            "counts": total.to_dict(),
            "estimates": _estimates(total),
            "without_normalization": {
                "counts": raw_total.to_dict(),
                "values": formula_values(raw_total),
            },
        },
        "sentences": sentence_entries,
    }
