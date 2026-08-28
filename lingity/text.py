"""Deterministic English token and sentence spans."""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Za-z](?:\.[A-Za-z])+\.?|[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|[A-Za-z]*\d+[A-Za-z0-9.-]*)"
    r"(?![A-Za-z0-9])"
)
ABBREVIATIONS = frozenset(
    {
        "dr.",
        "e.g.",
        "etc.",
        "fig.",
        "i.e.",
        "mr.",
        "mrs.",
        "ms.",
        "no.",
        "prof.",
        "vs.",
    }
)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Sentence:
    index: int
    text: str
    start: int
    end: int
    tokens: tuple[Token, ...]


def _is_sentence_boundary(text: str, index: int) -> bool:
    mark = text[index]
    if mark not in ".!?":
        return False
    if index + 1 < len(text) and not text[index + 1].isspace():
        return False
    if mark == ".":
        if 0 < index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit():
            return False
        prefix = text[max(0, index - 12):index + 1].lower()
        if any(prefix.endswith(abbreviation) for abbreviation in ABBREVIATIONS):
            return False
    return True


def _append_sentence(result: list[Sentence], text: str, start: int, end: int) -> None:
    raw = text[start:end]
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw.rstrip())
    adjusted_start = start + leading
    adjusted_end = start + trailing
    if adjusted_start >= adjusted_end:
        return
    sentence_text = text[adjusted_start:adjusted_end]
    tokens = tuple(
        Token(token.group(0), adjusted_start + token.start(), adjusted_start + token.end())
        for token in WORD_RE.finditer(sentence_text)
    )
    result.append(Sentence(len(result), sentence_text, adjusted_start, adjusted_end, tokens))


def sentences(text: str) -> tuple[Sentence, ...]:
    result: list[Sentence] = []
    start = 0
    index = 0
    while index < len(text):
        if _is_sentence_boundary(text, index):
            _append_sentence(result, text, start, index + 1)
            start = index + 1
        index += 1
    _append_sentence(result, text, start, len(text))
    return tuple(result)


def line_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    prior_newline = text.rfind("\n", 0, offset)
    column = offset + 1 if prior_newline < 0 else offset - prior_newline
    return line, column
