"""Deterministic English token and sentence spans."""

from __future__ import annotations

import re
from dataclasses import dataclass

WORD_RE = re.compile(r"\b(?:[A-Za-z]+(?:[-'][A-Za-z0-9]+)*|[A-Za-z]*\d+[A-Za-z0-9.-]*)\b")
SENTENCE_RE = re.compile(r"\S(?:.*?)(?:[.!?](?=\s|$)|$)", re.DOTALL)


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


def sentences(text: str) -> tuple[Sentence, ...]:
    result: list[Sentence] = []
    for match in SENTENCE_RE.finditer(text):
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw.rstrip())
        start = match.start() + leading
        end = match.start() + trailing
        if start >= end:
            continue
        sentence_text = text[start:end]
        tokens = tuple(
            Token(token.group(0), start + token.start(), start + token.end())
            for token in WORD_RE.finditer(sentence_text)
        )
        result.append(Sentence(len(result), sentence_text, start, end, tokens))
    return tuple(result)


def line_column(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    prior_newline = text.rfind("\n", 0, offset)
    column = offset + 1 if prior_newline < 0 else offset - prior_newline
    return line, column
