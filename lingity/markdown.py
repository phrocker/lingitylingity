"""Deterministic Markdown block segmentation.

Lingity scores prose. A Markdown document also carries structure that is not
prose: fenced code, tables, thematic breaks, headings, and list markers.
Reading that structure as prose glues a heading onto the paragraph beneath it,
parses a table row as a single long sentence, and reports an HTTP header name
as a noun stack. The score then measures the markup instead of the writing.

This module classifies the source into blocks and never alters it. Every block
records offsets into the original text, so a finding still locates itself in
the document the author wrote. Segmentation is pure and line-based, so an
analysis remains reproducible from the source alone.

Deliberate limits, so a caller knows what this does not claim to parse:
indented code blocks, setext headings, nested-list depth semantics, and raw
HTML blocks. A thematic break and a setext underline both classify as ``rule``
and carry no prose, which keeps them out of the parse without requiring this
module to resolve the ambiguity between them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

BlockKind = Literal[
    "prose",
    "heading",
    "list_item",
    "blockquote",
    "table",
    "code",
    "rule",
]

PROSE_KINDS: frozenset[str] = frozenset({"prose", "heading", "list_item", "blockquote"})
"""Kinds whose content the analyzer reads as prose."""

OPAQUE_KINDS: frozenset[str] = frozenset({"table", "code", "rule"})
"""Kinds that carry structure or literals rather than sentences."""

_FENCE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})")
_ATX = re.compile(
    r"^ {0,3}(?P<hashes>#{1,6})(?:[ \t]+(?P<title>.*?))?(?:[ \t]+#+)?[ \t]*$"
)
_RULE = re.compile(
    r"^ {0,3}(?:(?:-[ \t]*){3,}|(?:\*[ \t]*){3,}|(?:_[ \t]*){3,}|={3,}[ \t]*)$"
)
_MARKER = re.compile(r"^(?P<indent> *)(?P<marker>[-*+]|\d{1,9}[.)])(?P<gap> +)")
_QUOTE = re.compile(r"^(?P<indent> {0,3})>(?P<gap> ?)")
_TABLE_DELIM = re.compile(r"^ {0,3}\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")


@dataclass(frozen=True)
class Block:
    """One classified region of the source.

    ``start`` and ``end`` bound the whole block. ``content_start`` and
    ``content_end`` bound the part the analyzer may read, which excludes a
    heading's hashes, a list marker, and a blockquote's angle bracket.
    """

    kind: BlockKind
    start: int
    end: int
    content_start: int
    content_end: int

    @property
    def is_prose(self) -> bool:
        return self.kind in PROSE_KINDS


@dataclass(frozen=True)
class _Line:
    start: int
    end: int
    text: str


def _scan_lines(text: str) -> list[_Line]:
    lines: list[_Line] = []
    position = 0
    for raw in text.splitlines(keepends=True):
        content = raw.rstrip("\r\n")
        lines.append(_Line(position, position + len(content), content))
        position += len(raw)
    return lines


def _fence_kinds(lines: list[_Line]) -> list[bool]:
    """Mark every line a fenced code block encloses, fences included.

    A closing fence must repeat the opener's marker, run at least as long as
    the opener, and carry no info string. Without all three, a ``` line closes
    a ```` block and a ```python line inside one closes it early, after which
    the remaining code is scored as prose.
    """
    inside = False
    marker = ""
    width = 0
    flags: list[bool] = []
    for line in lines:
        match = _FENCE.match(line.text)
        if not inside:
            if match:
                inside = True
                marker = match.group("fence")[0]
                width = len(match.group("fence"))
                flags.append(True)
                continue
            flags.append(False)
            continue
        flags.append(True)
        if not match:
            continue
        fence = match.group("fence")
        closes = (
            fence[0] == marker
            and len(fence) >= width
            and not line.text[match.end():].strip()
        )
        if closes:
            inside = False
    return flags


def _table_flags(lines: list[_Line], fenced: list[bool]) -> list[bool]:
    """Mark the rows of a GFM table.

    The delimiter row counts only when it sits directly beneath the header row.
    Accepting one anywhere in a pipe-bearing run hides every line of that run,
    so a paragraph that merely contains a pipe would leave the analysis without
    raising anything.
    """
    flags = [False] * len(lines)
    index = 0
    while index < len(lines):
        header = lines[index]
        if fenced[index] or "|" not in header.text or not header.text.strip():
            index += 1
            continue
        delimiter = index + 1
        if (
            delimiter >= len(lines)
            or fenced[delimiter]
            or not _TABLE_DELIM.match(lines[delimiter].text)
        ):
            index += 1
            continue
        run_end = delimiter + 1
        while (
            run_end < len(lines)
            and not fenced[run_end]
            and "|" in lines[run_end].text
            and lines[run_end].text.strip()
        ):
            run_end += 1
        for position in range(index, run_end):
            flags[position] = True
        index = run_end
    return flags


def _classify(line: _Line) -> tuple[BlockKind, int, int | None]:
    """Return the line's kind, where its readable content starts, and where a
    single-line block's content ends. ``None`` defers the end to the block."""
    if _RULE.match(line.text):
        return "rule", line.end, line.end
    heading = _ATX.match(line.text)
    if heading:
        if heading.group("title") is None:
            return "heading", line.end, line.end
        return (
            "heading",
            line.start + heading.start("title"),
            line.start + heading.end("title"),
        )
    quote = _QUOTE.match(line.text)
    if quote:
        return "blockquote", line.start + quote.end(), None
    marker = _MARKER.match(line.text)
    if marker:
        return "list_item", line.start + marker.end(), None
    return "prose", line.start + (len(line.text) - len(line.text.lstrip(" "))), None


def segment(text: str) -> tuple[Block, ...]:
    """Classify ``text`` into contiguous blocks without modifying it."""
    lines = _scan_lines(text)
    if not lines:
        return ()
    fenced = _fence_kinds(lines)
    tables = _table_flags(lines, fenced)

    blocks: list[Block] = []
    open_kind: BlockKind | None = None
    open_start = 0
    open_content_start = 0
    open_content_end = 0

    def close() -> None:
        nonlocal open_kind
        if open_kind is None:
            return
        end = max(open_content_end, open_content_start)
        blocks.append(Block(open_kind, open_start, end, open_content_start, end))
        open_kind = None

    for index, line in enumerate(lines):
        if not line.text.strip():
            close()
            continue
        if fenced[index]:
            kind: BlockKind = "code"
            content_start = line.start
            fixed_end: int | None = line.end
        elif tables[index]:
            kind = "table"
            content_start = line.start
            fixed_end = line.end
        else:
            kind, content_start, fixed_end = _classify(line)

        # A blockquote opens its own block per line. Grouping them would leave
        # every marker after the first inside the readable span.
        opens_own_block = kind in {"heading", "list_item", "rule", "blockquote"}
        if open_kind is None or kind != open_kind or opens_own_block:
            close()
            open_kind = kind
            open_start = line.start
            open_content_start = content_start
        open_content_end = fixed_end if fixed_end is not None else line.end
    close()
    return tuple(blocks)


def prose_spans(blocks: tuple[Block, ...]) -> tuple[tuple[int, int], ...]:
    """Return the readable spans, in source order, that the analyzer may parse."""
    return tuple(
        (block.content_start, block.content_end)
        for block in blocks
        if block.is_prose and block.content_start < block.content_end
    )


def opaque_spans(blocks: tuple[Block, ...]) -> tuple[tuple[int, int], ...]:
    """Return the spans that carry structure or literals rather than prose."""
    return tuple(
        (block.start, block.end) for block in blocks if block.kind in OPAQUE_KINDS
    )


def inline_code_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Locate inline code spans so identifiers inside them stay protected.

    An identifier written as code is a name the author must not change, so a
    noun-stacking or compound-depth finding against it reports a defect that no
    rewrite may fix.
    """
    spans: list[tuple[int, int]] = []
    position = 0
    length = len(text)
    while position < length:
        if text[position] != "`":
            position += 1
            continue
        opener_end = position
        while opener_end < length and text[opener_end] == "`":
            opener_end += 1
        ticks = opener_end - position
        cursor = opener_end
        closed = -1
        while cursor < length:
            if text[cursor] != "`":
                cursor += 1
                continue
            closer_end = cursor
            while closer_end < length and text[closer_end] == "`":
                closer_end += 1
            if closer_end - cursor == ticks:
                closed = closer_end
                break
            cursor = closer_end
        if closed < 0:
            position = opener_end
            continue
        spans.append((position, closed))
        position = closed
    return tuple(spans)


def block_counts(blocks: tuple[Block, ...]) -> dict[str, int]:
    """Summarise the segmentation for publication inside an analysis artifact."""
    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.kind] = counts.get(block.kind, 0) + 1
    return dict(sorted(counts.items()))
