"""Markdown block segmentation, delegated to a CommonMark parser.

Lingity scores prose. A Markdown document also carries structure that is not
prose: fenced code, tables, thematic breaks, headings, and list markers.
Reading that structure as prose glues a heading onto the paragraph beneath it,
parses a table row as a single long sentence, and reports an identifier written
as code as a noun stack. The score then measures the markup instead of the
writing.

An earlier revision recognised that structure with hand-written patterns.
Review found six divergences from the specification in about fifty lines, and
every one of them failed silently: a block wrongly marked opaque is never
scored and never reported. Markdown is a specification, and hand-maintaining a
subset of one repeats the mistake this project already recorded about
protected-concept phrase lists.

Block structure therefore comes from ``markdown-it-py``, which is CommonMark
compliant and tested against the specification's own suite. This module decides
only which block kinds carry prose and where their content sits in the source.
It never alters the source, and every block records offsets into the original
text, so a finding still locates itself in the document the author wrote.

The parser is part of the analysis contract, exactly as the linguistic model
is. Its identity is published in every artifact and a major-version change is
refused rather than silently re-segmented.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from markdown_it import MarkdownIt
    from markdown_it.token import Token

PARSER_NAME = "markdown-it-py"
PARSER_MAJOR = 3
PARSER_INSTALL_HINT = (
    "Install the pinned parser with: python -m pip install "
    f"'markdown-it-py>={PARSER_MAJOR},<{PARSER_MAJOR + 1}'"
)

BlockKind = Literal[
    "prose",
    "heading",
    "list_item",
    "blockquote",
    "table",
    "code",
    "rule",
    "html",
]

PROSE_KINDS: frozenset[str] = frozenset({"prose", "heading", "list_item", "blockquote"})
"""Kinds whose content the analyzer reads as prose."""

OPAQUE_KINDS: frozenset[str] = frozenset({"table", "code", "rule", "html"})
"""Kinds that carry structure or literals rather than sentences."""

_CONTAINER_OPEN = {"blockquote_open": "blockquote", "list_item_open": "list_item"}
_CONTAINER_CLOSE = frozenset({"blockquote_close", "list_item_close"})
_OPAQUE_TOKENS = {
    "fence": "code",
    "code_block": "code",
    "hr": "rule",
    "html_block": "html",
}
_SEARCH_WINDOW = 3


class MarkdownParserError(RuntimeError):
    """Raised when the pinned Markdown parser cannot be used as published."""


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
    content_spans: tuple[tuple[int, int], ...] = ()
    """The readable pieces of this block, in source order.

    A wrapped list item or a blockquote paragraph is one block whose lines the
    container's markers separate in the source. The pieces are parsed as a
    single unit, so where an author wrapped a line cannot change a sentence
    boundary and therefore cannot change a score.
    """

    @property
    def is_prose(self) -> bool:
        return self.kind in PROSE_KINDS

    @property
    def readable(self) -> tuple[tuple[int, int], ...]:
        if self.content_spans:
            return self.content_spans
        if self.content_start < self.content_end:
            return ((self.content_start, self.content_end),)
        return ()


@dataclass(frozen=True)
class Segmentation:
    """The blocks of a source, and how confidently they were located."""

    blocks: tuple[Block, ...]
    unresolved_lines: int
    uncovered_lines: int
    """Non-blank source lines that no block claims.

    A token type this module does not handle would otherwise remove its lines
    from the analysis without raising anything. Counting what no block covers
    catches that for every construct at once, rather than for the ones somebody
    remembered to enumerate.
    """


@lru_cache(maxsize=1)
def _parser() -> "MarkdownIt":
    """Load the pinned parser, refusing a major version other than the published one."""
    try:
        import markdown_it
        from markdown_it import MarkdownIt
    except ImportError as error:  # pragma: no cover - exercised by packaging
        raise MarkdownParserError(
            f"{PARSER_NAME} is not installed. {PARSER_INSTALL_HINT}"
        ) from error
    major = markdown_it.__version__.split(".")[0]
    if major != str(PARSER_MAJOR):
        raise MarkdownParserError(
            f"{PARSER_NAME} version {markdown_it.__version__} is installed, but this "
            f"analyzer is pinned to major version {PARSER_MAJOR}. Segmentation is only "
            f"reproducible against the pinned major version. {PARSER_INSTALL_HINT}"
        )
    # CommonMark plus GFM tables. Nothing else is enabled, because every extra
    # rule changes what counts as prose and therefore what a score means.
    return MarkdownIt("commonmark").enable("table")


def parser_fingerprint() -> dict[str, str]:
    """Publish the parser identity that a segmentation is reproducible against."""
    # _parser() is the only gate that turns a missing or wrong-major install into
    # MarkdownParserError. Importing before calling it would let a bare
    # ImportError escape analyze_text() and the CLI's error handling, which
    # contradicts the contract that an unusable parser is refused with install
    # guidance rather than crashing. Once _parser() returns, the import cannot
    # fail.
    _parser()

    import markdown_it

    return {"name": PARSER_NAME, "version": markdown_it.__version__}


def _line_bounds(text: str) -> list[tuple[int, int]]:
    """Return each line's (start, end) offsets, excluding its terminator."""
    bounds: list[tuple[int, int]] = []
    position = 0
    for raw in text.splitlines(keepends=True):
        content = raw.rstrip("\r\n")
        bounds.append((position, position + len(content)))
        position += len(raw)
    return bounds


def _mapped_span(bounds: list[tuple[int, int]], token: "Token") -> tuple[int, int] | None:
    if token.map is None or not bounds:
        return None
    first = max(0, min(token.map[0], len(bounds) - 1))
    last = max(first, min(token.map[1] - 1, len(bounds) - 1))
    return bounds[first][0], bounds[last][1]


def _content_spans(
    text: str, token: "Token", bounds: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], int]:
    """Locate each line of an inline token's content in the source.

    The parser reports content with its container markers removed, so
    "> first\\n> second" arrives as "first\\nsecond". Locating each line
    separately keeps every marker outside the readable span without this module
    having to know which markers the parser stripped.
    """
    if token.map is None:
        return [], 0
    spans: list[tuple[int, int]] = []
    unresolved = 0
    line = max(0, token.map[0])
    for piece in token.content.split("\n"):
        stripped = piece.strip()
        if not stripped:
            line += 1
            continue
        located = -1
        for probe in range(line, min(line + _SEARCH_WINDOW, len(bounds))):
            start, end = bounds[probe]
            found = text.find(stripped, start, end)
            if found >= 0:
                located = found
                line = probe + 1
                break
        if located < 0:
            # Fail towards keeping the text. Dropping a line removes it from the
            # score in silence; keeping a marker only adds a token the rules can
            # see and report. The count is published so neither stays hidden.
            unresolved += 1
            index = min(line, len(bounds) - 1)
            if index < 0:
                break
            start, end = bounds[index]
            raw = text[start:end]
            spans.append((start + len(raw) - len(raw.lstrip()), start + len(raw.rstrip())))
            line += 1
            continue
        spans.append((located, located + len(stripped)))
    return spans, unresolved


def _uncovered(text: str, bounds: list[tuple[int, int]], blocks: list[Block]) -> int:
    """Count non-blank lines that no block claims.

    Blocks and lines are both ordered, so one forward sweep answers this. The
    earlier pass compared every line against every block, which degrades
    quadratically on a document of many one-line list items.

    Both ranges are half-open, so a block covers a line only when it starts
    before the line ends *and* ends after the line starts. Comparing them as
    though the ends were inclusive lets a block that stops exactly where a line
    begins be treated as covering that line, which reports a line as covered
    when nothing claims it -- the precise failure this count exists to surface.
    """
    ordered = sorted(blocks, key=lambda block: (block.start, block.end))
    count = 0
    index = 0
    for start, end in bounds:
        if not text[start:end].strip():
            continue
        while index < len(ordered) and ordered[index].end <= start:
            index += 1
        if index < len(ordered) and ordered[index].start < end:
            continue
        count += 1
    return count


def segment_source(text: str) -> Segmentation:
    """Classify ``text`` into contiguous blocks without modifying it."""
    bounds = _line_bounds(text)
    if not bounds:
        return Segmentation((), 0, 0)

    blocks: list[Block] = []
    containers: list[str] = []
    heading_depth = 0
    table_depth = 0
    unresolved = 0

    for token in _parser().parse(text):
        kind_name = token.type
        if kind_name == "table_open":
            table_depth += 1
            span = _mapped_span(bounds, token)
            if span is not None and table_depth == 1:
                blocks.append(Block("table", span[0], span[1], span[0], span[1]))
            continue
        if kind_name == "table_close":
            table_depth = max(0, table_depth - 1)
            continue
        if table_depth:
            continue
        if kind_name in _OPAQUE_TOKENS:
            span = _mapped_span(bounds, token)
            if span is None:
                continue
            opaque = _OPAQUE_TOKENS[kind_name]
            if opaque == "rule":
                blocks.append(Block("rule", span[0], span[1], span[1], span[1]))
            else:
                kind_opaque: BlockKind = "code" if opaque == "code" else "html"
                blocks.append(Block(kind_opaque, span[0], span[1], span[0], span[1]))
            continue
        if kind_name in _CONTAINER_OPEN:
            containers.append(_CONTAINER_OPEN[kind_name])
            continue
        if kind_name in _CONTAINER_CLOSE:
            if containers:
                containers.pop()
            continue
        if kind_name == "heading_open":
            heading_depth += 1
            continue
        if kind_name == "heading_close":
            heading_depth = max(0, heading_depth - 1)
            continue
        if kind_name != "inline":
            continue

        if heading_depth:
            kind: BlockKind = "heading"
        elif containers:
            kind = "list_item" if containers[-1] == "list_item" else "blockquote"
        else:
            kind = "prose"

        span = _mapped_span(bounds, token)
        spans, missed = _content_spans(text, token, bounds)
        unresolved += missed
        if not spans:
            if span is not None:
                # An empty heading such as "###" carries no readable content.
                blocks.append(Block(kind, span[0], span[1], span[1], span[1]))
            continue
        # One inline token is one logical block. Whatever separates its lines in
        # the source is container markup the parser already stripped, so the
        # pieces are recorded together rather than compared against a literal
        # separator. A wrapped line and a CRLF line ending are then identical.
        start = span[0] if span is not None else spans[0][0]
        end = span[1] if span is not None else spans[-1][1]
        blocks.append(
            Block(kind, start, end, spans[0][0], spans[-1][1], tuple(spans))
        )

    blocks.sort(key=lambda block: (block.start, block.end))
    return Segmentation(tuple(blocks), unresolved, _uncovered(text, bounds, blocks))


def segment(text: str) -> tuple[Block, ...]:
    """Classify ``text`` into contiguous blocks without modifying it."""
    return segment_source(text).blocks


def prose_spans(
    blocks: tuple[Block, ...],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return the readable spans the analyzer may parse, grouped by block.

    Each group is one logical block. The parser reads a group as a single unit,
    so a sentence wrapped across two source lines stays one sentence.
    """
    return tuple(
        block.readable for block in blocks if block.is_prose and block.readable
    )


def opaque_spans(blocks: tuple[Block, ...]) -> tuple[tuple[int, int], ...]:
    """Return the spans that carry structure or literals rather than prose."""
    return tuple(
        (block.start, block.end) for block in blocks if block.kind in OPAQUE_KINDS
    )


def _is_escaped(text: str, position: int) -> bool:
    """True when an odd number of backslashes precedes ``position``."""
    backslashes = 0
    index = position - 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _code_spans_within(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = start
    while position < end:
        if text[position] != "`" or _is_escaped(text, position):
            position += 1
            continue
        opener_end = position
        while opener_end < end and text[opener_end] == "`":
            opener_end += 1
        ticks = opener_end - position
        cursor = opener_end
        closed = -1
        while cursor < end:
            if text[cursor] != "`":
                cursor += 1
                continue
            closer_end = cursor
            while closer_end < end and text[closer_end] == "`":
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
    return spans


def inline_code_spans(
    text: str, spans: tuple[tuple[int, int], ...] = ()
) -> tuple[tuple[int, int], ...]:
    """Locate inline code spans so identifiers inside them stay protected.

    An identifier written as code is a name the author must not change, so a
    noun-stacking or compound-depth finding against it reports a defect that no
    rewrite may fix.

    ``spans`` restricts the scan to each range separately. Scanning the whole
    source at once lets an unmatched backtick in one block pair with a backtick
    in an unrelated block, and everything between them is then protected and
    every finding inside it suppressed. A code span cannot cross a block, so
    neither may the scan.

    A backtick preceded by an odd number of backslashes is a literal character
    and opens nothing.

    This stays a scanner rather than moving to the parser. Inline tokens report
    their children without source offsets, and a backtick run is otherwise
    unambiguous: an opener closes on the next run of exactly the same length.
    """
    if not spans:
        return tuple(_code_spans_within(text, 0, len(text)))
    located: list[tuple[int, int]] = []
    for start, end in spans:
        located.extend(_code_spans_within(text, start, end))
    return tuple(sorted(located))


def block_counts(blocks: tuple[Block, ...]) -> dict[str, int]:
    """Summarise the segmentation for publication inside an analysis artifact."""
    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.kind] = counts.get(block.kind, 0) + 1
    return dict(sorted(counts.items()))
