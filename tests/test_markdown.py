"""Markdown structure must not be scored as prose."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from lingity.analyzer import analyze_text
from lingity.markdown import (
    Block,
    MarkdownParserError,
    block_counts,
    inline_code_spans,
    opaque_spans,
    parser_fingerprint,
    prose_spans,
    segment,
    segment_source,
)
from lingity.models import JsonValue
from lingity.nlp import parse

TABLE_DOCUMENT = """# Findings

The reviewer must close every finding before sign-off.

| Finding | Owner | Status |
| --- | --- | --- |
| Token refresh fails silently | platform team | open |
| Quota governor starves tenants | platform team | open |

The owner must publish a remediation date.
"""

FENCE_DOCUMENT = """## Pipeline

The collector must record every response.

```text
ingest -> normalize without rewriting -> score baseline -> reserve attempt
```

The archive must never overwrite an observation.
"""


def _kinds(text: str) -> list[str]:
    return [block.kind for block in segment(text)]


def _rule_ids(text: str) -> set[str]:
    findings = cast(list[dict[str, JsonValue]], analyze_text(text)["findings"])
    return {cast(str, finding["rule_id"]) for finding in findings}


def test_segment_classifies_each_structural_kind() -> None:
    assert _kinds(TABLE_DOCUMENT) == ["heading", "prose", "table", "prose"]
    assert _kinds(FENCE_DOCUMENT) == ["heading", "prose", "code", "prose"]


def test_heading_content_excludes_its_markers() -> None:
    blocks = segment("### The collector must poll\n")
    assert len(blocks) == 1
    heading = blocks[0]
    assert heading.kind == "heading"
    assert "### The collector must poll\n"[heading.content_start : heading.content_end] == (
        "The collector must poll"
    )


def test_list_markers_do_not_join_items_into_one_sentence() -> None:
    document = "The owner must supply:\n\n- the ticket\n- the evidence\n- the deadline\n"
    blocks = segment(document)
    assert [block.kind for block in blocks] == [
        "prose",
        "list_item",
        "list_item",
        "list_item",
    ]
    parsed = parse(document, spans=prose_spans(blocks))
    assert len(parsed.sentences) == 4
    assert [sentence.text for sentence in parsed.sentences][1:] == [
        "the ticket",
        "the evidence",
        "the deadline",
    ]


def test_a_heading_never_merges_with_the_paragraph_beneath_it() -> None:
    document = "## Required actions\n\nThe owner must close the finding.\n"
    parsed = parse(document, spans=prose_spans(segment(document)))
    assert [sentence.text for sentence in parsed.sentences] == [
        "Required actions",
        "The owner must close the finding.",
    ]
    # Parsing the raw source glues the heading onto the sentence below it.
    unsegmented = parse(document)
    assert unsegmented.sentences[0].text.startswith("## Required actions")


def test_table_rows_are_never_parsed_as_prose() -> None:
    parsed = parse(TABLE_DOCUMENT, spans=prose_spans(segment(TABLE_DOCUMENT)))
    for sentence in parsed.sentences:
        assert "|" not in sentence.text


def test_fenced_code_is_never_parsed_as_prose() -> None:
    parsed = parse(FENCE_DOCUMENT, spans=prose_spans(segment(FENCE_DOCUMENT)))
    for sentence in parsed.sentences:
        assert "->" not in sentence.text
    assert opaque_spans(segment(FENCE_DOCUMENT))


def test_a_table_no_longer_trips_the_paragraph_rule() -> None:
    assert "LING-STRUCTURE-001" not in _rule_ids(TABLE_DOCUMENT)


def test_inline_code_spans_are_located_exactly() -> None:
    text = "Read `X-Business-Use-Case-Usage` on every response."
    spans = inline_code_spans(text)
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "`X-Business-Use-Case-Usage`"


def test_an_unclosed_backtick_opens_no_span() -> None:
    assert inline_code_spans("The header `X-App-Usage reports the quota.") == ()


def test_an_identifier_in_a_code_span_raises_no_finding() -> None:
    bare = "The collector reads X-Business-Use-Case-Usage on every response."
    coded = "The collector reads `X-Business-Use-Case-Usage` on every response."
    assert "LING-COMPOUND-DEPTH-001" in _rule_ids(bare)
    assert "LING-COMPOUND-DEPTH-001" not in _rule_ids(coded)


def test_offsets_still_locate_findings_in_the_original_source() -> None:
    result = analyze_text(TABLE_DOCUMENT)
    for finding in cast(list[dict[str, JsonValue]], result["findings"]):
        location = cast(dict[str, JsonValue], finding["location"])
        start = cast(int, location["start"])
        end = cast(int, location["end"])
        assert 0 <= start < end <= len(TABLE_DOCUMENT)


def test_prose_without_markdown_parses_exactly_as_before() -> None:
    prose = (
        "The reviewer must close every finding before sign-off. "
        "The owner must publish a remediation date."
    )
    segmented = parse(prose, spans=prose_spans(segment(prose)))
    whole = parse(prose)
    assert [sentence.text for sentence in segmented.sentences] == [
        sentence.text for sentence in whole.sentences
    ]
    assert [token.start for token in segmented.tokens] == [
        token.start for token in whole.tokens
    ]


def test_every_token_resolves_to_its_own_sentence() -> None:
    parsed = parse(TABLE_DOCUMENT, spans=prose_spans(segment(TABLE_DOCUMENT)))
    for token in parsed.tokens:
        sentence = parsed.sentence_of(token)
        assert sentence.start <= token.start
        assert token.end <= sentence.end


def test_the_artifact_publishes_its_segmentation() -> None:
    result = analyze_text(TABLE_DOCUMENT)
    ingest = cast(dict[str, JsonValue], result["ingest"])
    assert ingest["mode"] == "markdown"
    assert ingest["blocks"] == {"heading": 1, "prose": 2, "table": 1}
    assert cast(int, ingest["analyzed_characters"]) < len(TABLE_DOCUMENT)


def test_segmentation_is_deterministic() -> None:
    first = analyze_text(TABLE_DOCUMENT)
    second = analyze_text(TABLE_DOCUMENT)
    assert first["analysis_sha256"] == second["analysis_sha256"]


def test_block_counts_are_sorted_for_stable_hashing() -> None:
    counts = block_counts(segment(TABLE_DOCUMENT))
    assert list(counts) == sorted(counts)


def test_thematic_breaks_carry_no_prose() -> None:
    blocks = segment("The owner must act.\n\n---\n\nThe reviewer must confirm.\n")
    kinds = [block.kind for block in blocks]
    assert kinds == ["prose", "rule", "prose"]
    rule_block = next(block for block in blocks if block.kind == "rule")
    assert rule_block.content_start == rule_block.content_end
    assert isinstance(rule_block, Block)


# Regressions from review. Each of these silently dropped prose from the
# analysis or leaked markup into it, and neither failure raises anything.


def test_a_heading_keeps_a_trailing_hash_that_belongs_to_a_word() -> None:
    source = "### Use C#\n"
    heading = segment(source)[0]
    assert source[heading.content_start : heading.content_end] == "Use C#"


def test_a_heading_drops_a_whitespace_delimited_closing_sequence() -> None:
    source = "### Required actions ###\n"
    heading = segment(source)[0]
    assert source[heading.content_start : heading.content_end] == "Required actions"


def test_a_heading_with_no_title_carries_no_content() -> None:
    blocks = segment("###\n")
    assert blocks[0].kind == "heading"
    assert blocks[0].content_start == blocks[0].content_end


def test_a_short_fence_does_not_close_a_longer_one() -> None:
    source = "````\nThe owner must act.\n```\nstill code\n````\n\nThe reviewer must confirm.\n"
    parsed = parse(source, spans=prose_spans(segment(source)))
    assert [sentence.text for sentence in parsed.sentences] == [
        "The reviewer must confirm."
    ]


def test_an_info_string_does_not_close_a_fence() -> None:
    source = "```\nfirst\n```python\nsecond\n```\n\nThe reviewer must confirm.\n"
    parsed = parse(source, spans=prose_spans(segment(source)))
    assert [sentence.text for sentence in parsed.sentences] == [
        "The reviewer must confirm."
    ]


def test_a_tilde_fence_does_not_close_a_backtick_fence() -> None:
    source = "```\nfirst\n~~~\nsecond\n```\n\nThe reviewer must confirm.\n"
    parsed = parse(source, spans=prose_spans(segment(source)))
    assert [sentence.text for sentence in parsed.sentences] == [
        "The reviewer must confirm."
    ]


def test_a_spaced_thematic_break_is_not_a_list_item() -> None:
    assert [block.kind for block in segment("* * *\n")] == ["rule"]
    assert [block.kind for block in segment("- - -\n")] == ["rule"]
    assert [block.kind for block in segment("* item\n")] == ["list_item"]


def test_every_blockquote_marker_stays_out_of_the_parse() -> None:
    source = "> The owner must act.\n> The reviewer must confirm.\n"
    blocks = segment(source)
    assert [block.kind for block in blocks] == ["blockquote"]
    parsed = parse(source, spans=prose_spans(blocks))
    assert [sentence.text for sentence in parsed.sentences] == [
        "The owner must act.",
        "The reviewer must confirm.",
    ]
    for sentence in parsed.sentences:
        assert ">" not in sentence.text


def test_a_delimiter_row_counts_only_beneath_its_header() -> None:
    source = "--- | ---\nThe owner must act before sign-off.\n"
    blocks = segment(source)
    assert not [block for block in blocks if block.kind == "table"]
    parsed = parse(source, spans=prose_spans(blocks))
    assert any("The owner must act" in sentence.text for sentence in parsed.sentences)


def test_a_two_hyphen_row_is_a_table_delimiter_per_gfm() -> None:
    """The hand-written matcher demanded three hyphens. GFM requires one.

    Delegating to the parser means adopting the specification, including where
    the earlier implementation was deliberately stricter than it. Any renderer
    shows this as a table, so it is markup rather than prose.
    """
    source = "Owner | Status\n-- | --\nThe owner must act.\n"
    assert [block.kind for block in segment(source)] == ["table"]


def test_a_real_table_is_still_detected() -> None:
    source = "| Owner | Status |\n| --- | --- |\n| platform team | open |\n"
    assert [block.kind for block in segment(source)] == ["table"]


# The parser is part of the analysis contract, as the linguistic model is.


def test_an_indented_code_block_is_opaque() -> None:
    """CommonMark covers it. The hand-written segmenter did not."""
    source = "The owner must act.\n\n    indented = code\n\nThe reviewer confirms.\n"
    blocks = segment(source)
    assert [block.kind for block in blocks] == ["prose", "code", "prose"]
    parsed = parse(source, spans=prose_spans(blocks))
    for sentence in parsed.sentences:
        assert "indented = code" not in sentence.text


def test_a_setext_heading_is_a_heading() -> None:
    """Also unreachable for the hand-written segmenter, which called it a rule."""
    source = "Required actions\n===\n\nThe owner must act.\n"
    assert [block.kind for block in segment(source)] == ["heading", "prose"]


def test_nested_list_items_each_carry_their_own_content() -> None:
    source = "- outer item\n  - inner item\n"
    blocks = segment(source)
    assert [block.kind for block in blocks] == ["list_item", "list_item"]
    for block in blocks:
        assert not source[block.content_start : block.content_end].lstrip().startswith("-")


def test_the_parser_identity_is_published() -> None:
    fingerprint = parser_fingerprint()
    assert fingerprint["name"] == "markdown-it-py"
    assert fingerprint["version"].startswith("3.")


def test_the_artifact_publishes_the_parser_and_its_confidence() -> None:
    ingest = cast(dict[str, JsonValue], analyze_text(TABLE_DOCUMENT)["ingest"])
    assert ingest["parser"] == parser_fingerprint()
    assert ingest["unresolved_lines"] == 0


def test_content_is_located_for_every_line_of_the_repository_documents() -> None:
    """A line the segmenter cannot locate keeps its markers, so it must be rare."""
    for name in ("README.md", "DESIGN.md"):
        source = Path(__file__).resolve().parents[1].joinpath(name).read_text(
            encoding="utf-8"
        )
        assert segment_source(source).unresolved_lines == 0, name


def test_a_wrong_parser_major_version_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import markdown_it

    from lingity import markdown as markdown_module

    markdown_module._parser.cache_clear()
    monkeypatch.setattr(markdown_it, "__version__", "4.0.0")
    try:
        with pytest.raises(MarkdownParserError):
            markdown_module._parser()
    finally:
        markdown_module._parser.cache_clear()


def test_an_escaped_backtick_opens_no_span() -> None:
    text = r"Read \`X-Business-Use-Case-Usage\` today."
    assert inline_code_spans(text) == ()


def test_an_escaped_backtick_does_not_suppress_a_finding() -> None:
    escaped = r"The collector reads \`X-Business-Use-Case-Usage\` on every response."
    assert "LING-COMPOUND-DEPTH-001" in _rule_ids(escaped)


def test_a_code_span_cannot_cross_a_block() -> None:
    """An unmatched backtick must not protect everything up to the next one."""
    source = "The collector reads ` on every response.\n\nThe owner reads ` the log.\n"
    blocks = segment(source)
    flattened = tuple(span for group in prose_spans(blocks) for span in group)
    assert inline_code_spans(source, flattened) == ()
    # Scanned across the whole source the two stray backticks pair up.
    assert len(inline_code_spans(source)) == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "markdown-it-py diverges from cmark-gfm: an ATX heading line containing "
        "a pipe is consumed as a table header when the next line is a delimiter "
        "row. cmark-gfm builds a table header only from an open paragraph, so it "
        "yields a heading plus a paragraph. Pinned strictly so a parser upgrade "
        "that fixes it forces the win into the open."
    ),
)
def test_a_heading_is_not_a_table_header() -> None:
    source = "# Owner | Status\n--- | ---\nThe owner must act.\n"
    assert [block.kind for block in segment(source)] == ["heading", "prose"]


# Where an author wrapped a line must not change a score.


def test_a_wrapped_list_item_stays_one_sentence() -> None:
    source = "- The reviewer must close every\n  finding before sign-off.\n"
    blocks = segment(source)
    assert [block.kind for block in blocks] == ["list_item"]
    parsed = parse(source, spans=prose_spans(blocks))
    assert [sentence.text for sentence in parsed.sentences] == [
        "The reviewer must close every finding before sign-off."
    ]


def test_wrapping_does_not_change_the_score() -> None:
    wrapped = "- The reviewer must close every\n  finding before sign-off.\n"
    flat = "- The reviewer must close every finding before sign-off.\n"
    assert analyze_text(wrapped)["score"] == analyze_text(flat)["score"]


def test_a_crlf_line_ending_does_not_change_the_score() -> None:
    """The gap between wrapped lines is markup, whatever the line ending."""
    unix = "- The reviewer must close every\n  finding before sign-off.\n"
    windows = unix.replace("\n", "\r\n")
    assert analyze_text(windows)["score"] == analyze_text(unix)["score"]


def test_a_block_bounds_its_markers_even_though_its_content_does_not() -> None:
    source = "### Required actions ###\n"
    heading = segment(source)[0]
    assert source[heading.start : heading.end] == "### Required actions ###"
    assert source[heading.content_start : heading.content_end] == "Required actions"


def test_a_list_item_block_bounds_its_marker() -> None:
    source = "- the ticket\n"
    item = segment(source)[0]
    assert source[item.start : item.end] == "- the ticket"
    assert source[item.content_start : item.content_end] == "the ticket"


# Nothing may leave the analysis without being counted.


def test_a_raw_html_block_is_opaque_and_visible() -> None:
    source = "<div>\nThe owner must act.\n</div>\n\nThe reviewer must confirm.\n"
    segmentation = segment_source(source)
    assert [block.kind for block in segmentation.blocks] == ["html", "prose"]
    assert segmentation.uncovered_lines == 0
    counts = block_counts(segmentation.blocks)
    assert counts["html"] == 1


def test_no_line_of_the_repository_documents_goes_uncounted() -> None:
    for name in ("README.md", "DESIGN.md"):
        source = Path(__file__).resolve().parents[1].joinpath(name).read_text(
            encoding="utf-8"
        )
        segmentation = segment_source(source)
        assert segmentation.uncovered_lines == 0, name
        assert segmentation.unresolved_lines == 0, name


def test_the_artifact_publishes_what_no_block_claimed() -> None:
    ingest = cast(dict[str, JsonValue], analyze_text(TABLE_DOCUMENT)["ingest"])
    assert ingest["uncovered_lines"] == 0


def test_the_cli_reports_a_parser_failure_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from lingity import markdown as markdown_module
    from lingity.cli import main

    source = tmp_path / "source.md"
    source.write_text("The owner must act.\n", encoding="utf-8")

    def refuse() -> object:
        raise MarkdownParserError("pinned parser unavailable")

    monkeypatch.setattr(markdown_module, "_parser", refuse)
    assert main(["analyze", str(source), "--output", str(tmp_path / "out.json")]) == 2
    assert "pinned parser unavailable" in capsys.readouterr().err


# Grouped spans must not leak between coordinate systems or past a rule.


def test_a_code_span_may_open_and_close_on_different_lines() -> None:
    """A block is scanned whole, so a wrapped code span is still protected."""
    source = "The collector reads `\nX-Business-Use-Case-Usage\n` on every response.\n"
    groups = prose_spans(segment(source))
    extents = tuple((group[0][0], group[-1][1]) for group in groups)
    spans = inline_code_spans(source, extents)
    assert len(spans) == 1
    assert source[spans[0][0] : spans[0][1]].strip("`").strip() == (
        "X-Business-Use-Case-Usage"
    )
    assert "LING-COMPOUND-DEPTH-001" not in _rule_ids(source)


def test_an_observed_phrase_spanning_a_join_is_not_corrupted() -> None:
    """Sentence text is chunk-indexed; source offsets must never slice it.

    A sentence joined from two source lines has the container's markers removed,
    so every source offset inside it is shifted by an amount that varies across
    the sentence. Slicing with one would report a phrase off by those bytes.
    """
    source = "- The team must act in order\n  to close the finding.\n"
    findings = cast(list[dict[str, JsonValue]], analyze_text(source)["findings"])
    observed = [
        cast(dict[str, JsonValue], finding["observed_value"])["phrase"]
        for finding in findings
        if finding["rule_id"] in {"LING-FILLER-001", "LING-BUREAUCRACY-001"}
    ]
    assert observed
    assert set(observed) == {"in order to"}


def _long_blockquote_body() -> str:
    return (
        "the reviewer must close every finding and publish the remediation date "
        "before the owner approves the change "
    ) * 6


def test_wrapping_cannot_evade_the_paragraph_rule() -> None:
    """Each line of a wrapped block was counted as its own paragraph."""
    body = _long_blockquote_body()
    wrapped = "".join(f"> {word}\n" for word in body.split(" ") if word)
    one_line = "> " + body.strip() + "\n"
    assert len(body.split()) > 90
    for document in (wrapped, one_line):
        rules = [
            finding["rule_id"]
            for finding in cast(
                list[dict[str, JsonValue]], analyze_text(document)["findings"]
            )
        ]
        assert rules.count("LING-STRUCTURE-001") == 1


def test_many_one_line_items_are_still_fully_covered() -> None:
    """The uncovered sweep walks blocks and lines once each, in order."""
    source = "".join(f"- item number {index} must ship.\n" for index in range(400))
    segmentation = segment_source(source)
    assert len(segmentation.blocks) == 400
    assert segmentation.uncovered_lines == 0
    assert segmentation.unresolved_lines == 0
