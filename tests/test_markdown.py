"""Markdown structure must not be scored as prose."""

from __future__ import annotations

from typing import cast

from lingity.analyzer import analyze_text
from lingity.markdown import (
    Block,
    block_counts,
    inline_code_spans,
    opaque_spans,
    prose_spans,
    segment,
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
