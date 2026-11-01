"""Grade-level readability estimates: formulas, syllables, normalization, policy.

Every expected value below is computed by hand from the counts written beside
it, not read back from the implementation, so a changed coefficient, a changed
count or a changed rounding rule fails here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import socket
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

import lingity.profiles as profiles
import lingity.readability as readability
from lingity.analyzer import analyze_text
from lingity.cli import main
from lingity.models import JsonValue
from lingity.profiles import SCHEMA_DIR, Profile, canonical_json, load_profile, sha256_json
from lingity.readability import (
    FORMULA_COUNTS,
    FORMULAS,
    SyllableTableError,
    count_text,
    fallback_syllables,
    formula_values,
    syllables,
)

# Reach `scripts/` for this import only, as tests/test_release.py does: left on
# sys.path it would outrank the standard library for the rest of the session.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import build_syllable_table  # noqa: E402
finally:
    sys.path.remove(str(SCRIPTS))

FIXTURES = Path(__file__).parent / "fixtures"
GOLDENS = FIXTURES / "score-goldens.json"


def _block(text: str, profile: Profile | None = None) -> dict[str, Any]:
    return cast(dict[str, Any], analyze_text(text, profile)["readability_estimates"])


def _document_values(text: str) -> dict[str, float]:
    estimates = cast(dict[str, Any], _block(text)["document"]["estimates"])
    return {name: cast(float, estimates[name]["value"]) for name in FORMULAS}


def _graded(max_grade: float, **extra: float) -> Profile:
    data = copy.deepcopy(load_profile("web-copy").data)
    data["thresholds"]["max_grade"] = max_grade
    data["thresholds"].update(extra)
    return Profile(data=data, digest=sha256_json(data))


# --- known answers --------------------------------------------------------


def test_known_answer_for_one_short_sentence() -> None:
    """words=6, sentences=1, syllables=6, characters=17 (3+3+3+2+3+3), polysyllables=0.

    FK   = 0.39*6 + 11.8*1 - 15.59                      = -1.45
    FRE  = 206.835 - 1.015*6 - 84.6*1                   = 116.145 -> 116.15
    CLI  = 0.0588*(1700/6) - 0.296*(100/6) - 15.8       = -4.0733 -> -4.07
    SMOG = 1.0430*sqrt(0) + 3.1291                      = 3.13
    Fog  = 0.4*(6 + 0)                                  = 2.40
    ARI  = 4.71*(17/6) + 0.5*6 - 21.43                  = -5.085 -> -5.09
    """
    block = _block("The cat sat on the mat.")
    counts = block["document"]["counts"]
    assert {key: counts[key] for key in ("words", "sentences", "syllables", "characters", "polysyllables")} == {
        "words": 6,
        "sentences": 1,
        "syllables": 6,
        "characters": 17,
        "polysyllables": 0,
    }
    assert _document_values("The cat sat on the mat.") == {
        "flesch_kincaid_grade": -1.45,
        "flesch_reading_ease": 116.15,
        "coleman_liau_index": -4.07,
        "smog_grade": 3.13,
        "gunning_fog": 2.4,
        "automated_readability_index": -5.09,
    }


def test_known_answer_for_two_sentences_with_polysyllables() -> None:
    """communication 5, is 1, important 3 | we 1, understand 3, everything 3.

    words=6, sentences=2, syllables=16, polysyllables=4,
    characters = 13+2+9+2+10+10 = 46.

    FK   = 0.39*3 + 11.8*(16/6) - 15.59                 = 17.0467 -> 17.05
    FRE  = 206.835 - 1.015*3 - 84.6*(16/6)              = -21.81
    CLI  = 0.0588*(4600/6) - 0.296*(200/6) - 15.8       = 19.4133 -> 19.41
    SMOG = 1.0430*sqrt(4*30/2) + 3.1291                 = 11.2081 -> 11.21
    Fog  = 0.4*(3 + 100*4/6)                            = 27.8667 -> 27.87
    ARI  = 4.71*(46/6) + 0.5*3 - 21.43                  = 16.18
    """
    text = "Communication is important. We understand everything."
    for word, expected in {"communication": 5, "important": 3, "understand": 3, "everything": 3}.items():
        assert syllables(word) == (expected, True), word
    counts = _block(text)["document"]["counts"]
    assert (counts["words"], counts["sentences"], counts["syllables"], counts["characters"], counts["polysyllables"]) == (
        6,
        2,
        16,
        46,
        4,
    )
    assert _document_values(text) == {
        "flesch_kincaid_grade": 17.05,
        "flesch_reading_ease": -21.81,
        "coleman_liau_index": 19.41,
        "smog_grade": 11.21,
        "gunning_fog": 27.87,
        "automated_readability_index": 16.18,
    }


def test_each_estimate_carries_its_formula_and_the_counts_it_used() -> None:
    document = _block("Communication is important. We understand everything.")["document"]
    counts = document["counts"]
    for name, estimate in document["estimates"].items():
        assert estimate["formula"] == FORMULAS[name]
        assert estimate["counts"] == {key: counts[key] for key in FORMULA_COUNTS[name]}
        # The recorded counts are sufficient to replay the value.
        assert estimate["value"] == formula_values(readability.Counts(**counts))[name]


def test_values_are_null_rather_than_invented_when_nothing_is_counted() -> None:
    assert formula_values(readability.Counts()) == {name: None for name in FORMULAS}


# --- per sentence ---------------------------------------------------------


def test_per_sentence_entries_use_the_analyzer_sentences_and_offsets() -> None:
    text = "The cat sat on the mat. Communication is important."
    artifact = analyze_text(text)
    block = cast(dict[str, Any], artifact["readability_estimates"])
    sentences = cast(list[dict[str, Any]], artifact["sentences"])
    entries = block["sentences"]
    assert [entry["sentence_index"] for entry in entries] == [0, 1]
    for entry, sentence in zip(entries, sentences):
        assert (entry["start"], entry["end"]) == (sentence["start"], sentence["end"])
        assert text[entry["start"] : entry["end"]] == sentence["text"]
    assert entries[0]["values"]["flesch_kincaid_grade"] == -1.45
    # communication 5, is 1, important 3: FK = 0.39*3 + 11.8*3 - 15.59 = 20.98
    assert entries[1]["values"]["flesch_kincaid_grade"] == 20.98


def test_document_counts_are_the_sum_of_sentence_counts() -> None:
    fixture = json.loads((FIXTURES / "recommended-decision.json").read_text(encoding="utf-8"))
    block = _block(fixture["original"])
    summed: dict[str, int] = {}
    for entry in block["sentences"]:
        for key, value in entry["counts"].items():
            summed[key] = summed.get(key, 0) + value
    assert summed == block["document"]["counts"]


# --- syllables -----------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("make", 1),  # final silent e
        ("table", 2),  # consonant + le keeps its syllable
        ("jumped", 1),  # silent -ed
        ("wanted", 2),  # -ed after t is voiced
        ("yellow", 2),  # word-initial y is a consonant
        ("rhythm", 1),  # y inside a word is a vowel
        ("qwrtz", 1),  # never below one
        ("1234", 1),  # no letters at all
    ],
)
def test_fallback_rule(word: str, expected: int) -> None:
    assert fallback_syllables(word) == expected


def test_dictionary_wins_over_the_fallback_and_reports_which_was_used() -> None:
    # The spelling rule reads "business" as bus-i-ness; CMUdict says 2.
    assert fallback_syllables("business") == 3
    assert syllables("business") == (2, True)
    assert syllables("readability") == (5, True)
    assert syllables("Readability") == (5, True)
    assert syllables("lingity") == (3, False)


def test_compound_and_possessive_lookup() -> None:
    assert syllables("well-known") == (2, True)
    assert syllables("agency’s") == (3, True)
    assert syllables("reader's-guide") == (3, True)
    compound = count_text("Read the reader's-guide now.")
    assert compound.polysyllables == 1


def test_the_vendored_table_is_the_pinned_one() -> None:
    raw = readability.SYLLABLE_TABLE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == readability.SYLLABLE_SOURCE["table_sha256"]
    assert raw.count(b"\n") == readability.SYLLABLE_SOURCE["entries"]
    assert (readability.DATA_DIR / "CMUDICT-LICENSE").read_text(encoding="utf-8").startswith(
        "Copyright (C) 1993-2015 Carnegie Mellon University"
    )
    assert build_syllable_table.CMUDICT_COMMIT == readability.SYLLABLE_SOURCE["commit"]
    assert build_syllable_table.CMUDICT_SHA256 == readability.SYLLABLE_SOURCE["source_sha256"]


def test_an_altered_table_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    altered = tmp_path / "table.tsv"
    altered.write_bytes(readability.SYLLABLE_TABLE.read_bytes().replace(b"readability\t5", b"readability\t4"))
    monkeypatch.setattr(readability, "SYLLABLE_TABLE", altered)
    readability.syllable_table.cache_clear()
    try:
        with pytest.raises(SyllableTableError, match="not the pinned"):
            readability.syllable_table()
    finally:
        monkeypatch.undo()
        readability.syllable_table.cache_clear()
    assert readability.syllable_table()["readability"] == 5


def test_build_script_derivation_rule() -> None:
    source = (
        "every EH1 V ER0 IY0\n"
        "every(2) EH1 V R IY0\n"
        "aalborg AO1 L B AO0 R G # place, danish\n"
    )
    table = build_syllable_table.derive(source)
    # The first-listed pronunciation wins and the trailing comment is ignored.
    assert table == {"every": 3, "aalborg": 2}
    assert build_syllable_table.render({"b": 1, "a": 2}) == "a\t2\nb\t1\n"
    with pytest.raises(ValueError, match="before its primary"):
        build_syllable_table.derive("word(2) W ER1 D\n")


# --- normalization --------------------------------------------------------

FIGURES = "The enacted budget totals $11,898,367,000 across 412 programs in 2027 (p. P.1)."
REPLACED = "The enacted budget totals all across ten programs in May."
TOLERANCE_GRADES = 0.5
TOLERANCE_EASE = 3.0


def _within(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    """Return the formulas whose values differ by more than the stated tolerance."""
    out: dict[str, float] = {}
    for name in FORMULAS:
        limit = TOLERANCE_EASE if name == "flesch_reading_ease" else TOLERANCE_GRADES
        gap = abs(first[name] - second[name])
        if gap > limit:
            out[name] = gap
    return out


def test_figures_score_like_the_same_sentence_with_a_short_word() -> None:
    """Each figure counts as one word of 1 syllable and 3 characters, and the
    page citation is not counted, so a figure-heavy sentence lands within
    0.5 grade (3 Reading Ease points) of the sentence with a short word in
    place of each figure. The replacements here are 1 syllable and 3
    characters, the figure's own weight, so the two sentences count the same.

    A longer replacement moves the character formulas, by design of those
    formulas rather than of the rule: "some" (4 characters) for each of the
    three figures adds 0.3 characters per word to this 10-word sentence, which
    is 1.76 grades of Coleman-Liau."""
    figures = _block(FIGURES)["document"]
    assert figures["counts"]["figures"] == 3
    assert figures["counts"]["excluded_words"] == 2  # p, P.1
    assert _within(_document_values(FIGURES), _document_values(REPLACED)) == {}


def test_without_normalization_the_same_pair_is_far_apart() -> None:
    """The tolerance above is not loose: un-normalized, the figures inflate the
    character-based formulas well past it."""
    raw = cast(dict[str, float], _block(FIGURES)["document"]["without_normalization"]["values"])
    gaps = _within(raw, _document_values(REPLACED))
    assert {"coleman_liau_index", "automated_readability_index"} <= set(gaps)


def test_code_urls_and_references_are_not_counted_but_link_labels_are() -> None:
    counted = count_text("Read `config_value` at https://example.com/a/b and [ADR-12] (Smith, 2019).")
    # Read, at, and are counted; the code span, URL, reference and citation are not.
    assert (counted.words, counted.excluded_words) == (3, 7)
    link = count_text("Read the [guide](https://example.com) first.")
    assert link.words == 4  # Read, the, guide, first
    # Un-normalized, the URL is prose too: "https" and "example.com".
    assert count_text("Read the [guide](https://example.com) first.", normalize=False).words == 6


def test_code_scanner_handles_escaped_and_multi_backticks() -> None:
    escaped = count_text(r"Read \`plain words\` and ``code ` inner words`` now.")
    assert escaped.words == 5  # Read, plain, words, and, now
    assert count_text("Read ``code ` inner words`` now.").words == 2


def test_link_destinations_and_reference_targets_are_not_prose() -> None:
    assert count_text("Read the [guide](setup.md) first.").words == 4
    assert count_text("Read the [guide](#installation) first.").words == 4
    assert count_text("Read the [guide][ref] first.", references=frozenset({"REF"})).words == 4
    assert count_text("Read the [guide][] first.", references=frozenset({"GUIDE"})).words == 4
    assert count_text("Read the [guide] first.", references=frozenset({"GUIDE"})).words == 4
    assert count_text("Read [ADR-12] first.").words == 2


def test_invalid_commonmark_link_keeps_its_parenthesized_words() -> None:
    invalid = count_text("Read [note](important setup instructions) now.")
    assert (invalid.words, invalid.excluded_words) == (5, 1)
    assert count_text("Read [note](setup. md) now.").words == 4


def test_quoted_link_title_parenthesis_and_code_bracket() -> None:
    titled = count_text('Read the [guide](setup.md "Step 1) Setup") first.')
    assert (titled.words, titled.excluded_words) == (4, 4)
    coded_label = count_text("Read [Read `]` carefully](setup.md) now.")
    assert (coded_label.words, coded_label.excluded_words) == (4, 1)


def test_many_unmatched_brackets_do_not_hide_later_link() -> None:
    text = "[" * 10000 + " Read [guide](setup.md) now."
    counted = count_text(text)
    assert (counted.words, counted.excluded_words) == (3, 1)
    nested = count_text("See [the [guide](setup.md)] now.")
    assert (nested.words, nested.excluded_words) == (4, 1)


def test_exclusions_cross_sentence_boundaries_and_wrapped_lines() -> None:
    code = _block("Read `alpha. beta gamma` now.")["document"]["counts"]
    assert (code["words"], code["sentences"], code["excluded_words"]) == (2, 2, 3)

    wrapped = _block("Read `alpha. beta gamma` across\nwrapped lines and act.")["document"]["counts"]
    assert (wrapped["words"], wrapped["excluded_words"]) == (6, 3)

    linked = _block("Read [the guide. setup instructions](setup.md) now.")["document"]["counts"]
    assert (linked["words"], linked["excluded_words"]) == (6, 1)

    linked_wrapped = _block("Read [the guide. setup\ninstructions](setup.md) now.")["document"]["counts"]
    assert (linked_wrapped["words"], linked_wrapped["excluded_words"]) == (6, 1)


def test_document_reference_definition_preserves_visible_label() -> None:
    text = "Read the [guide][ref] first.\n\n[ref]: setup.md\n"
    assert _block(text)["document"]["counts"]["words"] == 4


def test_grade_output_cannot_remove_the_installed_syllable_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "data"
    store.mkdir()
    installed = store / "cmudict-syllables.tsv"
    installed.write_text("pinned contents", encoding="utf-8")
    monkeypatch.setattr(readability, "DATA_DIR", store)
    # The CLI imports DATA_DIR by value, so patch the guard's binding too.
    import lingity.cli as cli

    monkeypatch.setattr(cli, "DATA_DIR", store)
    source = tmp_path / "page.md"
    source.write_text("The cat sat on the mat.", encoding="utf-8")
    assert main(["grade", str(source), "--output", str(installed)]) == 2
    assert "installed data directory" in capsys.readouterr().err
    assert installed.read_text(encoding="utf-8") == "pinned contents"


def test_a_sentence_of_only_excluded_words_is_not_a_sentence() -> None:
    counts = count_text("(p. 12)")
    assert (counts.words, counts.sentences) == (0, 0)


def test_the_normalization_rule_is_recorded_in_the_artifact() -> None:
    method = _block("The cat sat on the mat.")["method"]
    assert method["normalization"] == {
        "applied": True,
        "rule": readability.NORMALIZATION_RULE,
        "figure_syllables": 1,
        "figure_characters": 3,
    }
    assert method["syllables"]["source"]["table_sha256"] == readability.SYLLABLE_SOURCE["table_sha256"]
    assert method["syllables"]["fallback"] == readability.FALLBACK_RULE
    assert method["formulas"] == FORMULAS


# --- determinism, offline, verify ----------------------------------------


def test_readability_block_is_byte_deterministic(recommendation_fixture: dict[str, str]) -> None:
    first = canonical_json(_block(recommendation_fixture["original"]))
    second = canonical_json(_block(recommendation_fixture["original"]))
    assert first == second


def test_analysis_makes_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("analysis opened a socket")

    analyze_text("Warm the pipeline first.")
    readability.syllable_table()
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    readability.syllable_table.cache_clear()
    assert _block(FIGURES)["document"]["counts"]["figures"] == 3


def test_verify_replays_the_readability_block(tmp_path: Path, recommendation_fixture: dict[str, str]) -> None:
    source = tmp_path / "source.txt"
    artifact = tmp_path / "analysis.json"
    source.write_text(recommendation_fixture["original"], encoding="utf-8")
    assert main(["analyze", str(source), "--output", str(artifact)]) == 0
    assert main(["verify", str(artifact)]) == 0

    value = cast(dict[str, Any], json.loads(artifact.read_text(encoding="utf-8")))
    value["readability_estimates"]["document"]["estimates"]["smog_grade"]["value"] += 0.01
    unhashed = {key: item for key, item in value.items() if key != "analysis_sha256"}
    value["analysis_sha256"] = hashlib.sha256(canonical_json(unhashed).encode("utf-8")).hexdigest()
    artifact.write_text(json.dumps(value), encoding="utf-8")
    # The hash is consistent, so only the replay can catch the altered value.
    assert main(["verify", str(artifact)]) == 2


def test_schema_requires_the_block() -> None:
    validator = Draft202012Validator(json.loads((SCHEMA_DIR / "analysis.schema.json").read_text(encoding="utf-8")))
    artifact = analyze_text("The cat sat on the mat.")
    validator.validate(artifact)
    del artifact["readability_estimates"]
    with pytest.raises(ValidationError):
        validator.validate(artifact)


# --- the opt-in grade threshold -------------------------------------------

HARD = "Comprehensive organizational documentation necessitates considerable institutional deliberation."
EASY = "The cat sat on the mat and then it ran home."
HEADING = "# Implementation Considerations\n\n"


def _grade_findings(text: str, profile: Profile) -> list[dict[str, Any]]:
    findings = cast(list[dict[str, Any]], analyze_text(text, profile)["findings"])
    return [finding for finding in findings if finding["rule_id"] == "LING-GRADE-001"]


def test_grade_finding_fires_only_above_the_threshold() -> None:
    text = f"{HEADING}{EASY} {HARD}"
    artifact = analyze_text(text, _graded(8))
    block = cast(dict[str, Any], artifact["readability_estimates"])
    grades = {entry["sentence_index"]: entry["values"]["flesch_kincaid_grade"] for entry in block["sentences"]}
    findings = _grade_findings(text, _graded(8))
    assert len(findings) == 1
    finding = findings[0]
    hard_index = finding["location"]["sentence_index"]
    assert text[finding["location"]["start"] : finding["location"]["end"]] == HARD
    assert grades[hard_index] > 8
    # The easy sentence is below the threshold, and the heading is above it but
    # shorter than min_words, so neither is reported.
    heading_grade, easy_grade = (grade for index, grade in sorted(grades.items()) if index != hard_index)
    assert easy_grade < 8 < heading_grade
    assert finding["observed_value"] == {
        "formula": "flesch_kincaid_grade",
        "grade": grades[hard_index],
        "words": 7,
    }
    assert finding["threshold"] == {"formula": "flesch_kincaid_grade", "max_grade": 8}
    assert finding["dimension"] == "sentence_load"
    assert finding["penalty"] == 15.0  # capped: (grade - 8) * 1.5 is far above 15
    assert finding["remediation"]


def test_a_short_heading_above_the_threshold_is_not_reported() -> None:
    profile = _graded(8)
    block = _block(HEADING + EASY, profile)
    assert block["sentences"][0]["values"]["flesch_kincaid_grade"] > 8
    assert _grade_findings(HEADING + EASY, profile) == []
    # Lowering min_words to the heading's length makes it reportable.
    assert len(_grade_findings(HEADING + EASY, _graded(8, grade_min_words=2))) == 1


def test_a_sentence_exactly_at_the_threshold_is_not_reported() -> None:
    grade = cast(float, _block(HARD)["sentences"][0]["values"]["flesch_kincaid_grade"])
    assert _grade_findings(HARD, _graded(grade)) == []
    assert len(_grade_findings(HARD, _graded(grade - 0.01))) == 1


def test_grade_penalty_feeds_sentence_load_and_lowers_the_score() -> None:
    text = f"{EASY} {HARD}"
    plain = cast(dict[str, Any], analyze_text(text, load_profile("web-copy"))["score"])
    graded = cast(dict[str, Any], analyze_text(text, _graded(8))["score"])
    load = next(component for component in graded["components"] if component["dimension"] == "sentence_load")
    assert {"rule_id": "LING-GRADE-001", "points": 15.0} in load["deductions"]
    assert graded["value"] < plain["value"]


def test_without_max_grade_no_profile_reports_grade() -> None:
    for name in json.loads(GOLDENS.read_text(encoding="utf-8"))["profiles"]:
        profile = load_profile(name)
        assert "max_grade" not in profile.thresholds
        assert _grade_findings(HARD, profile) == []
        assert _block(HARD, profile)["method"]["grade_threshold"] is None


def test_the_grade_policy_is_recorded_when_set() -> None:
    assert _block(EASY, _graded(8))["method"]["grade_threshold"] == {
        "rule_id": "LING-GRADE-001",
        "formula": "flesch_kincaid_grade",
        "max_grade": 8,
        "min_words": 5,
    }


def test_verify_replays_an_opted_in_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = tmp_path / "profiles"
    store.mkdir()
    for path in profiles.PROFILE_DIR.glob("*.json"):
        (store / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    data = copy.deepcopy(load_profile("web-copy").data)
    data["name"] = "web-copy-graded"
    data["thresholds"]["max_grade"] = 8
    (store / "web-copy-graded.v1.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(profiles, "PROFILE_DIR", store)

    source = tmp_path / "source.txt"
    artifact = tmp_path / "analysis.json"
    source.write_text(f"{EASY} {HARD}", encoding="utf-8")
    assert main(["analyze", str(source), "--profile", "web-copy-graded", "--output", str(artifact)]) == 0
    recorded = cast(dict[str, Any], json.loads(artifact.read_text(encoding="utf-8")))
    assert [f["rule_id"] for f in recorded["findings"]].count("LING-GRADE-001") == 1
    assert main(["verify", str(artifact)]) == 0


# --- existing profiles are unchanged --------------------------------------


def _goldens() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(GOLDENS.read_text(encoding="utf-8")))


@pytest.mark.parametrize("profile_name", sorted(_goldens()["profiles"]))
def test_existing_profiles_score_exactly_as_before(profile_name: str) -> None:
    """Snapshots taken from analyzer 1.7.0, before readability estimates existed.

    The profile itself must be byte-identical (same digest and version), and
    every score, finding, sentence record and protected manifest must hash to
    the same value it did then.
    """
    goldens = _goldens()
    expected = goldens["profiles"][profile_name]
    profile = load_profile(profile_name)
    assert (profile.digest, profile.version) == (expected["digest"], expected["version"])
    for key, text in goldens["texts"].items():
        artifact = analyze_text(text, profile)
        recorded = expected["artifacts"][key]

        def digest(value: JsonValue) -> str:
            return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

        observed = {
            "score_sha256": digest(artifact["score"]),
            "findings_sha256": digest(artifact["findings"]),
            "sentences_sha256": digest(artifact["sentences"]),
            "protected_sha256": cast(dict[str, Any], artifact["protected"])["sha256"],
            "score_value": cast(dict[str, Any], artifact["score"])["value"],
            "score_band": cast(dict[str, Any], artifact["score"])["band"],
        }
        assert observed == recorded, f"{profile_name} / {key} changed"


def test_goldens_cover_every_shipped_profile() -> None:
    shipped = {path.name.rsplit(".v", 1)[0] for path in profiles.PROFILE_DIR.glob("*.v*.json")}
    assert set(_goldens()["profiles"]) == shipped


# --- CLI ------------------------------------------------------------------


def test_grade_cli_prints_every_formula(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "page.md"
    source.write_text(f"{EASY} {FIGURES}", encoding="utf-8")
    assert main(["grade", str(source), "--sentences"]) == 0
    out = capsys.readouterr().out
    for label in (
        "Flesch-Kincaid grade",
        "Flesch Reading Ease",
        "Coleman-Liau index",
        "SMOG grade",
        "Gunning Fog index",
        "Automated Readability Index",
    ):
        assert label in out
    assert "none of them measures comprehension" in out
    assert "3 figure(s) counted as one short word" in out
    assert "[1]" in out


def test_grade_cli_json_is_the_artifact_block(tmp_path: Path) -> None:
    source = tmp_path / "page.md"
    output = tmp_path / "grade.json"
    source.write_text(FIGURES, encoding="utf-8")
    assert main(["grade", str(source), "--format", "json", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == _block(FIGURES)


def test_grade_cli_reports_the_profile_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "profiles"
    store.mkdir()
    data = copy.deepcopy(load_profile("web-copy").data)
    data["name"] = "graded"
    data["thresholds"]["max_grade"] = 8
    (store / "graded.v1.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(profiles, "PROFILE_DIR", store)
    source = tmp_path / "page.md"
    source.write_text(f"{EASY} {HARD}", encoding="utf-8")
    assert main(["grade", str(source), "--profile", "graded"]) == 0
    assert "1 sentence(s) above it (LING-GRADE-001)" in capsys.readouterr().out


def test_grade_cli_fails_explicitly_on_missing_input(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["grade", str(tmp_path / "missing.md")]) == 2
    assert "lingity grade failed" in capsys.readouterr().err
