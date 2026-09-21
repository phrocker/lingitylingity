"""The web-copy profile reads public marketing prose, not internal documents.

Its job is to catch the two things that make a public page read as machine
written: stacked nouns, and the stock vocabulary a model reaches for when asked
to sound enthusiastic. It should say nothing about prose that names a place, a
road, and a requirement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from lingity.analyzer import analyze_text
from lingity.models import JsonValue
from lingity.profiles import PROFILE_DIR, Profile, load_profile

# The shape a model produces when asked to write about an employer.
SLOP = (
    "Our world class team leverages cutting edge solutions to deliver seamless "
    "outcomes. We foster a dynamic team environment where passionate engineers "
    "harness the power of next generation technology. "
    "This is a testament to our robust, holistic approach."
)

# The same subject, written by someone who has been there.
CONCRETE = (
    "The office sits at 2720 Technology Drive, inside the park. "
    "Several other contractors are a short walk away. "
    "Most postings ask for an active clearance before you start. "
    "Many ask for a polygraph on top of it."
)


@pytest.fixture
def web_copy() -> Profile:
    return load_profile("web-copy")


def _rule_ids(text: str, profile: Profile) -> set[str]:
    findings = cast(list[dict[str, JsonValue]], analyze_text(text, profile)["findings"])
    return {cast(str, finding["rule_id"]) for finding in findings}


def _score(text: str, profile: Profile) -> float:
    score = cast(dict[str, JsonValue], analyze_text(text, profile)["score"])
    return cast(float, score["value"])


def test_slop_scores_below_concrete_prose(web_copy: Profile) -> None:
    assert _score(SLOP, web_copy) < _score(CONCRETE, web_copy)


def test_concrete_prose_reaches_the_clear_band(web_copy: Profile) -> None:
    # Naming a street, a requirement and a distance must not be penalised.
    assert _score(CONCRETE, web_copy) >= 85.0


def test_marketing_vocabulary_is_reported_as_jargon(web_copy: Profile) -> None:
    assert "LING-JARGON-001" in _rule_ids(SLOP, web_copy)


def test_concrete_prose_reports_no_jargon(web_copy: Profile) -> None:
    assert "LING-JARGON-001" not in _rule_ids(CONCRETE, web_copy)


def test_domain_nouns_are_not_treated_as_nominalizations(web_copy: Profile) -> None:
    # "intelligence", "government" and "infrastructure" carry a nominalization
    # suffix but name things rather than hide an actor, the same reasoning the
    # inherited list already applies to "architecture" and "governance".
    text = (
        "The company supports intelligence customers. "
        "It maintains the infrastructure a government office depends on."
    )
    assert "LING-NOMINALIZATION-001" not in _rule_ids(text, web_copy)


def test_profile_weights_favour_morphology_and_word_choice(web_copy: Profile) -> None:
    weights = web_copy.weights
    assert sum(weights.values()) == 100
    # Stacked nouns and stock phrases are the tells this profile exists to find,
    # so they must outweigh structure.
    assert weights["morphology"] > weights["structure"]
    assert weights["lexical_clarity"] > weights["structure"]


def _jargon_fixture() -> dict[str, dict[str, Any]]:
    path = Path(__file__).parent / "fixtures" / "web-copy-jargon.json"
    return cast(dict[str, dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


def _new_jargon(profile: Profile) -> set[str]:
    """Phrases this profile adds, ignoring the ones it inherits verbatim."""
    inherited = {
        phrase
        for phrases in cast(
            dict[str, list[str]], load_profile("product-strategy").rules["jargon"]
        ).values()
        for phrase in phrases
    }
    return {
        phrase
        for phrases in cast(dict[str, list[str]], profile.rules["jargon"]).values()
        for phrase in phrases
    } - inherited


def _fixture_cases() -> list[tuple[str, str, str, str]]:
    return [
        (surface, entry["lemma"], entry["classification"], example)
        for surface, entry in sorted(_jargon_fixture().items())
        for example in entry["examples"]
    ]


@pytest.mark.parametrize(
    "surface,lemma,classification,example",
    _fixture_cases(),
    ids=lambda value: str(value)[:40],
)
def test_every_added_phrase_fires_and_earns_its_place(
    surface: str, lemma: str, classification: str, example: str, web_copy: Profile
) -> None:
    """A stored phrase that never matches is a rule nobody enforces.

    Two things are checked, and the second is the one that catches real
    mistakes. First the phrase must fire on a natural sentence — the profile
    stores LEMMA sequences, so "bleeding edge" is stored as "bleed edge" and a
    surface form pinned to the wrong grammatical role is silent everywhere.
    Second, the finding must DISAPPEAR when this entry alone is removed. Without
    that, an entry duplicating one the profile already inherits looks alive
    while contributing nothing.
    """
    findings = cast(list[dict[str, JsonValue]], analyze_text(example, web_copy)["findings"])
    observed = [
        cast(dict[str, JsonValue], f["observed_value"])
        for f in findings
        if f["rule_id"] == "LING-JARGON-001"
    ]
    # Assert the pair, not just the rule id: moving a phrase into the wrong
    # user-visible category would otherwise leave this test green.
    assert {"phrase": surface, "classification": classification} in observed, example

    stripped = json.loads(json.dumps(web_copy.data))
    for group in stripped["rules"]["jargon"]:
        stripped["rules"]["jargon"][group] = [
            p for p in stripped["rules"]["jargon"][group] if p != lemma
        ]
    reduced = Profile(data=stripped, digest="test")
    survived = [
        cast(dict[str, JsonValue], f["observed_value"])["phrase"]
        for f in cast(list[dict[str, JsonValue]], analyze_text(example, reduced)["findings"])
        if f["rule_id"] == "LING-JARGON-001"
    ]
    assert surface not in survived, f"{lemma!r} is redundant: {example}"


def test_the_fixture_covers_every_phrase_the_profile_adds(web_copy: Profile) -> None:
    covered = {lemma for _surface, lemma, _cls, _ex in _fixture_cases()}
    assert _new_jargon(web_copy) == covered


def test_added_phrases_do_not_overlap_each_other(web_copy: Profile) -> None:
    """One phrase containing another double-counts a single occurrence."""
    stored = {
        phrase
        for phrases in cast(dict[str, list[str]], web_copy.rules["jargon"]).values()
        for phrase in phrases
    }
    overlaps = [
        (long, short)
        for long in stored
        for short in stored
        if long != short and f" {short} " in f" {long} "
    ]
    assert not overlaps, overlaps


# ── The implied second person ─────────────────────────────────────────────────
#
# Public copy addresses the reader and drops the subject to do it. "Find the
# median for your age band" names the reader as surely as "you find the median"
# does, so the profile reads a subjectless directive as its genre's actor. The
# permission is the one resume-review already uses for the implied first
# person; only its name changed, because the parse cannot tell the pronouns
# apart and does not need to.


@pytest.mark.parametrize(
    "text",
    [
        "Find the median for your age band.",
        "Ignore the mean entirely.",
        "Check the year behind any number you read.",
    ],
)
def test_an_imperative_addressed_to_the_reader_is_not_a_missing_actor(
    text: str, web_copy: Profile
) -> None:
    assert "LING-ACTOR-001" not in _rule_ids(text, web_copy)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The migration must be completed before the release.", "LING-PASSIVE-001"),
        ("A decision should be made about the rollout.", "LING-AGENCY-001"),
        ("Value must be delivered to the customer.", "LING-AGENCY-001"),
    ],
)
def test_the_permission_leaves_the_other_agency_passes_alone(
    text: str, expected: str, web_copy: Profile
) -> None:
    """It reaches the absent subject only, so hidden agency is still reported.

    Without this the reading would excuse the defect it exists to expose: copy
    that says a thing must happen and never says who does it. An imperative
    tells the reader to act; an agentless passive tells nobody to.
    """
    assert expected in _rule_ids(text, web_copy)


def test_the_permission_is_named_for_the_shape_not_the_pronoun(web_copy: Profile) -> None:
    assert web_copy.thresholds.get("allow_implied_subject") == 1
    assert "allow_implied_first_person" not in web_copy.thresholds


def test_the_name_this_shipped_under_is_still_read() -> None:
    """A profile written against 0.1.0 keeps its behaviour and its digest.

    resume-review still carries the old key, so this is the path it takes.
    """
    from lingity.analyzer import _allows_implied_subject

    legacy = Profile(data={"thresholds": {"allow_implied_first_person": 1}}, digest="test")
    renamed = Profile(data={"thresholds": {"allow_implied_subject": 1}}, digest="test")
    neither = Profile(data={"thresholds": {"max_sentence_words": 20}}, digest="test")

    assert _allows_implied_subject(legacy)
    assert _allows_implied_subject(renamed)
    assert not _allows_implied_subject(neither)


def test_the_shipped_resume_profile_still_reaches_the_reading() -> None:
    """The alias is load-bearing, not decorative -- resume-review depends on it."""
    from lingity.analyzer import _allows_implied_subject

    assert _allows_implied_subject(load_profile("resume-review"))


# ── Nouns that carry a suffix but name a thing ────────────────────────────────


@pytest.mark.parametrize("word", ["age", "average", "figure"])
def test_measurement_nouns_are_not_treated_as_nominalizations(
    word: str, web_copy: Profile
) -> None:
    """"age", "average" and "figure" end in a nominalization suffix and hide no actor.

    They are unavoidable in copy about money, where the subject is often an
    average figure for an age group. The list already applies this reasoning to
    "coverage", "percentage", "usage" and "measure".
    """
    assert word in cast(list[str], web_copy.rules["nominalization_exclusions"])


def test_a_sentence_of_those_nouns_reports_nothing(web_copy: Profile) -> None:
    text = "The average figure for that age group tells you little."
    assert "LING-NOMINALIZATION-001" not in _rule_ids(text, web_copy)


def test_the_exclusions_are_what_spares_that_sentence(web_copy: Profile) -> None:
    """Prove the guard can fail, so the test above cannot pass for another reason."""
    stripped = json.loads(json.dumps(web_copy.data))
    stripped["rules"]["nominalization_exclusions"] = [
        word
        for word in stripped["rules"]["nominalization_exclusions"]
        if word not in {"age", "average", "figure"}
    ]
    without = Profile(data=stripped, digest="test")
    text = "The average figure for that age group tells you little."
    assert "LING-NOMINALIZATION-001" in _rule_ids(text, without)
