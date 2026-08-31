"""The web-copy profile reads public marketing prose, not internal documents.

Its job is to catch the two things that make a public page read as machine
written: stacked nouns, and the stock vocabulary a model reaches for when asked
to sound enthusiastic. It should say nothing about prose that names a place, a
road, and a requirement.
"""

from __future__ import annotations

from typing import cast

import pytest

from lingity.analyzer import analyze_text
from lingity.models import JsonValue
from lingity.profiles import Profile, load_profile

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
