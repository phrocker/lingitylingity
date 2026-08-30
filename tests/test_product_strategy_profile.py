"""The product-strategy profile reads strategy prose, not architecture prose."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from lingity.analyzer import analyze_text
from lingity.models import JsonValue
from lingity.profiles import PROFILE_DIR, Profile, load_profile

HYPE = (
    "Our world class platform provides seamless value to customers. "
    "It is believed that the market wants a frictionless solution. "
    "Value will be delivered at the end of the day through our secret sauce. "
    "The decision should be made to prioritize this strategic initiative."
)

SPECIFIC = (
    "The buyer spends two thousand dollars a month and loses the account without warning. "
    "We read the status field every minute and name the cause within five minutes. "
    "We charge ninety dollars a month for each connected account. "
    "We do not promise that the vendor reinstates an account."
)


@pytest.fixture
def strategy() -> Profile:
    return load_profile("product-strategy")


def _rule_ids(text: str, profile: Profile) -> set[str]:
    findings = cast(list[dict[str, JsonValue]], analyze_text(text, profile)["findings"])
    return {cast(str, finding["rule_id"]) for finding in findings}


def _score(text: str, profile: Profile) -> float:
    score = cast(dict[str, JsonValue], analyze_text(text, profile)["score"])
    return cast(float, score["value"])


def _profile_names() -> list[str]:
    return sorted(path.name.rsplit(".v", 1)[0] for path in PROFILE_DIR.glob("*.v*.json"))


@pytest.mark.parametrize("name", _profile_names())
def test_every_installed_profile_loads_and_validates(name: str) -> None:
    profile = load_profile(name)
    assert profile.name == name
    assert sum(profile.weights.values()) == 100


def test_the_profile_is_installed_and_discoverable() -> None:
    assert "product-strategy" in _profile_names()


def test_bands_cover_the_whole_range(strategy: Profile) -> None:
    bands = cast(list[dict[str, float]], strategy.data["bands"])
    assert bands[0]["minimum"] == 0
    assert bands[-1]["maximum"] == 100
    for earlier, later in zip(bands, bands[1:]):
        assert earlier["maximum"] < later["minimum"]


def test_it_supplies_every_rule_the_analyzer_reads(strategy: Profile) -> None:
    """A new profile must answer every rule lookup the shipped profile answers."""
    reference = load_profile("architecture-review")
    assert set(reference.rules) <= set(strategy.rules)
    assert set(reference.thresholds) <= set(strategy.thresholds)


def _jargon_fixture() -> dict[str, dict[str, str]]:
    path = Path(__file__).parent / "fixtures" / "product-strategy-jargon.json"
    return cast(dict[str, dict[str, str]], json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("surface", sorted(_jargon_fixture()))
def test_every_jargon_phrase_fires_on_its_example(surface: str) -> None:
    """A stored phrase that never matches is a rule nobody enforces.

    The profile stores lemma sequences, and a lemma is not always the surface
    word: "best in class" parses as "good in class", and "cutting edge" as
    "cut edge". Deriving a phrase by eye therefore ships a rule that is silent.
    Every entry here is pinned to a sentence that provably raises the finding.
    """
    entry = _jargon_fixture()[surface]
    strategy = load_profile("product-strategy")
    assert "LING-JARGON-001" in _rule_ids(entry["example"], strategy)


def test_the_fixture_covers_every_jargon_phrase_in_the_profile(
    strategy: Profile,
) -> None:
    stored = {
        phrase
        for phrases in cast(dict[str, list[str]], strategy.rules["jargon"]).values()
        for phrase in phrases
    }
    covered = {entry["lemma"] for entry in _jargon_fixture().values()}
    assert stored == covered


def test_a_lemma_differs_from_its_surface_form(strategy: Profile) -> None:
    """Guards the reason the fixture exists, so nobody "corrects" the lemmas."""
    fixture = _jargon_fixture()
    assert fixture["best in class"]["lemma"] == "good in class"
    assert fixture["cutting edge"]["lemma"] == "cut edge"


def test_the_market_is_not_an_actor(strategy: Profile) -> None:
    """A sentence whose only actor is the market names nobody who can act."""
    actors = cast(list[str], strategy.rules["actor_terms"])
    for absent in ("market", "industry", "space", "ecosystem"):
        assert absent not in actors


def test_a_named_role_is_an_actor(strategy: Profile) -> None:
    actors = cast(list[str], strategy.rules["actor_terms"])
    for present in ("buyer", "customer", "competitor", "vendor", "team"):
        assert present in actors


def test_business_acronyms_do_not_fire(strategy: Profile) -> None:
    text = "The SMB buyer raises ARR and lowers CAC."
    assert "LING-ABBREVIATION-001" not in _rule_ids(text, strategy)


def test_go_to_market_is_an_allowed_compound(strategy: Profile) -> None:
    text = "The team owns the go-to-market plan."
    assert "LING-COMPOUND-001" not in _rule_ids(text, strategy)


def test_it_declares_no_protected_phrases(strategy: Profile) -> None:
    """The meaning gate reads claim signatures from the parse, not phrase lists."""
    assert strategy.rules["protected_concepts"] == []


def test_it_catches_hype_that_the_architecture_profile_certifies() -> None:
    """The shipped profile cannot see buzzwords, which is why this one exists."""
    architecture = load_profile("architecture-review")
    strategy = load_profile("product-strategy")
    assert _score(HYPE, architecture) > _score(HYPE, strategy)
    assert "LING-JARGON-001" not in _rule_ids(HYPE, architecture)
    assert "LING-JARGON-001" in _rule_ids(HYPE, strategy)


def test_specific_prose_survives_the_profile(strategy: Profile) -> None:
    """Naming a number, an actor, and a limit must not be penalised."""
    assert _score(SPECIFIC, strategy) >= 95.0


def test_analysis_is_deterministic_under_the_profile(strategy: Profile) -> None:
    first = analyze_text(HYPE, strategy)
    second = analyze_text(HYPE, strategy)
    assert first["analysis_sha256"] == second["analysis_sha256"]


def test_the_artifact_records_the_profile_it_used(strategy: Profile) -> None:
    reference = cast(dict[str, JsonValue], analyze_text(HYPE, strategy)["profile"])
    assert reference["name"] == "product-strategy"
    assert reference["version"] == "1.0.0"
    assert reference["digest"] == strategy.digest
