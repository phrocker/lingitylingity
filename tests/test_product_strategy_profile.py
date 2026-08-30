"""The product-strategy profile reads strategy prose, not architecture prose."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

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
    # Non-overlap is not coverage. Scores round to hundredths, so a gap wider
    # than 0.01 leaves scores no band can name and _band_for raises on them.
    for earlier, later in zip(bands, bands[1:]):
        assert round(later["minimum"] - earlier["maximum"], 2) == 0.01


def test_it_supplies_every_rule_the_analyzer_reads(strategy: Profile) -> None:
    """A new profile must answer every rule lookup the shipped profile answers."""
    reference = load_profile("architecture-review")
    assert set(reference.rules) <= set(strategy.rules)
    assert set(reference.thresholds) <= set(strategy.thresholds)


def _jargon_fixture() -> dict[str, dict[str, Any]]:
    path = Path(__file__).parent / "fixtures" / "product-strategy-jargon.json"
    return cast(dict[str, dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


def _fixture_cases() -> list[tuple[str, str, str]]:
    return [
        (surface, variant["lemma"], variant["example"])
        for surface, entry in sorted(_jargon_fixture().items())
        for variant in entry["variants"]
    ]


@pytest.mark.parametrize(
    "surface,lemma,example", _fixture_cases(), ids=lambda value: str(value)[:40]
)
def test_every_jargon_variant_fires_on_its_example(
    surface: str, lemma: str, example: str
) -> None:
    """A stored phrase that never matches is a rule nobody enforces.

    The profile stores lemma sequences, and a lemma is not always the surface
    word: "cutting edge" parses as "cut edge". Worse, a lemma depends on the
    grammatical role the phrase takes. "thought leadership" yields "thought
    leadership" as a subject and "think leadership" after a copula, so pinning
    one carrier ships a rule that is silent everywhere else. Every variant here
    is pinned to a natural sentence that provably raises the finding.
    """
    strategy = load_profile("product-strategy")
    findings = cast(
        list[dict[str, JsonValue]], analyze_text(example, strategy)["findings"]
    )
    observed = [
        cast(dict[str, JsonValue], finding["observed_value"])
        for finding in findings
        if finding["rule_id"] == "LING-JARGON-001"
    ]
    classification = _jargon_fixture()[surface]["classification"]
    # Assert the pair, not just the rule id: moving a phrase into the wrong
    # user-visible category would otherwise leave this test green.
    assert {"phrase": surface, "classification": classification} in observed


def test_the_fixture_covers_every_jargon_phrase_in_the_profile(
    strategy: Profile,
) -> None:
    stored = {
        phrase
        for phrases in cast(dict[str, list[str]], strategy.rules["jargon"]).values()
        for phrase in phrases
    }
    covered = {lemma for _surface, lemma, _example in _fixture_cases()}
    assert stored == covered


def test_a_role_dependent_phrase_fires_in_every_role(strategy: Profile) -> None:
    """The reported failure: one carrier pinned a lemma real prose never makes."""
    assert "LING-JARGON-001" in _rule_ids("Our thought leadership attracts buyers.", strategy)
    assert "LING-JARGON-001" in _rule_ids("The product is thought leadership.", strategy)
    assert "LING-JARGON-001" in _rule_ids("The platform is best in class.", strategy)
    assert "LING-JARGON-001" in _rule_ids("We ship a best in class platform.", strategy)


def test_a_lemma_differs_from_its_surface_form() -> None:
    """Guards the reason the fixture exists, so nobody "corrects" the lemmas."""
    lemmas = {
        surface: {lemma for _s, lemma, _e in _fixture_cases() if _s == surface}
        for surface in ("best in class", "cutting edge", "thought leadership")
    }
    assert "good in class" in lemmas["best in class"]
    assert lemmas["cutting edge"] == {"cut edge"}
    assert {"think leadership", "thought leadership"} <= lemmas["thought leadership"]


def test_the_market_is_not_an_actor(strategy: Profile) -> None:
    """A sentence whose only actor is the market names nobody who can act."""
    actors = cast(list[str], strategy.rules["actor_terms"])
    for absent in ("market", "industry", "space", "ecosystem"):
        assert absent not in actors


def test_a_market_only_directive_is_reported(strategy: Profile) -> None:
    """Omitting a term from actor_terms must change what the analyzer reports.

    Before `require_responsible_actor`, any overt subject cleared the rule, so
    the omission above decided nothing and this profile's actor list was
    decoration.
    """
    assert "LING-ACTOR-001" in _rule_ids("The market should prioritize retention.", strategy)
    assert "LING-ACTOR-001" in _rule_ids("The industry should adopt the standard.", strategy)


def test_a_named_actor_clears_the_same_directive(strategy: Profile) -> None:
    assert "LING-ACTOR-001" not in _rule_ids("The team should prioritize retention.", strategy)
    assert "LING-ACTOR-001" not in _rule_ids("The vendor should publish the limit.", strategy)


def test_a_named_entity_elsewhere_is_not_the_actor(strategy: Profile) -> None:
    """The directive's subject decides, not any organisation in the sentence."""
    assert "LING-ACTOR-001" in _rule_ids("The market should sell to Acme.", strategy)


def test_the_stricter_gate_is_opt_in() -> None:
    """architecture-review does not set the threshold, so it must not change."""
    architecture = load_profile("architecture-review")
    assert "require_responsible_actor" not in architecture.thresholds
    assert "LING-ACTOR-001" not in _rule_ids(
        "The market should prioritize retention.", architecture
    )


def test_a_benefit_without_a_mechanism_is_not_detected(strategy: Profile) -> None:
    """Documents a gap rather than a feature.

    The purpose_markers groups name a mechanism category, but the analyzer
    reads those groups only to report a sentence mixing more than two purposes.
    It never reports an absent one. This test fails the day that changes, which
    is the point.
    """
    assert _rule_ids("Customers save ten hours each week.", strategy) == set()


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
    assert _score(SPECIFIC, strategy) == 100.0


def test_analysis_is_deterministic_under_the_profile(strategy: Profile) -> None:
    first = analyze_text(HYPE, strategy)
    second = analyze_text(HYPE, strategy)
    assert first["analysis_sha256"] == second["analysis_sha256"]


def test_the_artifact_records_the_profile_it_used(strategy: Profile) -> None:
    reference = cast(dict[str, JsonValue], analyze_text(HYPE, strategy)["profile"])
    assert reference["name"] == "product-strategy"
    assert reference["version"] == "1.0.0"
    assert reference["digest"] == strategy.digest
