"""The resume-review profile reads accomplishment bullets, not prose paragraphs.

Every claim this file makes is a claim about what the analyzer *does*. A test
that only inspects a profile's configuration proved nothing on the previous
profile: it passed while the advertised behaviour was missing from the analyzer
entirely. Each assertion below therefore fails if the analyzer stops behaving as
documented, not merely if a JSON key moves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from lingity.analyzer import analyze_text
from lingity.models import JsonValue
from lingity.profiles import Profile, load_profile

# Generic prose. No employer, product, person, or real history appears here: the
# profile scores clarity, it does not judge a candidate or detect exaggeration.
WEAK = """## Summary

A results-driven self-starter and team player who is passionate about technology and wears many hats.

## Experience

- Responsible for the migration of the reporting platform.
- Was tasked with the implementation of the deployment pipeline.
- Involved in various projects that significantly improved performance.
- Duties included the coordination of releases across several teams.
- Helped to deliver numerous enhancements in a timely manner.
- Worked on the modernization of the integration layer in order to reduce cost.
"""

STRONG = """## Summary

I write payment software and lead the engineers who run it.

## Experience

- Migrated the reporting platform to a managed database in four months.
- Built the pipeline that ships thirty releases a week.
- Cut checkout latency from 1.2 seconds to 300 milliseconds.
- Rewrote the runbook and trained nine engineers to use it.
- Replaced the nightly batch job with a stream that updates every minute.
- Removed the integration layer and saved forty thousand dollars a year.
"""

# The rule each phrase family raises. A phrase filed under the wrong family
# would otherwise still look covered.
PHRASE_RULE_IDS = {
    "jargon": "LING-JARGON-001",
    "weak_verbs": "LING-WEAK-VERB-001",
    "hidden_agency": "LING-AGENCY-001",
    "bureaucratic_phrases": "LING-BUREAUCRACY-001",
    "filler_phrases": "LING-FILLER-001",
    "indirect_predicates": "LING-INDIRECT-PREDICATE-001",
}


@pytest.fixture
def resume() -> Profile:
    return load_profile("resume-review")


def _rule_ids(text: str, profile: Profile) -> set[str]:
    findings = cast(list[dict[str, JsonValue]], analyze_text(text, profile)["findings"])
    return {cast(str, finding["rule_id"]) for finding in findings}


def _score(text: str, profile: Profile) -> float:
    score = cast(dict[str, JsonValue], analyze_text(text, profile)["score"])
    return cast(float, score["value"])


def _band(text: str, profile: Profile) -> str:
    score = cast(dict[str, JsonValue], analyze_text(text, profile)["score"])
    return cast(str, score["band"])


def test_the_profile_is_installed_and_discoverable(resume: Profile) -> None:
    assert resume.name == "resume-review"
    assert resume.version == "1.0.0"
    assert sum(resume.weights.values()) == 100


def test_bands_cover_the_whole_range(resume: Profile) -> None:
    bands = cast(list[dict[str, float]], resume.data["bands"])
    assert bands[0]["minimum"] == 0
    assert bands[-1]["maximum"] == 100
    # Non-overlap is not coverage. Scores round to hundredths, so a gap wider
    # than 0.01 leaves scores no band can name and _band_for raises on them.
    for earlier, later in zip(bands, bands[1:]):
        assert round(later["minimum"] - earlier["maximum"], 2) == 0.01


def test_it_supplies_every_rule_the_analyzer_reads(resume: Profile) -> None:
    """A new profile must answer every rule lookup the shipped profile answers."""
    reference = load_profile("architecture-review")
    assert set(reference.rules) <= set(resume.rules)
    assert set(reference.thresholds) <= set(resume.thresholds)


def test_it_declares_no_protected_phrases(resume: Profile) -> None:
    """The meaning gate reads claim signatures from the parse, not phrase lists."""
    assert resume.rules["protected_concepts"] == []


# --------------------------------------------------------------------------
# The implied first person
# --------------------------------------------------------------------------


def test_a_subjectless_bullet_is_read_as_the_author(resume: Profile) -> None:
    """A resume bullet drops its subject by convention, not by evasion."""
    assert "LING-ACTOR-001" not in _rule_ids(
        "Cut checkout latency from 1.2 seconds to 300 milliseconds.", resume
    )
    assert "LING-ACTOR-001" not in _rule_ids(
        "Rewrote the runbook and trained nine engineers to use it.", resume
    )


def test_the_prose_profiles_still_report_the_same_bullet() -> None:
    """Without the threshold the rule taxes exactly the bullets that read best.

    Both sentences are active, specific, and quantified. They are reported only
    because the parser reads a bullet's leading verb as an imperative, which is
    why the threshold has to exist before the profile can be useful.
    """
    for name in ("architecture-review", "product-strategy"):
        profile = load_profile(name)
        assert "LING-ACTOR-001" in _rule_ids(
            "Cut checkout latency from 1.2 seconds to 300 milliseconds.", profile
        )
        assert "LING-ACTOR-001" in _rule_ids(
            "Rewrote the runbook and trained nine engineers to use it.", profile
        )


def test_the_threshold_does_not_excuse_the_passive(resume: Profile) -> None:
    """The distinction the whole profile depends on.

    A resume written in the passive hides the work, which is the defect this
    profile exists to find. Reading an absent subject as the author must
    therefore suppress the missing-subject finding and nothing else: a
    subjectless *passive* keeps both its agency and its voice findings.
    """
    subjectless_passive = "Must be completed before the release."
    reported = _rule_ids(subjectless_passive, resume)
    assert "LING-ACTOR-001" not in reported
    assert "LING-AGENCY-001" in reported
    assert "LING-PASSIVE-001" in reported

    # The same sentence reports the same voice findings under a profile that
    # does not set the threshold, so the threshold changed only the actor rule.
    architecture = _rule_ids(subjectless_passive, load_profile("architecture-review"))
    assert {"LING-AGENCY-001", "LING-PASSIVE-001"} <= architecture


def test_the_hidden_accomplishment_is_still_reported(resume: Profile) -> None:
    """"Was responsible for the migration" names no action and no actor."""
    assert "LING-AGENCY-001" in _rule_ids("Was responsible for the migration.", resume)
    assert "LING-AGENCY-001" in _rule_ids("Responsible for the migration.", resume)
    assert "LING-AGENCY-001" in _rule_ids("Was tasked with the migration.", resume)
    assert "LING-AGENCY-001" in _rule_ids("Duties included the release.", resume)


def test_the_prose_profiles_cannot_see_the_hidden_accomplishment() -> None:
    """The gap that justifies a third profile rather than a threshold alone."""
    architecture = load_profile("architecture-review")
    assert _rule_ids("Was responsible for the migration.", architecture) == set()
    assert _rule_ids("Responsible for the migration.", architecture) == set()


def test_the_threshold_is_opt_in() -> None:
    """The shipped profiles do not set it, so their behaviour must not change."""
    for name in ("architecture-review", "product-strategy"):
        profile = load_profile(name)
        assert "allow_implied_first_person" not in profile.thresholds


def test_an_overt_subject_is_never_read_as_the_author(resume: Profile) -> None:
    """The reading reaches an absent subject only, so a named one still decides."""
    assert "LING-ACTOR-001" in _rule_ids("The platform should own the runbook.", resume)
    assert "LING-ACTOR-001" not in _rule_ids("The team should own the runbook.", resume)


def test_an_impersonal_subject_is_not_an_absent_one(resume: Profile) -> None:
    """"It" is a subject the author wrote, not a subject the genre omits.

    The reading tests for a subject the parse does not carry, so an impersonal
    one still fails the actor rule instead of standing in for the author.
    """
    assert "LING-ACTOR-001" in _rule_ids("It should improve the runbook.", resume)


def test_the_recognised_actors_are_beneficiaries(resume: Profile) -> None:
    """The author is implied, so the terms that matter are who a bullet serves."""
    actors = cast(list[str], resume.rules["actor_terms"])
    for present in ("team", "customer", "user", "client", "stakeholder"):
        assert present in actors


# --------------------------------------------------------------------------
# Phrase rules: every stored lemma is demonstrated, never asserted
# --------------------------------------------------------------------------


def _phrase_fixture() -> dict[str, dict[str, Any]]:
    path = Path(__file__).parent / "fixtures" / "resume-review-phrases.json"
    return cast(dict[str, dict[str, Any]], json.loads(path.read_text(encoding="utf-8")))


def _fixture_cases() -> list[tuple[str, str, str, str, tuple[str, ...]]]:
    return [
        (surface, entry["rule"], entry["classification"], variant["lemma"], tuple(variant["examples"]))
        for surface, entry in sorted(_phrase_fixture().items())
        for variant in entry["variants"]
    ]


@pytest.mark.parametrize(
    "surface,rule,classification,lemma,examples",
    _fixture_cases(),
    ids=lambda value: str(value)[:40],
)
def test_every_phrase_variant_fires_on_every_recorded_example(
    surface: str, rule: str, classification: str, lemma: str, examples: tuple[str, ...]
) -> None:
    """A stored phrase that never matches is a rule nobody enforces.

    The profile stores lemma sequences, and a lemma is not the surface word:
    "detail-oriented" parses as "detail orient" and "duties included" as "duty
    include". A lemma also depends on the grammatical role the phrase takes, so
    "thought leader" yields "thought leader" in one position and "think leader"
    in another. Every variant here was derived by parsing a carrier sentence and
    reading the lemmas off it, never by eye, and every one is pinned to sentences
    that provably raise the finding.
    """
    resume = load_profile("resume-review")
    assert examples
    for example in examples:
        findings = cast(
            list[dict[str, JsonValue]], analyze_text(example, resume)["findings"]
        )
        observed = [
            cast(dict[str, JsonValue], finding["observed_value"])
            for finding in findings
            if finding["rule_id"] == PHRASE_RULE_IDS[rule]
        ]
        # Assert the pair, not just the rule id: moving a phrase into the wrong
        # user-visible category would otherwise leave this test green.
        matched = [
            entry
            for entry in observed
            if str(entry.get("phrase", "")).lower() == surface.lower()
            and entry.get("classification") == classification
        ]
        assert matched, (example, lemma, observed)


def test_the_fixture_covers_every_phrase_in_the_profile(resume: Profile) -> None:
    """A phrase added without a demonstration fails here, and so does a "fix".

    A reader who corrects a stored lemma back to its surface spelling silences
    the rule without breaking anything else. This is what catches that.
    """
    for rule in PHRASE_RULE_IDS:
        stored = {
            phrase
            for phrases in cast(dict[str, list[str]], resume.rules[rule]).values()
            for phrase in phrases
        }
        covered = {lemma for _s, family, _c, lemma, _e in _fixture_cases() if family == rule}
        assert stored == covered, rule


def test_a_lemma_differs_from_its_surface_form() -> None:
    """Guards the reason the fixture exists, so nobody "corrects" the lemmas."""
    lemmas = {
        surface: {lemma for _s, _f, _c, lemma, _e in _fixture_cases() if _s == surface}
        for surface in (
            "detail-oriented",
            "duties included",
            "thought leader",
            "wear many hats",
            "working knowledge of",
        )
    }
    assert lemmas["detail-oriented"] == {"detail orient"}
    assert lemmas["duties included"] == {"duty include"}
    assert lemmas["wear many hats"] == {"wear many hat"}
    assert lemmas["working knowledge of"] == {"work knowledge of"}
    # Role-dependent: pinning either one alone ships a rule silent in the other.
    assert lemmas["thought leader"] == {"think leader", "thought leader"}


def test_a_role_dependent_phrase_fires_in_both_roles(resume: Profile) -> None:
    """The reported failure: one carrier pins a lemma real prose never produces."""
    assert "LING-JARGON-001" in _rule_ids("Our thought leader impresses readers.", resume)
    assert "LING-JARGON-001" in _rule_ids("The writer is thought leader.", resume)
    assert "LING-JARGON-001" in _rule_ids("Our results driven approach shipped it.", resume)
    assert "LING-JARGON-001" in _rule_ids("The writer is results-driven.", resume)


# --------------------------------------------------------------------------
# The profile earns its existence
# --------------------------------------------------------------------------


def test_it_catches_a_resume_the_prose_profiles_certify() -> None:
    """Both shipped profiles call this bullet set clear. This one does not."""
    architecture = load_profile("architecture-review")
    strategy = load_profile("product-strategy")
    resume = load_profile("resume-review")
    assert _band(WEAK, architecture) == "clear"
    assert _band(WEAK, strategy) == "clear"
    assert _band(WEAK, resume) == "revision_required"
    assert _score(WEAK, resume) < _score(WEAK, strategy) < _score(WEAK, architecture)


def test_specific_active_prose_survives_the_profile(resume: Profile) -> None:
    """Naming the act, a number, and a result must not be penalised."""
    assert _score(STRONG, resume) == 100.0


def test_the_prose_profiles_penalise_the_stronger_resume() -> None:
    """The inversion the threshold corrects, measured on whole documents."""
    architecture = load_profile("architecture-review")
    resume = load_profile("resume-review")
    assert _score(STRONG, architecture) < _score(STRONG, resume)


def test_the_weak_resume_reports_the_defects_the_genre_has(resume: Profile) -> None:
    """Name them, so a rule that stops firing fails here rather than silently."""
    reported = _rule_ids(WEAK, resume)
    assert {
        "LING-AGENCY-001",
        "LING-BUREAUCRACY-001",
        "LING-FILLER-001",
        "LING-JARGON-001",
        "LING-NOMINALIZATION-001",
        "LING-PASSIVE-001",
        "LING-WEAK-VERB-001",
    } <= reported


def test_a_single_hedge_is_not_reported(resume: Profile) -> None:
    """Documents a gap rather than a feature.

    The profile sets `max_qualifiers_per_sentence` to 0, but LING-QUALIFIER-001
    needs at least two qualifier tokens in a sentence before it looks at the
    threshold at all. One unquantified hedge in a bullet is therefore not
    reported by that rule, and the profile instead routes the hollow
    intensifiers of this genre through LING-FILLER-001. This test fails the day
    that changes, which is the point.
    """
    assert "LING-QUALIFIER-001" not in _rule_ids(
        "Rewrote the runbook, which was largely successful.", resume
    )


SIX_SHORT_OPENERS = """- Led the reporting platform team.
- Led the payment platform team.
- Led the analytics platform team.
- Led the identity platform team.
- Led the messaging platform team.
- Led the search platform team.
"""

SIX_LONG_OPENERS = """- Coordinated the reporting rollout.
- Coordinated the payment rollout.
- Coordinated the analytics rollout.
- Managed the identity rollout.
- Managed the messaging rollout.
- Managed the search rollout.
"""


def test_a_short_repeated_opener_is_not_reported(resume: Profile) -> None:
    """Documents a gap rather than a feature.

    Six bullets opening with the same verb look like the redundancy rule's
    territory, but LING-REDUNDANCY-001 counts only content lemmas of at least
    seven characters, so "lead" never reaches the threshold however often it is
    repeated. What the rule reports on this text is the repeated noun.
    """
    findings = cast(
        list[dict[str, JsonValue]], analyze_text(SIX_SHORT_OPENERS, resume)["findings"]
    )
    repeated = {
        cast(str, cast(dict[str, JsonValue], finding["observed_value"])["term"])
        for finding in findings
        if finding["rule_id"] == "LING-REDUNDANCY-001"
    }
    assert repeated == {"platform"}


def test_a_long_repeated_opener_is_reported(resume: Profile) -> None:
    """The other side of the same boundary, so the gap above is located exactly."""
    findings = cast(
        list[dict[str, JsonValue]], analyze_text(SIX_LONG_OPENERS, resume)["findings"]
    )
    repeated = {
        cast(str, cast(dict[str, JsonValue], finding["observed_value"])["term"])
        for finding in findings
        if finding["rule_id"] == "LING-REDUNDANCY-001"
    }
    assert "coordinate" in repeated
    assert "manage" not in repeated


def test_the_hollow_intensifier_is_reported_as_filler(resume: Profile) -> None:
    """What the profile does instead, so the gap above is not the whole story."""
    reported = _rule_ids("Improved performance significantly.", resume)
    assert "LING-FILLER-001" in reported


def test_analysis_is_deterministic_under_the_profile(resume: Profile) -> None:
    first = analyze_text(WEAK, resume)
    second = analyze_text(WEAK, resume)
    assert first["analysis_sha256"] == second["analysis_sha256"]


def test_the_artifact_records_the_profile_it_used(resume: Profile) -> None:
    reference = cast(dict[str, JsonValue], analyze_text(WEAK, resume)["profile"])
    assert reference["name"] == "resume-review"
    assert reference["version"] == "1.0.0"
    assert reference["digest"] == resume.digest
