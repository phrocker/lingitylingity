"""The local-service profile reads a trade's county page.

`web-copy` already exists and is the wrong instrument here, which is worth
being precise about rather than assuming. Given a page about HVAC repair in
Howard County built entirely from stock phrases — family owned and operated,
no job too big or too small, licensed and insured, free estimates, we pride
ourselves on exceptional customer service — it scored 87.27 and called it
`clear`, with `lexical_clarity` at a clean 100. Its jargon lists are B2B: they
know `growth hack` and `hockey stick` and `viral loop`, and nothing a plumber
would ever write.

That matters more here than a bad score usually does. A business running
twenty-two county pages off one template is structurally a doorway farm, and
the judgement is made on the writing: two pages side by side, ninety percent
identical, is the test. Stock phrases are what make them identical, so an
instrument that cannot see them cannot see the risk.

The answer is not a `sales` or `marketing` profile. Generic marketing prose is
the same mistake as a generic county page — the value is in being specific
about the genre, and this genre fails in its own way.
"""

from __future__ import annotations

from typing import cast

import pytest

from lingity.analyzer import analyze_text
from lingity.models import JsonValue
from lingity.profiles import Profile, load_profile

# What a model writes when asked for a page about a trade in a county. Written
# at the length a real page is: the score is penalty-based, so a fragment scores
# better than the thing it is a fragment of, and asserting a band on four
# sentences tests the calibration rather than the profile.
STOCK = (
    "Looking for reliable HVAC service in Howard County? Our family owned and "
    "operated team has been serving Howard County homeowners since 1998. No job "
    "is too big or too small.\n\n"
    "We are licensed and insured, and we offer free estimates on all work. Our "
    "technicians are experts in heating and cooling, and we pride ourselves on "
    "providing exceptional customer service to every single customer we serve.\n\n"
    "Whether you need AC repair, furnace installation, or routine maintenance, "
    "we have you covered. We offer 24/7 emergency service because we know that "
    "your comfort cannot wait. Call us today for a free quote."
)

# The same page written by somebody who has worked in the county.
SPECIFIC = (
    "Howard County requires a mechanical permit for a furnace or condenser "
    "replacement. The Department of Inspections, Licenses and Permits charges "
    "$95 for the permit and inspects the work before it can be closed out. We "
    "pull the permit; you do not.\n\n"
    "Most of Columbia went up between 1967 and 1975, so a lot of the village "
    "homes still run the original ductwork in an unconditioned crawl space. "
    "That is why a system here can be sized correctly and still lose a third of "
    "its capacity before the air reaches the room.\n\n"
    "A furnace replacement in this county runs $4,200 to $7,800 depending on "
    "the unit and the ductwork. We quote the ductwork separately because it is "
    "where the surprise usually is.\n\n"
    "Call 410-555-0143. We answer the phone between 7am and 7pm."
)


@pytest.fixture
def local_service() -> Profile:
    return load_profile("local-service")


@pytest.fixture
def web_copy() -> Profile:
    return load_profile("web-copy")


def _rule_ids(text: str, profile: Profile) -> set[str]:
    findings = cast(list[dict[str, JsonValue]], analyze_text(text, profile)["findings"])
    return {cast(str, finding["rule_id"]) for finding in findings}


def _score(text: str, profile: Profile) -> float:
    score = cast(dict[str, JsonValue], analyze_text(text, profile)["score"])
    return cast(float, score["value"])


def _band(text: str, profile: Profile) -> str:
    score = cast(dict[str, JsonValue], analyze_text(text, profile)["score"])
    return cast(str, score["band"])


def test_web_copy_cannot_see_this_genre(web_copy: Profile) -> None:
    """The measurement that justifies a second profile.

    If `web-copy` caught this, `local-service` would be weights nobody needs.
    """
    assert "LING-JARGON-001" not in _rule_ids(STOCK, web_copy)


def test_stock_phrases_are_reported_as_jargon(local_service: Profile) -> None:
    assert "LING-JARGON-001" in _rule_ids(STOCK, local_service)


def test_a_page_of_stock_phrases_needs_revision(local_service: Profile) -> None:
    """Not "usable but improvable". A page built from phrases that would serve
    any county and any trade is the thing this profile exists to reject."""
    assert _band(STOCK, local_service) == "revision_required"


def test_specificity_scores_higher_than_stock(local_service: Profile) -> None:
    assert _score(SPECIFIC, local_service) > _score(STOCK, local_service)


def test_naming_a_permit_a_price_and_a_date_is_not_penalised(
    local_service: Profile,
) -> None:
    """The remedy for every phrase in the jargon list is a number beside it, so
    the profile must not then punish the number."""
    assert "LING-JARGON-001" not in _rule_ids(SPECIFIC, local_service)


def test_the_subject_matter_is_not_an_undefined_acronym(local_service: Profile) -> None:
    """`howardcountyhvac.com` cannot write a page without saying HVAC, and a
    profile that reports its own subject as jargon is unusable on the one
    domain it was built for."""
    text = (
        "A SEER 14 condenser and an 80 AFUE furnace suit most of these homes. "
        "We check static pressure in inches of water and CFM at the register."
    )
    assert "LING-ACRONYM-001" not in _rule_ids(text, local_service)


def test_a_trade_is_an_actor(local_service: Profile) -> None:
    """Agency is weighted heavily here, and it can only be measured against a
    vocabulary that contains the people who do the work."""
    profile = load_profile("local-service").data
    actors = profile["rules"]["actor_terms"]
    for term in ("homeowner", "plumber", "electrician", "technician"):
        assert term in actors


def test_trust_claims_are_flagged_for_substantiation_not_deletion(
    local_service: Profile,
) -> None:
    """"Licensed and insured" is true, legally meaningful and worth saying. It
    is flagged because it is worth saying *with the licence number* — the same
    move that stops twenty-two pages being one page."""
    profile = load_profile("local-service").data
    claims = profile["rules"]["jargon"]["unverifiable trust claim"]
    assert "license and insure" in claims
    assert "year of experience" in claims


def test_no_phrase_is_penalised_twice(local_service: Profile) -> None:
    """`family owned and operated` once matched both itself and `family owned`,
    so a page was penalised twice for saying a thing once."""
    profile = load_profile("local-service").data
    phrases = profile["rules"]["jargon"]["local service cliche"]
    for phrase in phrases:
        containing = [other for other in phrases if other != phrase and phrase in other]
        assert not containing, f"{phrase!r} is contained by {containing}"


# One surface sentence per phrase this profile adds. The reviewer on #22 asked
# for it and was right: five of the fifty-one lemmas shipped in a form that
# never fired — `trust name`, `full license`, `certified technician`,
# `years of experience`, `no hidden fee` — and a rule that never fires is
# indistinguishable from a rule nobody wrote. Hand-picked examples cannot catch
# that; only enumerating every phrase can.
SURFACES: dict[str, str] = {
    '100 percent satisfaction': 'We offer 100 percent satisfaction.',
    '24 7 emergency service': 'We offer 24 7 emergency service.',
    'a plus rating': 'We hold an A plus rating.',
    'attention to detail': 'Our attention to detail sets us apart.',
    'available 24 7': 'Our team is available 24 7.',
    'background check technician': 'Every background checked technician arrives in uniform.',
    'big or small': 'Every repair, big or small, gets the same crew.',
    'bond and insure': 'We are bonded and insured.',
    'certify technician': 'A certified technician performs the work.',
    'competitive rate': 'Our competitive rates beat the area.',
    'decade of experience': 'We bring decades of experience.',
    'exceptional customer service': 'We deliver exceptional customer service.',
    'factory train': 'Our factory trained technicians handle every brand.',
    'family own': 'We are a family owned business.',
    'fast and friendly': 'Our crews are fast and friendly.',
    'free estimate': 'We offer free estimates.',
    'free quote': 'Call today for a free quote.',
    'fully insure': 'Every technician is fully insured.',
    'fully license': 'Our company is fully licensed.',
    'hassle free': 'Booking is hassle free.',
    'here for you': 'We are here for you.',
    'hidden fee': 'We never charge a hidden fee.',
    'honest and reliable': 'We are honest and reliable.',
    'license and insure': 'We are licensed and insured.',
    'local expert': 'Our local experts arrive the same day.',
    'money back guarantee': 'We offer a money back guarantee.',
    'no job be too big': 'No job is too big for our crew.',
    'no job be too small': 'No job is too small for us.',
    'no obligation quote': 'Ask for a no obligation quote.',
    'one stop shop': 'We are your one stop shop for heating.',
    'peace of mind': 'A new system buys you peace of mind.',
    'pride ourselves': 'We pride ourselves on our service.',
    'prompt and professional': 'Our team is prompt and professional.',
    'quality workmanship': 'We stand behind our quality workmanship.',
    'same day service guarantee': 'Ask about our same day service guarantee.',
    'satisfaction be our top priority': 'Customer satisfaction is our top priority.',
    'satisfaction guarantee': 'Every visit carries a satisfaction guarantee.',
    'serve homeowner since': 'We have served homeowners since 1998.',
    'serve the area since': 'We have served the area since 1998.',
    'thousand of satisfied customer': 'We have thousands of satisfied customers.',
    'top notch': 'The workmanship is top notch.',
    'transparent pricing': 'Our transparent pricing has no surprises.',
    'trust by homeowner': 'We are trusted by homeowners across the county.',
    'trusted name': 'We are a trusted name in the county.',
    'unbeatable price': 'We offer unbeatable prices.',
    'upfront pricing': 'We believe in upfront pricing.',
    'we be here to help': 'We are here to help.',
    'we have you cover': 'Whatever breaks, we have you covered.',
    'we treat your home like our own': 'We treat your home like our own.',
    'year of experience': 'We bring thirty years of experience.',
    'your satisfaction be our': 'Your satisfaction is our promise.',
}


def _added_groups(profile: Profile) -> dict[str, list[str]]:
    """The groups this profile adds over `web-copy`, which it inherits from."""
    import json as _json
    from pathlib import Path as _Path

    from lingity.profiles import PROFILE_DIR

    inherited = _json.loads((_Path(PROFILE_DIR) / "web-copy.v1.json").read_text())
    base = set(inherited["rules"]["jargon"])
    return {
        group: phrases
        for group, phrases in profile.data["rules"]["jargon"].items()
        if group not in base
    }


@pytest.mark.parametrize("phrase", sorted(SURFACES))
def test_every_added_phrase_actually_fires(phrase: str, local_service: Profile) -> None:
    """A malformed lemma is silent, not loud. Five were."""
    assert "LING-JARGON-001" in _rule_ids(SURFACES[phrase], local_service), (
        f"{phrase!r} never fires on {SURFACES[phrase]!r}"
    )


def test_the_fixture_covers_every_phrase_the_profile_adds(local_service: Profile) -> None:
    """The half that keeps the test above honest: a phrase added to the profile
    and not to the fixture is a rule nobody has ever seen fire."""
    stored = {
        phrase for phrases in _added_groups(local_service).values() for phrase in phrases
    }
    assert stored == set(SURFACES)
