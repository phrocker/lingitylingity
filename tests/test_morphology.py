from __future__ import annotations

import pytest

from lingity import morphology
from lingity.morphology import (
    WordNetDataError,
    action_stem,
    canonical_action,
    canonical_action_info,
)


AGREEMENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("ratification", "ratify"),
    ("verification", "verify"),
    ("classification", "classify"),
    ("notification", "notify"),
    ("simplification", "simplify"),
    ("clarification", "clarify"),
    ("justification", "justify"),
    ("certification", "certify"),
    ("qualification", "qualify"),
    ("unification", "unify"),
    ("modification", "modify"),
    ("specification", "specify"),
    ("authorization", "authorize"),
    ("authorisation", "authorize"),
    ("organization", "organize"),
    ("organisation", "organize"),
    ("realization", "realize"),
    ("normalization", "normalize"),
    ("optimization", "optimize"),
    ("prioritization", "prioritize"),
    ("standardization", "standardize"),
    ("stabilization", "stabilize"),
    ("mobilization", "mobilize"),
    ("digitization", "digitize"),
    ("migration", "migrate"),
    ("deprecation", "deprecate"),
    ("communication", "communicate"),
    ("activation", "activate"),
    ("rotation", "rotate"),
    ("validation", "validate"),
    ("delegation", "delegate"),
    ("escalation", "escalate"),
    ("negotiation", "negotiate"),
    ("coordination", "coordinate"),
    ("allocation", "allocate"),
    ("creation", "create"),
    ("decision", "decide"),
    ("revision", "revise"),
    ("provision", "provide"),
    ("resolution", "resolve"),
    ("conclusion", "conclude"),
    ("inclusion", "include"),
    ("exclusion", "exclude"),
    ("collision", "collide"),
    ("reduction", "reduce"),
    ("production", "produce"),
    ("connection", "connect"),
    ("selection", "select"),
    ("detection", "detect"),
    ("completion", "complete"),
    ("adoption", "adopt"),
    ("assertion", "assert"),
    ("deployment", "deploy"),
    ("assessment", "assess"),
    ("development", "develop"),
    ("management", "manage"),
    ("agreement", "agree"),
    ("improvement", "improve"),
    ("replacement", "replace"),
    ("statement", "state"),
    ("requirement", "require"),
    ("acceptance", "accept"),
    ("resistance", "resist"),
    ("dependence", "depend"),
    ("reference", "refer"),
    ("emergence", "emerge"),
    ("occurrence", "occur"),
    ("preference", "prefer"),
    ("avoidance", "avoid"),
    ("appearance", "appear"),
    ("clearance", "clear"),
    ("performance", "perform"),
    ("guidance", "guide"),
    ("assistance", "assist"),
    ("closure", "close"),
    ("failure", "fail"),
    ("exposure", "expose"),
    ("departure", "depart"),
    ("seizure", "seize"),
    ("disclosure", "disclose"),
    ("composure", "compose"),
    ("erasure", "erase"),
    ("approval", "approve"),
    ("removal", "remove"),
    ("refusal", "refuse"),
    ("arrival", "arrive"),
    ("proposal", "propose"),
    ("disposal", "dispose"),
    ("reversal", "reverse"),
    ("renewal", "renew"),
    ("withdrawal", "withdraw"),
    ("portrayal", "portray"),
    ("retrieval", "retrieve"),
)

SEPARATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("approve", "reject"),
    ("deploy", "deprecate"),
    ("begin", "block"),
    ("verify", "revert"),
    ("close", "clone"),
    ("authorize", "audit"),
    ("accept", "deny"),
    ("migrate", "manage"),
    ("resolve", "reverse"),
    ("include", "exclude"),
    ("connect", "correct"),
    ("select", "detect"),
    ("create", "delete"),
    ("compose", "compare"),
    ("assess", "assert"),
    ("develop", "deploy"),
    ("organize", "optimize"),
    ("notify", "modify"),
    ("reduce", "refuse"),
    ("refer", "prefer"),
    ("appear", "approve"),
    ("state", "start"),
)

GOVERNANCE_SEPARATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("approve", "revoke"),
    ("ratify", "rescind"),
    ("commission", "decommission"),
    ("activate", "deactivate"),
    ("regulate", "deregulate"),
    ("close", "disclose"),
    ("defer", "expedite"),
    ("authorize", "prohibit"),
    ("permit", "forbid"),
    ("accept", "reject"),
    ("adopt", "abandon"),
    ("include", "exclude"),
    ("enable", "disable"),
    ("allow", "deny"),
    ("grant", "revoke"),
    ("start", "stop"),
    ("open", "close"),
    ("submit", "withdraw"),
    ("escalate", "deescalate"),
    ("enforce", "waive"),
    ("retain", "delete"),
    ("merge", "split"),
    ("provision", "deprovision"),
    ("connect", "disconnect"),
)

STABILITY_CASES: tuple[tuple[str, str], ...] = (
    ("", ""),
    ("   ", ""),
    (" Begin ", "begin"),
    ("block", "block"),
    ("address", "address"),
    ("resolve", "resolv"),
    ("resolution", "resolv"),
    ("connect", "connec"),
    ("connection", "connec"),
    ("close", "clos"),
    ("closure", "clos"),
    ("guide", "guid"),
    ("guidance", "guid"),
    ("make", "make"),
    ("use", "use"),
)

WORDNET_AGREEMENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("ratification", "ratify"),
    ("closure", "close"),
    ("authorization", "authorize"),
    ("deployment", "deploy"),
    ("remediation", "remediate"),
    ("decision", "decide"),
    ("verification", "verify"),
    ("resolution", "resolve"),
    ("approval", "approve"),
    ("recommendation", "recommend"),
    ("migration", "migrate"),
    ("assessment", "assess"),
    ("acceptance", "accept"),
    ("removal", "remove"),
    ("refusal", "refuse"),
    ("conclusion", "conclude"),
    ("provision", "provide"),
    ("revision", "revise"),
    ("classification", "classify"),
    ("enforcement", "enforce"),
    ("escalation", "escalate"),
    ("mitigation", "mitigate"),
    ("governance", "govern"),
    ("notification", "notify"),
    ("simplification", "simplify"),
    ("clarification", "clarify"),
    ("justification", "justify"),
    ("certification", "certify"),
    ("qualification", "qualify"),
    ("unification", "unify"),
    ("modification", "modify"),
    ("specification", "specify"),
    ("organization", "organize"),
    ("realization", "realize"),
    ("normalization", "normalize"),
    ("optimization", "optimize"),
    ("standardization", "standardize"),
    ("stabilization", "stabilize"),
    ("mobilization", "mobilize"),
    ("digitization", "digitize"),
    ("deprecation", "deprecate"),
    ("communication", "communicate"),
    ("activation", "activate"),
    ("rotation", "rotate"),
    ("validation", "validate"),
    ("delegation", "delegate"),
    ("negotiation", "negotiate"),
    ("coordination", "coordinate"),
    ("allocation", "allocate"),
    ("creation", "create"),
    ("inclusion", "include"),
    ("exclusion", "exclude"),
    ("collision", "collide"),
    ("reduction", "reduce"),
    ("production", "produce"),
    ("connection", "connect"),
    ("selection", "select"),
    ("detection", "detect"),
    ("completion", "complete"),
    ("adoption", "adopt"),
    ("assertion", "assert"),
    ("development", "develop"),
    ("management", "manage"),
    ("agreement", "agree"),
    ("improvement", "improve"),
    ("replacement", "replace"),
    ("statement", "state"),
    ("requirement", "require"),
    ("resistance", "resist"),
    ("reference", "refer"),
    ("emergence", "emerge"),
    ("avoidance", "avoid"),
    ("appearance", "appear"),
    ("clearance", "clear"),
    ("performance", "perform"),
    ("guidance", "guide"),
    ("assistance", "assist"),
    ("failure", "fail"),
    ("exposure", "expose"),
    ("departure", "depart"),
    ("seizure", "seize"),
    ("disclosure", "disclose"),
    ("composure", "compose"),
    ("erasure", "erase"),
    ("arrival", "arrive"),
    ("proposal", "propose"),
    ("disposal", "dispose"),
    ("reversal", "reverse"),
    ("renewal", "renew"),
    ("withdrawal", "withdraw"),
    ("portrayal", "portray"),
    ("retrieval", "retrieve"),
)

FALLBACK_AGREEMENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("cutover", "cutover"),
    ("shardification", "shardify"),
    ("bluegreenization", "bluegreenize"),
)

GERUND_ACTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("provisioning", "provision"),
    ("decommissioning", "decommission"),
    ("commissioning", "commission"),
    ("onboarding", "onboard"),
    ("offboarding", "offboard"),
    ("monitoring", "monitor"),
    ("reporting", "report"),
    ("logging", "log"),
    ("staging", "stage"),
    ("sequencing", "sequence"),
    ("sunsetting", "sunset"),
    ("throttling", "throttle"),
    ("auditing", "audit"),
    ("configuring", "configure"),
)

INVARIANT_ACTIONS: tuple[str, ...] = ("rollback", "cutover", "failover")


@pytest.mark.parametrize(("noun", "verb"), AGREEMENT_PAIRS)
def test_action_stem_agrees_for_nominalizations(noun: str, verb: str) -> None:
    assert action_stem(noun) == action_stem(verb)


@pytest.mark.parametrize(("left", "right"), SEPARATION_PAIRS)
def test_action_stem_keeps_different_actions_separate(left: str, right: str) -> None:
    assert action_stem(left) != action_stem(right)


@pytest.mark.parametrize(
    "lemma",
    tuple(
        sorted(
            {item for pair in AGREEMENT_PAIRS + SEPARATION_PAIRS for item in pair}
            | {lemma for lemma, _expected in STABILITY_CASES}
        )
    ),
)
def test_action_stem_is_idempotent(lemma: str) -> None:
    assert action_stem(action_stem(lemma)) == action_stem(lemma)


@pytest.mark.parametrize(("lemma", "expected"), STABILITY_CASES)
def test_action_stem_is_stable_for_base_forms(lemma: str, expected: str) -> None:
    assert action_stem(lemma) == expected


@pytest.mark.parametrize(("noun", "verb"), WORDNET_AGREEMENT_PAIRS)
def test_canonical_action_agrees_through_wordnet(noun: str, verb: str) -> None:
    noun_result = canonical_action_info(noun)
    verb_result = canonical_action_info(verb)

    assert noun_result.source == "wordnet"
    assert verb_result.source == "wordnet"
    assert noun_result.key == verb_result.key


def test_canonical_action_handles_multi_verb_wordnet_components() -> None:
    approval = canonical_action_info("approval")
    approve = canonical_action_info("approve")
    provision = canonical_action_info("provision")
    provide = canonical_action_info("provide")

    assert approval.wordnet_component == approve.wordnet_component
    assert approval.wordnet_component == ("approbate", "approve")
    assert approval.key == approve.key == "approbate"
    assert provision.wordnet_component == provide.wordnet_component
    assert provision.wordnet_component == ("provide", "provision")
    assert provision.key == provide.key == "provide"


@pytest.mark.parametrize(("left", "right"), SEPARATION_PAIRS)
def test_canonical_action_keeps_different_actions_separate(left: str, right: str) -> None:
    assert canonical_action(left) != canonical_action(right)


@pytest.mark.parametrize(("left", "right"), GOVERNANCE_SEPARATION_PAIRS)
def test_canonical_action_keeps_governance_antonyms_separate(
    left: str,
    right: str,
) -> None:
    assert canonical_action(left) != canonical_action(right)


@pytest.mark.parametrize(("noun", "verb"), FALLBACK_AGREEMENT_PAIRS)
def test_canonical_action_falls_back_to_stemmer_for_jargon(noun: str, verb: str) -> None:
    noun_result = canonical_action_info(noun)
    verb_result = canonical_action_info(verb)

    assert noun_result.source == "stemmer"
    assert verb_result.source == "stemmer"
    assert noun_result.key == verb_result.key
    assert noun_result.key == action_stem(noun)
    assert verb_result.key == action_stem(verb)


@pytest.mark.parametrize(("gerund", "base"), GERUND_ACTION_PAIRS)
def test_canonical_action_retries_wordnet_after_stripping_gerunds(
    gerund: str,
    base: str,
) -> None:
    assert canonical_action(gerund) == canonical_action(base)


def test_canonical_action_reaches_wordnet_after_gerund_stripping() -> None:
    provisioning = canonical_action_info("provisioning")

    assert provisioning.source == "wordnet"
    assert provisioning.key == canonical_action("provision")


def test_canonical_action_keeps_decommission_as_an_unstemmed_fallback() -> None:
    decommissioning = canonical_action_info("decommissioning")
    decommission = canonical_action_info("decommission")

    assert decommissioning.source == "stemmer"
    assert decommission.source == "stemmer"
    assert decommissioning.key == decommission.key == "decommission"
    assert canonical_action("commission") != decommission.key


@pytest.mark.parametrize("lemma", INVARIANT_ACTIONS)
def test_canonical_action_preserves_invariant_jargon(lemma: str) -> None:
    assert canonical_action_info(lemma).source == "stemmer"
    assert canonical_action(lemma) == lemma


@pytest.mark.parametrize(
    "lemma",
    tuple(
        sorted(
            {
                item
                for pair in (
                    WORDNET_AGREEMENT_PAIRS
                    + FALLBACK_AGREEMENT_PAIRS
                    + GERUND_ACTION_PAIRS
                    + GOVERNANCE_SEPARATION_PAIRS
                )
                for item in pair
            }
            | {item for pair in SEPARATION_PAIRS for item in pair}
            | set(INVARIANT_ACTIONS)
        )
    ),
)
def test_canonical_action_is_idempotent(lemma: str) -> None:
    assert canonical_action(canonical_action(lemma)) == canonical_action(lemma)


def test_canonical_action_raises_named_error_when_wordnet_data_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_action_info.cache_clear()

    def missing_wordnet() -> object:
        raise LookupError("missing wordnet corpus")

    monkeypatch.setattr(morphology, "_load_wordnet", missing_wordnet)

    with pytest.raises(WordNetDataError, match="WordNet corpus data is unavailable"):
        canonical_action("ratification")

    canonical_action_info.cache_clear()
