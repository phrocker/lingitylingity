from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
import pytest

from lingity.analyzer import DESIGN_SIGNAL_RULES, DESIGN_TABLE_RULES, RULE_DIMENSIONS
from lingity.analyzer import analyze_text
from lingity.models import JsonValue
from lingity.profiles import load_profile

NEW_RULE_IDS = frozenset(DESIGN_SIGNAL_RULES.values())
EXPECTED_DIMENSIONS = {
    "sentence_load",
    "morphology",
    "agency",
    "lexical_clarity",
    "structure",
    "redundancy",
}


def _findings(text: str) -> list[dict[str, JsonValue]]:
    return cast(list[dict[str, JsonValue]], analyze_text(text)["findings"])


def _rule_ids(text: str) -> set[str]:
    return {cast(str, finding["rule_id"]) for finding in _findings(text)}


def _sentences(text: str) -> list[dict[str, JsonValue]]:
    return cast(list[dict[str, JsonValue]], analyze_text(text)["sentences"])


def _noun_stack_texts(text: str) -> set[str]:
    return {
        cast(str, cast(dict[str, JsonValue], finding["observed_value"])["text"])
        for finding in _findings(text)
        if finding["rule_id"] == "LING-NOUN-STACK-001"
    }


def test_rewrite_scores_better_and_reduces_findings(
    recommendation_fixture: dict[str, str],
) -> None:
    original = analyze_text(recommendation_fixture["original"])
    rewrite = analyze_text(recommendation_fixture["rewrite"])
    original_score = cast(dict[str, JsonValue], original["score"])["value"]
    rewrite_score = cast(dict[str, JsonValue], rewrite["score"])["value"]
    assert isinstance(original_score, float)
    assert isinstance(rewrite_score, float)
    assert rewrite_score > original_score
    # No absolute band is pinned here. A pin records where the current weights
    # happen to land rather than measuring the rewrite, and the pin this test
    # used to carry held only because three fabricated noun-stack findings
    # depressed the original's score. A rewrite that carries every governed
    # term forward cannot shed the findings those terms cause, so assert the
    # direction and the margin instead.
    bands = ["unusable", "revision_required", "usable_but_improvable", "clear"]
    original_band = cast(str, cast(dict[str, JsonValue], original["score"])["band"])
    rewrite_band = cast(str, cast(dict[str, JsonValue], rewrite["score"])["band"])
    assert bands.index(rewrite_band) >= bands.index(original_band)
    assert rewrite_score - original_score >= 10.0
    assert len(cast(list[JsonValue], rewrite["findings"])) < len(
        cast(list[JsonValue], original["findings"])
    )


def test_required_rule_families_are_detected(
    recommendation_fixture: dict[str, str],
) -> None:
    analysis = analyze_text(recommendation_fixture["original"])
    findings = cast(list[dict[str, JsonValue]], analysis["findings"])
    rule_ids = {cast(str, finding["rule_id"]) for finding in findings}
    assert {
        "LING-SENTENCE-001",
        "LING-ACTION-001",
        "LING-NOMINALIZATION-001",
        "LING-NOUN-STACK-001",
        "LING-AGENCY-001",
        "LING-WEAK-VERB-001",
        "LING-JARGON-001",
        "LING-BUREAUCRACY-001",
    } <= rule_ids


def test_every_finding_is_fully_attributed(
    recommendation_fixture: dict[str, str],
) -> None:
    analysis = analyze_text(recommendation_fixture["original"])
    findings = cast(list[dict[str, JsonValue]], analysis["findings"])
    required = {
        "rule_id", "severity", "location", "observed_value", "threshold", "remediation"
    }
    assert findings
    assert all(required <= finding.keys() for finding in findings)


def test_score_arithmetic_is_attributed(
    recommendation_fixture: dict[str, str],
) -> None:
    analysis = analyze_text(recommendation_fixture["original"])
    score = cast(dict[str, JsonValue], analysis["score"])
    components = cast(list[dict[str, JsonValue]], score["components"])
    contribution = round(
        sum(cast(float, component["weighted_contribution"]) for component in components),
        2,
    )
    assert contribution == score["value"]
    assert sum(cast(int, component["weight"]) for component in components) == 100


def test_analysis_publishes_linguistic_model_fingerprint() -> None:
    model = cast(dict[str, JsonValue], analyze_text("The team ships the change.")["linguistic_model"])
    assert set(model) == {"name", "version", "runtime", "digest"}
    assert all(isinstance(model[field], str) and model[field] for field in model)


def test_passive_voice_is_located() -> None:
    text = "The architecture was approved without a named owner."
    analysis = analyze_text(text)
    findings = cast(list[dict[str, JsonValue]], analysis["findings"])
    passive = next(
        finding for finding in findings if finding["rule_id"] == "LING-PASSIVE-001"
    )
    location = cast(dict[str, JsonValue], passive["location"])
    assert text[cast(int, location["start"]):cast(int, location["end"])] == "was approved"


def test_general_architecture_review_corpus_detects_both_directions() -> None:
    path = Path(__file__).parent / "fixtures" / "architecture-review-corpus.json"
    corpus = cast(dict[str, list[dict[str, JsonValue]]], json.loads(path.read_text(encoding="utf-8")))
    for item in corpus["positive"]:
        text = cast(str, item["text"])
        required = set(cast(list[str], item["required_rule_ids"]))
        assert required <= _rule_ids(text), cast(str, item["name"])
    for item in corpus["negative"]:
        text = cast(str, item["text"])
        forbidden = set(cast(list[str], item["forbidden_rule_ids"]))
        assert _rule_ids(text).isdisjoint(forbidden), cast(str, item["name"])


def test_new_rules_have_positive_and_negative_corpus_coverage() -> None:
    path = Path(__file__).parent / "fixtures" / "architecture-review-corpus.json"
    corpus = cast(dict[str, list[dict[str, JsonValue]]], json.loads(path.read_text(encoding="utf-8")))
    positive_hits: set[str] = set()
    positive_passes = 0
    for item in corpus["positive"]:
        text = cast(str, item["text"])
        required = set(cast(list[str], item["required_rule_ids"]))
        matched = required <= _rule_ids(text)
        positive_passes += int(matched)
        positive_hits.update(required & NEW_RULE_IDS)
    negative_hits: set[str] = set()
    negative_passes = 0
    for item in corpus["negative"]:
        text = cast(str, item["text"])
        forbidden = set(cast(list[str], item["forbidden_rule_ids"]))
        matched = _rule_ids(text).isdisjoint(forbidden)
        negative_passes += int(matched)
        negative_hits.update(forbidden & NEW_RULE_IDS)

    assert positive_hits == NEW_RULE_IDS
    assert negative_hits == NEW_RULE_IDS
    assert positive_hits <= set(RULE_DIMENSIONS)
    assert positive_passes == len(corpus["positive"])
    assert negative_passes == len(corpus["negative"])


def test_all_rule_ids_are_reachable_from_labelled_corpus() -> None:
    path = Path(__file__).parent / "fixtures" / "architecture-review-corpus.json"
    corpus = cast(dict[str, list[dict[str, JsonValue]]], json.loads(path.read_text(encoding="utf-8")))
    required_rule_ids = {
        rule_id
        for item in corpus["positive"]
        for rule_id in cast(list[str], item["required_rule_ids"])
    }
    assert required_rule_ids == set(RULE_DIMENSIONS)


def test_design_signal_mapping_and_dimension_coverage() -> None:
    assert set(DESIGN_SIGNAL_RULES) == {
        "sentence_load.punctuation_depth",
        "noun_stacking.consecutive_noun_modifiers",
        "noun_stacking.compound_depth",
        "agency.explicit_actor_action_pairs",
        "voice.indirect_predicates",
        "lexical_clarity.uncommon_compounds",
        "lexical_clarity.abbreviation_density",
        "structure.list_suitability",
        "structure.mixed_purpose_sentences",
        "redundancy.filler_phrases",
        "redundancy.duplicated_recommendations",
        "redundancy.repeated_qualifiers",
    }
    assert set(RULE_DIMENSIONS.values()) == EXPECTED_DIMENSIONS
    assert NEW_RULE_IDS <= set(RULE_DIMENSIONS)
    assert set(DESIGN_TABLE_RULES) == {
        "sentence_load",
        "morphology",
        "noun_stacking",
        "agency",
        "voice",
        "lexical_clarity",
        "structure",
        "redundancy",
    }
    mapped_rule_ids = {
        rule_id
        for rule_ids in DESIGN_TABLE_RULES.values()
        for rule_id in rule_ids
    }
    assert mapped_rule_ids <= set(RULE_DIMENSIONS)
    assert {RULE_DIMENSIONS[rule_id] for rule_id in DESIGN_TABLE_RULES["noun_stacking"]} == {"morphology"}
    assert {RULE_DIMENSIONS[rule_id] for rule_id in DESIGN_TABLE_RULES["voice"]} == {"agency"}


def test_analysis_schema_accepts_new_findings() -> None:
    path = Path(__file__).parent / "fixtures" / "architecture-review-corpus.json"
    schema_path = Path(__file__).parents[1] / "lingity" / "schemas" / "v1" / "analysis.schema.json"
    schema = cast(dict[str, Any], json.loads(schema_path.read_text(encoding="utf-8")))
    properties = cast(dict[str, Any], schema["properties"])
    linguistic_model = cast(dict[str, Any], properties["linguistic_model"])
    model_fields = set(cast(list[str], linguistic_model["required"]))
    assert model_fields == {"name", "version", "runtime", "digest"}
    validator = Draft202012Validator(schema)
    corpus = cast(dict[str, list[dict[str, JsonValue]]], json.loads(path.read_text(encoding="utf-8")))
    texts = [
        cast(str, item["text"])
        for item in corpus["positive"]
        if NEW_RULE_IDS & set(cast(list[str], item["required_rule_ids"]))
    ]
    for text in texts:
        validator.validate(analyze_text(text))


def test_profile_analysis_phrases_are_plain_lemma_strings() -> None:
    profile = load_profile()
    for group_name in (
        "hidden_agency",
        "weak_verbs",
        "jargon",
        "bureaucratic_phrases",
        "indirect_predicates",
        "purpose_markers",
        "filler_phrases",
        "recommendation_action_groups",
    ):
        phrase_map = cast(dict[str, list[str]], profile.rules[group_name])
        for expressions in phrase_map.values():
            for expression in expressions:
                assert expression
                assert "\\" not in expression
                assert not any(character in expression for character in "^$*+?[]{}|")


def test_action_count_ignores_participles_and_counts_predicates() -> None:
    noun_phrase = (
        "Evidence-based planning, advanced design, completed documentation, "
        "critical final quality, several operational constraints."
    )
    assert cast(int, _sentences(noun_phrase)[0]["action_count"]) == 0
    assert "LING-ACTION-001" not in _rule_ids(noun_phrase)

    predicate_series = (
        "Teams assess risks, build controls, run tests, fix defects, deploy changes, "
        "and support operations."
    )
    sentence = _sentences(predicate_series)[0]
    assert cast(int, sentence["action_count"]) == 6
    assert "LING-ACTION-001" in _rule_ids(predicate_series)
    assert "LING-NOUN-STACK-001" not in _rule_ids(predicate_series)


def test_nominalization_detection_ignores_ordinary_adjectives() -> None:
    text = "The critical final quality and several operational constraints guide the review."
    assert "LING-NOMINALIZATION-001" not in _rule_ids(text)


def test_noun_stacks_reset_at_punctuation_and_predicates() -> None:
    text = (
        "Clear, concise, direct recommendations help reviewers. "
        "Teams assess risks, document decisions, implement controls, monitor telemetry, "
        "ensure recovery, and support operations."
    )
    assert "LING-NOUN-STACK-001" not in _rule_ids(text)


def test_noun_stacks_stop_at_finite_s_predicates() -> None:
    false_positive_sentences = [
        "The platform team owns this component.",
        "The security group requires an owner.",
        "The migration plan needs approval.",
        "The gateway service handles retries.",
        "The billing system provides usage data.",
    ]
    for text in false_positive_sentences:
        assert _noun_stack_texts(text) == set(), text
        assert cast(int, _sentences(text)[0]["action_count"]) == 1

    assert _noun_stack_texts(
        "The identity provider certificate rotation process failed."
    ) == {"identity provider certificate rotation process"}
    assert _noun_stack_texts(
        "Use the repository-evidenced hybrid topology only as a baseline."
    ) == set()


def test_compound_depth_is_morphology_and_not_uncommon_compound_double_count() -> None:
    text = "The risk-adjusted-capacity-planning-review needs an owner."
    findings = _findings(text)
    depth = [
        finding for finding in findings
        if finding["rule_id"] == "LING-COMPOUND-DEPTH-001"
    ]
    assert len(depth) == 1
    assert depth[0]["dimension"] == "morphology"
    observed = cast(dict[str, JsonValue], depth[0]["observed_value"])
    assert observed["compound"] == "risk-adjusted-capacity-planning-review"
    assert observed["parts"] == 5
    assert "LING-COMPOUND-001" not in {cast(str, finding["rule_id"]) for finding in findings}


def test_abstract_evidence_rule_generalises_beyond_the_fixture_wording() -> None:
    """The rule must detect the construction, not the phrase it was written for.

    "closure evidence" appears in the shipped fixture. A rule that lists that
    phrase scores perfectly on the fixture and detects nothing else, so the
    check here is on vocabulary the profile has never seen.
    """

    def phrases(text: str) -> list[str]:
        analysis = analyze_text(text)
        findings = cast(list[dict[str, JsonValue]], analysis["findings"])
        return [
            cast(str, cast(dict[str, JsonValue], finding["observed_value"])["phrase"])
            for finding in findings
            if cast(dict[str, JsonValue], finding["observed_value"]).get("classification")
            == "abstract evidence phrase"
        ]

    unseen = {
        "Require completion evidence for the escalated items.": "completion evidence",
        "The committee requires attestation evidence before onboarding.": "attestation evidence",
        "Provide remediation evidence for the flagged controls.": "remediation evidence",
    }
    for text, expected in unseen.items():
        assert phrases(text) == [expected], (
            f"the abstract-evidence rule did not generalise to {expected!r}"
        )

    # A concrete modifier is not a nominalization standing in for an act.
    assert phrases("Provide the test evidence to the auditor.") == []
    assert phrases("The team collected evidence from the logs.") == []


def test_abstract_evidence_finding_is_fully_attributed() -> None:
    analysis = analyze_text("Require closure evidence for the governed recommendations.")
    findings = cast(list[dict[str, JsonValue]], analysis["findings"])
    matches = [
        finding
        for finding in findings
        if cast(dict[str, JsonValue], finding["observed_value"]).get("classification")
        == "abstract evidence phrase"
    ]
    assert len(matches) == 1
    finding = matches[0]
    assert finding["rule_id"] == "LING-JARGON-001"
    assert finding["severity"] == "medium"
    assert finding["threshold"]
    assert cast(str, finding["remediation"]).strip()
    location = cast(dict[str, JsonValue], finding["location"])
    start = cast(int, location["start"])
    end = cast(int, location["end"])
    assert (
        "Require closure evidence for the governed recommendations."[start:end]
        == "closure evidence"
    )


def test_mixed_purpose_detects_common_review_vocabulary_without_clean_false_positives() -> None:
    mixed = (
        "We recommend deferring ratification, and the team must patch the authorization "
        "gap, add regression tests, publish the runbook, and obtain sign-off before the "
        "exit criteria are considered met."
    )
    assert "LING-MIXED-PURPOSE-001" in _rule_ids(mixed)

    clean_sentences = [
        "We fixed the bug before the release.",
        "The team patched the endpoint before release.",
        "The platform team will harden the gateway before rollout.",
        "We are deferring ratification until the next review.",
        "The team will remove the old feature flag after the release.",
    ]
    for text in clean_sentences:
        assert "LING-MIXED-PURPOSE-001" not in _rule_ids(text), text


def test_passive_voice_rejects_adjectives_and_detects_irregulars() -> None:
    assert "LING-PASSIVE-001" not in _rule_ids("The service is a token. The endpoint is open.")

    text = "The gateway was built by the platform team. The decision was made yesterday."
    observed = {
        cast(str, finding["observed_value"])
        for finding in _findings(text)
        if finding["rule_id"] == "LING-PASSIVE-001"
    }
    assert {"was built", "was made"} <= observed


def test_passive_voice_allows_intervening_negation_and_adverbs() -> None:
    observed = {
        cast(str, finding["observed_value"])
        for finding in _findings("The architecture was not formally approved.")
        if finding["rule_id"] == "LING-PASSIVE-001"
    }
    assert "was not formally approved" in observed


def test_passive_voice_distinguishes_perfect_aspect_from_passive() -> None:
    perfect_actives = [
        "The token has expired.",
        "The team have completed the migration.",
        "The job has finished.",
        "The request had arrived late.",
        "Latency has increased.",
        "Traffic has grown steadily.",
    ]
    for text in perfect_actives:
        assert "LING-PASSIVE-001" not in _rule_ids(text), text

    true_passives = [
        "The change has been approved.",
        "The change was approved.",
        "The error is handled upstream.",
        "The nodes were removed.",
        "The design is being reviewed.",
        "The data will be migrated overnight.",
        "The change must be approved.",
        "The endpoint got deprecated last year.",
    ]
    for text in true_passives:
        assert "LING-PASSIVE-001" in _rule_ids(text), text

    clean_adjectival_or_transitive_got = [
        "The service is responsible.",
        "The service is available.",
        "The task is complete.",
        "The request was late.",
        "The team got approval.",
        "The team is responsible for uptime.",
        "The report is available now.",
        "The migration is complete.",
        "The delivery was late.",
    ]
    for text in clean_adjectival_or_transitive_got:
        assert "LING-PASSIVE-001" not in _rule_ids(text), text


def test_noun_stacks_stop_at_adverbs() -> None:
    adverbs = [
        "again",
        "always",
        "annually",
        "daily",
        "later",
        "monthly",
        "never",
        "often",
        "once",
        "quarterly",
        "soon",
        "twice",
        "weekly",
    ]
    for adverb in adverbs:
        text = f"We tested the rollback path {adverb} and it restored service in under four minutes."
        assert "LING-NOUN-STACK-001" not in _rule_ids(text), text


def test_sentence_and_clause_segmentation_handle_abbreviations_and_lists() -> None:
    text = (
        "The cache uses a regional store, e.g. Redis, for session hints. "
        "Teams monitor latency."
    )
    records = _sentences(text)
    assert len(records) == 2
    assert "e.g. Redis" in cast(str, records[0]["text"])

    list_text = "Authentication, authorization, logging, monitoring, resilience, and deployment are required."
    list_record = _sentences(list_text)[0]
    assert cast(int, list_record["clause_count"]) <= 3
    assert "LING-CLAUSE-001" not in _rule_ids(list_text)


def test_clean_prose_sweep_has_zero_findings() -> None:
    clean_sentences = [
        "The team shipped the change on Tuesday and monitored it closely.",
        "We measured latency three times and the results were consistent.",
        "The database team owns the migration and will run it next week.",
        "Sarah reviewed the design document and approved it without changes.",
        "The service returns an error when the token has expired.",
        "Our goal is to cut the deploy time from thirty minutes to five.",
        "The cache holds session data for one hour and then evicts it.",
        "We will not proceed until the security team signs off.",
        "The report lists costs, owners, and deadlines for each workstream.",
        "This design uses a read-through cache and a write-behind queue.",
        "The build failed twice yesterday because a dependency was missing.",
        "Engineers rotate the on-call pager weekly and hand off open incidents.",
        "The load balancer routes traffic to the healthy nodes only.",
        "We compared three options and chose the simplest one.",
        "The vendor confirmed the fix will ship in version 4.2 next month.",
        "The payments team will migrate the billing service to the new cluster on 3 March.",
        "Alice owns the schema change and will land it behind a feature flag.",
    ]
    for text in clean_sentences:
        assert _findings(text) == [], text


def _noun_stacks(text: str) -> list[tuple[str, int, int]]:
    stacks: list[tuple[str, int, int]] = []
    for finding in _findings(text):
        if finding["rule_id"] != "LING-NOUN-STACK-001":
            continue
        observed = cast(dict[str, JsonValue], finding["observed_value"])
        stacks.append(
            (
                cast(str, observed["text"]),
                cast(int, observed["words"]),
                cast(int, observed["units"]),
            )
        )
    return stacks


def test_a_noun_stack_span_is_the_text_it_reports() -> None:
    # The reported span used to be sliced between the first and last stack
    # token, so intervening words were swallowed and the reported word count
    # disagreed with the reported text. observed_value is the contract this
    # project rests on, so "words" must count the words actually in the span
    # and "units" must be the naming units compared against the threshold.
    for text in (
        "Recommended decision: Propose deferring architecture ratification and any "
        "irreversible V2 cutover.",
        "Require closure evidence for the governed recommendations before a target "
        "architecture returns for human decision.",
        "The customer data retention policy review board met.",
        "The Azure Kubernetes Service cluster owners met.",
    ):
        for span, words, units in _noun_stacks(text):
            assert span in text, span
            assert len(span.split()) == words, (span, words)
            assert 0 < units <= words, (span, units, words)
            assert not any(
                token in span.split() for token in ("and", "or", "the", "a", "for")
            )


def test_a_named_entity_counts_as_one_naming_unit() -> None:
    # "Azure Kubernetes Service" is one product name, so the phrase is three
    # units (name + cluster + owners) across five words. Counting the words
    # would report a denser stack than a reader experiences.
    stacks = _noun_stacks("The Azure Kubernetes Service cluster owners met.")
    assert stacks == [("Azure Kubernetes Service cluster owners", 5, 3)]


@pytest.mark.parametrize(
    "text",
    [
        "Tom Zimmermann and Eirini Kalliamvakou published a study.",
        "The Azure Kubernetes Service cluster failed.",
        "Rudrajit Choudhuri wrote the summary.",
    ],
)
def test_a_named_entity_is_not_a_noun_stack(text: str) -> None:
    # A person or product name is a single naming unit, not stacked modifiers.
    # "Unpack the noun stack with a verb or preposition" is unactionable advice
    # for "Tom Zimmermann", so the entity must not inflate the stack depth.
    assert _noun_stacks(text) == [], text


@pytest.mark.parametrize(
    "text",
    [
        "The critical security review passed.",
        "A large red button appeared.",
        "The final approved budget shipped.",
    ],
)
def test_an_adjective_is_not_a_noun_stack(text: str) -> None:
    assert _noun_stacks(text) == [], text


def test_a_noun_stack_is_detected_regardless_of_modifier_dependency_label() -> None:
    # The parser labels the first modifier of "messaging loss hypotheses"
    # "compound" in one context and "amod" in another. The phrase is the same
    # noun stack either way, so detection must not depend on which label the
    # parser chose.
    embedded = (
        "Use the hybrid topology as a baseline, address the authorization concerns, "
        "and verify the two critical messaging loss hypotheses."
    )
    standalone = "Verify the messaging loss hypotheses."
    assert ("messaging loss hypotheses", 3, 3) in _noun_stacks(embedded)
    assert ("messaging loss hypotheses", 3, 3) in _noun_stacks(standalone)


def _redundancy_terms(text: str) -> list[tuple[str, int]]:
    terms: list[tuple[str, int]] = []
    for finding in _findings(text):
        if finding["rule_id"] != "LING-REDUNDANCY-001":
            continue
        observed = cast(dict[str, JsonValue], finding["observed_value"])
        terms.append((cast(str, observed["term"]), cast(int, observed["occurrences"])))
    return sorted(terms)


def _fingerprints(text: str) -> list[str]:
    """Identify each finding by what it reports, not by where the offsets land."""
    return sorted(
        f"{finding['rule_id']}:{json.dumps(finding['observed_value'], sort_keys=True)}"
        for finding in _findings(text)
    )


def test_a_repeated_term_is_reported_within_one_block() -> None:
    # Three uses of one content word in a single paragraph is the local
    # repetition the rule exists to catch, and it still is.
    text = (
        "The governance board reviewed governance rules and published governance advice."
    )
    assert ("governance", 3) in _redundancy_terms(text)


def test_a_term_repeated_across_blocks_is_not_redundancy() -> None:
    # Governance prose has to call one concept by one name in every section, so
    # a term recurring across paragraphs is the document being consistent. When
    # this was counted per document, naming the subject of a long document was
    # itself enough to be reported as redundant.
    blocks = [
        "The governance board met in March.",
        "Each team publishes governance rules.",
        "The auditor reviews governance evidence.",
        "Engineers follow governance practice daily.",
        "The council records governance decisions.",
    ]
    assert _redundancy_terms("\n\n".join(blocks)) == []


def test_a_document_reports_exactly_what_its_blocks_report() -> None:
    # A finding has to describe the passage it points at. A rule that counted
    # across the whole document made a paragraph score differently alone than it
    # did inside the document containing it, so concatenating clean prose
    # manufactured findings that no paragraph had.
    blocks = [
        "The governance board reviewed governance rules and published governance advice.",
        "A decision was made by the committee regarding the deferral of the migration.",
        "The auditor reviews governance evidence and records governance decisions.",
        "Engineers ship the gateway each week.",
    ]
    together = _fingerprints("\n\n".join(blocks))
    apart = sorted(item for block in blocks for item in _fingerprints(block))
    assert together == apart

def _observed_texts(text: str) -> list[str]:
    texts: list[str] = []
    for finding in _findings(text):
        observed = finding["observed_value"]
        if isinstance(observed, str):
            texts.append(observed)
        elif isinstance(observed, dict):
            texts.extend(value for value in observed.values() if isinstance(value, str))
    return texts


def test_a_finding_quotes_text_the_way_the_parser_read_it() -> None:
    # The parser joins a block's wrapped lines with one space before reading
    # them, so a finding that echoed the raw source slice reported a line break
    # and a list item's indentation that the analysis never saw.
    text = "- The results are not\n  conflated with the source data by the team.\n"
    texts = _observed_texts(text)
    assert texts, "expected the wrapped passive construction to be reported"
    assert all("\n" not in value for value in texts), texts
