from __future__ import annotations

from typing import cast

from lingity.analyzer import analyze_text
from lingity.invariants import compare_protected
from lingity.models import JsonValue


def _protected(analysis: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], analysis["protected"])


def test_clearer_rewrite_preserves_protected_meaning(
    recommendation_fixture: dict[str, str],
) -> None:
    source = analyze_text(recommendation_fixture["original"])
    candidate = analyze_text(recommendation_fixture["rewrite"])
    comparison = compare_protected(_protected(source), _protected(candidate))
    assert comparison["equivalent"] is True
    signature = cast(list[str], _protected(source)["semantic_signature"])
    assert "identifier:identifier:V2" in signature
    assert "quantity:count:2" in signature
    assert "negation:governance_polarity:architecture_approval_deferred" in signature
    assert "governance:authority:target_architecture_requires_human_approval" in signature


def test_changed_quantity_is_rejected(
    recommendation_fixture: dict[str, str],
) -> None:
    source = analyze_text(recommendation_fixture["original"])
    changed = recommendation_fixture["rewrite"].replace("either messaging path", "three messaging paths")
    candidate = analyze_text(changed)
    comparison = compare_protected(_protected(source), _protected(candidate))
    assert comparison["equivalent"] is False
    assert "quantity:count:2" in cast(list[str], comparison["missing"])
    assert "quantity:count:3" in cast(list[str], comparison["added"])


def test_dropped_duplicate_protected_value_is_rejected() -> None:
    source = analyze_text("ADR-42 must retain two paths and two owners.")
    candidate = analyze_text("ADR-42 must retain two paths and owners.")
    comparison = compare_protected(_protected(source), _protected(candidate))
    assert comparison["equivalent"] is False
    assert cast(list[str], comparison["missing"]) == ["quantity:count:2"]


def test_extracts_modal_citation_and_governance() -> None:
    text = "ADR-42 must not change by more than 15% [E-7]; approval requires a waiver."
    manifest = _protected(analyze_text(text))
    signature = set(cast(list[str], manifest["semantic_signature"]))
    assert "identifier:identifier:ADR-42" in signature
    assert "modal:modal_strength:must not" in signature
    assert "quantity:percentage:15%" in signature
    assert "citation:reference:[E-7]" in signature
    assert "negation:lexical_negation:not" in signature
    assert "governance:term:approval" in signature
    assert "governance:term:waiver" in signature
