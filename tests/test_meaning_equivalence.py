"""The protected-meaning gate must generalise beyond memorised wording.

Every pair in ``meaning-equivalence-corpus.json`` was authored from the
semantics of governance directives rather than from any profile pattern or
shipped fixture. A gate that recognises phrasings instead of propositions
scores near the degenerate baseline on this corpus: answering "changed" for
every pair scores 16 of 33, so the changed-pair count alone proves nothing. The
equivalent-pair count is what distinguishes a semantic gate from a lookup
table, and both are asserted here.

The two counts above are checked against the corpus by
``test_the_documented_baseline_matches_the_corpus``, because a number written
into prose beside a file that grows is a number that goes quietly wrong.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from lingity.invariants import compare_protected, extract_protected
from lingity.profiles import Profile, load_profile

CORPUS_PATH = Path(__file__).parent / "fixtures" / "meaning-equivalence-corpus.json"


def _corpus() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return loaded


def _comparison(profile: Profile, left: str, right: str) -> dict[str, Any]:
    return compare_protected(
        extract_protected(left, profile),
        extract_protected(right, profile),
    )


def _disposition(profile: Profile, left: str, right: str) -> str:
    return str(_comparison(profile, left, right)["disposition"])


@pytest.fixture(scope="module")
def profile() -> Profile:
    return load_profile("architecture-review")


def _ids(key: str) -> list[str]:
    return [str(pair["id"]) for pair in _corpus()[key]]


def _pairs(key: str) -> list[dict[str, Any]]:
    return list(_corpus()[key])


def _known_gaps() -> dict[str, str]:
    return dict(_corpus().get("known_gaps", {}))


def _apply_known_gap(pair: dict[str, Any]) -> None:
    """Fail the test as expected when the pair names a documented extractor gap.

    These are recorded as strict expected failures rather than deleted, so the
    corpus keeps measuring honestly. Every gap below is conservative: the gate
    answers "unresolved" or "changed", never "equivalent", so no gap can let a
    meaning-changing rewrite through. If a change fixes one, the strict xfail
    turns into a failure and forces the win to be acknowledged here.
    """

    gaps = _known_gaps()
    gap = gaps.get(str(pair["id"]))
    if gap is not None:
        pytest.xfail(gap)


@pytest.mark.parametrize("pair", _pairs("equivalent_pairs"), ids=_ids("equivalent_pairs"))
def test_faithful_rewrites_compare_equivalent(profile: Profile, pair: dict[str, Any]) -> None:
    _apply_known_gap(pair)
    disposition = _disposition(profile, str(pair["left"]), str(pair["right"]))
    assert disposition == "equivalent", (
        f"{pair['id']}: a faithful rewrite was reported {disposition!r}. "
        f"{pair['rationale']}\n  left:  {pair['left']}\n  right: {pair['right']}"
    )


@pytest.mark.parametrize("pair", _pairs("changed_pairs"), ids=_ids("changed_pairs"))
def test_meaning_changes_are_detected(profile: Profile, pair: dict[str, Any]) -> None:
    _apply_known_gap(pair)
    disposition = _disposition(profile, str(pair["left"]), str(pair["right"]))
    assert disposition == "changed", (
        f"{pair['id']}: a {pair['mutation']} mutation was reported {disposition!r} "
        f"and would have been accepted. {pair['rationale']}\n"
        f"  left:  {pair['left']}\n  right: {pair['right']}"
    )


def test_no_meaning_change_is_ever_reported_equivalent(profile: Profile) -> None:
    """The one property that must hold with no exceptions.

    A gap that answers "unresolved" costs a rewrite that should have been
    accepted. A gap that answers "equivalent" certifies a rewrite that changed
    what the text commits to. Only the second is a safety failure, so it is
    asserted over the whole corpus including the pairs listed in known_gaps.
    """

    for pair in _corpus()["changed_pairs"]:
        disposition = _disposition(profile, str(pair["left"]), str(pair["right"]))
        assert disposition != "equivalent", (
            f"{pair['id']}: a {pair['mutation']} mutation was certified as "
            f"meaning-preserving. {pair['rationale']}\n"
            f"  left:  {pair['left']}\n  right: {pair['right']}"
        )


def test_known_gaps_are_all_real_and_conservative(profile: Profile) -> None:
    """known_gaps must stay an honest ledger, not a place to hide failures.

    Every listed id must exist, must still fail, and must fail conservatively.
    An entry that starts passing has to be removed rather than left to make the
    corpus score look worse than it is.
    """

    corpus = _corpus()
    expected = {"equivalent_pairs": "equivalent", "changed_pairs": "changed"}
    by_id = {
        str(pair["id"]): (key, pair)
        for key in expected
        for pair in corpus[key]
    }
    for gap_id, reason in _known_gaps().items():
        assert gap_id in by_id, f"known_gaps names {gap_id!r}, which is not in the corpus"
        assert reason.strip(), f"known_gaps[{gap_id!r}] has no stated reason"
        key, pair = by_id[gap_id]
        disposition = _disposition(profile, str(pair["left"]), str(pair["right"]))
        assert disposition != expected[key], (
            f"{gap_id} now passes and must be removed from known_gaps"
        )
        assert disposition != "equivalent" or key == "equivalent_pairs", (
            f"{gap_id} is not a conservative gap: it certifies a changed meaning"
        )


def test_comparison_is_symmetric(profile: Profile) -> None:
    """Order of arguments must not decide the verdict, except when naming an actor.

    Comparison is directional: supplying an actor the source left unnamed is a
    permitted specification, while dropping a named actor is a violation. That
    asymmetry is the whole point of the actor policy, so it is asserted rather
    than waived -- a pair may only disagree by direction when the comparison
    actually reports a specification.
    """

    corpus = _corpus()
    for key in ("equivalent_pairs", "changed_pairs"):
        for pair in corpus[key]:
            left, right = str(pair["left"]), str(pair["right"])
            forward = _comparison(profile, left, right)
            backward = _comparison(profile, right, left)
            if forward["disposition"] == backward["disposition"]:
                continue
            specified = list(forward["specified"]) + list(backward["specified"])
            assert specified, (
                f"{pair['id']}: asymmetric verdict {forward['disposition']!r} forward "
                f"vs {backward['disposition']!r} reversed, with no actor "
                f"specification to account for it"
            )


def test_identity_is_equivalent(profile: Profile) -> None:
    """A text always preserves its own meaning."""
    corpus = _corpus()
    for key in ("equivalent_pairs", "changed_pairs"):
        for pair in corpus[key]:
            for side in ("left", "right"):
                text = str(pair[side])
                assert _disposition(profile, text, text) == "equivalent", (
                    f"{pair['id']} ({side}): text compared unequal to itself"
                )


BASELINE_CLAIM = re.compile(r'answering "changed" for\s*\n?every pair scores (\d+) of (\d+)', re.MULTILINE)


def test_the_documented_baseline_matches_the_corpus() -> None:
    """The module docstring states a degenerate score, so it must stay true.

    The corpus grows. A hard number written beside it in prose does not, and a
    stale one would misstate the very argument this file exists to make: that
    the changed-pair count alone proves nothing.
    """
    claim = BASELINE_CLAIM.search(__doc__ or "")
    assert claim is not None, "the module docstring no longer states a baseline score"

    corpus = _corpus()
    changed = len(corpus["changed_pairs"])
    total = changed + len(corpus["equivalent_pairs"])

    assert (int(claim.group(1)), int(claim.group(2))) == (changed, total), (
        f"the docstring claims a degenerate baseline of {claim.group(1)} of {claim.group(2)}, "
        f"but the corpus scores {changed} of {total}"
    )
