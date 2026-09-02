"""The release script's refusals, checked one at a time.

Every assertion here is about the script declining to publish. That bias is
deliberate: an upload cannot be undone and its version number can never be
reused, so a check that fails open is worse than no check, because the silence
reads as assurance. Each refusal therefore gets a case proving it fires, and a
matching case proving it does not fire on a healthy artifact -- a function that
rejected everything would satisfy the first half alone.
"""

from __future__ import annotations

import email
import io
import sys
import urllib.error
from email.message import Message
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import release  # noqa: E402

HEALTHY = """\
Metadata-Version: 2.4
Name: lingity
Version: 0.1.0
License-Expression: Apache-2.0
License-File: LICENSE
License-File: NOTICE
Classifier: Typing :: Typed
Requires-Dist: spacy>=3.8,<3.9
Requires-Dist: mypy>=1.13,<2; extra == "dev"
"""


def built(text: str) -> Message:
    """Parse metadata the way the script reads it out of a wheel."""
    return email.message_from_string(text)


def test_healthy_metadata_passes_every_artifact_check() -> None:
    """The refusals below mean nothing if the real thing is also refused."""
    healthy = built(HEALTHY)

    release.require_no_direct_reference(healthy)
    release.require_license(healthy)
    release.require_known_classifiers(healthy)


@pytest.mark.parametrize(
    "requirement",
    [
        "en_core_web_sm @ https://example.invalid/en_core_web_sm-3.8.0-py3-none-any.whl",
        'en_core_web_sm @ file:../wheels/x.whl; extra == "dev"',
    ],
)
def test_a_direct_reference_stops_the_release(requirement: str) -> None:
    """The failure this script exists for, including when parked in an extra.

    An extra contributes to the same `Requires-Dist` field and is rejected on
    the same terms, so scanning only the required dependencies would pass an
    artifact the index refuses.
    """
    artifact = built(HEALTHY + f"Requires-Dist: {requirement}\n")

    with pytest.raises(release.ReleaseError, match="direct URL dependencies"):
        release.require_no_direct_reference(artifact)


def test_metadata_without_a_license_stops_the_release() -> None:
    """Publishing unlicensed work leaves nobody able to legally use it."""
    artifact = built(HEALTHY.replace("License-Expression: Apache-2.0\n", ""))

    with pytest.raises(release.ReleaseError, match="states no license"):
        release.require_license(artifact)


def test_a_license_expression_with_no_file_stops_the_release() -> None:
    """An identifier with no text behind it states terms it does not supply."""
    artifact = built(
        HEALTHY.replace("License-File: LICENSE\n", "").replace("License-File: NOTICE\n", "")
    )

    with pytest.raises(release.ReleaseError, match="ships no license file"):
        release.require_license(artifact)


def test_the_legacy_license_field_is_still_accepted() -> None:
    """Older metadata states its terms in `License`, and that is not a defect.

    The accepted terms are returned so the caller reports what was checked. When
    the status line read `License-Expression` independently it printed
    `license None` for exactly this artifact -- a run reporting no license while
    the guard it just passed had certified one.
    """
    artifact = built(HEALTHY.replace("License-Expression: Apache-2.0", "License: Apache-2.0"))

    assert release.require_license(artifact) == "Apache-2.0"


def test_the_reported_license_is_the_one_that_was_checked() -> None:
    """A status line that can disagree with its own check is worse than silence."""
    for field in ("License-Expression", "License"):
        artifact = built(HEALTHY.replace("License-Expression: Apache-2.0", f"{field}: Apache-2.0"))

        assert release.require_license(artifact) == artifact.get(field)


def test_a_dist_path_that_is_not_a_directory_stops_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`shutil.rmtree` raises `NotADirectoryError` on a file, bypassing the refusal path."""
    dist = tmp_path / "dist"
    dist.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(release, "DIST", dist)

    with pytest.raises(release.ReleaseError, match="is not a directory"):
        release.build()


def test_a_dist_directory_is_cleared_rather_than_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refusal must not fire on the ordinary case it sits in front of."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "lingity-0.0.1-py3-none-any.whl").write_text("stale", encoding="utf-8")
    monkeypatch.setattr(release, "DIST", dist)
    monkeypatch.setattr(release, "_run", lambda *_, **__: None)

    with pytest.raises(release.ReleaseError, match="expected a wheel and an sdist"):
        release.build()

    assert not dist.exists(), "the stale artifact survived, so `twine upload dist/*` would ship it"


def test_an_unknown_classifier_stops_the_release() -> None:
    """PyPI validates classifiers at upload; nothing local does."""
    artifact = built(HEALTHY + "Classifier: Topic :: Not A Real Classifier\n")

    with pytest.raises(release.ReleaseError, match="classifiers PyPI rejects"):
        release.require_known_classifiers(artifact)


def test_a_version_the_index_already_holds_stops_the_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused version is rejected, and the answer arrives mid-transfer."""
    monkeypatch.setattr(release, "index_holds", lambda *_: True)

    with pytest.raises(release.ReleaseError, match="cannot be reused"):
        release.require_version_absent("pypi", "lingity", "0.1.0")


def test_an_absent_version_does_not_stop_the_release(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(release, "index_holds", lambda *_: False)

    release.require_version_absent("pypi", "lingity", "0.1.0")


def test_an_unreachable_index_stops_the_release_rather_than_reading_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distinction this test protects is the whole point of the check.

    "The index said no such version" and "the index could not be asked" are
    different answers. Collapsing them would let a network failure read as
    permission to upload, and the check would be quietest exactly when it had
    stopped working.
    """

    def unreachable(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("name resolution failed")

    monkeypatch.setattr("urllib.request.urlopen", unreachable)

    with pytest.raises(release.ReleaseError, match="could not be reached"):
        release.index_holds("pypi", "lingity", "0.1.0")


def test_a_server_error_stops_the_release_rather_than_reading_as_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only 404 means absent. A 503 means the question went unanswered."""

    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError("url", 503, "Service Unavailable", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", unavailable)

    with pytest.raises(release.ReleaseError, match="answered 503"):
        release.index_holds("pypi", "lingity", "0.1.0")


def test_a_dirty_worktree_stops_the_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """An upload from uncommitted work cannot be reproduced from any commit."""
    monkeypatch.setattr(release, "_run", lambda *_, **__: " M lingity/nlp.py\n")

    with pytest.raises(release.ReleaseError, match="uncommitted changes"):
        release.require_clean_worktree()


def test_a_refused_confirmation_stops_the_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typing something else must abort, not fall through to the upload."""
    monkeypatch.setattr("sys.stdin", io.StringIO("lingity 9.9.9\n"))

    with pytest.raises(release.ReleaseError, match="nothing was uploaded"):
        release.confirm("pypi", "lingity", "0.1.0")


def test_the_matching_confirmation_proceeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("lingity 0.1.0\n"))

    release.confirm("pypi", "lingity", "0.1.0")


def test_the_repository_argument_has_no_default() -> None:
    """A default would eventually publish to the wrong index quietly."""
    with pytest.raises(SystemExit):
        release.main([])


def test_the_declared_version_is_the_one_that_would_be_released() -> None:
    """The script must release what pyproject.toml declares, not a guess."""
    name, version = release.declared_version()

    assert name == "lingity"
    assert version.count(".") >= 1


@pytest.mark.parametrize(
    ("pyproject", "expected"),
    [
        ("name = 'lingity'\n", "no \\[project\\] table"),
        ("[project]\nversion = '0.1.0'\n", "no project name"),
        ("[project]\nname = 'lingity'\n", "no version"),
        ("[project\nname =", "not valid TOML"),
    ],
)
def test_an_unreadable_pyproject_stops_the_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pyproject: str, expected: str
) -> None:
    """A refusal must arrive as a sentence, not as whatever the parser raised.

    Without this the missing-table case surfaces as `KeyError: 'project'` -- a
    traceback naming a dictionary lookup, from a script whose entire contract is
    to stop and say why.
    """
    (tmp_path / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)

    with pytest.raises(release.ReleaseError, match=expected):
        release.declared_version()


def test_the_release_extra_can_run_the_guards_it_depends_on() -> None:
    """The documented install must support the script's default behaviour.

    `scripts/release.py` runs the packaging guards unless `--skip-guards` is
    passed, so an extra that cannot run pytest breaks the exact command the
    READMEs give. Depending on `dev` rather than restating pieces of it keeps
    that true when a guard grows a new dependency.
    """
    import tomllib

    extras = tomllib.loads(
        (release.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["optional-dependencies"]

    assert "release" in extras, "pyproject.toml declares no release extra"
    assert any(requirement.startswith("lingity[") for requirement in extras["release"]), (
        "the release extra does not pull in the dev extra, so `pip install -e '.[release]'` "
        f"cannot run the guards the script invokes: {extras['release']}"
    )
