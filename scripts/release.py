"""Build and upload a release, refusing to do so on anything it cannot verify.

An upload is irreversible in the way that matters: a version number, once used,
can never be reused on that index. Deleting a release does not free it. So every
check here runs *before* the upload, and anything unverifiable stops the run
rather than being assumed benign.

The checks are not decoration. `twine check` validates README rendering and
nothing else -- it passes distributions PyPI rejects outright, which is how a
PEP 440 direct reference survived every local check in this repository until a
test caught it. `python -m build` and `pip install` accept the same artifacts.
The metadata inspection below covers the three failures that would otherwise
appear for the first time as an HTTP 400 from the index.

Credentials are never read, written, or accepted as arguments. `twine` resolves
them itself from `~/.pypirc`, the system keyring, or `TWINE_*` environment
variables, so this runs on a machine that is already logged in and carries no
secret of its own.
"""

from __future__ import annotations

import argparse
import email
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Sequence
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parent.parent
DIST = REPO_ROOT / "dist"

INDEXES = {
    "pypi": "https://pypi.org",
    "testpypi": "https://test.pypi.org",
}


class ReleaseError(Exception):
    """A condition that must stop the release rather than be worked around."""


def _run(command: Sequence[str], *, capture: bool = False) -> str:
    """Run `command` in the repository root, failing loudly on a non-zero exit.

    Output is streamed rather than captured unless the caller wants the value,
    because `build`, `twine` and the guard run are slow enough that hiding their
    progress until they finish would be worse than not repeating it here. The
    refusal therefore says where the explanation is instead of implying there
    is none.
    """
    printable = " ".join(command)
    print(f"  $ {printable}", flush=True)
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=capture)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        reason = f":\n{detail}" if detail else " (its output is above)"
        raise ReleaseError(f"`{printable}` exited {result.returncode}{reason}")
    return (result.stdout or "") if capture else ""


def require_clean_worktree() -> str:
    """Return the HEAD commit, refusing to release anything uncommitted.

    An upload built from a dirty tree cannot be reproduced from the repository,
    and the version it claims will never be available again to correct it.
    """
    status = _run(["git", "status", "--porcelain"], capture=True).strip()
    if status:
        raise ReleaseError(
            "the working tree has uncommitted changes, so the upload could not be "
            f"reproduced from any commit:\n{status}"
        )
    return _run(["git", "rev-parse", "HEAD"], capture=True).strip()


def declared_version() -> tuple[str, str]:
    """Return the distribution name and version declared in pyproject.toml.

    Every failure here is reported as a refusal rather than raised as whatever
    the parser happened to produce. A missing `[project]` table would otherwise
    surface as a bare `KeyError: 'project'` -- a traceback naming a dictionary
    lookup, from a script whose contract is to stop with a reason.
    """
    path = REPO_ROOT / "pyproject.toml"
    try:
        parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ReleaseError(f"{path} could not be read: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ReleaseError(f"{path} is not valid TOML: {error}") from error

    project = parsed.get("project")
    if not isinstance(project, dict):
        raise ReleaseError(f"{path} declares no [project] table, so there is nothing to release")

    name, version = project.get("name"), project.get("version")
    if not isinstance(name, str) or not name:
        raise ReleaseError(f"{path} declares no project name to release")
    if not isinstance(version, str) or not version:
        raise ReleaseError(f"{path} declares no version to release")
    return name, version


def index_holds(index: str, name: str, version: str) -> bool:
    """Report whether `index` already holds `name` at `version`.

    A network failure is raised, never reported as absence. Treating an
    unreachable index as "the version is free" would be the exact shape of
    success-shaped fallback this project refuses: the check would fall silent at
    the moment it stopped working, and the caller would read that as assurance.
    """
    url = f"{INDEXES[index]}/pypi/{name}/{version}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise ReleaseError(
            f"{index} answered {error.code} for {url}, so it could not be checked"
        ) from error
    except urllib.error.URLError as error:
        raise ReleaseError(
            f"{index} could not be reached to check {name} {version}: {error.reason}"
        ) from error
    return True


def require_version_absent(index: str, name: str, version: str) -> None:
    """Refuse to reuse a version the index already holds.

    A repeated version is rejected with `400 File already exists`, and that
    answer arrives only after the build and part of the transfer. Asking first
    turns it into a sentence.
    """
    if index_holds(index, name, version):
        raise ReleaseError(
            f"{index} already holds {name} {version}. A version number cannot be reused "
            "even after deletion -- raise the version in pyproject.toml."
        )


def build() -> list[Path]:
    """Build a wheel and an sdist from a cleared `dist/`.

    Clearing it first is not tidiness. `twine upload dist/*` uploads whatever is
    present, so an artifact left from an earlier version rides along unnoticed.
    """
    if DIST.exists() and not DIST.is_dir():
        raise ReleaseError(f"{DIST} exists but is not a directory, so it cannot be cleared or built into")
    if DIST.exists():
        shutil.rmtree(DIST)
    _run([sys.executable, "-m", "build"])

    artifacts = sorted(DIST.glob("*"))
    suffixes = {path.suffix for path in artifacts}
    if not {".whl", ".gz"} <= suffixes:
        raise ReleaseError(f"expected a wheel and an sdist in dist/, found {artifacts}")
    return artifacts


def wheel_metadata(wheel: Path) -> Message:
    """Return the METADATA of `wheel`, which is what the index actually reads."""
    with ZipFile(wheel) as archive:
        names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ReleaseError(f"{wheel.name} carries {len(names)} METADATA files, expected 1")
        return email.message_from_string(archive.read(names[0]).decode("utf-8"))


def require_no_direct_reference(metadata: Message) -> None:
    """A public index rejects any distribution whose metadata carries one.

    PEP 440: "Public index servers SHOULD NOT allow the use of direct references
    in uploaded distributions." PyPI answers `400 Can't have direct dependency`.
    Extras are covered because they contribute to this same field and are
    rejected on the same terms.
    """
    direct = [value for value in metadata.get_all("Requires-Dist", []) if "@" in value]
    if direct:
        raise ReleaseError(
            "the built metadata declares direct URL dependencies, which the index "
            f"rejects at upload: {direct}"
        )


def require_license(metadata: Message) -> str:
    """Without stated terms, default copyright forbids use of what is published.

    Returns the terms it accepted so the caller reports the value that was
    actually checked. Reading the field a second time at the print site let the
    two disagree: this accepts either spelling, so a legacy `License` field
    passed here and then printed `license None`.
    """
    expression = metadata.get("License-Expression") or metadata.get("License")
    if not expression:
        raise ReleaseError(
            "the built metadata states no license, so nobody could legally use the upload"
        )
    if not metadata.get_all("License-File"):
        raise ReleaseError(f"the built metadata declares {expression} but ships no license file")
    return expression


def require_known_classifiers(metadata: Message) -> None:
    """PyPI validates classifiers at upload and rejects any it does not know."""
    try:
        from trove_classifiers import classifiers as known
    except ImportError as error:
        raise ReleaseError(
            "trove-classifiers is not installed, so the classifiers cannot be checked "
            "before upload. Install the release extra: pip install -e '.[release]'"
        ) from error

    unknown = [value for value in metadata.get_all("Classifier", []) if value not in known]
    if unknown:
        raise ReleaseError(f"the built metadata declares classifiers PyPI rejects: {unknown}")


def upload(index: str, artifacts: Sequence[Path]) -> None:
    """Hand the artifacts to twine, which resolves credentials itself."""
    _run(
        [sys.executable, "-m", "twine", "upload", "--repository", index, "--non-interactive"]
        + [str(path) for path in artifacts]
    )


def confirm(index: str, name: str, version: str) -> None:
    """Require a typed confirmation before an upload that cannot be undone."""
    expected = f"{name} {version}"
    print(
        f"\nAbout to upload {expected} to {index} ({INDEXES[index]}).\n"
        "This cannot be undone, and the version number can never be reused.\n"
        "Type the name and version to continue: ",
        end="",
        flush=True,
    )
    if sys.stdin.readline().strip() != expected:
        raise ReleaseError("confirmation did not match, so nothing was uploaded")


def _release(arguments: argparse.Namespace) -> int:
    name, version = declared_version()
    print(f"Releasing {name} {version} to {arguments.repository}\n")

    print("Checking the repository")
    head = require_clean_worktree()
    print(f"  HEAD {head[:12]}, working tree clean")
    if arguments.skip_guards:
        print("  guard tests SKIPPED by request")
    else:
        # These pin the declared license against the file beside it, the
        # documented model wheel against the version the loader enforces, and
        # the READMEs against the workflow. They are claims about the repository
        # rather than the artifact, so they run from the suite that owns them.
        _run([sys.executable, "-m", "pytest", "tests/test_documentation.py", "-q"])

    print("\nChecking the index")
    require_version_absent(arguments.repository, name, version)
    print(f"  {arguments.repository} does not hold {name} {version}")

    print("\nBuilding")
    artifacts = build()

    print("\nChecking the built metadata")
    wheel = next(path for path in artifacts if path.suffix == ".whl")
    metadata = wheel_metadata(wheel)
    require_no_direct_reference(metadata)
    license_terms = require_license(metadata)
    require_known_classifiers(metadata)
    print(
        f"  no direct references, license {license_terms}, "
        f"{len(metadata.get_all('Classifier', []))} classifiers all known"
    )
    # Last of the artifact checks because it is the weakest: it reads the README
    # rendering and nothing the three checks above cover.
    _run([sys.executable, "-m", "twine", "check", "--strict"] + [str(p) for p in artifacts])

    if arguments.dry_run:
        print(f"\nDry run: {len(artifacts)} artifacts built and verified, nothing uploaded.")
        for path in artifacts:
            print(f"  {path.relative_to(REPO_ROOT)}")
        return 0

    if not arguments.yes:
        confirm(arguments.repository, name, version)

    print("\nUploading")
    upload(arguments.repository, artifacts)

    print("\nConfirming the index now holds it")
    if not index_holds(arguments.repository, name, version):
        raise ReleaseError(
            "the upload reported success but the index does not hold the version. "
            "Do not retry blindly -- check the index first."
        )
    print(f"  {INDEXES[arguments.repository]}/project/{name}/{version}/")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build, verify and upload a release. Credentials come from twine.",
    )
    parser.add_argument(
        "--repository",
        required=True,
        choices=sorted(INDEXES),
        help="which index to upload to. Required, because there is no safe default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run every check and build the artifacts, but do not upload.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the typed confirmation, for an automated release.",
    )
    parser.add_argument(
        "--skip-guards",
        action="store_true",
        help="do not run the packaging guard tests. They are why this is safe.",
    )
    arguments = parser.parse_args(argv)

    try:
        return _release(arguments)
    except ReleaseError as error:
        print(f"\nRelease stopped: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
