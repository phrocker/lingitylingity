"""The documented development commands must be the commands CI actually runs.

README.md and README_AI.md both state that the block under `## Development` is
what CI runs. That is a claim about another file, so it can rot silently the
moment either side is edited alone -- which is exactly what happened before this
test existed. Pinning it here turns the claim into something that fails loudly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
READMES = ("README.md", "README_AI.md")


def _documented_commands(readme: Path) -> list[str]:
    """Return the fenced command block that follows the `## Development` heading."""
    text = readme.read_text(encoding="utf-8")
    heading = re.search(r"^## Development$", text, re.MULTILINE)
    if heading is None:
        raise AssertionError(f"{readme.name} has no '## Development' section")

    fence = re.search(r"^```[a-z]*\n(.*?)^```", text[heading.end() :], re.MULTILINE | re.DOTALL)
    if fence is None:
        raise AssertionError(f"{readme.name} '## Development' section has no fenced command block")

    return [line.strip() for line in fence.group(1).splitlines() if line.strip()]


def _workflow_commands() -> list[str]:
    """Return every single-line `run:` command in the CI workflow, in order.

    Parsed with a regex rather than a YAML loader so the test adds no dependency
    the package does not otherwise need. Every step in this workflow uses the
    single-line `run: <command>` form; a block scalar would not be picked up, so
    the count assertion below fails rather than silently ignoring it.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    commands = [match.group(1).strip() for match in re.finditer(r"^\s*run:\s*(\S.*)$", text, re.MULTILINE)]
    if not commands:
        raise AssertionError(f"{WORKFLOW.name} declared no `run:` steps")
    return commands


def test_ci_workflow_exists() -> None:
    assert WORKFLOW.is_file(), f"{WORKFLOW} is missing; the READMEs point at it by name"


@pytest.mark.parametrize("readme_name", READMES)
def test_documented_commands_are_the_commands_ci_runs(readme_name: str) -> None:
    documented = _documented_commands(REPO_ROOT / readme_name)
    workflow = _workflow_commands()

    # The workflow guards the NLTK download behind a cache-hit condition, so it
    # runs conditionally there but is unconditional advice locally. Order is
    # otherwise identical, and every documented command must appear verbatim.
    assert documented == workflow, (
        f"{readme_name} says these are the commands CI runs, but they differ.\n"
        f"  documented: {documented}\n"
        f"  ci.yml:     {workflow}"
    )


@pytest.mark.parametrize("readme_name", READMES)
def test_documented_extras_spec_is_quoted(readme_name: str) -> None:
    """`.[dev]` unquoted is a glob in zsh, so the documented form must quote it."""
    for command in _documented_commands(REPO_ROOT / readme_name):
        if "pip install" in command:
            assert "'.[dev]'" in command or '".[dev]"' in command, (
                f"{readme_name} documents an unquoted extras spec: {command!r}"
            )
            break
    else:
        raise AssertionError(f"{readme_name} documents no pip install step")


@pytest.mark.parametrize("readme_name", READMES)
def test_documented_tools_run_under_the_current_interpreter(readme_name: str) -> None:
    """Bare `pytest`/`mypy` resolve off PATH, which need not be this interpreter."""
    for command in _documented_commands(REPO_ROOT / readme_name):
        first = command.split()[0]
        assert first == "python", f"{readme_name} documents {command!r}, which does not go through `python -m`"
