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


BLOCK_SCALAR = re.compile(r"^[|>][+-]?\d*$")


def _parse_run_steps(text: str) -> list[str]:
    """Return every single-line `run:` command in a workflow, in order.

    Parsed with a regex rather than a YAML loader so the test adds no dependency
    the package does not otherwise need. That parser understands only the
    single-line `run: <command>` form. A block scalar (`run: |`) is rejected here
    by name, because the regex would otherwise capture the bare `|` and compare
    it against the README as though it were a command -- a confusing failure
    that names the wrong culprit.
    """
    commands = [
        match.group(1).strip() for match in re.finditer(r"^[ \t]*(?:-[ \t]+)?run:[ \t]*(\S.*)$", text, re.MULTILINE)
    ]

    block_scalars = [command for command in commands if BLOCK_SCALAR.match(command)]
    if block_scalars:
        raise AssertionError(
            f"{WORKFLOW.name} uses a block scalar (`run: {block_scalars[0]}`), which this regex "
            "parser cannot read. Teach _parse_run_steps to fold block scalars before relying on it."
        )

    if not commands:
        raise AssertionError(f"{WORKFLOW.name} declared no `run:` steps")
    return commands


def _workflow_commands() -> list[str]:
    return _parse_run_steps(WORKFLOW.read_text(encoding="utf-8"))


def _conditional_run_steps(text: str) -> list[str]:
    """Return the `run:` commands whose step is guarded by an `if:` condition.

    The READMEs claim CI runs the documented block in order and differs only in
    when it downloads the corpora. That is a second claim about the workflow, so
    it is pinned here rather than left to rot alongside the first.
    """
    conditional = []
    for step in re.split(r"^[ \t]*-[ \t]", text, flags=re.MULTILINE)[1:]:
        run = re.search(r"^[ \t]*(?:-[ \t]+)?run:[ \t]*(\S.*)$", step, re.MULTILINE)
        if run is None:
            continue
        if re.search(r"^[ \t]*if:[ \t]*\S", step, re.MULTILINE):
            conditional.append(run.group(1).strip())
    return conditional


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


def test_single_line_run_steps_parse_in_document_order() -> None:
    workflow = "steps:\n  - name: Tests\n    run: python -m pytest\n  - name: Types\n    run: python -m mypy\n"
    assert _parse_run_steps(workflow) == ["python -m pytest", "python -m mypy"]


def test_a_run_step_written_as_a_list_entry_is_not_skipped() -> None:
    """`- run: ...` is legal YAML; missing it would drop a CI step from the parity check."""
    assert _parse_run_steps("steps:\n  - run: python -m pytest\n") == ["python -m pytest"]


@pytest.mark.parametrize("indicator", ["|", "|-", ">", ">-", "|2"])
def test_a_block_scalar_run_step_is_named_rather_than_misread(indicator: str) -> None:
    """Without this guard the bare indicator is captured and compared as a command."""
    workflow = f"steps:\n  - run: {indicator}\n      python -m pytest\n"
    with pytest.raises(AssertionError, match="block scalar"):
        _parse_run_steps(workflow)


def test_a_workflow_declaring_no_run_steps_is_rejected() -> None:
    with pytest.raises(AssertionError, match="no `run:` steps"):
        _parse_run_steps("steps:\n  - uses: actions/checkout@v4\n")


def test_the_corpora_download_is_the_only_conditional_command() -> None:
    """The READMEs say CI differs from the documented block only in this one step."""
    assert _conditional_run_steps(WORKFLOW.read_text(encoding="utf-8")) == [
        "python -m nltk.downloader wordnet omw-1.4"
    ]


def test_an_unguarded_step_is_not_reported_as_conditional() -> None:
    workflow = "steps:\n  - name: Tests\n    run: python -m pytest\n  - name: Types\n    run: python -m mypy\n"
    assert _conditional_run_steps(workflow) == []


def test_a_guarded_step_is_attributed_to_its_own_command() -> None:
    """An `if:` must not leak onto the neighbouring step's command."""
    workflow = (
        "steps:\n"
        "  - name: Download\n"
        "    if: steps.cache.outputs.cache-hit != 'true'\n"
        "    run: python -m nltk.downloader wordnet\n"
        "  - name: Tests\n"
        "    run: python -m pytest\n"
    )
    assert _conditional_run_steps(workflow) == ["python -m nltk.downloader wordnet"]


@pytest.mark.parametrize("readme_name", READMES)
def test_the_readme_does_not_claim_the_commands_run_verbatim(readme_name: str) -> None:
    """One step is cache-gated, so "verbatim" overstates what CI guarantees."""
    text = (REPO_ROOT / readme_name).read_text(encoding="utf-8")
    assert "verbatim" not in text, (
        f"{readme_name} claims the documented commands run verbatim, but "
        f"{_conditional_run_steps(WORKFLOW.read_text(encoding='utf-8'))} is conditional in CI"
    )
