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


def _development_section(readme: Path) -> str:
    """Return the `## Development` section's text, bounded by the next `##` heading.

    Shared so that every claim graded against the workflow is read from the same
    slice of the file. Without the bound the first match *anywhere* below the
    heading wins, so an unrelated section could either supply the command block
    or trip a check that was only ever about this one.
    """
    text = readme.read_text(encoding="utf-8")
    heading = re.search(r"^## Development$", text, re.MULTILINE)
    if heading is None:
        raise AssertionError(f"{readme.name} has no '## Development' section")

    section = text[heading.end() :]
    next_heading = re.search(r"^## ", section, re.MULTILINE)
    return section if next_heading is None else section[: next_heading.start()]


def _documented_commands(readme: Path) -> list[str]:
    """Return the fenced command block inside the `## Development` section."""
    fence = re.search(r"^```[a-z]*\n(.*?)^```", _development_section(readme), re.MULTILINE | re.DOTALL)
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


def _step_blocks(text: str) -> list[str]:
    """Split the workflow's `steps:` list into one chunk per step.

    Bounded to the `steps:` section and split only at the indentation of its own
    items. Splitting on any `- ` in the file would break a step apart at a nested
    list under `with:`, stranding its `if:` in one chunk and its `run:` in the
    next -- so a guarded command would be read as unguarded. Lists elsewhere in
    the workflow, such as a block-style `matrix.python-version`, would create
    spurious chunks for the same reason.
    """
    heading = re.search(r"^([ \t]*)steps:[ \t]*$", text, re.MULTILINE)
    if heading is None:
        raise AssertionError("workflow declares no `steps:` block")

    depth = len(heading.group(1))
    body_lines = []
    for line in text[heading.end() :].splitlines():
        if line.strip() and len(line) - len(line.lstrip()) <= depth:
            break
        body_lines.append(line)
    body = "\n".join(body_lines)

    item = re.search(r"^([ \t]*)-[ \t]", body, re.MULTILINE)
    if item is None:
        raise AssertionError("`steps:` block declares no steps")

    return re.split(rf"^{re.escape(item.group(1))}-[ \t]", body, flags=re.MULTILINE)[1:]


def _conditional_run_steps(text: str) -> list[str]:
    """Return the `run:` commands whose step is guarded by an `if:` condition.

    The READMEs claim CI runs the documented block in order and differs only in
    when it downloads the corpora. That is a second claim about the workflow, so
    it is pinned here rather than left to rot alongside the first.
    """
    conditional = []
    for step in _step_blocks(text):
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
    """One step is cache-gated, so "verbatim" overstates what CI guarantees.

    Scoped to the `## Development` section: the overclaim is specifically about
    this command block, and the word is unremarkable anywhere else.
    """
    section = _development_section(REPO_ROOT / readme_name)
    assert "verbatim" not in section, (
        f"{readme_name} claims the documented commands run verbatim, but "
        f"{_conditional_run_steps(WORKFLOW.read_text(encoding='utf-8'))} is conditional in CI"
    )


DEVELOPMENT_FENCE = "```text\npython -m pytest\n```\n"
LATER_SECTION_FENCE = "## Prior art\n\n```text\nsome other snippet\n```\n"


def test_a_fence_in_a_later_section_is_not_borrowed(tmp_path: Path) -> None:
    """An empty `## Development` section must fail, not reach into the next one."""
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n## Development\n\nProse only.\n\n{LATER_SECTION_FENCE}", encoding="utf-8")

    with pytest.raises(AssertionError, match="no fenced command block"):
        _documented_commands(readme)


def test_the_development_fence_is_read_rather_than_a_later_one(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n## Development\n\n{DEVELOPMENT_FENCE}\n{LATER_SECTION_FENCE}", encoding="utf-8")

    assert _documented_commands(readme) == ["python -m pytest"]


def test_a_readme_without_a_development_section_is_rejected(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n{LATER_SECTION_FENCE}", encoding="utf-8")

    with pytest.raises(AssertionError, match="no '## Development' section"):
        _documented_commands(readme)


def test_the_section_is_bounded_by_the_next_heading(tmp_path: Path) -> None:
    """Scoping is what keeps an unrelated section from answering for this one."""
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Title\n\n## Development\n\n{DEVELOPMENT_FENCE}\n## Prior art\n\nQuoted verbatim from the paper.\n",
        encoding="utf-8",
    )

    section = _development_section(readme)
    assert "verbatim" not in section
    assert "Prior art" not in section
    assert "python -m pytest" in section


def test_a_development_section_at_the_end_of_the_file_is_read_whole(tmp_path: Path) -> None:
    """With no following heading the section runs to EOF rather than coming back empty."""
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\n## Development\n\n{DEVELOPMENT_FENCE}", encoding="utf-8")

    assert _documented_commands(readme) == ["python -m pytest"]


BLOCK_STYLE_MATRIX_WORKFLOW = """jobs:
  check:
    strategy:
      matrix:
        python-version:
          - "3.11"
          - "3.12"
    steps:
      - uses: actions/checkout@v4
      - name: Download
        if: steps.cache.outputs.cache-hit != 'true'
        run: python -m nltk.downloader wordnet
      - name: Tests
        run: python -m pytest
"""

NESTED_LIST_WORKFLOW = """jobs:
  check:
    steps:
      - name: Restore
        if: github.event_name == 'push'
        with:
          paths:
            - ~/nltk_data
            - ~/.cache/pip
        run: python -m nltk.downloader wordnet
      - name: Tests
        run: python -m pytest
"""


def test_a_list_outside_the_steps_block_does_not_split_steps() -> None:
    """A block-style matrix is a list too; only the steps' own items may split."""
    assert _conditional_run_steps(BLOCK_STYLE_MATRIX_WORKFLOW) == ["python -m nltk.downloader wordnet"]
    assert _parse_run_steps(BLOCK_STYLE_MATRIX_WORKFLOW) == [
        "python -m nltk.downloader wordnet",
        "python -m pytest",
    ]


def test_a_nested_list_inside_a_step_does_not_strand_its_guard() -> None:
    """Splitting on any `- ` would separate this step's `if:` from its `run:`."""
    assert _conditional_run_steps(NESTED_LIST_WORKFLOW) == ["python -m nltk.downloader wordnet"]


def test_a_workflow_without_a_steps_block_is_rejected() -> None:
    with pytest.raises(AssertionError, match="no `steps:` block"):
        _conditional_run_steps("jobs:\n  check:\n    runs-on: ubuntu-latest\n")


def test_an_empty_steps_block_is_rejected() -> None:
    with pytest.raises(AssertionError, match="declares no steps"):
        _conditional_run_steps("jobs:\n  check:\n    steps:\n\n  other:\n    runs-on: x\n")
