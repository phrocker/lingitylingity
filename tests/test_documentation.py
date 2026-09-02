"""Claims the documentation makes about other files must stay true.

README.md and README_AI.md both state that the block under `## Development` is
what CI runs. That is a claim about another file, so it can rot silently the
moment either side is edited alone -- which is exactly what happened before this
test existed. Pinning it here turns the claim into something that fails loudly.

The same is true of the profile count. Two branches independently wrote "Three
profiles ship" and the merge that landed a fourth made both wrong, without
touching either sentence.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import cast

import pytest
from trove_classifiers import classifiers as trove_classifiers

from lingity.invariants import compare_protected, extract_protected
from lingity.nlp import MODEL_NAME, MODEL_VERSION
from lingity.profiles import load_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
READMES = ("README.md", "README_AI.md")
PROFILE_DIR = REPO_ROOT / "lingity" / "profiles"
PROFILE_COUNT_CLAIM = re.compile(r"\b(\w+) profiles ship\b", re.IGNORECASE)
NUMBER_WORDS = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
    "Six": 6,
    "Seven": 7,
    "Eight": 8,
    "Nine": 9,
    "Ten": 10,
}


def _shipped_profiles() -> list[str]:
    """Return the name of every profile that ships, without its version suffix."""
    profiles = sorted(path.name.split(".")[0] for path in PROFILE_DIR.glob("*.json"))
    if not profiles:
        raise AssertionError(f"{PROFILE_DIR} contains no profiles")
    return profiles


def _documented_profile_count(document: Path) -> int:
    """Return the profile count `document` claims, as a number.

    Read case-insensitively throughout: README.md capitalises the claim at a line
    start and DESIGN.md states it mid-sentence, and a document that capitalised
    more of the phrase would otherwise be reported as making no claim at all --
    a failure raised against a document that does state one.

    A word this function cannot read is reported by name rather than skipped,
    because a claim the test cannot parse is a claim the test is not checking.
    """
    text = document.read_text(encoding="utf-8")
    claim = PROFILE_COUNT_CLAIM.search(text)
    if claim is None:
        raise AssertionError(
            f"{document.name} no longer states how many profiles ship. Either restore the "
            "claim or drop it from PROFILE_COUNT_DOCUMENTS, rather than leaving it unchecked."
        )

    word = claim.group(1)
    if word.isdigit():
        return int(word)
    if word.capitalize() not in NUMBER_WORDS:
        raise AssertionError(
            f"{document.name} says '{word} profiles ship' and this test cannot read "
            f"'{word}' as a number. Add it to NUMBER_WORDS."
        )
    return NUMBER_WORDS[word.capitalize()]


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
RUN_LINE = re.compile(r"^[ \t]*(?:-[ \t]+)?run:[ \t]*(\S.*)$", re.MULTILINE)


def _strip_inline_comment(value: str) -> str:
    """Drop a YAML inline comment from a plain scalar.

    YAML starts a comment at a `#` preceded by whitespace, so `run: pytest  # smoke`
    is the command `pytest`. Taking the rest of the line instead would compare the
    comment against the README and fail while naming the wrong culprit.

    Quote state is tracked because a `#` inside quotes is content, not a comment:
    `run: echo "a # b"` is one command. A blind split on `#` would truncate it and
    then report a mismatch the workflow does not have.
    """
    quote = ""
    for index, char in enumerate(value):
        if quote:
            if char == quote:
                quote = ""
        elif char in "\"'":
            quote = char
        elif char == "#" and index > 0 and value[index - 1] in " \t":
            return value[:index].strip()
    return value.strip()


def _run_commands(text: str) -> list[str]:
    """Return every single-line `run:` command in `text`, in order."""
    return [_strip_inline_comment(match.group(1)) for match in RUN_LINE.finditer(text)]


def _parse_run_steps(text: str) -> list[str]:
    """Return every single-line `run:` command in a workflow, in order.

    Parsed with a regex rather than a YAML loader so the test adds no dependency
    the package does not otherwise need. That parser understands only the
    single-line `run: <command>` form. A block scalar (`run: |`) is rejected here
    by name, because the regex would otherwise capture the bare `|` and compare
    it against the README as though it were a command -- a confusing failure
    that names the wrong culprit.
    """
    commands = _run_commands(text)

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
    """Split every `steps:` list in the workflow into one chunk per step.

    Bounded to each `steps:` section and split only at the indentation of that
    section's own items. Splitting on any `- ` in the file would break a step
    apart at a nested list under `with:`, stranding its `if:` in one chunk and
    its `run:` in the next -- so a guarded command would be read as unguarded.
    Lists elsewhere, such as a block-style `matrix.python-version`, would create
    spurious chunks for the same reason.

    Every job is read, not just the first. `_parse_run_steps` already collects
    `run:` commands from the whole file, so stopping at one job here would let a
    second job's conditional steps go unseen while its commands were still being
    graded.
    """
    blocks: list[str] = []
    for heading in re.finditer(r"^([ \t]*)steps:[ \t]*$", text, re.MULTILINE):
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

        blocks.extend(re.split(rf"^{re.escape(item.group(1))}-[ \t]", body, flags=re.MULTILINE)[1:])

    if not blocks:
        raise AssertionError("workflow declares no `steps:` block")
    return blocks


def _conditional_run_steps(text: str) -> list[str]:
    """Return the `run:` commands whose step is guarded by an `if:` condition.

    The READMEs claim CI runs the documented block in order and differs only in
    when it downloads the corpora. That is a second claim about the workflow, so
    it is pinned here rather than left to rot alongside the first.
    """
    conditional = []
    for step in _step_blocks(text):
        run = RUN_LINE.search(step)
        if run is None:
            continue
        if re.search(r"^[ \t]*if:[ \t]*\S", step, re.MULTILINE):
            conditional.append(_strip_inline_comment(run.group(1)))
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


TWO_JOB_WORKFLOW = """jobs:
  check:
    steps:
      - name: Tests
        run: python -m pytest
  publish:
    steps:
      - name: Upload
        if: github.ref == 'refs/heads/main'
        run: python -m twine upload dist/*
"""


def test_a_second_job_is_read_too() -> None:
    """`_parse_run_steps` grades every job, so guards must be read from every job.

    Stopping at the first `steps:` block would let a later job's conditional
    command be graded as unconditional -- the same under-report as a stranded
    guard, reached a different way.
    """
    assert _conditional_run_steps(TWO_JOB_WORKFLOW) == ["python -m twine upload dist/*"]
    assert _parse_run_steps(TWO_JOB_WORKFLOW) == ["python -m pytest", "python -m twine upload dist/*"]


COMMENTED_WORKFLOW = """
jobs:
  build:
    steps:
      - name: Tests
        run: python -m pytest  # smoke only
      - name: Echo
        run: echo "a # b"
      - name: Corpora
        if: steps.cache.outputs.cache-hit != 'true'
        run: python -m nltk.downloader wordnet  # first run only
"""


def test_an_inline_comment_is_not_part_of_the_command() -> None:
    """YAML ends a plain scalar at a ` #`, so the comment is not the command.

    Capturing it would compare `python -m pytest  # smoke only` against the
    README's `python -m pytest` and fail -- a real mismatch reported against a
    workflow that matches, naming the wrong culprit.
    """
    assert _parse_run_steps(COMMENTED_WORKFLOW)[0] == "python -m pytest"


def test_a_quoted_hash_is_content_not_a_comment() -> None:
    """A `#` inside quotes is part of the command, so stripping must be quote-aware.

    This is the failure a blind split on `#` would introduce while fixing the
    other one: it would truncate this command to `echo "a` and report a mismatch
    the workflow does not have.
    """
    assert _parse_run_steps(COMMENTED_WORKFLOW)[1] == 'echo "a # b"'


def test_the_conditional_reader_strips_comments_too() -> None:
    """Both parsers read the same `run:` line, so both must read it the same way.

    They were separate regexes with separate strips; a fix applied to one and not
    the other would leave the conditional check comparing a commented command.
    """
    assert _conditional_run_steps(COMMENTED_WORKFLOW) == ["python -m nltk.downloader wordnet"]


PROFILE_COUNT_DOCUMENTS = ("README.md", "DESIGN.md")


@pytest.mark.parametrize("name", PROFILE_COUNT_DOCUMENTS)
def test_the_documented_profile_count_matches_the_profiles_that_ship(name: str) -> None:
    """A stated count drifts whenever a profile lands, and nothing else has to change.

    Two branches independently wrote "Three profiles ship"; the merge that added
    a fourth falsified both sentences without editing either. Neither branch had
    done anything wrong on its own, which is why review did not catch it -- the
    claim only became false in the merge.
    """
    document = REPO_ROOT / name
    shipped = _shipped_profiles()

    assert _documented_profile_count(document) == len(shipped), (
        f"{name} states a profile count that no longer matches the "
        f"{len(shipped)} profiles in lingity/profiles: {', '.join(shipped)}"
    )


def test_every_shipped_profile_is_named_in_the_readme() -> None:
    """The count alone does not catch a rename, which leaves it right and the prose wrong.

    README.md introduces each profile by name, so a profile renamed or replaced
    keeps the count correct while the paragraph describes something that is no
    longer there -- and a reader following the README would load a profile that
    does not exist.
    """
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    missing = [profile for profile in _shipped_profiles() if f"`{profile}`" not in text]

    assert not missing, (
        f"README.md never names {', '.join(missing)}, so a shipped profile is undocumented"
    )


@pytest.mark.parametrize(
    "claim",
    ["Four profiles ship.", "four profiles ship.", "Four Profiles Ship.", "FOUR PROFILES SHIP."],
)
def test_the_count_claim_is_found_however_it_is_capitalised(claim: str, tmp_path: Path) -> None:
    """Capitalisation is a prose choice, not a change to what the document claims.

    A case-sensitive pattern reports a document that capitalises more of the
    phrase as making no claim at all, which raises "no longer states how many
    profiles ship" against a document that plainly does -- a false failure that
    sends the reader looking for a deleted sentence still sitting in the file.
    """
    document = tmp_path / "DOC.md"
    document.write_text(f"Intro line. {claim}\n", encoding="utf-8")

    assert _documented_profile_count(document) == 4


@pytest.mark.parametrize(
    "sentence",
    [
        "Four profiles shipped in the first release.",
        "Four profiles shipping in the next release.",
        "Four profiles ships nothing.",
    ],
)
def test_a_word_that_merely_starts_with_ship_is_not_the_count_claim(
    sentence: str, tmp_path: Path
) -> None:
    """The claim is what ships now, not what shipped once.

    Without a boundary after "ship" the pattern also matches "shipped" and
    "shipping", so a sentence recording past or planned releases would be read
    as the current count -- and because the first match wins, such a sentence
    earlier in the file would hide the real claim entirely.
    """

    document = tmp_path / "DOC.md"
    document.write_text(f"{sentence}\nFive profiles ship.\n", encoding="utf-8")

    assert _documented_profile_count(document) == 5




JUDGE_EXAMPLE = re.compile(r"\$ lingity judge.*?\n(.*?)```", re.DOTALL)
DROPPED_COUNT = re.compile(r"(\d+) protected element\(s\) dropped")
MISSING_LINE = re.compile(r"^\s*MISSING (\S.*?)\s*$", re.MULTILINE)
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "recommended-decision.json"


@pytest.mark.parametrize("name", READMES)
def test_the_documented_rejection_lists_elements_the_gate_really_reports(
    name: str,
) -> None:
    """The sample rejection must be output the code can still produce.

    This example went stale twice without anyone noticing: once when a
    governance term stopped being reported, and once when ordering relations
    began reading from the governing clause, which changed the shape of every
    `order:sequence` signature. Both times the block kept claiming an output
    the gate could no longer emit, because nothing compared it to a real run.
    """
    text = (REPO_ROOT / name).read_text(encoding="utf-8")
    block = JUDGE_EXAMPLE.search(text)
    assert block is not None, f"{name} no longer shows a judge rejection to check"
    body = block.group(1)

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    profile = load_profile()
    comparison = compare_protected(
        extract_protected(fixture["original"], profile),
        extract_protected(fixture["unfaithful_rewrite"], profile),
    )
    missing = cast(list[str], comparison["missing"])

    documented = MISSING_LINE.findall(body)
    assert documented, f"{name} shows a rejection that names no dropped element"
    for element in documented:
        assert element in missing, (
            f"{name} says the gate reports {element!r}, but it reports {missing!r}"
        )

    count = DROPPED_COUNT.search(body)
    assert count is not None, f"{name} no longer states how many elements were dropped"
    assert int(count.group(1)) == len(missing), (
        f"{name} claims {count.group(1)} dropped element(s); the gate reports {len(missing)}"
    )


PYPROJECT = REPO_ROOT / "pyproject.toml"
# PEP 508 marks a direct reference with `@` separating the name from a URL. The
# space before it is optional in practice (`name@git+https://...` parses), and
# the `//` is not required either: `name @ file:../wheels/x.whl` is a direct
# reference with a scheme and a relative path, and PyPI rejects it on the same
# terms. Matching on the scheme's colon rather than on `://` covers both. An
# ordinary specifier carries no `@` at all, so this cannot collide with one.
DIRECT_REFERENCE = re.compile(r"@\s*[a-zA-Z][a-zA-Z0-9+.\-]*:")
# The trailing filename is matched in full, not just up to the version. Ending
# the pattern at the version's hyphen accepted any artifact that happened to
# start the right way: an ABI-specific wheel, a `.zip`, or a bare truncated URL
# all satisfied it. Those fail at install time rather than at documentation
# time, which is the failure this guard exists to move earlier.
MODEL_WHEEL_URL = re.compile(
    r"https://github\.com/explosion/spacy-models/releases/download/"
    r"(?P<name>[a-z_]+)-(?P<tag>[0-9.]+)/(?P=name)-(?P<version>[0-9.]+)"
    r"-py3-none-any\.whl(?![\w.\-])"
)


def _documented_model_command(commands: list[str]) -> str:
    """Return the one documented command that installs the linguistic model."""
    matches = [command for command in commands if "spacy-models" in command]
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one command installing the linguistic model, found {matches!r}"
        )
    return matches[0]


@pytest.mark.parametrize("readme_name", READMES)
def test_the_documented_model_wheel_matches_the_version_the_loader_requires(
    readme_name: str,
) -> None:
    """The pinned model version is stated in four files and enforced in one.

    `lingity/nlp.py` refuses any model but `MODEL_VERSION`, so a README or CI
    line naming a different wheel does not degrade the analysis -- it fails
    closed at load time. It does something worse: it tells a reader to install
    the exact thing that will be rejected, and the error names the loader rather
    than the instruction that caused it. Pinning the URL against the constant
    keeps the advice and the enforcement in step.
    """
    command = _documented_model_command(_documented_commands(REPO_ROOT / readme_name))

    url = MODEL_WHEEL_URL.search(command)
    assert url is not None, f"{readme_name} installs a model from an unrecognised URL: {command!r}"
    assert url.group("name") == MODEL_NAME, (
        f"{readme_name} installs {url.group('name')!r}, but the loader requires {MODEL_NAME!r}"
    )
    assert url.group("version") == MODEL_VERSION == url.group("tag"), (
        f"{readme_name} installs {MODEL_NAME} {url.group('version')} from release tag "
        f"{url.group('tag')}, but lingity/nlp.py requires exactly {MODEL_VERSION}"
    )


def test_ci_installs_the_same_model_wheel_the_readmes_document() -> None:
    """Parity already covers this, but only while the command stays in the block."""
    command = _documented_model_command(_workflow_commands())

    url = MODEL_WHEEL_URL.search(command)
    assert url is not None, f"ci.yml installs a model from an unrecognised URL: {command!r}"
    assert (url.group("name"), url.group("version")) == (MODEL_NAME, MODEL_VERSION), (
        f"ci.yml installs {url.group('name')} {url.group('version')}, but lingity/nlp.py "
        f"requires {MODEL_NAME} {MODEL_VERSION}"
    )


def _string_list(name: str, value: object) -> list[str]:
    """Return `value` as a list of strings, refusing to guess at anything else.

    Reading a malformed table as though it were well formed is the one failure
    these guards must not have. A `dependencies` key holding a bare string is
    iterable, so a scan would walk it character by character, match nothing, and
    report the file clean -- the guard would be loudest about safety at exactly
    the moment it had stopped looking. Failing here instead keeps a shape
    problem from being reported as an absence of findings, and keeps the
    message about the shape rather than about forty stray characters.
    """
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AssertionError(
            f"pyproject.toml declares {name} as {type(value).__name__}, not a list of "
            "strings. This guard cannot read it, so it cannot vouch for it."
        )
    return value


def test_no_dependency_is_declared_as_a_direct_url() -> None:
    """A public index rejects any distribution whose metadata carries one.

    `en_core_web_sm` is not on PyPI, so declaring it here needs a PEP 440 direct
    reference -- and PyPI answers `400 Can't have direct dependency`. Nothing
    local catches it: the build succeeds, `pip install` succeeds, and `twine
    check` passes because it validates only README rendering. The first signal
    would be the upload itself, so the guard belongs here.

    Every declared dependency list is checked, not just the required one. An
    extra contributes `Requires-Dist: name @ url; extra == "..."` to the same
    metadata field and is rejected on the same terms, so a direct reference
    parked in `[project.optional-dependencies]` would fail an upload exactly as
    a required one does.
    """
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]

    groups: dict[str, list[str]] = {
        "dependencies": _string_list("dependencies", project.get("dependencies", []))
    }
    extras = project.get("optional-dependencies", {})
    assert isinstance(extras, dict), (
        f"pyproject.toml declares optional-dependencies as {type(extras).__name__}, "
        "not a table. This guard cannot read it, so it cannot vouch for it."
    )
    for extra, requirements in extras.items():
        name = f"optional-dependencies.{extra}"
        groups[name] = _string_list(name, requirements)

    direct = [
        f"{group}: {requirement}"
        for group, requirements in groups.items()
        for requirement in requirements
        if DIRECT_REFERENCE.search(requirement)
    ]

    assert not direct, (
        "pyproject.toml declares a direct URL dependency, which PyPI rejects at upload: "
        f"{direct}. Install it as a documented step instead."
    )


@pytest.mark.parametrize(
    "requirement",
    [
        'en_core_web_sm @ https://example.invalid/en_core_web_sm-3.8.0-py3-none-any.whl',
        "lingity@git+https://github.com/phrocker/lingitylingity",
        "thing @ file:///tmp/thing-1.0-py3-none-any.whl",
        "thing @ file:../wheels/thing-1.0-py3-none-any.whl",
    ],
)
def test_a_direct_reference_is_recognised_in_any_form(requirement: str) -> None:
    """The guard is worthless if it only matches the exact line that was removed.

    The `file:` form without `//` is included deliberately: a scheme followed by
    a relative path is still a direct reference, and a matcher keyed on `://`
    would let it through while reporting the file clean.
    """
    assert DIRECT_REFERENCE.search(requirement) is not None


@pytest.mark.parametrize("requirement", ["spacy>=3.8,<3.9", "nltk>=3.9,<4", 'mypy<2,>=1.13'])
def test_an_ordinary_requirement_is_not_mistaken_for_a_direct_reference(
    requirement: str,
) -> None:
    assert DIRECT_REFERENCE.search(requirement) is None


MODEL_RELEASE = (
    "https://github.com/explosion/spacy-models/releases/download/"
    f"{MODEL_NAME}-{MODEL_VERSION}/{MODEL_NAME}-{MODEL_VERSION}"
)


@pytest.mark.parametrize(
    "url",
    [
        f"{MODEL_RELEASE}-cp311-cp311-win_amd64.whl",
        f"{MODEL_RELEASE}-py3-none-any.zip",
        f"{MODEL_RELEASE}-py3-none-any.whl.asc",
        f"{MODEL_RELEASE}-",
        f"{MODEL_RELEASE}.tar.gz",
    ],
)
def test_an_artifact_that_is_not_the_pinned_wheel_is_rejected(url: str) -> None:
    """Naming the right version is not the same as naming the right artifact.

    Every URL here carries the correct model name and version, so a matcher that
    stops at the version accepts all of them. They install something other than
    the pinned universal wheel -- a platform build, a signature, a source
    archive, or nothing at all -- and the mismatch surfaces as a download
    failure in CI rather than as a documentation defect here.
    """
    assert MODEL_WHEEL_URL.search(url) is None


def test_the_pinned_wheel_url_is_accepted() -> None:
    """The negative cases above prove nothing if the real URL fails too."""
    match = MODEL_WHEEL_URL.search(f"{MODEL_RELEASE}-py3-none-any.whl")

    assert match is not None
    assert (match.group("name"), match.group("version")) == (MODEL_NAME, MODEL_VERSION)


@pytest.mark.parametrize(
    "value",
    [
        "en_core_web_sm @ https://example.invalid/en_core_web_sm-3.8.0-py3-none-any.whl",
        ["spacy>=3.8,<3.9", 3],
        {"spacy": ">=3.8"},
        None,
    ],
)
def test_a_malformed_dependency_table_is_refused_rather_than_scanned(value: object) -> None:
    """A shape this guard cannot read must not be reported as clean.

    The string case is the dangerous one: it carries a direct reference and it
    is iterable, so scanning it walks single characters, matches nothing, and
    returns an empty finding list. The guard would then pass while the very
    thing it exists to catch sat in the file.
    """
    with pytest.raises(AssertionError, match="cannot vouch for it"):
        _string_list("dependencies", value)


def test_a_well_formed_dependency_table_is_returned_unchanged() -> None:
    """Refusing everything would satisfy the test above and guard nothing."""
    requirements = ["spacy>=3.8,<3.9", "nltk>=3.9,<4"]

    assert _string_list("dependencies", requirements) == requirements
    assert _string_list("optional-dependencies.dev", []) == []


def test_every_declared_classifier_is_a_real_trove_classifier() -> None:
    """PyPI validates classifiers at upload and rejects any it does not know.

    This is the same shape as the direct-reference blocker: the build succeeds,
    the install succeeds, and `twine check` passes, because none of them consult
    the classifier list. A typo would surface for the first time as a rejected
    upload. `trove-classifiers` is the canonical list PyPI validates against, so
    checking membership here answers the question locally.
    """
    classifiers = _string_list(
        "classifiers",
        tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"].get("classifiers", []),
    )
    assert classifiers, "pyproject.toml declares no classifiers"

    unknown = [name for name in classifiers if name not in trove_classifiers]

    assert not unknown, (
        f"pyproject.toml declares classifiers PyPI does not recognise: {unknown}. "
        "An unknown classifier is rejected at upload."
    )


# A licence identifier is a legal statement about the whole distribution, and
# nothing in the toolchain checks that it describes the file shipped beside it.
# `twine check`, `build` and `pip` all accept `license = "MIT"` over an Apache
# text without comment. Each identifier is therefore mapped to phrases only its
# own text contains, so the declaration is checked against the licence rather
# than against itself.
LICENCE_MARKERS = {
    "Apache-2.0": (
        "Apache License",
        "Version 2.0, January 2004",
        "http://www.apache.org/licenses/",
        'distributed on an "AS IS" BASIS',
    ),
    "MIT": ("MIT License", "Permission is hereby granted, free of charge"),
}
# What the README must state as the project's own terms. Kept separate from the
# markers above because that map answers "is this file that licence?" while this
# one answers "does the prose say so?", and the two are checked against
# different text.
LICENCE_README_PHRASES = {
    "Apache-2.0": "Apache License 2.0",
    "MIT": "MIT License",
}


def test_the_licence_text_is_the_licence_that_is_declared() -> None:
    """The identifier and the file are two claims that must agree.

    They are written in different places by different edits, and a mismatch is
    not a build failure -- it is a distribution that tells users they may do
    something the bundled terms do not permit. No packaging tool compares them.
    """
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    declared = project.get("license")

    assert isinstance(declared, str) and declared, (
        "pyproject.toml declares no SPDX license expression. Without one the "
        "distribution states no terms, and default copyright forbids use."
    )
    assert declared in LICENCE_MARKERS, (
        f"pyproject.toml declares {declared!r}, which this guard has no text to "
        f"check it against. Add its markers to LICENCE_MARKERS: {sorted(LICENCE_MARKERS)}"
    )

    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    missing = [marker for marker in LICENCE_MARKERS[declared] if marker not in text]

    assert not missing, (
        f"pyproject.toml declares {declared!r}, but LICENSE is missing text that "
        f"licence contains: {missing}. The identifier and the file disagree."
    )


def test_every_declared_licence_file_is_shipped() -> None:
    """`license-files` names what the build copies into the distribution."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    declared = _string_list("license-files", project.get("license-files", []))

    assert declared, "pyproject.toml declares no license-files"

    missing = [name for name in declared if not (REPO_ROOT / name).is_file()]

    assert not missing, (
        f"pyproject.toml lists license files that do not exist: {missing}. "
        "The build would ship a licence claim with nothing behind it."
    )


def test_no_licence_classifier_accompanies_the_licence_expression() -> None:
    """Each is valid alone; together they are a hard build error.

    `test_every_declared_classifier_is_a_real_trove_classifier` cannot catch
    this, because `License :: OSI Approved :: Apache Software License` is a
    perfectly real classifier. PEP 639 superseded it, and setuptools refuses to
    build when both are present -- so the failure lands as an
    `InvalidConfigError` from the backend rather than as anything naming the
    combination that caused it.
    """
    classifiers = _string_list(
        "classifiers",
        tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"].get("classifiers", []),
    )

    licence_classifiers = [name for name in classifiers if name.startswith("License ::")]

    assert not licence_classifiers, (
        f"pyproject.toml declares both a license expression and {licence_classifiers}. "
        "PEP 639 superseded license classifiers and setuptools refuses to build "
        "with both. Keep the expression and drop the classifier."
    )


@pytest.mark.parametrize("readme_name", READMES)
def test_the_readmes_name_the_licence_that_is_declared(readme_name: str) -> None:
    """A reader decides whether they may use this from the README, not the metadata.

    The check is anchored to the opening statement of the `## License` section
    rather than to the whole file. A bare substring search over the README
    passes on any mention anywhere -- and this README names the MIT and WordNet
    licences of the two data artifacts, so searching the file for `MIT` would
    have reported a MIT declaration as correctly documented while the section
    said Apache. The claim being checked is what the project states as its own
    terms, so that is the line to read.
    """
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    declared = cast(str, project["license"])
    assert declared in LICENCE_README_PHRASES, (
        f"pyproject.toml declares {declared!r}, which this guard has no README "
        f"phrase for. Add one to LICENCE_README_PHRASES: {sorted(LICENCE_README_PHRASES)}"
    )

    text = (REPO_ROOT / readme_name).read_text(encoding="utf-8")
    section = re.search(r"^## License\s*\n(.*?)(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    assert section is not None, f"{readme_name} has no `## License` section"

    opening = next(
        (line for line in section.group(1).splitlines() if line.strip()),
        "",
    )
    phrase = LICENCE_README_PHRASES[declared]

    assert phrase in opening, (
        f"{readme_name} opens its License section with {opening.strip()!r}, which does "
        f"not state {phrase!r} as declared in pyproject.toml."
    )
