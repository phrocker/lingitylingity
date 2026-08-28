# Lingity

Lingity currently provides deterministic, governed text analysis for
LLM-authored or human-authored content. This milestone ships two local CLI
commands: `analyze` produces a reproducible JSON analysis artifact, and
`verify` validates and replays that artifact.

There are no network calls or LLM calls in the current runtime. Provider
interfaces and schemas exist for planned rewrite-proposal and semantic-drift
adapters, but they are not invoked by `analyze` or `verify`.

## Core contract

1. The source text is immutable.
2. Every score is reproducible from a versioned profile and analyzer.
3. The current runtime must not emit success-shaped fallback results.
4. Protected facts, identifiers, quantities, modality, negation, citations,
   ownership, and governance status must survive unchanged.
5. Planned rewrite candidates must improve the configured linguistic
   thresholds without introducing a hard-gate violation.
6. Planned rewrite runs must surface semantic uncertainty as `needs_human`,
   not disguise it as success.

## Current workflow

```text
source text
  -> deterministic analysis
  -> schema-valid analysis artifact
  -> deterministic verification replay
```

## Planned workflow

```text
source text
  -> deterministic analysis
  -> bounded LLM proposal
  -> invariant validation
  -> deterministic rescoring
  -> semantic-drift challenge
  -> accept, iterate, reject, or require human review
```

The first target profile is `architecture-review`: recommendations, ADR
summaries, findings, risks, and review decisions that must remain precise while
reading like professional human communication.

## CLI

```text
lingity analyze review.md --profile architecture-review
lingity verify analysis.json
```

`analyze` emits a deterministic, schema-valid JSON artifact containing located
findings, the attributed Human Readability Index, protected-element manifests,
and content/profile hashes. `verify` validates the schema and hashes, resolves
the recorded profile, and replays the analysis; altered or non-reproducible
artifacts fail explicitly.

The current analyzer is a versioned English dependency-parse model covering
every deterministic signal published in the [DESIGN.md](DESIGN.md) dimension
table:

- **Sentence load** — words, clauses, punctuation depth, and actions per
  sentence.
- **Morphology** — nominalization density and weak verb constructions.
- **Noun stacking** — consecutive noun modifiers and hyphenated compound depth.
- **Agency** — agentless directives and missing explicit actor-action pairs.
- **Voice** — passive constructions and indirect predicates. Passive detection
  is structural: it requires a passive auxiliary or passive subject relation
  (`auxpass`/`nsubjpass`), so the active perfect (`has expired`) can never be
  mistaken for the passive (`has been approved`).
- **Lexical clarity** — jargon, uncommon compounds, and undefined abbreviation
  density.
- **Structure** — paragraph length, list suitability, and mixed-purpose
  sentences.
- **Redundancy** — repeated qualifiers, duplicated recommendations, and filler
  phrases.

Noun stacking findings are reported under the `morphology` dimension and voice
findings under `agency`, so the score always resolves to the six weighted
dimensions.

Every finding carries a rule ID, severity, character location, observed value,
threshold, and remediation. Overlapping spans within a dimension are
de-duplicated so a single defect is not penalised twice.

Rules read sentence structure — predicates, subjects, auxiliaries, negation,
and modifier chains — rather than matching surface strings, so detection
generalises to unseen wording. Because the parse is part of the analysis
contract, the pipeline is pinned: `en_core_web_sm` at an exact version, loaded
fail-closed. Its name, version, runtime, and digest are recorded as
`linguistic_model` inside the hashed artifact, and `verify` refuses any
artifact produced by a different pipeline instead of silently re-analysing it.

The Human Readability Index weights those six dimensions and converts each
dimension's deducted points into a score with a half-life decay, so worse text
never scores higher than better text. The exact arithmetic is published in the
artifact's `score.formula` field.

Provider protocols exist for future proposal and semantic-challenge adapters,
but this milestone performs no network or LLM calls.

See [DESIGN.md](DESIGN.md), the
[AgenticTuner comparison](docs/agentictuner-comparison.md), and the
[implementation plan](docs/implementation-plan.md).

## Development

```text
python -m pip install -e .[dev]
pytest
mypy
python -m compileall -q lingity tests
```
