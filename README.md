# Lingity

Lingity is a governed text-improvement engine for LLM-authored content.

It combines deterministic linguistic analysis with bounded LLM rewrite
proposals. The LLM may propose clearer wording, but deterministic code controls
which attempt is accepted, when iteration stops, and whether protected meaning
has changed.

## Core contract

1. The source text is immutable.
2. Every score is reproducible from a versioned profile and analyzer.
3. Every LLM rewrite is an append-only attempt with input and output hashes.
4. Protected facts, identifiers, quantities, modality, negation, citations,
   ownership, and governance status must survive unchanged.
5. A candidate must improve the configured linguistic thresholds without
   introducing a hard-gate violation.
6. The system never silently accepts semantic uncertainty. It selects the
   original text or requires human review when no candidate passes.

## Intended workflow

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

## Planned CLI

```text
lingity analyze review.md --profile architecture-review
lingity improve review.md --profile architecture-review --max-attempts 3
lingity verify .lingity/runs/<run-id>/record.json
```

See [DESIGN.md](DESIGN.md), the
[AgenticTuner comparison](docs/agentictuner-comparison.md), and the
[implementation plan](docs/implementation-plan.md).

## Status

The repository currently contains the initial design and implementation plan.

