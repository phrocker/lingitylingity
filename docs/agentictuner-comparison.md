# AgenticTuner Comparison

## Summary

AgenticTuner is a broad prompt trust and compliance service. Lingity should
reuse its strongest concepts--hybrid analysis, weighted scoring, refinement,
guardrails, diffing, and stability analysis--but solve a narrower problem with
stronger acceptance governance.

AgenticTuner evaluates whether a prompt is suitable and trustworthy. Lingity
governs whether an LLM-produced wording change may replace a source text.

## What AgenticTuner does

The implementation in `C:\Dev\AgenticTuner` provides:

- ATPL ratings for purpose, safety, compliance, provenance, and autonomy
- a deterministic weighted score over those five categories
- shallow rule-based linguistic checks using TextBlob and regular expressions
- optional LLM evaluation merged with the rule-based ratings
- one-shot LLM prompt refinement followed by reevaluation
- score and pattern guardrails
- prompt diff attribution
- randomized prompt-variant stability scoring
- policy, risk, integration, metering, authentication, and UI capabilities

Relevant implementation:

- `prompt_advisor/scoring.py`
- `prompt_advisor/llm_evaluator.py`
- `prompt_advisor/guardrails.py`
- `prompt_advisor/prompt_diff.py`
- `prompt_advisor/stability_scorer.py`
- `prompt_advisor/main.py`

## Comparison

| Concern | AgenticTuner | Lingity direction |
|---|---|---|
| Primary object | Prompts | Any governed human-facing text |
| Primary goal | Trust/compliance fitness | Clearer wording with preserved meaning |
| Deterministic analysis | Length, spelling, symbols, punctuation, weighted ATPL score | Located morphology, syntax, agency, structure, jargon, and redundancy rules |
| LLM role | Evaluator and one-shot refiner | Candidate generator and non-authoritative drift challenger |
| Acceptance | Returns refined prompt and its score | Code selects only candidates that pass hard invariants and improvement gates |
| Iteration | Caller-driven | Bounded, stateful, deterministic stopping policy |
| Semantic preservation | Prompt asks the model to maintain intent | Protected-element manifest plus drift challenge and human escalation |
| Provenance | Query and metering history | Immutable source, attempt ledger, hashes, profile/model versions, and dispositions |
| Failure behavior | Several paths return neutral/default results or the original prompt | Explicit rejected, unchanged, failed, or needs-human terminal state |
| Stability | Randomized textual perturbations scored by the evaluator | Seeded or exhaustive tests with recorded variant set and rule versions |
| Scope | Full hosted service with auth, policy, integrations, UI, and persistence | Initially a small library and CLI with provider adapters |

## Useful ideas to carry forward

1. Separate category ratings from the aggregate score.
2. Support versioned, configurable weights.
3. Report why a score changed between source and candidate.
4. Keep deterministic guardrails separate from LLM evaluation.
5. Analyze stability rather than trusting a single evaluation.
6. Return structured recommendations and typed findings.

## Gaps Lingity must close

### Linguistic depth

AgenticTuner's rule layer checks sentence length, spelling corrections,
non-ASCII characters, symbols, and punctuation ratios. It does not currently
measure the morphology that made the architecture recommendation sound
inhuman: nominalization density, noun stacks, weak predicates, hidden agency,
or too many actions in one sentence.

### Score alignment

The public scoring engine calculates the aggregate from the five ATPL
categories. Linguistic ratings produced by the hybrid evaluator are not part
of that aggregate. Lingity needs one versioned calculation in which every
published dimension has a defined contribution and hard gates remain outside
the score.

### Governed replacement

AgenticTuner's `/refine_prompt` flow asks the LLM to preserve intent, then
scores the returned text. It does not compare protected facts, obligations,
negation, actors, quantities, evidence references, or governance status before
returning the rewrite.

### Iteration and audit

AgenticTuner supports reevaluation and diffing but does not own an immutable,
bounded sequence of rewrite attempts with deterministic acceptance and
stopping. Lingity should make this the central runtime abstraction.

### Failure semantics

Governed improvement cannot use neutral ratings or an unchanged source as an
implicit success. Failure and uncertainty must remain visible and auditable.

## Relationship

Lingity should be designed as a focused engine rather than a fork of
AgenticTuner. A future adapter could allow AgenticTuner to call Lingity for
governed refinement while retaining its broader policy, account, integration,
and UI capabilities.

