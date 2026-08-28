# Implementation Plan

## Product slice

Build a Python library and CLI that can deterministically analyze text, request
up to three LLM rewrite candidates, reject semantic or governance drift, and
select an improved human-facing projection.

The first profile is `architecture-review`.

## Phase 0: contracts and corpus

Deliver:

- JSON Schemas for analysis, profile, attempt, challenge, and run records
- a versioned `architecture-review` profile
- a labeled corpus of human and machine-like architecture-review sentences
- expected metric values and acceptance outcomes for every corpus case

Exit criteria:

- schemas validate under JSON Schema Draft 2020-12
- each score can be reproduced from the fixture without an LLM
- corpus includes modality, negation, quantities, IDs, evidence references,
  ownership, uncertainty, and ratification examples

## Phase 1: deterministic analyzer

Implement:

- sentence and token spans
- clause and action density
- nominalization detection
- noun-stack detection
- passive and hidden-agency detection
- weak-verb and bureaucratic-phrase rules
- glossary and jargon checks
- paragraph and list-suitability checks
- score calculation with complete attribution

Exit criteria:

- identical input, profile, package version, and linguistic-model digest produce
  byte-equivalent analysis JSON
- every deduction has a rule ID and source span
- tests pin all thresholds and arithmetic

## Phase 2: protected-element manifest

Implement extraction and comparison for:

- identifiers, citations, URLs, and code spans
- quantities, dates, percentages, durations, and thresholds
- modal verbs, negation, and conditional clauses
- named actors, owners, systems, and components
- approval, risk, waiver, applicability, and confidence terms
- profile-defined exact phrases

Exit criteria:

- candidates that strengthen or weaken obligations are rejected
- candidates that add, remove, or alter protected elements are rejected
- violations identify the exact source and candidate spans

## Phase 3: governed attempt engine

Implement:

- append-only run and attempt records
- attempt reservation before provider invocation
- SHA-256 binding of source, prompt, response, candidate, and profile
- provider interface with a fake deterministic provider for tests
- bounded iteration and deterministic stopping
- explicit `selected`, `unchanged`, `needs_human`, and `failed` outcomes

Exit criteria:

- retries cannot reuse or overwrite an attempt ID
- interrupted and failed calls remain visible
- selection can be replayed without invoking the provider
- no candidate is selected solely because an LLM recommends it

## Phase 4: semantic-drift challenge

Implement a separate structured challenger that compares source and candidate.
Its output identifies omitted, added, weakened, strengthened, or scope-shifted
claims.

Deterministic code reconciles each challenge against protected elements and
profile policy. The challenger can block or escalate but cannot independently
approve.

Exit criteria:

- every challenge receives a disposition
- unresolved material challenges result in `needs_human`
- model failures do not become pass results
- architecture-review recommendations preserve decision strength and
  uncertainty

## Phase 5: CLI and reports

Implement:

```text
lingity analyze
lingity improve
lingity verify
lingity render
```

Render:

- baseline and selected scores
- located findings
- protected-element comparison
- attempt timeline
- source/candidate diff
- rejection and selection rationale

Exit criteria:

- `verify` detects altered artifacts and broken hashes
- reports are projections of canonical JSON
- the source remains available even when every attempt fails

## Phase 6: calibration

Run blinded comparison with architecture reviewers:

- original machine-authored text
- Lingity-selected text
- manually edited text

Measure:

- preference
- comprehension time
- perceived professionalism
- semantic-error rate
- missed obligation or uncertainty changes
- inter-rater agreement

Do not tune only for preference. Any configuration that improves style while
increasing semantic-error rate is rejected.

## Initial repository shape

```text
lingity/
  analyzer.py
  invariants.py
  scoring.py
  profiles.py
  attempts.py
  providers/
  schemas/
  rendering.py
  cli.py
profiles/
  architecture-review.json
schemas/
tests/
  corpus/
docs/
```

## First implementation milestone

The first executable milestone should:

1. Analyze one text file without network access.
2. Produce deterministic JSON with sentence load, nominalizations, noun stacks,
   agency, and weak-verb findings.
3. Calculate the Human Readability Index with full attribution.
4. Extract and hash protected identifiers, quantities, modal terms, and
   negation.
5. Pass a fixture based on the original architecture recommendation and show
   why the clearer rewrite scores better without changing protected meaning.

