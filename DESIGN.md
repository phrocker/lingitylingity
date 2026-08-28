# Governed Text Improvement Design

## Objective

Improve the clarity and human readability of LLM-authored text without
allowing a rewrite to change governed meaning.

This is not a general paraphraser. Lingity is an acceptance-controlled
pipeline in which models generate candidates and deterministic code governs
selection.

## Design principles

- **Immutable source:** Store the original text once and identify it by SHA-256.
- **Versioned analysis:** Record the analyzer version, linguistic model digest,
  profile version, metric values, findings, and score calculation.
- **Proposal-only LLM:** An LLM cannot approve its own rewrite or alter policy.
- **Hard gates before scores:** A higher readability score cannot compensate
  for changed facts, weakened obligations, missing evidence, or new claims.
- **Append-only attempts:** Never overwrite a failed or superseded candidate.
- **Explicit uncertainty:** Semantic uncertainty results in `needs_human`, not
  a success-shaped fallback.
- **Projection boundary:** Governed source records remain canonical. Lingity
  improves human-facing text projections.

## Pipeline

```text
ingest
  -> normalize without rewriting
  -> extract protected elements
  -> score baseline
  -> reserve attempt
  -> generate candidate
  -> validate lexical and structural invariants
  -> score candidate
  -> challenge semantic drift
  -> select or reject
  -> stop on success, attempt limit, or no material improvement
```

### 1. Ingest

The run records:

- source text and SHA-256
- profile name and version
- analyzer and linguistic-model versions
- requested provider/model parameters
- maximum attempts and minimum required improvement

### 2. Protect meaning

Deterministic extraction identifies elements that a candidate must preserve:

- identifiers and evidence references
- numbers, percentages, dates, durations, and thresholds
- named actors, owners, systems, and components
- modal strength: `must`, `must not`, `should`, `may`, and equivalents
- negation and conditional language
- approval, ratification, waiver, risk, and applicability status
- quoted text, code spans, URLs, and citations
- profile-defined terminology

Profiles may add exact protected phrases and approved substitutions.

### 3. Analyze language

Each finding includes a rule ID, severity, location, observed value, threshold,
and remediation. Initial metrics are:

| Dimension | Deterministic signals |
|---|---|
| Sentence load | words, clauses, punctuation depth, and actions per sentence |
| Morphology | nominalization density and weak verb constructions |
| Noun stacking | consecutive noun modifiers and compound depth |
| Agency | explicit actor-action pairs and agentless directives |
| Voice | passive constructions and indirect predicates |
| Lexical clarity | jargon, uncommon compounds, abbreviation density |
| Structure | paragraph length, list suitability, and mixed-purpose sentences |
| Redundancy | repeated qualifiers, duplicated recommendations, filler phrases |

Dependency parsing may be used, but the parser package and model digest become
part of the reproducibility contract.

The implementation takes that option: every signal above is computed from a
dependency parse rather than surface pattern matching, so detection depends on
sentence structure instead of wording. The parser package is pinned to an exact
model version and loaded fail-closed, and its name, version, runtime, and
digest are recorded as `linguistic_model` inside the hashed analysis artifact.
Verification rejects an artifact produced by a different pipeline rather than
re-analysing it under new assumptions.

## Scoring

The score is an explanation aid, not the acceptance authority.

```text
Human Readability Index =
    25% sentence load
  + 20% morphology
  + 20% agency
  + 15% lexical clarity
  + 10% structure
  + 10% redundancy
```

Every component is calculated from published rules. Profiles may change
weights and thresholds, but weights must total 100 and profile versions are
immutable.

Suggested profile bands:

- `85-100`: clear
- `70-84`: usable but improvable
- `0-69`: revision required

Acceptance additionally requires:

- no hard-gate violations
- minimum total-score improvement
- no regression beyond configured per-dimension tolerances
- no new high-severity linguistic finding
- semantic-drift result of `no_material_change`

## LLM iteration

The proposal prompt contains the source text, deterministic findings, protected
elements, approved glossary, and the prior attempt's rejection reasons.

The model must return structured output:

```json
{
  "candidate_text": "...",
  "addressed_rule_ids": ["LING-SENTENCE-001"],
  "claimed_preservations": ["MODALITY", "QUANTITIES", "EVIDENCE_REFS"]
}
```

The runtime ignores self-reported preservation claims when deciding acceptance.

Iteration stops deterministically when:

1. A candidate passes all gates and reaches the target score.
2. The maximum attempt count is reached.
3. Two consecutive candidates fail to achieve the minimum improvement.
4. A protected element changes.
5. The semantic challenge requires human judgment.

The default maximum is three attempts.

## Semantic-drift challenge

Some meaning cannot be proved equivalent with lexical rules alone. A separate
challenger may compare source and candidate and return typed claims:

- omitted claim
- added claim
- changed modality
- changed actor or ownership
- changed condition or scope
- changed uncertainty
- changed recommendation or decision strength

The challenger may block acceptance but cannot approve a candidate by itself.
High-consequence profiles can require human review even when the challenger
reports no drift.

## Run record

Each run should contain:

- immutable source
- protected-element manifest
- baseline analysis
- append-only attempts
- candidate hashes and model metadata
- deterministic invariant results
- before/after scores and findings
- semantic challenge
- selection rationale
- terminal status

Proposed statuses:

```text
draft -> analyzed -> improving -> selected
                            |-> unchanged
                            |-> needs_human
                            |-> failed
```

## Outputs

- `record.json`: canonical machine-readable run
- `analysis.json`: deterministic findings and score calculation
- `report.md`: human-readable explanation
- `selected.txt`: accepted text, or the source when no rewrite is accepted

## Non-goals

- Proving semantic equivalence from an LLM judgment
- Replacing human approval for consequential communication
- Optimizing text solely to maximize one aggregate score
- Editing canonical architecture or governance records
- Hiding failed attempts or model uncertainty

