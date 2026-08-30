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

The gate compares meaning as propositions rather than as wording. It parses
each sentence into a claim signature: action, actor, target, modality,
polarity, and status. It also records the ordering relations between claims.
Two texts are equivalent when their signatures match, so "close the findings
before sign-off" and "sign-off happens only after the findings are closed"
agree, while "approve" and "ratify" do not.

Profiles carry no protected sentence patterns. An earlier revision matched
protected concepts against literal phrases copied from the test fixture. The
fixture passed, but the gate understood nothing. Those patterns were removed
and the gate rebuilt on the parse.

The comparison is deliberately directional. Naming an actor the source left
unnamed is a permitted specification and is reported under `specified` —
otherwise the remediation `LING-AGENCY-001` itself recommends would be
rejected. Dropping or swapping a named actor stays a violation.

The gate is fail-closed. When a sentence yields no proposition and no protected
element, the gate reports `unresolved` rather than assuming a match. Two texts
that both fail to parse would otherwise compare equal and certify a meaning
nobody read. An `unresolved` verdict costs a rewrite that the gate should have
accepted. A false `equivalent` certifies a rewrite that changed what the text
commits to. Only the second is a safety failure.

A text is committed to what it asserts a thing *is*, not only to what it says
should be done. The gate therefore reads a linking verb and its complement as a
state claim. "The fix is complete and fail-closed" records `complete` and
`fail close` against the fix.

Coordinated complements attach inconsistently. The second complement sometimes
hangs off the first and sometimes off the verb, so the gate collects both
attachment points. A conjunct with its own subject or its own tense belongs to
the ordinary claim extractor, and the gate leaves it there.

That extraction closed a real false `equivalent`. Before it, "the fix is
complete and fail-closed" and "the fix is incomplete and fail-open" compared
equivalent on live text. The held-out corpus did not catch it. Its copular pair
failed the coverage guard for an unrelated reason, so an `unresolved` verdict
masked the hole and we recorded it as a conservative gap. A corpus that scores
well is evidence, not proof: a gap in the corpus can hide a gap in the gate.
`tests/test_meaning_equivalence.py` asserts that the corpus never reports a
meaning change as `equivalent`, and that assertion covers the documented gaps
too. It can still speak only for the pairs it contains.

A held-out corpus of 32 pairs
(`tests/fixtures/meaning-equivalence-corpus.json`) measures how well the gate
generalises. Its pairs come from governance semantics rather than from any
profile or shipped fixture. Answering "changed" for every pair scores 16/32, so
the changed-pair count alone proves nothing. The equivalent-pair count is what
separates a semantic gate from a lookup table.

The corpus lists the eight pairs the gate does not resolve under `known_gaps`,
with the linguistic reason for each. Each gap is a strict expected failure, so
fixing one forces the win into the open rather than absorbing it silently.
Every listed gap answers `unresolved` or `changed`, never `equivalent`.

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

The implementation takes that option. A dependency parse produces every signal
above, so detection depends on sentence structure instead of wording. The
loader pins the parser package to an exact model version and fails closed. It
records the parser name, version, runtime, and digest as `linguistic_model`
inside the hashed analysis artifact. Verification rejects an artifact produced
by a different pipeline rather than re-analysing it under new assumptions.

## Profiles

A profile carries the weights, thresholds, and vocabulary for one kind of
document. The analyzer is shared; the profile decides what counts as a defect
and how much it costs. Two profiles ship, and a project may install more.

`architecture-review` reads review decisions. `product-strategy` reads need
statements, value propositions, and positioning. The second weights agency and
lexical clarity at 25 each and structure at 8, because a strategy document
mostly fails by claiming something unfalsifiable or by claiming it without
naming who acts.

### A profile can require a recognised actor

`product-strategy` omits "market", "industry", and "space" from its actor terms.
That omission decided nothing on its own. `LING-ACTOR-001` cleared a directive
whenever the verb carried any overt subject, so "the market should prioritise
retention" reported nothing and the profile's choice of actor terms never
reached the rule.

A profile may now set `require_responsible_actor`. Under that threshold a
directive clears the rule only when its subject is an actor the profile
recognises. `architecture-review` does not set it and behaves as before.

`product-strategy` stores no protected phrases, and declares an empty
`protected_concepts` array. `architecture-review` still carries one governance
entry from before the phrase matching was removed. Protection belongs to the
meaning gate, which reads claim signatures from the parse.

Lingity does not detect a benefit asserted without a mechanism. The
`purpose_markers` groups name a `mechanism` category, but the analyzer reads
those groups only to report a sentence that mixes more than two purposes. It
never reports an absent one. "Customers save ten hours each week" therefore
passes, and a rule that reports it would be new analyzer work rather than
profile vocabulary.

### Phrases are stored as lemmas, and a lemma is not the surface word

The phrase rules match lemma sequences, so a phrase written the way an author
types it will never fire. The pinned pipeline lemmatises "cutting edge" to "cut
edge". A phrase list derived by eye therefore ships rules that are silently
dead, and a passing suite says nothing about it, because a rule that never
matches breaks nothing.

A lemma also depends on the grammatical role the phrase takes, which is the
subtler trap. "Thought leadership" lemmatises to "thought leadership" as a
subject and to "think leadership" after a copula, because the parser reads
"thought" as a noun in one and a verb in the other. "Best in class" produces
three forms across its roles. Pinning a phrase to a single carrier therefore
ships a rule that fires in one construction and stays silent everywhere else,
and an unnatural carrier pins the form that real prose never produces.

Every jargon phrase in `product-strategy` is exercised across six natural
carriers spanning subject, object, prepositional object, complement, and verb
positions. Every distinct lemma that fires is stored, and
`tests/fixtures/product-strategy-jargon.json` records each one beside the
sentence that produced it. A test asserts that the fixture and the profile cover
exactly the same lemma set, so a phrase added without a demonstration fails the
suite, and a reader who "corrects" a lemma back to its surface spelling fails it
too.

One phrase was dropped rather than shipped. The pinned pipeline lemmatises
"force multiplier" to "force multipli", which matches correctly but reads as a
typo, so a future reader would repair it and silence the rule. A phrase that
cannot be stored legibly is not worth the maintenance hazard.

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

A rejected candidate comes back with `protected_delta`, which names the exact
elements that the rewrite dropped, introduced, or left unresolved. A verdict
that only says "meaning changed" gives a caller nothing to act on. Naming the
elements is what lets an iterative loop converge instead of guessing. A
candidate that scores higher but drops governed content is rejected on that
evidence. The fixture in `tests/fixtures/recommended-decision.json` keeps such a
rewrite as `unfaithful_rewrite` for exactly this reason.

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

