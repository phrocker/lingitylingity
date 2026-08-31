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

Two profiles ship. `architecture-review` reads recommendations, ADR summaries,
findings, risks, and review decisions that must remain precise while reading
like professional human communication. `product-strategy` reads need
statements, value propositions, positioning, and go-to-market plans.

A strategy document fails differently from an architecture review. It claims
something unfalsifiable, or it claims it without naming who acts, so
`product-strategy` weights agency and lexical clarity highest and structure
lowest.

It also sets `require_responsible_actor`. Under that threshold a directive must
name an actor the profile recognises, rather than any noun at all, and
`product-strategy` omits "market", "industry", and "space" from its actor
terms. "The market should prioritize retention" therefore reports
`LING-ACTOR-001`, because a sentence whose only actor is the market names
nobody who can act.

The difference is measurable. On the same hyped paragraph,
`architecture-review` scores 88.65 and reports no jargon at all, while
`product-strategy` scores 69.53 and reports five jargon findings. Neither
profile penalises prose that names a number, an actor, and a limit.

## CLI

```text
lingity analyze review.md --profile architecture-review
lingity verify analysis.json

lingity critique review.md --output brief.json
lingity judge review.md --candidate rewrite.md
lingity improve review.md --provider subagent --candidate rewrite.md
```

`analyze` emits a deterministic, schema-valid JSON artifact containing located
findings, the attributed Human Readability Index, protected-element manifests,
and content/profile hashes. `verify` validates the schema and hashes, resolves
the recorded profile, and replays the analysis; altered or non-reproducible
artifacts fail explicitly. Both commands are pure and offline.

`critique`, `judge`, and `improve` drive rewriting. `critique` emits an
improvement brief — the ranked defects and the elements a rewrite may not
change. `judge` decides a single candidate. `improve` runs the bounded loop,
feeding each rejection back into the next brief. All three exit `0` on success,
`1` on a reasoned rejection, and `2` on an error, so a host agent can branch on
the exit code alone.

The current analyzer is a versioned English dependency-parse model covering
every deterministic signal published in the [DESIGN.md](DESIGN.md) dimension
table:

- **Sentence load** — words, clauses, punctuation depth, and actions per
  sentence.
- **Morphology** — nominalization density and weak verb constructions.
- **Noun stacking** — consecutive noun modifiers and hyphenated compound depth.
  A stack must be contiguous, and a named entity counts as one unit, so
  `Azure Kubernetes Service cluster` is two units rather than four and a
  person's name is never reported as a stack. Detection reads the dependency
  relation rather than the part-of-speech tag, because the tagger reads
  `messaging` in `messaging loss hypotheses` as a noun in one sentence and a
  verb in another. The finding reports `words` for the span and `units` for the
  naming units the threshold compares.
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
  phrases. Lingity counts a repeated content word within one block, not across
  the whole text. Governance prose has to call one concept by one name in every
  section, so a term that recurs between sections shows consistency. Counting
  document-wide made a finding depend on wording far away from it. Joining clear
  paragraphs then manufactured findings that no paragraph had alone.

Every rule is block-scoped: the findings for a document are exactly the findings
of its blocks. A passage therefore scores the same alone as it does inside the
document that contains it.

A finding quotes source text the way the parser read it. The parser joins a
block's wrapped lines with a single space, so an observed value never carries a
line break or a list marker's indentation.

Noun stacking findings are reported under the `morphology` dimension and voice
findings under `agency`, so the score always resolves to the six weighted
dimensions.

Lingity reads Markdown structure before it parses prose. Headings, list items,
blockquotes, and paragraphs carry prose. Fenced code, indented code, tables, and
thematic breaks do not, so the analyzer never reads them. Each prose block
parses on its own, so a heading cannot glue itself onto the paragraph beneath it
and a table row cannot register as one long sentence. An identifier written
inside a code span is a name a rewrite must not change, so a finding falling
wholly inside one is dropped.

Each block is parsed as one unit, so a sentence wrapped across two source lines
stays one sentence and neither the wrap point nor the line-ending style changes
a score.

Block structure comes from `markdown-it-py`, which is CommonMark compliant and
tested against the specification's own suite. The parser is part of the analysis
contract exactly as the linguistic model is: its identity is published in every
artifact under `ingest`, and a major-version change is refused rather than
silently re-segmented. The artifact also publishes `unresolved_lines` and
`uncovered_lines`, so text that left the analysis is counted rather than lost in
silence. `verify` replays the segmentation.

Every finding carries a rule ID, severity, character location, observed value,
threshold, and remediation. Overlapping spans within a dimension are
de-duplicated so a single defect is not penalised twice.

Rules read sentence structure rather than surface strings, so detection
generalises to unseen wording. They inspect predicates, subjects, auxiliaries,
negation, and modifier chains. The parse is part of the analysis contract, so
Lingity pins the pipeline to `en_core_web_sm` at an exact version and loads it
fail-closed. It records the parser name, version, runtime, and digest as
`linguistic_model` inside the hashed artifact. `verify` refuses any artifact
produced by a different pipeline instead of silently re-analysing it.

The Human Readability Index weights those six dimensions and converts each
dimension's deducted points into a score with a half-life decay, so worse text
never scores higher than better text. The exact arithmetic is published in the
artifact's `score.formula` field.

Provider protocols exist for future proposal and semantic-challenge adapters,
but `analyze` and `verify` themselves perform no network or LLM calls.

See [DESIGN.md](DESIGN.md), the
[AgenticTuner comparison](docs/agentictuner-comparison.md), and the
[implementation plan](docs/implementation-plan.md).

## Rewriting

A model may *propose* a rewrite. Only deterministic code decides whether to
accept one. A candidate is accepted when, and only when, all of the following
hold:

- protected meaning is equivalent to the source,
- the Human Readability Index strictly improves,
- no new high-severity finding appears,
- and no semantic-drift challenge raised material doubt.

Lingity never accepts a regression, never accepts a tie, and never accepts an
unresolved meaning comparison. When nothing qualifies, it returns the source
text unchanged together with the reasons every candidate failed. It rejects a
candidate that scores a perfect 100 but drops a protected claim. A higher score
never buys a change in meaning.

Rejections are actionable. Every verdict carries `protected_delta`, naming the
exact elements dropped, introduced, or left unresolved, so the next attempt can
restore them by name instead of guessing:

```text
$ lingity judge source.txt --candidate shorter.txt
accepted False   68.17 -> 89.59
  reason: protected meaning is changed: 5 protected element(s) dropped
  MISSING quantity:count:2
  MISSING governance:term:recommend
  MISSING order:sequence:earlier=govern recommendation;later=target architecture return human decision
```

The gate compares meaning as propositions rather than as wording. It parses
each sentence into a claim signature: action, actor, target, modality,
polarity, and status. It also records the ordering relations between claims.
"Close the findings before sign-off" therefore agrees with "sign-off happens
only after the findings are closed". "Approve" and "ratify" do not agree. The
gate reads linking verbs as state claims. "The fix is complete and
fail-closed" therefore disagrees with "the fix is incomplete and fail-open".
No profile contains protected sentence patterns.

A held-out corpus of 32 pairs measures how well the gate generalises. It shares
no wording with any profile or fixture. The corpus documents the eight pairs
the gate does not resolve, with the linguistic reason for each. Every one of
them answers `unresolved` or `changed` rather than `equivalent`. The corpus is
evidence, not proof: it once masked a false `equivalent` on copular text behind
an unrelated coverage failure. Treat a passing corpus as a floor.

Providers are transports, never authorities:

- **`subagent`** (default) — no network and no API key. The host agent, such as
  Agency, *is* the model: Lingity hands it a brief, the host writes a candidate,
  and Lingity judges the result. Use `critique` and `judge` interactively, or
  pass `--candidate` files to `improve`.
- **`openai`** and **`anthropic`** — direct API calls over the standard library.
  Credentials come only from `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, and are
  never accepted as arguments, logged, or written to an artifact. There is no
  default model: omitting `--model` is an error rather than a guess.

A drift challenger may only *raise* doubt. It can block an acceptance, but it
can never clear a deterministic failure, and an unparseable challenge response
is an error rather than a quiet `no_material_change`.

## Development

```text
python -m pip install -e .[dev]
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4
pytest
mypy --strict lingity tests
python -m compileall -q lingity tests
```

Analysis needs the spaCy model and the WordNet corpora present locally. Both are
install-time steps on purpose: nothing downloads anything at analysis time, so a
run cannot silently depend on the network or quietly change behaviour when a
corpus is missing. Missing data is an error, not a fallback.

WordNet drives morphology — deriving the verb behind a nominalization
("ratification" → "ratify") and separating a word from its antonyms — rather
than a hand-maintained suffix list.

## Prior art

The rule families follow published work on requirements and plain-language
quality:

- Femmer, Méndez Fernández, Wagner, Eder, *Rapid Quality Assurance with
  Requirements Smells* (Journal of Systems and Software, 2017) — the
  smell-detection framing behind nominalization, passive voice, and vague-term
  rules.
- INCOSE-TP-010-009, *Guide to Writing Requirements* (2019) — rules on
  imperatives, ambiguity, and quantification.
- U.S. Federal Plain Language Guidelines (PLAIN) — actor-first sentences, active
  voice, and short sentence targets.

No existing package was found that detects nominalizations, noun stacks, hidden
agency, or bureaucratic phrasing as attributed findings, or that gates a rewrite
on preserved governed meaning, so those are implemented here.
