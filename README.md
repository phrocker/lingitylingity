# Lingity

Lingity provides deterministic, governed analysis and rewrite validation for
English Markdown or plain text written by people or language models. Use it
when readability may improve but facts, identifiers, quantities, modality,
negation, citations, ownership, and governance status must not change.

The local `analyze` and `verify` commands make no network or LLM calls.
Rewriting commands can use a host agent or an explicitly configured provider,
but deterministic code remains the authority that accepts or rejects a
candidate.

## Prerequisites and installation

Lingity is not yet published to a package index. Install it from a clone with
Python 3.11 or later:

```text
python -m pip install .
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
python -m nltk.downloader wordnet omw-1.4
```

All three commands are required. The spaCy model must be exactly
`en_core_web_sm` 3.8.0; do not substitute
`python -m spacy download en_core_web_sm`, which can resolve a different
version. WordNet and `omw-1.4` are corpus data and must be downloaded
separately.

Lingity fails closed when these prerequisites are wrong or absent.
`LinguisticModelError` names the model installation command, and
`WordNetDataError` names the corpus command. Analysis never downloads data or
quietly falls back to a different model or rule.

## Run the first analysis

1. Analyze a document with a profile:

   ```text
   lingity analyze review.md --profile architecture-review
   ```

   The command writes a deterministic, schema-valid JSON artifact to standard
   output. Redirect it when you want to verify or retain it:

   ```text
   lingity analyze review.md --profile architecture-review > analysis.json
   ```

2. Verify the artifact:

   ```text
   lingity verify analysis.json
   ```

   `verify` validates the schema and hashes, resolves the recorded profile, and
   replays the analysis. An altered artifact, unavailable profile, different
   parser pipeline, or non-reproducible result fails explicitly.

The artifact contains located findings, the attributed Human Readability Index,
protected-element manifests, source and profile hashes, ingest coverage, and
the pinned linguistic model identity. The same source, profile, analyzer, model,
and parser produce the same result.

## Read grade-level estimates

Every `analyze` artifact carries a `readability_estimates` block with six
standard formulas: Flesch-Kincaid grade, Flesch Reading Ease, Coleman-Liau,
SMOG, Gunning Fog, and the Automated Readability Index. Each one reports its
value, its exact formula, and the counts it used. The block covers the whole
document and each sentence, keyed by `sentence_index` with source offsets.

To print the estimates without reading JSON:

```text
lingity grade page.md
lingity grade page.md --sentences
lingity grade page.md --format json
```

Read the numbers as estimates, not measurements:

- The formulas disagree. On dense prose they can differ by four grades or
  more, so the command prints all of them side by side.
- They count surface features: sentence length, syllables, and characters.
  They do not measure whether a reader understands the text.
- Other tools disagree with Lingity's counts too. The same formula name can
  mean a different syllable source, sentence split, or rounding.

The counts are deterministic and offline:

- Syllables come from a vendored table derived from a pinned CMUdict release,
  with a documented spelling rule for words the dictionary does not list.
- Figures, codes, and citations are normalized. A figure such as
  `$11,898,367,000` or a code such as `W00A01` counts as one short word. Inline
  code, URLs, and citations such as `(p. P.1)` are not counted.
- The artifact records the syllable source and its hash, the fallback rule, and
  the normalization rule. It also reports document values without
  normalization for comparison.

The estimates do not change the Human Readability Index. A profile can opt in
to a grade limit by setting `max_grade` in its thresholds. Lingity then reports
`LING-GRADE-001` for each sentence whose Flesch-Kincaid grade exceeds it, and
those findings cost points like any other finding. No shipped profile sets
`max_grade`. See [DESIGN.md](DESIGN.md) for the counting rules and caveats.

To make a rewrite aim for a lower grade, pass `--target-grade` through the
critique, judge, or improvement loop:

```text
lingity critique page.md --target-grade 8 --output brief.json
lingity judge page.md --candidate shorter.md --target-grade 8 --output verdict.json
lingity improve page.md --provider anthropic --model <model> --target-grade 8
```

The brief asks for the main point first, shorter concrete sentences, familiar
words, and removal of repeated context. The judge requires the candidate's
Flesch-Kincaid grade to decrease and reach the target. It also requires the
existing score not to fall and applies the protected-meaning, high-severity,
word-growth, and optional semantic-drift checks. A score already at 100 can
therefore stay at 100 while the grade falls. `--target-grade` is a rewrite goal;
the profile's `max_grade` remains an independent sentence finding policy.

These checks compare the rewrite with its source; they do not establish that
the source itself is true. For source-backed fact review, save the reference
material in an evidence bundle and pass it to `judge` or `improve`:

```json
{"sources": [{"id": "report", "url": "https://example.org/report", "retrieved_at": "2026-11-02T12:00:00Z", "content": "The exact source text used for review."}]}
```

```text
lingity judge page.md --candidate shorter.md --target-grade 8 \
  --evidence evidence.json --fact-check-provider openai \
  --fact-check-model <model> --output verdict.json
```

`anthropic` is also supported. Lingity asks the model to list the factual
claims in every sentence of both the source and rewrite and assess them against
the saved snapshots. A contradicted or unresolved claim blocks acceptance.
Every cited quote must occur in its named source;
the verdict records source hashes and the evidence bundle hash. The model's
judgment of whether a quote actually supports a claim still needs human review
before publication, especially for consequential claims. Lingity does not
fetch URLs or verify a source's authority. `--challenge-provider` remains a
separate check for meaning drift between source and rewrite.

## Choose a profile

Five profiles ship:

| Profile | Use it for |
| --- | --- |
| `architecture-review` | Recommendations, ADR summaries, findings, risks, and review decisions. |
| `product-strategy` | Need statements, value propositions, positioning, and go-to-market plans. |
| `web-copy` | Landing pages, product descriptions, and job listings. |
| `resume-review` | Resume and CV accomplishment bullets. |
| `local-service` | Location-specific pages for trades and local services. |

Profiles are versioned analysis policy, not interchangeable labels. They set
different weights, thresholds, vocabulary, and genre rules. For example,
`product-strategy` requires a responsible actor, `resume-review` recognizes the
implied first-person subject of accomplishment bullets, and `local-service`
distinguishes trade terms from generic promotional language.

Run the same document under another profile only when that profile matches the
document's purpose:

```text
lingity analyze strategy.md --profile product-strategy
```

See [DESIGN.md](DESIGN.md) for profile rationale, threshold behavior, scoring,
parser boundaries, and rule details.

## Rewrite a document

Lingity separates proposal generation from acceptance. A model may propose a
rewrite; only deterministic checks decide whether to accept it.

### Create a critique

```text
lingity critique review.md --output brief.json
lingity critique review.md --profile architecture-review --style technical-writer --output styled-brief.json
```

`critique` ranks defects and records the elements a rewrite may not change.
Add `--style` to include a versioned style contract and deterministic provider
instructions. Omitting `--style` preserves the unstyled brief behavior.

### Judge one candidate

```text
lingity judge review.md --candidate rewrite.md --profile architecture-review
```

A candidate is accepted only when protected meaning is equivalent, the Human
Readability Index strictly improves, no new high-severity finding appears, the
profile's readable-word growth budget is met, and no semantic-drift challenge
raises material doubt. A higher score never compensates for changed meaning.

Every rejection reports its reasons and a `protected_delta` so the next edit can
restore named elements instead of guessing:

```text
$ lingity judge source.txt --candidate shorter.txt
accepted False   70.46 -> 89.50
  reason: protected meaning is changed: 9 protected element(s) dropped
  MISSING quantity:count:2
  MISSING governance:term:ratify
  MISSING order:sequence:earlier=require closure evidence govern recommendation;later=target architecture return human decision
  ...
```

When no candidate qualifies, Lingity returns the source unchanged with the
rejection reasons. It never accepts a regression, tie, unresolved meaning
comparison, or candidate that drops a protected claim.

### Run the bounded improvement loop

```text
lingity improve review.md --provider subagent --candidate rewrite.md
lingity improve review.md --provider anthropic --model <model> --style technical-writer
```

`improve` feeds each rejection into the next critique until a candidate is
accepted or the bounded run ends. `--style` passes the contract to an API
provider. The `subagent` provider serves candidates written before the loop
runs, so it rejects `--style`; give the host agent `lingity critique --style`
instead. The copy-ready editing prompt for writing one candidate is in
[`docs/conservative-editing-prompt.md`](docs/conservative-editing-prompt.md).

To drive the whole loop from an agent harness such as Claude Code, follow
[`docs/agent-harness.md`](docs/agent-harness.md). It covers the critique, judge,
and repair steps, how to read a verdict, a copy-ready system prompt, and a
Claude Code skill in [`skills/lingity`](skills/lingity/SKILL.md).

Proposal providers are transports, never authorities:

- `subagent` is the default. It makes no network call and needs no API key. The
  host agent writes a candidate and Lingity judges it.
- `openai` and `anthropic` make direct API calls. Credentials come only from
  `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`; they are not accepted as arguments,
  logged, or written to artifacts. These providers require `--model`; Lingity
  has no default model.

A drift challenger can block acceptance but cannot clear a deterministic
failure. An unparseable challenge response is an error.

### Exit codes

`critique`, `judge`, and `improve` use:

- `0`: success
- `1`: reasoned rejection
- `2`: error

This contract lets a host process branch on the exit code without parsing
human-readable output.

## Choose a style contract

Four versioned JSON style contracts ship:

- `architecture-review`
- `conservative-web-editor`
- `local-service-guide`
- `technical-writer`

List and inspect them with:

```text
lingity styles
lingity style technical-writer --format json
lingity style technical-writer --format prompt
```

Style contracts guide proposal generation. They do not score rhetorical fit,
and `judge` remains style-independent. Profiles validate deterministic
readability and protected meaning; a profile score is not a style-contract fit
score. Optional style guidance never changes the acceptance gates.

See [`docs/style-contract-experiment.md`](docs/style-contract-experiment.md)
for the style-contract evaluation and examples.

## Analysis and safety contract

Lingity's current workflow is:

```text
source text
  -> deterministic analysis
  -> schema-valid analysis artifact
  -> deterministic verification replay
```

Rewrite validation adds a bounded proposal between analysis and deterministic
acceptance:

```text
source text
  -> deterministic analysis
  -> bounded proposal
  -> invariant validation
  -> deterministic rescoring
  -> semantic-drift challenge
  -> accept, iterate, reject, or require human review
```

The core guarantees are:

1. The source text is immutable.
2. Every score is reproducible from a versioned profile and analyzer.
3. The runtime does not emit success-shaped fallback results.
4. Protected facts, identifiers, quantities, modality, negation, citations,
   ownership, and governance status must survive unchanged.
5. A candidate must improve the configured thresholds without a hard-gate
   violation.
6. Semantic uncertainty is `needs_human`, not success.

The analyzer covers sentence load, morphology, noun stacking, agency, voice,
lexical clarity, structure, and redundancy. It reads Markdown prose blocks but
not fenced or indented code, tables, or thematic breaks. Inline-code identifiers
are protected. Findings include a rule ID, severity, source location, observed
value, threshold, and remediation.

The parser, linguistic model, profile, analyzer, and ingest coverage are part of
the hashed analysis contract. `verify` refuses incompatible inputs instead of
silently reanalyzing them. The exact dimensions, score formula, block-scoping
rules, CommonMark behavior, protected-meaning model, held-out corpus limits, and
provider protocols are documented in [DESIGN.md](DESIGN.md).

Additional background is available in the
[AgenticTuner comparison](docs/agentictuner-comparison.md) and the
[implementation plan](docs/implementation-plan.md).

## Development

```text
python -m pip install -e '.[dev]'
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
python -m nltk.downloader wordnet omw-1.4
python -m pytest
python -m mypy
python -m compileall -q lingity tests
```

These are the commands CI runs, in this order, on Python 3.11 and 3.12 for every
push to `main` and every pull request; see `.github/workflows/ci.yml`. CI skips
the corpus download when its `~/nltk_data` cache is restored. Locally, rerunning
the download after the corpora are present is a no-op.

Use `python -m` so each tool runs under the interpreter where Lingity is
installed. Keep the extras spec quoted because `zsh` treats brackets as a glob.
`pytest` and `mypy` read their settings from `pyproject.toml`.

## Prior art

The rule families follow:

- Femmer, Méndez Fernández, Wagner, Eder, *Rapid Quality Assurance with
  Requirements Smells* (Journal of Systems and Software, 2017).
- INCOSE-TP-010-009, *Guide to Writing Requirements* (2019).
- U.S. Federal Plain Language Guidelines (PLAIN).

These sources inform the smell-detection, imperative, ambiguity,
quantification, actor-first, active-voice, and sentence-length rules. Lingity
adds attributed findings and governed-meaning rewrite gates.

## License

Apache License 2.0. The full text is in [LICENSE](LICENSE), and [NOTICE](NOTICE)
carries the copyright statement and required section 4(d) attribution.

Lingity does not redistribute or declare its two linguistic data artifacts as
dependencies. Users download the `en_core_web_sm` spaCy model under the MIT
License and the NLTK WordNet corpus under the WordNet 3.0 License.

Lingity does bundle one derived data file: a syllable-count table built from
the Carnegie Mellon Pronouncing Dictionary (CMUdict). CMUdict is distributed
under a BSD-style license, and its full text ships as
`lingity/data/CMUDICT-LICENSE`. `NOTICE` records all three.

## Release

Publishing requires an authenticated `twine` configuration. Credentials are
resolved from `~/.pypirc`, the system keyring, or `TWINE_*` environment
variables; no credential is passed as an argument or written by the script.

Run:

```text
python -m pip install -e ".[release]"
python scripts/release.py --repository testpypi --dry-run
python scripts/release.py --repository testpypi
python scripts/release.py --repository pypi
```

`--repository` is required; there is no default. Use `--dry-run` to run every
check and build the artifacts without uploading.

The release script stops on an unclean working tree, failing guard test,
already-published version, direct URL dependency, missing license metadata,
unknown classifier, or unreachable index. Uploads cannot be undone and a
version cannot be reused, so an unavailable check is an error rather than an
assumed success.
