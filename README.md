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
lingity improve review.md --provider subagent --candidate rewrite.md --style technical-writer
```

`improve` feeds each rejection into the next critique until a candidate is
accepted or the bounded run ends. The copy-ready host-agent instructions are in
[`docs/conservative-editing-prompt.md`](docs/conservative-editing-prompt.md).

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

Lingity does not redistribute or declare its two data artifacts as
dependencies. Users download the `en_core_web_sm` spaCy model under the MIT
License and the NLTK WordNet corpus under the WordNet 3.0 License. `NOTICE`
records both.

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
