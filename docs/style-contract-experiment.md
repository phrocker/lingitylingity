# Bounded style-contract experiment

Date: 2026-09-23

This experiment asks one narrow question: can explicit style contracts produce
usefully different edits before Lingity grows a new style engine? It uses the
existing profiles unchanged and one fictional, neutral facilities notice. The
canonical inputs are in [`style-contract-experiment/`](style-contract-experiment/).

The original experiment covers the three contracts reproduced below. Those
contracts remain executable, versioned JSON guidance in
[`lingity/styles/`](../lingity/styles/), alongside the separately added
`technical-writer` contract. Four contracts now ship and are validated by
[`style-contract.schema.json`](../lingity/schemas/v1/style-contract.schema.json).
They can be listed with `lingity styles`, rendered with
`lingity style <name> --format prompt`, and passed to `critique`, or to
`improve` with an API provider, with `--style <name>`. Their digest-bound
structured data and rendered instructions enter the critique brief and proposal
prompt. The `subagent` provider serves candidates written before the loop runs,
so `improve` rejects `--style` for it; the host agent reads the styled brief
from `critique --style` instead.

This execution path does not turn the contracts into deterministic authorities.
They guide proposal generation only. `judge` remains style-independent, and
the profile score continues to measure shared clarity, economy, and protected
meaning rather than rhetorical fit.

## Copy-ready contracts

### Conservative web editor

```text
Reader outcome: A general web reader can identify the service impact, planned
work, cost, requirements, notice obligation, and delay plan on a quick scan.

Content policy: Preserve every source fact and condition. Add no explanation,
benefit, reassurance, heading, recommendation, or background.

Rhetorical shape: Lead with immediate reader impact. Follow with the planned
change and cost, then requirements and owner obligations, then contingency.

Editing posture: Make the minimum necessary change. Split overloaded sentences,
reorder only to improve scan order, and prefer short paragraphs over expansion.

Observable tendencies: Impact-first ordering; one or two facts per sentence;
short paragraphs; no promotional language.

Positive guidance: Put closures and continued access before implementation
detail. Keep dates, quantities, actors, modality, conditions, and scope exact.

Negative guidance: Do not add a summary, call to action, friendly framing,
benefits, transitions, examples, or inferred consequences.
```

### Local-service guide

```text
Reader outcome: A person planning around the work can find the permit and
inspection requirements first, then timing, cost, access effects, notice, and
the weather fallback.

Content policy: Preserve every source fact and condition. Do not infer local
rules, prices, schedules, contacts, advice, or service claims.

Rhetorical shape: Lead with the practical requirement. Group timing and cost,
then access and notice, then contingency.

Editing posture: Act as a factual guide, not a marketer. Reorder around the
reader's likely planning questions and split overloaded sentences without
adding answers the source does not provide.

Observable tendencies: Requirement-first ordering; concrete local nouns near
the front; access and notice kept together; contingency last.

Positive guidance: Surface the county permit, inspection count, date, budget,
closure duration, notice period, and fallback exactly.

Negative guidance: Do not add national averages, recommendations, urgency,
claims about typical projects, or unsupported permit guidance.
```

### Architecture review

```text
Reader outcome: A reviewer can distinguish the proposed change, constraints,
operational controls, and contingency without losing implementation facts.

Content policy: Preserve every source fact, actor, requirement, quantity,
condition, and commitment. Add no decision, approval state, risk rating,
evidence claim, owner, or recommendation.

Rhetorical shape: Keep the change first. Group permit and budget constraints,
then operational impact and required notice controls, then contingency.

Editing posture: Clarify the review record rather than persuade. Split mixed
purposes, preserve causal and conditional relationships, and avoid decorative
labels that could be parsed as new claims.

Observable tendencies: Change-first ordering; constraints grouped together;
control owner beside the control; contingency isolated at the end.

Positive guidance: Keep the facilities team, project manager, permit,
inspections, budget, closure, notice period, weather condition, and fallback.

Negative guidance: Do not invent architecture terminology, assurance,
ratification, mitigation, evidence, trade-offs, or a review recommendation.
```

## Inputs and candidates

The source deliberately compresses several purposes into two long sentences.
The generated candidates preserve its facts but make different editorial
decisions:

| Contract | Candidate | Distinct decision |
|---|---|---|
| Conservative web editor | [`conservative-web-editor.md`](style-contract-experiment/conservative-web-editor.md) | Leads with continued access and the three-day entrance closure. |
| Local-service guide | [`local-service-guide.md`](style-contract-experiment/local-service-guide.md) | Leads with the county permit and two inspections. |
| Architecture review | [`architecture-review.md`](style-contract-experiment/architecture-review.md) | Keeps the proposed change first and groups constraints, operational controls, and contingency. |

These are ordering and grouping changes, not adjective substitutions. The
architecture candidate also keeps the access contrast in one sentence, while
the other two split it for faster scanning.

## Relevant-profile results

Finding counts use `RULE_ID:severity x count`.

| Contract / profile | Score | Readable words | Judge | Protected meaning | Source findings | Candidate findings |
|---|---:|---:|---|---|---|---|
| Conservative web editor / `web-copy` | 81.76 -> 98.33 | 90 -> 87 (-3; maximum 102) | accepted | equivalent; no delta | `LING-ACTION-001:high x1`, `LING-ACTION-001:medium x1`, `LING-CLAUSE-001:high x1`, `LING-LIST-001:high x1`, `LING-LIST-001:low x1`, `LING-NOMINALIZATION-001:high x1`, `LING-SENTENCE-001:medium x2`, `LING-STRUCTURE-001:low x1` | `LING-NOMINALIZATION-001:high x1` |
| Local-service guide / `local-service` | 85.66 -> 98.14 | 90 -> 87 (-3; maximum 102) | accepted | equivalent; no delta | `LING-ACTION-001:high x1`, `LING-ACTION-001:medium x1`, `LING-CLAUSE-001:high x1`, `LING-LIST-001:high x1`, `LING-LIST-001:low x1`, `LING-NOMINALIZATION-001:high x1`, `LING-SENTENCE-001:high x1`, `LING-SENTENCE-001:medium x1`, `LING-STRUCTURE-001:medium x1` | `LING-NOMINALIZATION-001:high x1`, `LING-SENTENCE-001:low x2` |
| Architecture review / `architecture-review` | 86.36 -> 100.00 | 90 -> 88 (-2; maximum 102) | accepted | equivalent; no delta | `LING-ACTION-001:low x1`, `LING-CLAUSE-001:low x1`, `LING-LIST-001:medium x1`, `LING-NOMINALIZATION-001:medium x1`, `LING-REDUNDANCY-001:low x2`, `LING-SENTENCE-001:medium x2` | none |

Each economy gate passed. Every protected delta had empty `missing`, `added`,
`unresolved`, and `specified` arrays.

## Cross-style comparison

| Analysis profile | Source | Web candidate | Local-service candidate | Architecture candidate |
|---|---:|---:|---:|---:|
| `web-copy` | 81.76 | 98.33 | 98.33 | 98.33 |
| `local-service` | 85.66 | 98.14 | 98.14 | 98.14 |
| `architecture-review` | 86.36 | 100.00 | 100.00 | 100.00 |

The contracts produced distinct editorial decisions that are plausibly useful
for their intended readers. The current profiles did **not** produce a valid
cross-style inversion: within a profile, all three candidates received the
same score and findings. They reward the shared sentence splitting and reduced
overload, but they do not currently measure whether impact, requirements, or a
proposed change appears first, nor how facts are grouped across short
paragraphs.

That is a useful bounded result, not a reason to tune thresholds until one
candidate wins. Structured contracts can guide generation today, while current
Lingity profiles validate shared clarity, economy, and protected meaning rather
than contract-specific rhetorical fit. A later design should require broader
examples and an explicit observable before adding a style-fit signal.

The executable JSON slice implements that bounded conclusion: the original
three experimental contracts and the later `technical-writer` contract are
loadable, discoverable, renderable, and optional in proposal generation, but no
style-fit score or profile tuning has been added.

## Reproduce

From the repository root in PowerShell:

```powershell
$root = "docs\style-contract-experiment"
$out = Join-Path $env:TEMP "lingity-style-contract-experiment"
New-Item -ItemType Directory -Force $out | Out-Null

python -m lingity analyze "$root\source.md" --profile web-copy --output "$out\web-source.json"
python -m lingity analyze "$root\conservative-web-editor.md" --profile web-copy --output "$out\web-candidate.json"
python -m lingity judge "$root\source.md" --candidate "$root\conservative-web-editor.md" --profile web-copy --output "$out\web-verdict.json"

python -m lingity analyze "$root\source.md" --profile local-service --output "$out\local-source.json"
python -m lingity analyze "$root\local-service-guide.md" --profile local-service --output "$out\local-candidate.json"
python -m lingity judge "$root\source.md" --candidate "$root\local-service-guide.md" --profile local-service --output "$out\local-verdict.json"

python -m lingity analyze "$root\source.md" --profile architecture-review --output "$out\architecture-source.json"
python -m lingity analyze "$root\architecture-review.md" --profile architecture-review --output "$out\architecture-candidate.json"
python -m lingity judge "$root\source.md" --candidate "$root\architecture-review.md" --profile architecture-review --output "$out\architecture-verdict.json"
```

For the cross-style matrix, analyze each of the three candidate files once
under each of the three profiles. No profile or engine setting was changed for
this experiment.
