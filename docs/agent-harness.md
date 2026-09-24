# Using Lingity from an agent harness

This guide is for an agent host, such as Claude Code or another coding agent,
that improves a document with Lingity. The agent writes the rewrite. Lingity
decides whether to accept it. The agent never makes that decision itself.

The workflow uses the default `subagent` path: no API key and no network call
from Lingity. Each step is a separate command, so the host keeps control of
the conversation and branches on exit codes.

## Before you start

Confirm that Lingity runs:

```text
lingity --help
```

If the command is missing or exits with an error about the spaCy model or
WordNet, follow the installation steps in [README.md](../README.md). Lingity
fails closed when a prerequisite is wrong; do not work around the error.

Choose a profile for the kind of document:

| Profile | Use for |
| --- | --- |
| `architecture-review` | Recommendations, ADR summaries, findings, risks, and review decisions. The default. |
| `product-strategy` | Need statements, value propositions, positioning, and go-to-market plans. |
| `web-copy` | Landing pages, product descriptions, and job listings. |
| `resume-review` | Resume and CV accomplishment bullets. |
| `local-service` | Location-specific pages for trades and local services. |

## The loop

Work on a copy of the paths below. Never edit the source file.

1. **Create the brief.**

   ```text
   lingity critique source.md --profile architecture-review --output brief.json
   ```

   Exit `0` means the brief was written. Exit `2` is an error: report it and
   stop.

2. **Write one candidate** to `candidate.md`, following the
   [conservative editing prompt](conservative-editing-prompt.md) and the brief.

3. **Judge the candidate.**

   ```text
   lingity judge source.md --candidate candidate.md --profile architecture-review --output verdict.json
   ```

   Use the same `--profile` as the critique.

4. **Branch on the exit code.**

   - `0`: accepted. `candidate.md` is the result. Stop.
   - `1`: rejected with reasons. Go to step 5.
   - `2`: error. Report the message and stop. Do not retry the same command.

5. **Repair and retry.** Read `rejection_reasons` and `protected_delta` in
   `verdict.json`. Record the attempt in `attempts.json`, then create a new
   brief that includes it:

   ```json
   [
     {"index": 1, "rejection_reasons": ["protected meaning is changed: 2 protected element(s) dropped"]}
   ]
   ```

   ```text
   lingity critique source.md --profile architecture-review --prior-attempts attempts.json --output brief.json
   ```

   Write the next candidate from the **source**, not from the rejected
   candidate, and restore every element the verdict names. Return to step 3.

6. **Stop after three attempts.** If no candidate is accepted, keep the source
   unchanged and report the last verdict's reasons. Three matches the default
   bound of `lingity improve`.

## Reading the brief

| Field | What to do with it |
| --- | --- |
| `defects` | Fix these, highest ranked first. Each has an `excerpt`, a `location`, and a `remediation`. |
| `must_preserve` | Keep every element exactly: identifiers, quantities, modal terms, negation, citations, governance terms. |
| `rewrite_constraints.maximum_candidate_readable_words` | Hard ceiling on the candidate's length. Prefer shorter. |
| `current_score` | The score the candidate must strictly beat. |
| `prior_attempts` | Earlier rejection reasons. Do not repeat those mistakes. |
| `style` | Present only with `--style`. Generation guidance, not an acceptance rule. |

## Reading the verdict

| Field | Meaning |
| --- | --- |
| `accepted` | The decision. Only this field decides the outcome. |
| `rejection_reasons` | One sentence per failed gate. |
| `protected_delta.missing` | Elements the candidate dropped. Put each one back. |
| `protected_delta.added` | Elements the candidate introduced. Remove them or restore the source wording. |
| `protected_delta.unresolved` | Doubts Lingity could not settle, most often a sentence it could not read as meaning. Rewrite that sentence as a plain statement with an explicit actor and action, or keep the source sentence. |
| `protected_delta.specified` | Actors you named that the source left unnamed. Allowed. |
| `economy` | Word counts against the growth budget. `passed: false` means the candidate is too long. |

Delta entries are signatures, not prose. For example,
`quantity:count:2` means the count "two" is missing, and
`claim:action=ratify;actor=unspecified;modality=must;polarity=negative;status=deferred;target=architecture`
means the source's deferred obligation not to ratify the architecture is
missing.

## Rules

- Do not accept a candidate yourself. Only exit code `0` from `lingity judge`
  is an acceptance.
- Do not edit the source file, the brief, or the verdict.
- Do not change `--profile` between the critique and the judge to get a
  different answer.
- Do not add content to raise the score. The economy gate rejects growth past
  the budget, and new claims are reported as `added`.
- Do not describe a rejected candidate as an improvement. When nothing is
  accepted, the correct result is the unchanged source plus the reasons.

## Reporting the result

Tell the user:

- whether a candidate was accepted, and after how many attempts;
- the score before and after (`source_score`, `candidate_score`);
- when nothing was accepted, the last `rejection_reasons`.

## Styles

A style contract adds generation guidance to the brief:

```text
lingity styles
lingity critique source.md --profile architecture-review --style technical-writer --output brief.json
```

Follow the brief's `style.instructions` while writing. A style never overrides
`must_preserve`, the word budget, or the verdict. Do not pass `--style` to
`lingity improve --provider subagent`; that combination is rejected, because
subagent candidates are written before the loop runs.

## Copy-ready system prompt

```text
You improve documents with Lingity. You write candidate rewrites; Lingity alone
decides whether one is accepted.

For each document:
1. Run `lingity critique <source> --profile <profile> --output brief.json`.
2. Write a candidate to a new file. Act as a conservative editor: fix the
   brief's defects with the smallest sufficient change, keep every element in
   `must_preserve` exactly, stay under
   `rewrite_constraints.maximum_candidate_readable_words`, and add no new
   content.
3. Run `lingity judge <source> --candidate <file> --profile <profile> --output verdict.json`.
   Exit 0 means accepted: stop. Exit 2 means error: report it and stop.
4. On exit 1, read `rejection_reasons` and `protected_delta`. Restore every
   `missing` element and remove every `added` one. Record the attempt in a
   prior-attempts file, re-run critique with `--prior-attempts`, and write the
   next candidate from the source.
5. Stop after three attempts. If none was accepted, keep the source unchanged
   and report the last reasons.

Never edit the source, never treat your own judgement as acceptance, and never
call a rejected candidate an improvement.
```

## Claude Code skill

[`skills/lingity/SKILL.md`](../skills/lingity/SKILL.md) packages this loop as a
Claude Code skill. Copy the `skills/lingity` directory to
`~/.claude/skills/lingity` for every project, or to `.claude/skills/lingity`
inside one project. Claude Code then loads it when asked to improve a document
with Lingity.

## Pre-written candidates

When the host has already written candidate files, `lingity improve` runs the
judging loop over them in order:

```text
lingity improve source.md --provider subagent --candidate first.md --candidate second.md --output result.json
```

It exits `0` when a candidate is accepted and `1` when none is. The accepted
text is `selected_text` in the output, and `attempts` records every verdict.
