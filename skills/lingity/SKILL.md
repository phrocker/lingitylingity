---
name: lingity
description: Improve the readability of a Markdown or plain-text document with Lingity while keeping its protected meaning (facts, identifiers, quantities, modality, negation, citations, governance terms) unchanged. Use when the user asks to improve, tighten, or clarify a document with Lingity, or to check whether a rewrite preserves a document's meaning.
---

# Lingity

You write candidate rewrites. Lingity alone decides whether one is accepted.
The full guide is `docs/agent-harness.md` in the Lingity repository.

## Setup

Run `lingity --help`. If it fails, tell the user to install Lingity as its
README describes, and stop. Do not work around a prerequisite error.

Pick a profile: `architecture-review` (default), `product-strategy`,
`resume-review`, `web-copy`, or `local-service`. Use the same profile in every
command.

## Loop

Never edit the source file. Write candidates to new files.

1. `lingity critique <source> --profile <profile> --output brief.json`
2. Write a candidate as a conservative editor:
   - fix the brief's `defects` with the smallest sufficient change;
   - keep every element in `must_preserve` exactly;
   - stay under `rewrite_constraints.maximum_candidate_readable_words`;
   - add no introductions, summaries, examples, or new claims.
3. `lingity judge <source> --candidate <candidate> --profile <profile> --output verdict.json`
   - exit `0`: accepted. Stop.
   - exit `1`: rejected. Go to step 4.
   - exit `2`: error. Report it and stop.
4. Read `rejection_reasons` and `protected_delta` in the verdict. Restore every
   `missing` element, remove every `added` one, and rewrite `unresolved`
   sentences as plain statements or keep the source sentence. Add the attempt
   to `attempts.json` as `{"index": N, "rejection_reasons": [...]}`, re-run
   step 1 with `--prior-attempts attempts.json`, and write the next candidate
   from the source.
5. Stop after three attempts. If none was accepted, keep the source unchanged.

## Rules

- Only exit code `0` from `lingity judge` is an acceptance.
- Do not change the profile between critique and judge.
- Do not call a rejected candidate an improvement.

## Report

Say whether a candidate was accepted and after how many attempts, give the
score before and after (`source_score`, `candidate_score`), and when nothing was
accepted, give the last `rejection_reasons`.
