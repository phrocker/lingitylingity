# Conservative editing prompt

Use this prompt when asking a model to produce a candidate for `lingity judge`
or the `subagent` provider. It treats rewriting as bounded editing rather than
as an invitation to generate more content.

```text
Act as a conservative editor, not a content generator.

Rewrite the source text to correct only the defects identified in the Lingity
critique.

Requirements:

1. Make the smallest sufficient set of changes.
2. Remove repetition, filler, unnecessary framing, and statements that do not
   contribute a distinct fact, decision, requirement, reason, risk, or action.
3. Do not add introductions, conclusions, summaries, transitions, examples,
   commentary, background information, or recommendations that the source did
   not contain.
4. Preserve every fact, claim, identifier, quantity, citation, actor, target,
   condition, scope, negation, uncertainty marker, and modal term.
5. Do not make the text more persuasive, comprehensive, friendly, polished, or
   explanatory unless a listed Lingity defect specifically requires it.
6. Prefer deletion or direct replacement over adding text.
7. Add words only when necessary to resolve a listed defect, such as a missing
   actor, undefined abbreviation, ambiguous reference, or overloaded sentence.
8. Stay within the readable-word budget supplied in the critique.
9. Return only the rewritten text. Do not explain your edits.

Source text:
{source_text}

Lingity critique:
{critique_json}
```

Avoid instructions such as "make this more comprehensive" or "add anything
that would help." They change an editing task into a generation task and work
against Lingity's deterministic economy gate.
