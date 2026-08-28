"""Small deterministic lexicons for English architecture-review analysis."""

from __future__ import annotations

ACTION_VERBS = frozenset(
    {
        "accept",
        "add",
        "address",
        "adopt",
        "align",
        "analyze",
        "approve",
        "assess",
        "assign",
        "audit",
        "avoid",
        "begin",
        "bring",
        "build",
        "carry",
        "change",
        "check",
        "choose",
        "chose",
        "clarify",
        "collect",
        "complete",
        "configure",
        "confirm",
        "constrain",
        "control",
        "coordinate",
        "create",
        "decide",
        "defer",
        "define",
        "deliver",
        "deploy",
        "design",
        "detect",
        "determine",
        "document",
        "enforce",
        "ensure",
        "evaluate",
        "fix",
        "found",
        "govern",
        "handle",
        "help",
        "identify",
        "implement",
        "improve",
        "isolate",
        "keep",
        "limit",
        "lose",
        "maintain",
        "make",
        "manage",
        "matter",
        "migrate",
        "mitigate",
        "monitor",
        "move",
        "name",
        "observe",
        "operate",
        "plan",
        "preserve",
        "prevent",
        "propose",
        "provide",
        "publish",
        "ratify",
        "recommend",
        "reduce",
        "reject",
        "require",
        "resolve",
        "review",
        "run",
        "secure",
        "select",
        "separate",
        "split",
        "state",
        "support",
        "test",
        "track",
        "treat",
        "unblock",
        "update",
        "use",
        "validate",
        "verify",
        "write",
    }
)

AUXILIARIES = frozenset(
    {
        "am",
        "are",
        "be",
        "been",
        "being",
        "did",
        "do",
        "does",
        "had",
        "has",
        "have",
        "is",
        "was",
        "were",
    }
)

MODALS = frozenset(
    {
        "can",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
    }
)

DETERMINERS = frozenset(
    {
        "a",
        "all",
        "an",
        "any",
        "each",
        "every",
        "no",
        "our",
        "several",
        "some",
        "that",
        "the",
        "these",
        "this",
        "those",
    }
)

PREPOSITIONS = frozenset(
    {
        "about",
        "after",
        "against",
        "as",
        "at",
        "before",
        "between",
        "by",
        "during",
        "for",
        "from",
        "in",
        "into",
        "of",
        "on",
        "through",
        "to",
        "under",
        "with",
        "without",
    }
)

CONJUNCTIONS = frozenset({"and", "but", "or", "because", "although", "while", "unless"})

SUBJECT_PRONOUNS = frozenset({"he", "it", "she", "they", "we", "who", "you"})

COMMON_SUBJECT_NOUNS = frozenset(
    {
        "architecture",
        "architects",
        "component",
        "components",
        "design",
        "endpoint",
        "gateway",
        "owner",
        "owners",
        "platform",
        "review",
        "service",
        "services",
        "system",
        "systems",
        "team",
        "teams",
        "telemetry",
    }
)

COMMON_ADJECTIVES = frozenset(
    {
        "advanced",
        "available",
        "clear",
        "concise",
        "critical",
        "current",
        "direct",
        "eventual",
        "existing",
        "explicit",
        "final",
        "formal",
        "future",
        "immediate",
        "irreversible",
        "named",
        "new",
        "next",
        "open",
        "operational",
        "plain",
        "provisional",
        "regional",
        "recommended",
        "required",
        "responsible",
        "several",
        "asynchronous",
        "target",
    }
)

IRREGULAR_PAST_PARTICIPLES = frozenset(
    {
        "built",
        "done",
        "found",
        "given",
        "known",
        "made",
        "run",
        "sent",
        "set",
        "shown",
        "taken",
        "written",
    }
)


def third_person_singular(verb: str) -> str:
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return f"{verb[:-1]}ies"
    if verb.endswith(("ch", "sh", "s", "x", "z", "o")):
        return f"{verb}es"
    return f"{verb}s"


def regular_past(verb: str) -> str:
    if verb.endswith("e"):
        return f"{verb}d"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return f"{verb[:-1]}ied"
    return f"{verb}ed"


FINITE_ACTION_FORMS = frozenset(third_person_singular(verb) for verb in ACTION_VERBS)
PAST_ACTION_FORMS = frozenset(regular_past(verb) for verb in ACTION_VERBS) | IRREGULAR_PAST_PARTICIPLES
PASSIVE_PARTICIPLES = PAST_ACTION_FORMS | frozenset(
    {
        "approved",
        "built",
        "completed",
        "documented",
        "implemented",
        "made",
        "recommended",
        "required",
        "resolved",
        "reviewed",
        "validated",
        "verified",
    }
)
