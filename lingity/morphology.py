"""Canonical action keys from WordNet, with symmetric stemming as fallback.

Lingity compares parsed claims by action key, so verb lemmas and their
nominalizations need to converge on the same representation.  The primary path
uses WordNet derivational links and canonicalizes every verb in a small
derivational component to the same deterministic representative.  This avoids
asymmetries such as ``approval`` linking to both ``approve`` and ``approbate``.

When WordNet has no derivational action for a lemma, domain jargon falls back
to an ordered symmetric stemmer.  Those fallback stems are comparison keys, not
English words: for example, ``conclude`` and ``conclusion`` become ``conclus``.

The fallback minimum generated stem length is four characters.  That is the
shortest length needed by the validation corpus for productive pairs such as
``close``/``closure``, ``fail``/``failure``, and ``apply``-like final-y verb
endings, while protecting short base actions such as ``be``, ``use``, and
``make`` from destructive final-vowel stripping.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

_MIN_STEM_LENGTH = 4
_MIN_GERUND_BASE_LENGTH = 3
_WORDNET_CLOSURE_DEPTH = 1
_VOWELS = "aeiou"
_WORDNET_INSTALL_HINT = (
    "Install the corpus without analysis-time network access by preparing the "
    "environment with: python -m nltk.downloader wordnet"
)

CanonicalSource = Literal["wordnet", "stemmer"]


class WordNetDataError(RuntimeError):
    """Raised when the WordNet corpus cannot be used for canonicalization."""


@dataclass(frozen=True)
class ActionCanonicalization:
    """Canonical action key plus provenance for analyzer reporting."""

    key: str
    source: CanonicalSource
    wordnet_component: tuple[str, ...] = ()


def canonical_action(lemma: str) -> str:
    """Canonical key for an action, shared by a verb and its nominalizations."""

    return canonical_action_info(lemma).key


@lru_cache(maxsize=4096)
def canonical_action_info(lemma: str) -> ActionCanonicalization:
    """Return the canonical action key and the layer that produced it."""

    word = _normalize_lemma(lemma)
    if not word:
        return ActionCanonicalization(key="", source="stemmer")

    wordnet_component = _wordnet_verb_component(word, _WORDNET_CLOSURE_DEPTH)
    if wordnet_component:
        return ActionCanonicalization(
            key=wordnet_component[0],
            source="wordnet",
            wordnet_component=wordnet_component,
        )

    for gerund_base in _gerund_base_candidates(word):
        wordnet_component = _wordnet_verb_component(
            gerund_base,
            _WORDNET_CLOSURE_DEPTH,
        )
        if wordnet_component:
            return ActionCanonicalization(
                key=wordnet_component[0],
                source="wordnet",
                wordnet_component=wordnet_component,
            )

    return ActionCanonicalization(key=action_stem(word), source="stemmer")


def action_stem(lemma: str) -> str:
    """Reduce an action lemma to a shared verb/nominalization stem."""

    word = _normalize_lemma(lemma)
    if not word:
        return ""

    previous = ""
    while word != previous:
        previous = word
        word = _stem_once(word)

    return word


def _wordnet_verb_component(word: str, max_depth: int) -> tuple[str, ...]:
    try:
        return _wordnet_verb_component_unchecked(word, max_depth)
    except LookupError as error:
        raise WordNetDataError(
            f"WordNet corpus data is unavailable. {_WORDNET_INSTALL_HINT}"
        ) from error


def _wordnet_verb_component_unchecked(word: str, max_depth: int) -> tuple[str, ...]:
    wordnet = _load_wordnet()
    seen = _wordnet_verb_seeds(wordnet, word)
    if not seen:
        return ()

    frontier = set(seen)
    for _step in range(max_depth):
        next_frontier: set[str] = set()
        for verb in frontier:
            for noun in _related_nouns(wordnet, verb):
                next_frontier.update(_related_verbs(wordnet, noun))
        next_frontier -= seen
        if not next_frontier:
            break
        seen.update(next_frontier)
        frontier = next_frontier

    return tuple(sorted(seen))


def _wordnet_verb_seeds(wordnet: Any, word: str) -> set[str]:
    verbs: set[str] = set()
    for verb_lemma in _matching_wordnet_lemmas(wordnet, word, wordnet.VERB):
        if _has_related_pos(verb_lemma, wordnet.NOUN):
            verbs.add(_wordnet_surface(verb_lemma.name()))

    for noun_lemma in _matching_wordnet_lemmas(wordnet, word, wordnet.NOUN):
        for related in noun_lemma.derivationally_related_forms():
            if related.synset().pos() == wordnet.VERB:
                verbs.add(_wordnet_surface(related.name()))

    return verbs


def _related_nouns(wordnet: Any, verb: str) -> set[str]:
    nouns: set[str] = set()
    for verb_lemma in _matching_wordnet_lemmas(wordnet, verb, wordnet.VERB):
        for related in verb_lemma.derivationally_related_forms():
            if related.synset().pos() == wordnet.NOUN:
                nouns.add(_wordnet_surface(related.name()))
    return nouns


def _related_verbs(wordnet: Any, noun: str) -> set[str]:
    verbs: set[str] = set()
    for noun_lemma in _matching_wordnet_lemmas(wordnet, noun, wordnet.NOUN):
        for related in noun_lemma.derivationally_related_forms():
            if related.synset().pos() == wordnet.VERB:
                verbs.add(_wordnet_surface(related.name()))
    return verbs


def _matching_wordnet_lemmas(wordnet: Any, word: str, pos: str) -> tuple[Any, ...]:
    key = _wordnet_key(word)
    lemmas: list[Any] = []
    for synset in wordnet.synsets(key, pos=pos):
        for lemma in synset.lemmas():
            if str(lemma.name()).lower() == key:
                lemmas.append(lemma)
    return tuple(lemmas)


def _has_related_pos(lemma: Any, pos: str) -> bool:
    return any(
        related.synset().pos() == pos
        for related in lemma.derivationally_related_forms()
    )


def _load_wordnet() -> Any:
    try:
        corpus = importlib.import_module("nltk.corpus")
    except ImportError as error:
        raise WordNetDataError(
            f"nltk is not installed, so WordNet canonicalization is unavailable. "
            f"{_WORDNET_INSTALL_HINT}"
        ) from error

    wordnet = getattr(corpus, "wordnet")
    wordnet.ensure_loaded()
    return wordnet


def _normalize_lemma(lemma: str) -> str:
    return lemma.strip().lower()


def _wordnet_key(word: str) -> str:
    return word.replace(" ", "_")


def _wordnet_surface(word: str) -> str:
    return word.lower().replace("_", " ")


def _stem_once(word: str) -> str:
    gerund_base = _strip_gerund(word)
    if gerund_base is not None:
        return gerund_base

    candidate = _replace_suffix(word, "ification", "if")
    if candidate is not None:
        return candidate

    candidate = _replace_suffix(word, "ization", "iz")
    if candidate is not None:
        return candidate

    candidate = _replace_suffix(word, "isation", "iz")
    if candidate is not None:
        return candidate

    candidate = _replace_suffix(word, "ation", "at")
    if candidate is not None:
        return candidate

    candidate = _replace_suffix(word, "lution", "lv")
    if candidate is not None:
        return candidate

    candidate = _replace_suffix(word, "ction", "c")
    if candidate is not None:
        return candidate

    candidate = _replace_suffix_after_vowel(word, "sion", "s")
    if candidate is not None:
        return candidate

    candidate = _replace_suffix(word, "tion", "t")
    if candidate is not None:
        return candidate

    candidate = _strip_suffix(word, "ment")
    if candidate is not None:
        return candidate

    candidate = _strip_suffix(word, "ance", undouble=True)
    if candidate is not None:
        return candidate

    candidate = _strip_suffix(word, "ence", undouble=True)
    if candidate is not None:
        return candidate

    candidate = _strip_suffix(word, "ure")
    if candidate is not None:
        return candidate

    candidate = _strip_suffix(word, "al")
    if candidate is not None:
        return candidate

    if (word.endswith("ide") or word.endswith("ude")) and not word.endswith("uide"):
        candidate = f"{word[:-2]}s"
        if _is_long_enough(candidate):
            return candidate

    if word.endswith("ct"):
        candidate = word[:-1]
        if _is_long_enough(candidate):
            return candidate

    if word.endswith("y") and len(word) > 1 and word[-2] not in _VOWELS:
        candidate = word[:-1]
        if _is_long_enough(candidate):
            return candidate

    if word.endswith("e"):
        candidate = word[:-1]
        if _is_long_enough(candidate):
            return candidate

    return word


def _replace_suffix(word: str, suffix: str, replacement: str) -> str | None:
    if not word.endswith(suffix):
        return None

    candidate = f"{word[: -len(suffix)]}{replacement}"
    if not _is_long_enough(candidate):
        return None

    return candidate


def _replace_suffix_after_vowel(
    word: str,
    suffix: str,
    replacement: str,
) -> str | None:
    prefix = word[: -len(suffix)] if word.endswith(suffix) else ""
    if not prefix or prefix[-1] not in _VOWELS:
        return None

    return _replace_suffix(word, suffix, replacement)


def _strip_suffix(word: str, suffix: str, *, undouble: bool = False) -> str | None:
    if not word.endswith(suffix):
        return None

    candidate = word[: -len(suffix)]
    if undouble:
        candidate = _undouble_final_consonant(candidate)

    if not _is_long_enough(candidate):
        return None

    return candidate


def _undouble_final_consonant(stem: str) -> str:
    if len(stem) < 2 or stem[-1] != stem[-2] or stem[-1] in _VOWELS:
        return stem

    return stem[:-1]


def _gerund_base_candidates(word: str) -> tuple[str, ...]:
    stripped = _raw_gerund_base(word)
    if stripped is None:
        return ()

    candidates = [stripped]
    with_e = f"{stripped}e"
    if with_e not in candidates:
        candidates.append(with_e)
    return tuple(candidates)


def _strip_gerund(word: str) -> str | None:
    stripped = _raw_gerund_base(word)
    if stripped is None:
        return None

    if (
        stripped.endswith(("ag", "dg", "c", "v", "u"))
        or stripped.endswith(("bl", "dl", "gl", "kl", "pl", "sl", "tl", "zl"))
    ):
        return f"{stripped}e"

    return stripped


def _raw_gerund_base(word: str) -> str | None:
    if not word.endswith("ing"):
        return None

    stem = word[:-3]
    if len(stem) < _MIN_GERUND_BASE_LENGTH:
        return None

    undoubled = _undouble_final_consonant(stem)
    if len(undoubled) < _MIN_GERUND_BASE_LENGTH:
        return None

    return undoubled


def _is_long_enough(stem: str) -> bool:
    return len(stem) >= _MIN_STEM_LENGTH
