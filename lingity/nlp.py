"""Deterministic linguistic pipeline backed by a pinned spaCy model.

The analyzer resolves grammar from part-of-speech tags and dependency arcs
rather than from surface patterns. Reproducibility therefore depends on the
exact pipeline that produced a parse, so the pipeline identity is fingerprinted
and published in every analysis artifact.

Loading is fail-closed: a missing, unreadable, or unexpected model raises
``LinguisticModelError`` instead of degrading to a weaker analysis.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spacy.language import Language

MODEL_NAME = "en_core_web_sm"
MODEL_VERSION = "3.8.0"
MODEL_INSTALL_HINT = (
    "Install the pinned model with: python -m pip install "
    "https://github.com/explosion/spacy-models/releases/download/"
    f"{MODEL_NAME}-{MODEL_VERSION}/{MODEL_NAME}-{MODEL_VERSION}-py3-none-any.whl"
)


class LinguisticModelError(RuntimeError):
    """Raised when the pinned linguistic pipeline cannot be used as published."""


@dataclass(frozen=True)
class Token:
    """A parsed token with its grammatical annotation.

    ``index`` is the document-level token offset and ``head`` is the
    document-level offset of this token's syntactic head; a token that heads its
    own clause refers to itself.
    """

    index: int
    text: str
    start: int
    end: int
    lemma: str
    pos: str
    tag: str
    dep: str
    head: int
    morph: str
    entity_type: str
    is_alpha: bool
    is_stop: bool
    is_punct: bool
    sentence_index: int

    @property
    def lower(self) -> str:
        return self.text.lower()

    @property
    def is_word(self) -> bool:
        """True for tokens that count as words for load and density measures."""
        return self.is_alpha or any(character.isalnum() for character in self.text)


@dataclass(frozen=True)
class Sentence:
    index: int
    text: str
    start: int
    end: int
    tokens: tuple[Token, ...]
    root: int

    @property
    def words(self) -> tuple[Token, ...]:
        return tuple(token for token in self.tokens if token.is_word)


@dataclass(frozen=True)
class Document:
    """A parsed document exposing sentence and dependency structure."""

    text: str
    sentences: tuple[Sentence, ...]
    tokens: tuple[Token, ...]

    def token(self, index: int) -> Token:
        return self.tokens[index]

    def head_of(self, token: Token) -> Token:
        return self.tokens[token.head]

    def children(self, token: Token) -> tuple[Token, ...]:
        return tuple(
            candidate
            for candidate in self.tokens
            if candidate.head == token.index and candidate.index != token.index
        )

    def subtree(self, token: Token) -> tuple[Token, ...]:
        collected: dict[int, Token] = {token.index: token}
        frontier = [token]
        while frontier:
            current = frontier.pop()
            for child in self.children(current):
                if child.index not in collected:
                    collected[child.index] = child
                    frontier.append(child)
        return tuple(collected[index] for index in sorted(collected))

    def sentence_of(self, token: Token) -> Sentence:
        return self.sentences[token.sentence_index]

    def __iter__(self) -> Iterator[Token]:
        return iter(self.tokens)


def _model_directory(package_path: Path) -> Path:
    candidates = sorted(
        entry
        for entry in package_path.iterdir()
        if entry.is_dir() and entry.name.startswith(f"{MODEL_NAME}-")
    )
    if not candidates:
        raise LinguisticModelError(
            f"Installed {MODEL_NAME} package contains no model data directory. "
            f"{MODEL_INSTALL_HINT}"
        )
    return candidates[-1]


@lru_cache(maxsize=1)
def _digest() -> str:
    """Hash the on-disk model so a silent model swap cannot go unnoticed."""
    import spacy.util

    try:
        package_path = Path(spacy.util.get_package_path(MODEL_NAME))
    except (OSError, ImportError) as error:
        raise LinguisticModelError(
            f"Linguistic model {MODEL_NAME} is not installed. {MODEL_INSTALL_HINT}"
        ) from error
    if not package_path.is_dir():
        raise LinguisticModelError(
            f"Installed {MODEL_NAME} package has no readable location at {package_path}. "
            f"{MODEL_INSTALL_HINT}"
        )
    directory = _model_directory(package_path)
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_pipeline() -> "Language":
    """Load the pinned pipeline, refusing any version other than the published one."""
    try:
        import spacy
    except ImportError as error:
        raise LinguisticModelError(
            "spaCy is not installed. Install project dependencies with: "
            "python -m pip install -e ."
        ) from error
    try:
        pipeline = spacy.load(MODEL_NAME)
    except OSError as error:
        raise LinguisticModelError(
            f"Linguistic model {MODEL_NAME} is not installed. {MODEL_INSTALL_HINT}"
        ) from error
    installed_version = str(pipeline.meta["version"])
    if installed_version != MODEL_VERSION:
        raise LinguisticModelError(
            f"Linguistic model {MODEL_NAME} version {installed_version} is installed, "
            f"but this analyzer is pinned to {MODEL_VERSION}. Analyses are only "
            f"reproducible against the pinned version. {MODEL_INSTALL_HINT}"
        )
    required = ("tagger", "parser", "attribute_ruler", "lemmatizer")
    missing = [name for name in required if name not in pipeline.pipe_names]
    if missing:
        raise LinguisticModelError(
            f"Linguistic model {MODEL_NAME} is missing required components: "
            f"{', '.join(missing)}. {MODEL_INSTALL_HINT}"
        )
    return pipeline


def model_fingerprint() -> dict[str, str]:
    """Publish the pipeline identity that a parse is reproducible against."""
    from importlib.metadata import version

    load_pipeline()
    return {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "runtime": f"spacy-{version('spacy')}",
        "digest": _digest(),
    }


def parse(text: str) -> Document:
    """Parse ``text`` into sentences and dependency-annotated tokens."""
    pipeline = load_pipeline()
    doc = pipeline(text)
    tokens: list[Token] = []
    sentence_index_of: dict[int, int] = {}
    for sentence_index, span in enumerate(doc.sents):
        for token in span:
            sentence_index_of[token.i] = sentence_index
    for token in doc:
        tokens.append(
            Token(
                index=token.i,
                text=token.text,
                start=token.idx,
                end=token.idx + len(token.text),
                lemma=token.lemma_.lower(),
                pos=token.pos_,
                tag=token.tag_,
                dep=token.dep_,
                head=token.head.i,
                morph=str(token.morph),
                entity_type=token.ent_type_,
                is_alpha=token.is_alpha,
                is_stop=token.is_stop,
                is_punct=token.is_punct,
                sentence_index=sentence_index_of[token.i],
            )
        )
    frozen = tuple(tokens)
    sentences: list[Sentence] = []
    for sentence_index, span in enumerate(doc.sents):
        span_tokens = tuple(
            frozen[token.i] for token in span if not (token.is_space and not token.text.strip())
        )
        if not span_tokens:
            continue
        start = span_tokens[0].start
        end = span_tokens[-1].end
        sentences.append(
            Sentence(
                index=len(sentences),
                text=text[start:end],
                start=start,
                end=end,
                tokens=span_tokens,
                root=span.root.i,
            )
        )
    return Document(text=text, sentences=tuple(sentences), tokens=frozen)
