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
from typing import TYPE_CHECKING, Any, Iterator, Sequence

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
    spans: tuple[tuple[int, int], ...] = ()
    """The character ranges of ``text`` that were parsed as prose."""

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


def parse(
    text: str, spans: Sequence[Sequence[tuple[int, int]]] | None = None
) -> Document:
    """Parse ``text`` into sentences and dependency-annotated tokens.

    ``spans`` restricts the parse to the listed groups of character ranges. Each
    group is parsed as one unit and each group separately, so a sentence can
    never run across a structural boundary such as a heading or a table, while a
    sentence wrapped across two source lines of the same block stays one
    sentence. The pieces of a group are joined by a single space, because what
    separates them in the source is container markup.

    Offsets stay relative to ``text``, so a finding still locates itself in the
    source the author wrote. Passing ``None`` parses the whole text as one group.
    """
    pipeline = load_pipeline()
    if spans is None:
        groups: tuple[tuple[tuple[int, int], ...], ...] = (((0, len(text)),),)
    else:
        groups = tuple(
            tuple((start, end) for start, end in group if text[start:end].strip())
            for group in spans
        )

    tokens: list[Token] = []
    sentences: list[Sentence] = []
    for group in groups:
        if not group:
            continue
        # Build the parse unit and the map back to source offsets. A token never
        # straddles a join, because the join is a space and the tokenizer splits
        # on whitespace.
        parts: list[str] = []
        placement: list[tuple[int, int, int]] = []
        cursor = 0
        for start, end in group:
            piece = text[start:end]
            if parts:
                parts.append(" ")
                cursor += 1
            placement.append((cursor, cursor + len(piece), start))
            parts.append(piece)
            cursor += len(piece)
        chunk = "".join(parts)

        def to_source(index: int, placement: list[tuple[int, int, int]] = placement) -> int:
            for chunk_start, chunk_end, source_start in placement:
                if chunk_start <= index < chunk_end:
                    return source_start + (index - chunk_start)
            last_start, last_end, last_source = placement[-1]
            return last_source + (last_end - last_start)

        doc = pipeline(chunk)
        base = len(tokens)
        sentence_base = len(sentences)

        kept_per_sentence: list[list[Any]] = []
        provisional: dict[int, int] = {}
        for local_index, span in enumerate(doc.sents):
            kept_per_sentence.append(
                [token for token in span if not (token.is_space and not token.text.strip())]
            )
            for token in span:
                provisional[token.i] = local_index

        # A sentence that holds only whitespace is dropped, so translate the
        # provisional index to the index the sentence actually receives.
        translate: dict[int, int] = {}
        surviving = 0
        latest = sentence_base
        for local_index, kept in enumerate(kept_per_sentence):
            if kept:
                latest = sentence_base + surviving
                translate[local_index] = latest
                surviving += 1
            else:
                translate[local_index] = latest

        for token in doc:
            start = to_source(token.idx)
            tokens.append(
                Token(
                    index=base + token.i,
                    text=token.text,
                    start=start,
                    end=start + len(token.text),
                    lemma=token.lemma_.lower(),
                    pos=token.pos_,
                    tag=token.tag_,
                    dep=token.dep_,
                    head=base + token.head.i,
                    morph=str(token.morph),
                    entity_type=token.ent_type_,
                    is_alpha=token.is_alpha,
                    is_stop=token.is_stop,
                    is_punct=token.is_punct,
                    sentence_index=translate[provisional[token.i]],
                )
            )

        for span, kept in zip(doc.sents, kept_per_sentence):
            if not kept:
                continue
            span_tokens = tuple(tokens[base + token.i] for token in kept)
            # The sentence reads as parsed, not as sliced from the source: a
            # sentence spanning a join would otherwise show the markup between
            # its lines.
            sentences.append(
                Sentence(
                    index=len(sentences),
                    text=chunk[kept[0].idx : kept[-1].idx + len(kept[-1].text)],
                    start=span_tokens[0].start,
                    end=span_tokens[-1].end,
                    tokens=span_tokens,
                    root=base + span.root.i,
                )
            )

    return Document(
        text=text,
        sentences=tuple(sentences),
        tokens=tuple(tokens),
        spans=tuple(span for group in groups for span in group),
    )
