"""Evidence-grounded, fail-closed claim review for rewrites.

The model assesses whether each sentence is supported by supplied source
snapshots. Deterministic code checks coverage and citation provenance; it does
not claim that a model's entailment judgment proves a fact true.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final, Protocol, cast
from urllib.parse import urlsplit

from lingity.models import JsonValue
from lingity.profiles import sha256_json
from lingity.providers.base import ProviderError, ProviderResponseError
from lingity.text import sentences

CLAIM_STATUSES: Final = frozenset({"supported", "contradicted", "unresolved"})


class EvidenceError(ValueError):
    """Invalid or insufficient evidence supplied for a fact check."""


@dataclass(frozen=True)
class EvidenceSource:
    id: str
    url: str
    retrieved_at: str
    content: str
    sha256: str

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "url": self.url,
            "retrieved_at": self.retrieved_at,
            "content": self.content,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class EvidenceBundle:
    sources: tuple[EvidenceSource, ...]
    digest: str

    @staticmethod
    def from_payload(payload: object) -> EvidenceBundle:
        if not isinstance(payload, dict) or set(payload) != {"sources"}:
            raise EvidenceError("evidence bundle must contain only a sources array")
        raw_sources = payload["sources"]
        if not isinstance(raw_sources, list) or not raw_sources:
            raise EvidenceError("evidence bundle requires at least one source")
        sources: list[EvidenceSource] = []
        seen: set[str] = set()
        for index, raw in enumerate(raw_sources):
            if not isinstance(raw, dict) or set(raw) != {"id", "url", "retrieved_at", "content"}:
                raise EvidenceError(f"evidence source {index} requires id, url, retrieved_at, and content")
            values = [raw[key] for key in ("id", "url", "retrieved_at", "content")]
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise EvidenceError(f"evidence source {index} has an empty or invalid field")
            source_id, url, retrieved_at, content = cast(tuple[str, str, str, str], tuple(values))
            if source_id in seen:
                raise EvidenceError(f"duplicate evidence source id {source_id!r}")
            parsed_url = urlsplit(url)
            if parsed_url.scheme not in ("https", "http") or not parsed_url.netloc:
                raise EvidenceError(f"evidence source {source_id!r} needs an HTTP(S) URL")
            try:
                retrieved = datetime.fromisoformat(retrieved_at.replace("Z", "+00:00"))
            except ValueError as error:
                raise EvidenceError(
                    f"evidence source {source_id!r} needs an ISO 8601 retrieval time"
                ) from error
            if retrieved.tzinfo is None:
                raise EvidenceError(
                    f"evidence source {source_id!r} needs a timezone in retrieved_at"
                )
            seen.add(source_id)
            sources.append(EvidenceSource(
                source_id, url, retrieved_at, content,
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
            ))
        return EvidenceBundle(tuple(sources), sha256_json([item.to_dict() for item in sources]))

    @staticmethod
    def from_file(path: Path) -> EvidenceBundle:
        return EvidenceBundle.from_payload(json.loads(path.read_text(encoding="utf-8")))


class FactChecker(Protocol):
    @property
    def name(self) -> str: ...

    def check(self, text: str, bundle: EvidenceBundle) -> dict[str, JsonValue]: ...


def _validated_review(
    payload: Mapping[str, JsonValue], text: str, bundle: EvidenceBundle,
    provider: str, model: str,
) -> dict[str, JsonValue]:
    if set(payload) != {"provider", "model", "sentences"}:
        raise ProviderResponseError("fact-check response has missing or unexpected fields")
    if payload["provider"] != provider or payload["model"] != model:
        raise ProviderResponseError("fact-check response has an unexpected provider or model")
    raw_reviews = payload["sentences"]
    units = sentences(text)
    if not isinstance(raw_reviews, list) or len(raw_reviews) != len(units):
        raise ProviderResponseError("fact-check response must review every sentence exactly once")
    source_by_id = {source.id: source for source in bundle.sources}
    reviews: list[dict[str, JsonValue]] = []
    for index, (raw, unit) in enumerate(zip(raw_reviews, units)):
        if not isinstance(raw, dict) or set(raw) != {"index", "text", "claims", "reason"}:
            raise ProviderResponseError(f"fact-check sentence {index} has missing or unexpected fields")
        if isinstance(raw["index"], bool) or raw["index"] != index or raw["text"] != unit.text:
            raise ProviderResponseError(f"fact-check sentence {index} does not match the input")
        reason = raw["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ProviderResponseError(f"fact-check sentence {index} requires a reason")
        raw_claims = raw["claims"]
        if not isinstance(raw_claims, list):
            raise ProviderResponseError(f"fact-check sentence {index} needs a claims array")
        clean_claims: list[dict[str, JsonValue]] = []
        for claim_index, claim in enumerate(raw_claims):
            label = f"fact-check sentence {index} claim {claim_index}"
            if not isinstance(claim, dict) or set(claim) != {"text", "status", "citations", "reason"}:
                raise ProviderResponseError(f"{label} has missing or unexpected fields")
            claim_text, status = claim["text"], claim["status"]
            if not isinstance(claim_text, str) or not claim_text.strip() or claim_text not in unit.text:
                raise ProviderResponseError(f"{label} is not quoted from its sentence")
            if not isinstance(status, str) or status not in CLAIM_STATUSES:
                raise ProviderResponseError(f"{label} has an invalid status")
            claim_reason = claim["reason"]
            if not isinstance(claim_reason, str) or not claim_reason.strip():
                raise ProviderResponseError(f"{label} requires a reason")
            citations = claim["citations"]
            if not isinstance(citations, list):
                raise ProviderResponseError(f"{label} needs a citations array")
            if status in ("supported", "contradicted") and not citations:
                raise ProviderResponseError(f"{label} needs cited evidence")
            clean_citations: list[dict[str, JsonValue]] = []
            for citation in citations:
                if not isinstance(citation, dict) or set(citation) != {"source_id", "quote"}:
                    raise ProviderResponseError(f"{label} has an invalid citation")
                source_id, quote = citation["source_id"], citation["quote"]
                if not isinstance(source_id, str) or source_id not in source_by_id:
                    raise ProviderResponseError(f"{label} cites an unknown source")
                if not isinstance(quote, str) or not quote.strip() or quote not in source_by_id[source_id].content:
                    raise ProviderResponseError(f"{label} cites a quote absent from its source")
                clean_citations.append({"source_id": source_id, "quote": quote})
            clean_claims.append({
                "text": claim_text, "status": status,
                "citations": cast(JsonValue, clean_citations), "reason": claim_reason,
            })
        statuses = {item["status"] for item in clean_claims}
        sentence_status = (
            "contradicted" if "contradicted" in statuses else
            "unresolved" if "unresolved" in statuses else
            "supported" if clean_claims else "not_factual"
        )
        reviews.append({
            "index": index, "text": unit.text, "status": sentence_status,
            "claims": cast(JsonValue, clean_claims), "reason": reason,
        })
    return {
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "evidence_sha256": bundle.digest,
        "provider": provider,
        "model": model,
        "passed": bool(reviews) and all(item["status"] in ("supported", "not_factual") for item in reviews),
        "sentences": cast(JsonValue, reviews),
    }


class ModelFactChecker:
    """Ask a model for sentence assessments and validate its source citations."""

    def __init__(
        self, provider: str, model: str,
        complete_json: Callable[[str], dict[str, JsonValue]],
    ) -> None:
        if not provider or not model:
            raise ProviderError("fact checker needs an explicit provider and model")
        self._provider = provider
        self._model = model
        self._complete_json = complete_json

    @property
    def name(self) -> str:
        return self._provider

    def check(self, text: str, bundle: EvidenceBundle) -> dict[str, JsonValue]:
        units = sentences(text)
        if not units:
            raise EvidenceError("fact check requires text with at least one sentence")
        prompt = (
            "Review EVERY numbered sentence against ONLY the supplied source snapshots. "
            "Assess every factual assertion within each sentence, including dates, "
            "numbers, attribution, scope, and causal claims. Mark supported only if "
            "the cited excerpts support ALL of them. Mark contradicted for a conflict, "
            "unresolved for missing, ambiguous, or conflicting evidence, and not_factual "
            "only for a sentence with no factual assertion. Never use the input text "
            "as evidence. Never invent a source or quote.\n"
            "Return one JSON object with provider, model, and sentences. Each sentence "
            "must have index, exact text, claims, and a brief reason. For each "
            "atomic factual assertion, include a claim with text copied exactly "
            "from its sentence, status (supported, contradicted, or unresolved), "
            "citations (source_id and exact quote), and reason. Use an empty claims "
            "array only when the sentence makes no factual assertion. Include "
            "citations for supported and contradicted claims.\n"
            f"Provider: {self._provider}\nModel: {self._model}\n"
            f"Sentences: {json.dumps([{'index': unit.index, 'text': unit.text} for unit in units], ensure_ascii=False)}\n"
            f"Evidence: {json.dumps([source.to_dict() for source in bundle.sources], ensure_ascii=False)}"
        )
        return _validated_review(
            self._complete_json(prompt), text, bundle, self._provider, self._model
        )


def create_fact_checker(provider: str, model: str) -> ModelFactChecker:
    """Use an existing API transport for an evidence-backed review."""
    if provider == "openai":
        from lingity.providers.openai_provider import _OpenAIClient

        from lingity.providers.openai_provider import DEFAULT_BASE_URL, DEFAULT_TIMEOUT_SECONDS

        openai_client = _OpenAIClient(
            model=model, base_url=DEFAULT_BASE_URL,
            timeout=DEFAULT_TIMEOUT_SECONDS, transport=None,
        )
        return ModelFactChecker(provider, openai_client.model, lambda prompt: openai_client.complete_json(prompt, "fact check"))
    if provider == "anthropic":
        from lingity.providers.anthropic_provider import (
            DEFAULT_ANTHROPIC_VERSION, DEFAULT_BASE_URL, DEFAULT_TIMEOUT_SECONDS,
            _AnthropicMessagesClient,
        )

        anthropic_client = _AnthropicMessagesClient(
            model=model, base_url=DEFAULT_BASE_URL, timeout=DEFAULT_TIMEOUT_SECONDS,
            anthropic_version=DEFAULT_ANTHROPIC_VERSION, transport=None,
        )
        return ModelFactChecker(provider, anthropic_client.model, lambda prompt: anthropic_client.send_prompt(prompt, 4096))
    raise ProviderError(f"unknown fact-check provider {provider!r}; choose openai or anthropic")
