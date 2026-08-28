"""Subagent handoff provider.

This is the default transport, and the one Lingity expects to run under inside
an agent host such as Agency. There is no network call and no API key. The
host agent *is* the model: Lingity hands it a critique brief, the host writes a
candidate, and Lingity judges the result.

Two shapes are supported:

``lingity critique`` / ``lingity judge``
    The interactive shape. The host agent reads a brief, writes a candidate,
    and asks Lingity for a verdict. Each command is a separate process, so the
    host keeps full control of the conversation.

:class:`SubagentProvider`
    The scripted shape, used by :func:`lingity.improve.improve_text` when the
    host has already produced one or more candidate files. Each attempt
    consumes the next candidate in order.

The provider is deliberately incapable of judging anything. It only carries
text. When it runs out of candidates it says so explicitly rather than
resubmitting the last one or returning the source unchanged, either of which
would let a bounded loop report a success it never earned.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from lingity.models import JsonValue
from lingity.providers.base import (
    ProposalRequest,
    ProposalResponse,
    ProviderError,
    ProviderExhausted,
)

SUBAGENT_MODEL: Final = "host-agent"


class SubagentProvider:
    """Serves candidates written by the host agent, in order."""

    def __init__(
        self,
        candidate_paths: Sequence[str | Path],
        *,
        model: str = SUBAGENT_MODEL,
        encoding: str = "utf-8",
    ) -> None:
        if not candidate_paths:
            raise ProviderError(
                "the subagent provider needs at least one candidate file; "
                "the host agent must write a candidate before the loop runs"
            )
        if not model:
            raise ProviderError("the subagent provider requires a non-empty model label")
        self._paths = [Path(path) for path in candidate_paths]
        self._model = model
        self._encoding = encoding
        self._served = 0

    @property
    def name(self) -> str:
        return "subagent"

    @property
    def remaining(self) -> int:
        return len(self._paths) - self._served

    def propose(self, request: ProposalRequest) -> ProposalResponse:
        # Touch the brief so a malformed one fails here rather than downstream.
        request.critique_sha256

        if self._served >= len(self._paths):
            raise ProviderExhausted(
                f"the subagent provider has already served all "
                f"{len(self._paths)} candidate file(s); the host agent must "
                "write another candidate before another attempt can be judged"
            )

        path = self._paths[self._served]
        self._served += 1

        if not path.is_file():
            raise ProviderError(
                f"candidate file {path} does not exist; the host agent was "
                "expected to write the rewritten text there"
            )
        text = path.read_text(encoding=self._encoding)
        if not text.strip():
            raise ProviderError(
                f"candidate file {path} is empty; an empty rewrite cannot be "
                "judged and is never treated as an improvement"
            )

        payload: dict[str, JsonValue] = {
            "candidate_text": text,
            "addressed_rule_ids": [],
            "claimed_preservations": [],
            "provider": self.name,
            "model": self._model,
        }
        return ProposalResponse.from_payload(payload, self.name)


def create_subagent_provider(**options: object) -> SubagentProvider:
    """Registry factory for the subagent transport."""

    raw_paths = options.get("candidate_paths")
    if raw_paths is None:
        raise ProviderError(
            "the subagent provider requires a 'candidate_paths' option naming "
            "the file(s) the host agent wrote"
        )
    if isinstance(raw_paths, (str, Path)):
        paths: Sequence[str | Path] = [raw_paths]
    elif isinstance(raw_paths, Sequence):
        paths = [item for item in raw_paths if isinstance(item, (str, Path))]
        if len(paths) != len(raw_paths):
            raise ProviderError("every candidate path must be a string or Path")
    else:
        raise ProviderError(
            "'candidate_paths' must be a path or a sequence of paths; received "
            f"{type(raw_paths).__name__}"
        )

    model = options.get("model", SUBAGENT_MODEL)
    if not isinstance(model, str):
        raise ProviderError("'model' must be a string label for the host agent")
    return SubagentProvider(paths, model=model)
