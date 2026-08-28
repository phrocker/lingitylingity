"""Typed records used by the deterministic analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True)
class Location:
    start: int
    end: int
    line: int
    column: int
    sentence_index: int | None = None

    def to_dict(self) -> dict[str, JsonValue]:
        value: dict[str, JsonValue] = {
            "start": self.start,
            "end": self.end,
            "line": self.line,
            "column": self.column,
        }
        if self.sentence_index is not None:
            value["sentence_index"] = self.sentence_index
        return value


@dataclass(frozen=True)
class Finding:
    rule_id: str
    dimension: str
    severity: str
    location: Location
    observed_value: JsonValue
    threshold: JsonValue
    remediation: str
    penalty: float

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "rule_id": self.rule_id,
            "dimension": self.dimension,
            "severity": self.severity,
            "location": self.location.to_dict(),
            "observed_value": self.observed_value,
            "threshold": self.threshold,
            "remediation": self.remediation,
            "penalty": self.penalty,
        }
