"""Versioned profile loading and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from lingity.models import JsonValue

PACKAGE_DIR = Path(__file__).resolve().parent
PROFILE_DIR = PACKAGE_DIR / "profiles"
SCHEMA_DIR = PACKAGE_DIR / "schemas" / "v1"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Profile:
    data: dict[str, Any]
    digest: str

    @property
    def name(self) -> str:
        return cast(str, self.data["name"])

    @property
    def version(self) -> str:
        return cast(str, self.data["version"])

    @property
    def thresholds(self) -> dict[str, int | float]:
        return cast(dict[str, int | float], self.data["thresholds"])

    @property
    def weights(self) -> dict[str, int]:
        return cast(dict[str, int], self.data["weights"])

    @property
    def rules(self) -> dict[str, Any]:
        return cast(dict[str, Any], self.data["rules"])

    def reference(self) -> dict[str, JsonValue]:
        return {"name": self.name, "version": self.version, "digest": self.digest}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Required JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], value)


def load_profile(name: str = "architecture-review") -> Profile:
    matches = sorted(PROFILE_DIR.glob(f"{name}.v*.json"))
    if not matches:
        raise ValueError(f"Unknown profile: {name}")
    if len(matches) != 1:
        raise ValueError(f"Profile name is ambiguous: {name}")
    data = _load_json(matches[0])
    schema = _load_json(SCHEMA_DIR / "profile.schema.json")
    Draft202012Validator(schema).validate(data)
    weights = cast(dict[str, int], data["weights"])
    if sum(weights.values()) != 100:
        raise ValueError(f"Profile weights must total 100, got {sum(weights.values())}")
    return Profile(data=data, digest=sha256_json(data))
