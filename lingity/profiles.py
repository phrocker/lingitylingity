"""Versioned profile loading and validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from lingity.models import JsonValue

PACKAGE_DIR = Path(__file__).resolve().parent
PROFILE_DIR = PACKAGE_DIR / "profiles"
SCHEMA_DIR = PACKAGE_DIR / "schemas" / "v1"
PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


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


def _profile_options(profile_paths: dict[str, list[Path]]) -> str:
    names = sorted(profile_paths)
    if not names:
        return "(none installed)"
    return ", ".join(names)


def _available_profile_paths() -> dict[str, list[Path]]:
    profiles: dict[str, list[Path]] = {}
    for path in sorted(PROFILE_DIR.glob("*.v*.json")):
        name = path.name.rsplit(".v", 1)[0]
        if name:
            profiles.setdefault(name, []).append(path)
    return profiles


def _schema_location(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    return path or "<root>"


def _validate_profile_schema(data: dict[str, Any], path: Path) -> None:
    schema = _load_json(SCHEMA_DIR / "profile.schema.json")
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    if errors:
        details = "; ".join(f"{_schema_location(error)}: {error.message}" for error in errors[:5])
        if len(errors) > 5:
            details = f"{details}; and {len(errors) - 5} more error(s)"
        raise ValueError(f"Profile {path.name} is invalid: {details}")


def load_profile(name: str = "architecture-review") -> Profile:
    profile_paths = _available_profile_paths()
    options = _profile_options(profile_paths)
    if not PROFILE_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid profile name: {name!r}. Valid profiles: {options}")
    matches = profile_paths.get(name, [])
    if not matches:
        raise ValueError(f"Unknown profile: {name}. Valid profiles: {options}")
    if len(matches) != 1:
        raise ValueError(f"Profile name is ambiguous: {name}. Valid profiles: {options}")
    data = _load_json(matches[0])
    _validate_profile_schema(data, matches[0])
    if data["name"] != name:
        raise ValueError(f"Profile filename name {name!r} does not match profile data name {data['name']!r}")
    weights = cast(dict[str, int], data["weights"])
    if sum(weights.values()) != 100:
        raise ValueError(f"Profile weights must total 100, got {sum(weights.values())}")
    return Profile(data=data, digest=sha256_json(data))
