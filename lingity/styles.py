"""Versioned style-contract loading, validation, and rendering."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from lingity.models import JsonValue
from lingity.profiles import SCHEMA_DIR, sha256_json

PACKAGE_DIR = Path(__file__).resolve().parent
STYLE_DIR = PACKAGE_DIR / "styles"
STYLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class StyleContract:
    """A validated style contract and the digest of its canonical JSON."""

    data: dict[str, Any]
    digest: str

    @property
    def name(self) -> str:
        return cast(str, self.data["name"])

    @property
    def version(self) -> str:
        return cast(str, self.data["version"])

    @property
    def title(self) -> str:
        return cast(str, self.data["title"])

    def reference(self) -> dict[str, JsonValue]:
        return {"name": self.name, "version": self.version, "digest": self.digest}

    def render(self) -> str:
        return render_style_instructions(self)


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


def _schema_location(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    return path or "<root>"


def _validate_style_schema(data: dict[str, Any], path: Path) -> None:
    schema = _load_json(SCHEMA_DIR / "style-contract.schema.json")
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(data), key=lambda error: list(error.absolute_path)
    )
    if errors:
        details = "; ".join(
            f"{_schema_location(error)}: {error.message}" for error in errors[:5]
        )
        if len(errors) > 5:
            details = f"{details}; and {len(errors) - 5} more error(s)"
        raise ValueError(f"Style contract {path.name} is invalid: {details}")


def _available_style_paths() -> dict[str, list[Path]]:
    styles: dict[str, list[Path]] = {}
    for path in sorted(STYLE_DIR.glob("*.v*.json")):
        name = path.name.rsplit(".v", 1)[0]
        if name:
            styles.setdefault(name, []).append(path)
    return styles


def _style_options(style_paths: dict[str, list[Path]]) -> str:
    names = sorted(style_paths)
    return ", ".join(names) if names else "(none installed)"


def load_style(name: str) -> StyleContract:
    """Load one installed style contract, failing on every ambiguity."""

    style_paths = _available_style_paths()
    options = _style_options(style_paths)
    if not STYLE_NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid style name: {name!r}. Valid styles: {options}")
    matches = style_paths.get(name, [])
    if not matches:
        raise ValueError(f"Unknown style: {name}. Valid styles: {options}")
    if len(matches) != 1:
        raise ValueError(f"Style name is ambiguous: {name}. Valid styles: {options}")

    path = matches[0]
    data = _load_json(path)
    _validate_style_schema(data, path)
    if data["name"] != name:
        raise ValueError(
            f"Style filename name {name!r} does not match style data name "
            f"{data['name']!r}"
        )
    return StyleContract(data=data, digest=sha256_json(data))


def available_style_names() -> tuple[str, ...]:
    """Return validated installed style names in deterministic order."""

    names = tuple(sorted(_available_style_paths()))
    for name in names:
        load_style(name)
    return names


def _numbered(values: list[str]) -> str:
    return "\n".join(f"{index}. {value}" for index, value in enumerate(values, 1))


def _examples(values: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{index}. Text: {item['text']}\n   Reason: {item['reason']}"
        for index, item in enumerate(values, 1)
    )


def render_style_instructions(style: StyleContract) -> str:
    """Render deterministic provider guidance from structured contract fields."""

    data = style.data
    outcome = cast(dict[str, Any], data["reader_outcome"])
    policy = cast(dict[str, Any], data["content_policy"])
    shape = cast(dict[str, Any], data["rhetorical_shape"])
    posture = cast(dict[str, Any], data["editing_posture"])
    tendencies = cast(list[dict[str, str]], data["observable_tendencies"])
    tendency_lines = [
        f"{item['tendency']}: {item['reason']}" for item in tendencies
    ]

    return (
        f"Lingity style contract: {style.title} "
        f"({style.name} v{style.version}, sha256:{style.digest})\n"
        "Status: generation guidance only; this contract is not a deterministic "
        "style-fit score or acceptance authority.\n\n"
        "Reader outcome\n"
        f"Audience: {outcome['audience']}\n"
        f"Outcome: {outcome['outcome']}\n"
        f"Scan priority:\n{_numbered(cast(list[str], outcome['scan_priority']))}\n\n"
        "Content policy\n"
        f"Preserve:\n{_numbered(cast(list[str], policy['preserve']))}\n"
        f"Do not add:\n{_numbered(cast(list[str], policy['do_not_add']))}\n\n"
        "Rhetorical shape\n"
        f"Lead: {shape['lead']}\n"
        f"Sequence:\n{_numbered(cast(list[str], shape['sequence']))}\n"
        f"Ending: {shape['ending']}\n\n"
        "Editing posture\n"
        f"Role: {posture['role']}\n"
        f"Change scope: {posture['change_scope']}\n"
        f"Allowed moves:\n{_numbered(cast(list[str], posture['allowed_moves']))}\n"
        "Prohibited moves:\n"
        f"{_numbered(cast(list[str], posture['prohibited_moves']))}\n\n"
        "Observable tendencies\n"
        f"{_numbered(tendency_lines)}\n\n"
        "Positive guidance\n"
        f"{_numbered(cast(list[str], data['positive_guidance']))}\n\n"
        "Negative guidance\n"
        f"{_numbered(cast(list[str], data['negative_guidance']))}\n\n"
        "Positive examples\n"
        f"{_examples(cast(list[dict[str, str]], data['positive_examples']))}\n\n"
        "Negative examples\n"
        f"{_examples(cast(list[dict[str, str]], data['negative_examples']))}"
    )
