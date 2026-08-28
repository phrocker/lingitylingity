"""Lingity command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence, cast
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from lingity.analyzer import analyze_text
from lingity.profiles import SCHEMA_DIR, canonical_json, load_profile


def _schema(name: str) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8")),
    )


def _write_json(value: object, output: Path | None) -> None:
    rendered = json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(rendered, encoding="utf-8")
            temporary.replace(output)
        finally:
            if temporary.exists():
                temporary.unlink()


def _normalized_path(path: Path) -> str:
    return os.path.normcase(str(path.expanduser().resolve(strict=False)))


def _reject_input_output_alias(input_path: Path, output: Path | None) -> None:
    if output is not None and _normalized_path(input_path) == _normalized_path(output):
        raise ValueError("input and output paths must be different")


def _remove_stale_output(output: Path | None) -> None:
    if output is None:
        return
    try:
        output.unlink()
    except FileNotFoundError:
        return


def _analyze(args: argparse.Namespace) -> int:
    path = cast(Path, args.input)
    output = cast(Path | None, args.output)
    try:
        _reject_input_output_alias(path, output)
        _remove_stale_output(output)
        text = path.read_text(encoding="utf-8")
        result = analyze_text(text, load_profile(cast(str, args.profile)))
        Draft202012Validator(_schema("analysis.schema.json")).validate(result)
        _write_json(result, output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, SchemaError, ValidationError) as exc:
        print(f"lingity analyze failed: {exc}", file=sys.stderr)
        return 2
    return 0


def _verify(args: argparse.Namespace) -> int:
    path = cast(Path, args.analysis)
    output = cast(Path | None, args.output)
    try:
        _reject_input_output_alias(path, output)
        _remove_stale_output(output)
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict):
            raise ValueError("analysis artifact must be a JSON object")
        Draft202012Validator(_schema("analysis.schema.json")).validate(artifact)
        recorded_hash = artifact.get("analysis_sha256")
        unhashed = dict(artifact)
        unhashed.pop("analysis_sha256", None)
        import hashlib

        actual_hash = hashlib.sha256(canonical_json(unhashed).encode("utf-8")).hexdigest()
        if recorded_hash != actual_hash:
            raise ValueError("analysis_sha256 does not match artifact content")
        source = artifact.get("source")
        profile_ref = artifact.get("profile")
        if not isinstance(source, dict) or not isinstance(source.get("text"), str):
            raise ValueError("analysis source text is missing")
        if not isinstance(profile_ref, dict) or not isinstance(profile_ref.get("name"), str):
            raise ValueError("analysis profile reference is missing")
        profile = load_profile(profile_ref["name"])
        if profile.digest != profile_ref.get("digest") or profile.version != profile_ref.get("version"):
            raise ValueError("analysis profile digest or version does not match the installed profile")
        replayed = analyze_text(source["text"], profile)
        if replayed != artifact:
            raise ValueError("analysis is not reproducible with the installed analyzer")
        verification = {
            "schema_version": "1.0.0",
            "valid": True,
            "analysis_sha256": recorded_hash,
            "source_sha256": source.get("sha256"),
            "profile": profile.reference(),
        }
        Draft202012Validator(_schema("verification.schema.json")).validate(verification)
        _write_json(verification, output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, SchemaError, ValidationError) as exc:
        print(f"lingity verify failed: {exc}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lingity")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze a UTF-8 text file")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--profile", default="architecture-review")
    analyze.add_argument("--output", type=Path)
    analyze.set_defaults(handler=_analyze)

    verify = subparsers.add_parser("verify", help="validate and replay an analysis artifact")
    verify.add_argument("analysis", type=Path)
    verify.add_argument("--output", type=Path)
    verify.set_defaults(handler=_verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Any, args.handler)
    return cast(int, handler(args))
