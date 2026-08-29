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
from lingity.critique import CritiqueError, build_critique
from lingity.improve import ImprovementError, improve_text, judge_candidate
from lingity.nlp import LinguisticModelError, model_fingerprint
from lingity.profiles import SCHEMA_DIR, canonical_json, load_profile
from lingity.providers import (
    ProviderError,
    available_proposal_providers,
    create_challenge_provider,
    create_proposal_provider,
)

CLI_ERRORS = (
    OSError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
    SchemaError,
    ValidationError,
    LinguisticModelError,
    CritiqueError,
    ImprovementError,
    ProviderError,
)


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
    except (OSError, TypeError, ValueError, json.JSONDecodeError, SchemaError, ValidationError, LinguisticModelError) as exc:
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
        recorded_model = artifact.get("linguistic_model")
        if not isinstance(recorded_model, dict):
            raise ValueError("analysis linguistic model reference is missing")
        installed_model = model_fingerprint()
        if recorded_model != installed_model:
            raise ValueError(
                "analysis was produced by a different linguistic pipeline "
                f"({recorded_model.get('name')} {recorded_model.get('version')} "
                f"{recorded_model.get('runtime')} digest {recorded_model.get('digest')}) "
                f"than the installed one ({installed_model['name']} {installed_model['version']} "
                f"{installed_model['runtime']} digest {installed_model['digest']}); "
                "analyses are only reproducible against the pipeline that produced them"
            )
        replayed = analyze_text(source["text"], profile)
        if replayed != artifact:
            raise ValueError("analysis is not reproducible with the installed analyzer")
        verification = {
            "schema_version": "1.0.0",
            "valid": True,
            "analysis_sha256": recorded_hash,
            "source_sha256": source.get("sha256"),
            "profile": profile.reference(),
            "linguistic_model": installed_model,
        }
        Draft202012Validator(_schema("verification.schema.json")).validate(verification)
        _write_json(verification, output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, SchemaError, ValidationError, LinguisticModelError) as exc:
        print(f"lingity verify failed: {exc}", file=sys.stderr)
        return 2
    return 0


def _load_prior_attempts(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("attempts")
    if not isinstance(payload, list):
        raise ValueError(
            f"prior-attempts file {path} must hold a JSON array of attempt "
            "records, or an object with an 'attempts' array"
        )
    return [cast(dict[str, Any], item) for item in payload]


def _critique(args: argparse.Namespace) -> int:
    path = cast(Path, args.input)
    output = cast(Path | None, args.output)
    try:
        _reject_input_output_alias(path, output)
        _remove_stale_output(output)
        text = path.read_text(encoding="utf-8")
        analysis = analyze_text(text, load_profile(cast(str, args.profile)))
        brief = build_critique(
            analysis,
            prior_attempts=cast(
                Any, _load_prior_attempts(cast(Path | None, args.prior_attempts))
            ),
        )
        Draft202012Validator(_schema("critique.schema.json")).validate(brief)
        _write_json(brief, output)
    except CLI_ERRORS as exc:
        print(f"lingity critique failed: {exc}", file=sys.stderr)
        return 2
    return 0


def _judge(args: argparse.Namespace) -> int:
    source_path = cast(Path, args.source)
    candidate_path = cast(Path, args.candidate)
    output = cast(Path | None, args.output)
    try:
        _reject_input_output_alias(source_path, output)
        _reject_input_output_alias(candidate_path, output)
        _remove_stale_output(output)
        profile = load_profile(cast(str, args.profile))
        source_text = source_path.read_text(encoding="utf-8")
        candidate_text = candidate_path.read_text(encoding="utf-8")
        if not candidate_text.strip():
            raise ValueError(
                f"candidate file {candidate_path} is empty; an empty rewrite is "
                "never an improvement"
            )
        challenger = None
        challenge_provider = cast(str | None, args.challenge_provider)
        if challenge_provider is not None:
            challenger = create_challenge_provider(
                challenge_provider, model=cast(str | None, args.challenge_model)
            )
        accepted, reasons, evidence = judge_candidate(
            source_text, candidate_text, profile, challenger=challenger
        )
        verdict = {
            "schema_version": "1.0.0",
            "accepted": accepted,
            "rejection_reasons": list(reasons),
            "source_score": evidence["source_score"],
            "candidate_score": evidence["candidate_score"],
            "protected_disposition": evidence["protected_disposition"],
            "protected_delta": evidence["protected_delta"],
            "challenge": evidence["challenge"],
            "profile": profile.reference(),
            "linguistic_model": model_fingerprint(),
        }
        Draft202012Validator(_schema("verdict.schema.json")).validate(verdict)
        _write_json(verdict, output)
        return 0 if accepted else 1
    except CLI_ERRORS as exc:
        print(f"lingity judge failed: {exc}", file=sys.stderr)
        return 2


def _improve(args: argparse.Namespace) -> int:
    source_path = cast(Path, args.source)
    output = cast(Path | None, args.output)
    try:
        _reject_input_output_alias(source_path, output)
        _remove_stale_output(output)
        profile = load_profile(cast(str, args.profile))
        source_text = source_path.read_text(encoding="utf-8")

        options: dict[str, Any] = {}
        provider_name = cast(str, args.provider)
        if provider_name == "subagent":
            candidates = cast(list[Path] | None, args.candidate)
            if not candidates:
                raise ValueError(
                    "the subagent provider requires at least one --candidate "
                    "file written by the host agent"
                )
            options["candidate_paths"] = candidates
        else:
            model = cast(str | None, args.model)
            if not model:
                raise ValueError(
                    f"provider {provider_name!r} requires an explicit --model; "
                    "Lingity does not choose a model for you"
                )
            options["model"] = model
        provider = create_proposal_provider(provider_name, **options)

        challenger = None
        challenge_provider = cast(str | None, args.challenge_provider)
        if challenge_provider is not None:
            challenger = create_challenge_provider(
                challenge_provider, model=cast(str | None, args.challenge_model)
            )

        result = improve_text(
            source_text,
            profile,
            provider,
            max_attempts=cast(int, args.max_attempts),
            challenger=challenger,
        )
        record = result.to_dict()
        record["profile"] = cast(Any, profile.reference())
        record["linguistic_model"] = cast(Any, model_fingerprint())
        _write_json(record, output)
        return 0 if result.accepted else 1
    except CLI_ERRORS as exc:
        print(f"lingity improve failed: {exc}", file=sys.stderr)
        return 2


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

    critique = subparsers.add_parser(
        "critique",
        help="emit a deterministic improvement brief for a host agent or model",
    )
    critique.add_argument("input", type=Path)
    critique.add_argument("--profile", default="architecture-review")
    critique.add_argument("--prior-attempts", type=Path, dest="prior_attempts")
    critique.add_argument("--output", type=Path)
    critique.set_defaults(handler=_critique)

    judge = subparsers.add_parser(
        "judge",
        help="decide a single candidate rewrite; exits 1 when it is rejected",
    )
    judge.add_argument("source", type=Path)
    judge.add_argument("--candidate", type=Path, required=True)
    judge.add_argument("--profile", default="architecture-review")
    judge.add_argument("--challenge-provider", dest="challenge_provider")
    judge.add_argument("--challenge-model", dest="challenge_model")
    judge.add_argument("--output", type=Path)
    judge.set_defaults(handler=_judge)

    improve = subparsers.add_parser(
        "improve",
        help="run the bounded improvement loop; exits 1 when nothing is accepted",
    )
    improve.add_argument("source", type=Path)
    improve.add_argument(
        "--provider",
        default="subagent",
        choices=list(available_proposal_providers()),
    )
    improve.add_argument("--model")
    improve.add_argument("--candidate", type=Path, action="append")
    improve.add_argument("--max-attempts", type=int, default=3, dest="max_attempts")
    improve.add_argument("--profile", default="architecture-review")
    improve.add_argument("--challenge-provider", dest="challenge_provider")
    improve.add_argument("--challenge-model", dest="challenge_model")
    improve.add_argument("--output", type=Path)
    improve.set_defaults(handler=_improve)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = cast(Any, args.handler)
    return cast(int, handler(args))
