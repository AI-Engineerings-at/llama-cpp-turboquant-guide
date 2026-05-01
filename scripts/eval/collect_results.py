from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "results" / "schema" / "run-result.schema.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate qllama benchmark result files against the run-result schema."
    )
    parser.add_argument(
        "--validate",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more files or glob patterns to validate.",
    )
    return parser.parse_args(argv)


class ResultCollectorError(RuntimeError):
    """Raised when result collection or validation fails."""


class SchemaValidator:
    def __init__(self, schema_path: Path) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self._validator = Draft202012Validator(
            schema, format_checker=Draft202012Validator.FORMAT_CHECKER
        )

    def validate(self, payload: dict[str, Any]) -> None:
        self._validator.validate(payload)


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = (
            sorted(Path().glob(pattern))
            if any(ch in pattern for ch in "*?[")
            else [Path(pattern)]
        )
        for path in matched:
            if path.is_dir():
                continue
            if path not in paths:
                paths.append(path)
    if not paths:
        raise ResultCollectorError("No result files matched the provided paths")
    return paths


def load_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ResultCollectorError(f"Result file is empty: {path}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _load_jsonl_records(path)

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        if not all(isinstance(item, dict) for item in parsed):
            raise ResultCollectorError(
                f"JSON array in {path} must contain only objects"
            )
        return list(parsed)
    raise ResultCollectorError(
        f"Unsupported JSON payload in {path}: expected object, array, or JSONL"
    )


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except (
            json.JSONDecodeError
        ) as exc:  # pragma: no cover - error path exercised via CLI test
            raise ResultCollectorError(
                f"Invalid JSON on line {line_number} of {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ResultCollectorError(
                f"Line {line_number} of {path} is not a JSON object"
            )
        records.append(payload)
    if not records:
        raise ResultCollectorError(f"No JSON records found in {path}")
    return records


def validate_files(paths: list[Path]) -> tuple[int, int]:
    validator = SchemaValidator(SCHEMA_PATH)
    validated_files = 0
    validated_records = 0
    for path in paths:
        records = load_records(path)
        for record in records:
            validator.validate(record)
            validated_records += 1
        validated_files += 1
    return validated_files, validated_records


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        paths = expand_paths(args.validate)
        files, records = validate_files(paths)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"validated {records} records across {files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
