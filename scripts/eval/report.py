from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "results" / "schema" / "run-result.schema.json"


@dataclass(frozen=True)
class SummaryStats:
    record_count: int
    success_count: int
    success_rate: float
    median_ttft_ms: float | None
    median_tokens_per_sec: float | None
    peak_vram_gb: float | None
    fallback_rate: float
    error_types: dict[str, int]


class ReportGenerationError(RuntimeError):
    """Raised when report generation cannot proceed."""


class SchemaValidator:
    def __init__(self, schema_path: Path) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self._validator = Draft202012Validator(
            schema, format_checker=Draft202012Validator.FORMAT_CHECKER
        )

    def validate(self, payload: dict[str, Any]) -> None:
        self._validator.validate(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a qllama benchmark summary report from schema-valid result files."
    )
    parser.add_argument(
        "--input-glob",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more files or glob patterns containing benchmark result records.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path for the generated markdown report.",
    )
    parser.add_argument(
        "--gates-out",
        default=None,
        help="Optional path for a machine-readable JSON summary artifact.",
    )
    return parser.parse_args(argv)


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
        raise ReportGenerationError(
            "No result files matched the provided input patterns"
        )
    return paths


def load_records(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ReportGenerationError(f"Result file is empty: {path}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _load_jsonl_records(path)

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        if not all(isinstance(item, dict) for item in parsed):
            raise ReportGenerationError(
                f"JSON array in {path} must contain only objects"
            )
        return list(parsed)

    raise ReportGenerationError(
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
        except json.JSONDecodeError as exc:
            raise ReportGenerationError(
                f"Invalid JSON on line {line_number} of {path}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ReportGenerationError(
                f"Line {line_number} of {path} is not a JSON object"
            )
        records.append(payload)
    if not records:
        raise ReportGenerationError(f"No JSON records found in {path}")
    return records


def validate_records(paths: list[Path]) -> list[dict[str, Any]]:
    validator = SchemaValidator(SCHEMA_PATH)
    records: list[dict[str, Any]] = []
    for path in paths:
        file_records = load_records(path)
        for record in file_records:
            validator.validate(record)
            records.append(record)
    if not records:
        raise ReportGenerationError("No valid records were loaded from input files")
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall = compute_stats(records)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["model"]), str(record["mode"]))].append(record)

    by_model_mode = []
    for (model, mode), group_records in sorted(grouped.items()):
        stats = compute_stats(group_records)
        by_model_mode.append(
            {
                "model": model,
                "mode": mode,
                **stats_to_dict(stats),
            }
        )

    return {
        "generated_at": iso_now(),
        "record_count": len(records),
        "overall": stats_to_dict(overall),
        "by_model_mode": by_model_mode,
    }


def compute_stats(records: list[dict[str, Any]]) -> SummaryStats:
    success_records = [record for record in records if bool(record["success"])]
    ttft_values = [
        float(record["ttft_ms"])
        for record in success_records
        if record["ttft_ms"] is not None
    ]
    tps_values = [
        float(record["tokens_per_sec"])
        for record in success_records
        if record["tokens_per_sec"] is not None
    ]
    vram_values = [
        float(record["vram_gb"]) for record in records if record["vram_gb"] is not None
    ]
    error_types = Counter(
        str(record["error_type"])
        for record in records
        if record["error_type"] not in (None, "")
    )
    fallback_count = sum(1 for record in records if bool(record["fallback_triggered"]))

    return SummaryStats(
        record_count=len(records),
        success_count=len(success_records),
        success_rate=safe_ratio(len(success_records), len(records)),
        median_ttft_ms=median_or_none(ttft_values),
        median_tokens_per_sec=median_or_none(tps_values),
        peak_vram_gb=max(vram_values) if vram_values else None,
        fallback_rate=safe_ratio(fallback_count, len(records)),
        error_types=dict(sorted(error_types.items())),
    )


def stats_to_dict(stats: SummaryStats) -> dict[str, Any]:
    return {
        "record_count": stats.record_count,
        "success_count": stats.success_count,
        "success_rate": round(stats.success_rate, 4),
        "median_ttft_ms": round(stats.median_ttft_ms, 3)
        if stats.median_ttft_ms is not None
        else None,
        "median_tokens_per_sec": round(stats.median_tokens_per_sec, 3)
        if stats.median_tokens_per_sec is not None
        else None,
        "peak_vram_gb": round(stats.peak_vram_gb, 3)
        if stats.peak_vram_gb is not None
        else None,
        "fallback_rate": round(stats.fallback_rate, 4),
        "error_types": stats.error_types,
    }


def render_markdown(summary: dict[str, Any], source_paths: list[Path]) -> str:
    overall = summary["overall"]
    lines = [
        "# Benchmark Summary",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- `{path.as_posix()}`" for path in source_paths)
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Records: {overall['record_count']}",
            f"- Success count: {overall['success_count']}",
            f"- Success rate: {format_percent(overall['success_rate'])}",
            f"- Median TTFT (ms): {format_optional_number(overall['median_ttft_ms'])}",
            f"- Median tokens/s: {format_optional_number(overall['median_tokens_per_sec'])}",
            f"- Peak VRAM (GB): {format_optional_number(overall['peak_vram_gb'])}",
            f"- Fallback rate: {format_percent(overall['fallback_rate'])}",
            "",
            "## Error Types",
            "",
        ]
    )

    if overall["error_types"]:
        for error_type, count in overall["error_types"].items():
            lines.append(f"- `{error_type}`: {count}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## By model and mode",
            "",
            "| Model | Mode | Records | Success rate | Median TTFT ms | Median tok/s | Peak VRAM GB | Fallback rate |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in summary["by_model_mode"]:
        lines.append(
            "| {model} | {mode} | {record_count} | {success_rate} | {ttft} | {tps} | {vram} | {fallback_rate} |".format(
                model=row["model"],
                mode=row["mode"],
                record_count=row["record_count"],
                success_rate=format_percent(row["success_rate"]),
                ttft=format_optional_number(row["median_ttft_ms"]),
                tps=format_optional_number(row["median_tokens_per_sec"]),
                vram=format_optional_number(row["peak_vram_gb"]),
                fallback_rate=format_percent(row["fallback_rate"]),
            )
        )

    return "\n".join(lines) + "\n"


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def format_optional_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        source_paths = expand_paths(args.input_glob)
        records = validate_records(source_paths)
        summary = summarize(records)
        write_text(Path(args.out), render_markdown(summary, source_paths))
        if args.gates_out:
            write_json(Path(args.gates_out), summary)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"generated report from {len(source_paths)} files and {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
