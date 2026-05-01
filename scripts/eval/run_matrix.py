from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from jsonschema import Draft202012Validator

PHASES: tuple[str, ...] = ("cold", "warm", "restart-warm")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKEND_HOST = "http://127.0.0.1:8000"
SCHEMA_PATH = ROOT / "results" / "schema" / "run-result.schema.json"


@dataclass(frozen=True)
class PromptRow:
    id: str
    category: str
    input: str
    expected_markers: list[str]


@dataclass(frozen=True)
class RunConfig:
    profile: str
    model: str
    mode: str
    context_target: int
    prompt_pack: Path
    repeats: int
    output: Path
    dry_run: bool
    backend_host: str
    selected_backend: str
    request_timeout_seconds: float
    max_tokens: int
    restart_command: str | None


class RunMatrixError(RuntimeError):
    """Raised when the runner cannot produce a valid run matrix."""


class SchemaValidator:
    def __init__(self, schema_path: Path) -> None:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self._validator = Draft202012Validator(
            schema, format_checker=Draft202012Validator.FORMAT_CHECKER
        )

    def validate(self, payload: dict[str, Any]) -> None:
        self._validator.validate(payload)


class PackRunner:
    def __init__(self, config: RunConfig, validator: SchemaValidator) -> None:
        self.config = config
        self.validator = validator
        self.prompts = load_prompt_pack(config.prompt_pack)
        self.git_rev = git_rev()

    def build_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for repeat_index in range(1, self.config.repeats + 1):
            for phase in PHASES:
                if self.config.dry_run:
                    record = self._dry_run_record(phase, repeat_index)
                else:
                    record = self._execute_phase(phase, repeat_index)
                self.validator.validate(record)
                records.append(record)
        return records

    def _dry_run_record(self, phase: str, repeat_index: int) -> dict[str, Any]:
        return base_record(
            config=self.config,
            git_rev_value=self.git_rev,
            phase=phase,
            repeat_index=repeat_index,
            success=True,
            ttft_ms=0.0,
            tokens_per_sec=0.0,
            latency_ms=0.0,
            vram_gb=0.0,
            rss_gb=0.0,
            error_type=None,
            fallback_triggered=False,
            selected_backend=self.config.selected_backend,
        )

    def _execute_phase(self, phase: str, repeat_index: int) -> dict[str, Any]:
        if phase == "restart-warm" and not self.config.restart_command:
            return base_record(
                config=self.config,
                git_rev_value=self.git_rev,
                phase=phase,
                repeat_index=repeat_index,
                success=False,
                ttft_ms=None,
                tokens_per_sec=None,
                latency_ms=None,
                vram_gb=None,
                rss_gb=None,
                error_type="restart_command_missing",
                fallback_triggered=False,
                selected_backend=None,
            )

        if phase == "restart-warm" and self.config.restart_command:
            run_restart_command(self.config.restart_command)

        ttft_values: list[float] = []
        tps_values: list[float] = []
        latency_values: list[float] = []

        for prompt in self.prompts:
            try:
                result = execute_prompt(
                    backend_host=self.config.backend_host,
                    model=self.config.model,
                    prompt=prompt,
                    timeout_seconds=self.config.request_timeout_seconds,
                    max_tokens=self.config.max_tokens,
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised by CLI integration tests
                return base_record(
                    config=self.config,
                    git_rev_value=self.git_rev,
                    phase=phase,
                    repeat_index=repeat_index,
                    success=False,
                    ttft_ms=None,
                    tokens_per_sec=None,
                    latency_ms=None,
                    vram_gb=None,
                    rss_gb=None,
                    error_type=classify_error(exc),
                    fallback_triggered=False,
                    selected_backend=None,
                )
            ttft_values.append(result["ttft_ms"])
            tps_values.append(result["tokens_per_sec"])
            latency_values.append(result["latency_ms"])

        return base_record(
            config=self.config,
            git_rev_value=self.git_rev,
            phase=phase,
            repeat_index=repeat_index,
            success=True,
            ttft_ms=round(mean(ttft_values), 3),
            tokens_per_sec=round(mean(tps_values), 3),
            latency_ms=round(mean(latency_values), 3),
            vram_gb=None,
            rss_gb=None,
            error_type=None,
            fallback_triggered=False,
            selected_backend=self.config.selected_backend,
        )


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description="Execute a qllama benchmark matrix and emit schema-valid JSONL results."
    )
    parser.add_argument("--profile", required=True, help="Profile path or identifier.")
    parser.add_argument(
        "--model", required=True, help="Model identifier used for the run."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("f16", "turbo2", "turbo3", "turbo4"),
        help="Benchmark mode.",
    )
    parser.add_argument(
        "--ctx",
        required=True,
        type=int,
        dest="context_target",
        help="Target context length for this run.",
    )
    parser.add_argument(
        "--prompt-pack",
        required=True,
        dest="prompt_pack",
        help="Path to the JSONL prompt pack to execute.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many repeat groups to execute across cold/warm/restart-warm phases.",
    )
    parser.add_argument(
        "--output", required=True, help="Output path for JSONL result records."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write schema-valid dry-run records without contacting a backend.",
    )
    parser.add_argument(
        "--backend-host",
        default=DEFAULT_BACKEND_HOST,
        help="Base URL for the OpenAI-compatible backend (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--selected-backend",
        default="primary",
        help="Backend label to record for successful runs (default: primary).",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=60.0,
        help="HTTP timeout for non-dry-run prompt execution.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="`max_tokens` sent to the chat completion endpoint during non-dry runs.",
    )
    parser.add_argument(
        "--restart-command",
        default=None,
        help="Optional shell command used before the restart-warm phase in non-dry-run mode.",
    )

    args = parser.parse_args(argv)

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    return RunConfig(
        profile=args.profile,
        model=args.model,
        mode=args.mode,
        context_target=args.context_target,
        prompt_pack=Path(args.prompt_pack),
        repeats=args.repeats,
        output=Path(args.output),
        dry_run=args.dry_run,
        backend_host=args.backend_host.rstrip("/"),
        selected_backend=args.selected_backend,
        request_timeout_seconds=args.request_timeout_seconds,
        max_tokens=args.max_tokens,
        restart_command=args.restart_command,
    )


def load_prompt_pack(path: Path) -> list[PromptRow]:
    if not path.exists():
        raise RunMatrixError(f"Prompt pack not found: {path}")

    rows: list[PromptRow] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RunMatrixError(
                f"Prompt pack row {line_number} in {path} is not an object"
            )
        rows.append(
            PromptRow(
                id=str(payload["id"]),
                category=str(payload["category"]),
                input=str(payload["input"]),
                expected_markers=[
                    str(item) for item in payload.get("expected_markers", [])
                ],
            )
        )

    if not rows:
        raise RunMatrixError(f"Prompt pack is empty: {path}")
    return rows


def execute_prompt(
    *,
    backend_host: str,
    model: str,
    prompt: PromptRow,
    timeout_seconds: float,
    max_tokens: int,
) -> dict[str, float]:
    started_at = time.perf_counter()
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(
            f"{backend_host}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt.input}],
                "max_tokens": max_tokens,
            },
            headers={"Content-Type": "application/json"},
        )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    response.raise_for_status()
    payload = response.json()
    timings = payload.get("timings") if isinstance(payload, dict) else None
    ttft_ms = extract_timing_ms(timings, elapsed_ms)
    tokens_per_sec = extract_tokens_per_second(timings)
    return {
        "ttft_ms": ttft_ms,
        "tokens_per_sec": tokens_per_sec,
        "latency_ms": elapsed_ms,
    }


def extract_timing_ms(timings: object, default_ms: float) -> float:
    if isinstance(timings, dict):
        for key in ("prompt_ms", "first_token_ms", "ttft_ms"):
            value = timings.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return float(default_ms)


def extract_tokens_per_second(timings: object) -> float:
    if isinstance(timings, dict):
        for key in ("predicted_per_second", "tokens_per_second"):
            value = timings.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return 0.0


def run_restart_command(command: str) -> None:
    result = subprocess.run(command, shell=True, check=False, text=True)
    if result.returncode != 0:
        raise RunMatrixError(
            f"Restart command failed with exit code {result.returncode}: {command}"
        )


def base_record(
    *,
    config: RunConfig,
    git_rev_value: str,
    phase: str,
    repeat_index: int,
    success: bool,
    ttft_ms: float | None,
    tokens_per_sec: float | None,
    latency_ms: float | None,
    vram_gb: float | None,
    rss_gb: float | None,
    error_type: str | None,
    fallback_triggered: bool,
    selected_backend: str | None,
) -> dict[str, Any]:
    return {
        "model": config.model,
        "backend_host": config.backend_host,
        "profile": config.profile,
        "mode": config.mode,
        "context_target": config.context_target,
        "phase": phase,
        "prompt_pack": config.prompt_pack.as_posix(),
        "run_id": build_run_id(config, phase, repeat_index),
        "ttft_ms": ttft_ms,
        "tokens_per_sec": tokens_per_sec,
        "latency_ms": latency_ms,
        "vram_gb": vram_gb,
        "rss_gb": rss_gb,
        "success": success,
        "error_type": error_type,
        "fallback_triggered": fallback_triggered,
        "selected_backend": selected_backend,
        "git_rev": git_rev_value,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }


def build_run_id(config: RunConfig, phase: str, repeat_index: int) -> str:
    profile_slug = Path(config.profile).stem.replace("_", "-")
    return f"{profile_slug}-{config.mode}-{config.context_target}-{phase}-r{repeat_index:02d}-{uuid4().hex[:8]}"


def git_rev() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    value = result.stdout.strip()
    return value or "unknown"


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def classify_error(exc: Exception) -> str:
    if isinstance(exc, RunMatrixError):
        return "runner_error"
    if isinstance(exc, httpx.TimeoutException):
        return "upstream_timeout"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"http_{exc.response.status_code}"
    if isinstance(exc, httpx.HTTPError):
        return "upstream_transport_error"
    return exc.__class__.__name__.lower()


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        validator = SchemaValidator(SCHEMA_PATH)
        runner = PackRunner(config, validator)
        records = runner.build_records()
        write_records(config.output, records)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"wrote {len(records)} result records to {config.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
