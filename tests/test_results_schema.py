from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "results" / "schema" / "run-result.schema.json"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    )


@pytest.fixture()
def valid_success_payload() -> dict[str, object]:
    return {
        "model": "mistral-small3.2:24b",
        "backend_host": "http://10.40.10.90:8010",
        "profile": "profiles/mistral24b-turbo3-100k.yaml",
        "mode": "turbo3",
        "context_target": 100000,
        "phase": "warm",
        "prompt_pack": "scripts/eval/prompts/long-context.jsonl",
        "run_id": "mistral24b-turbo3-100k-run-001",
        "ttft_ms": 842.5,
        "tokens_per_sec": 47.2,
        "latency_ms": 12894.2,
        "vram_gb": 17.1,
        "rss_gb": 5.4,
        "success": True,
        "error_type": None,
        "fallback_triggered": False,
        "selected_backend": "primary",
        "git_rev": "abc1234",
        "timestamp": "2026-04-11T02:45:00Z",
    }


@pytest.fixture()
def valid_failure_payload() -> dict[str, object]:
    return {
        "model": "gemma4:26b-a4b-it-q4_K_M",
        "backend_host": "http://10.40.10.99:11434",
        "profile": "profiles/gemma4-64k-3090.yaml",
        "mode": "f16",
        "context_target": 65536,
        "phase": "restart-warm",
        "prompt_pack": "scripts/eval/prompts/runtime-regression.jsonl",
        "run_id": "gemma4-64k-run-009",
        "ttft_ms": None,
        "tokens_per_sec": None,
        "latency_ms": None,
        "vram_gb": None,
        "rss_gb": None,
        "success": False,
        "error_type": "upstream_timeout",
        "fallback_triggered": True,
        "selected_backend": None,
        "git_rev": "def5678-dirty",
        "timestamp": "2026-04-11T02:46:00Z",
    }


def test_schema_accepts_valid_success_run(
    validator: Draft202012Validator, valid_success_payload: dict[str, object]
) -> None:
    validator.validate(valid_success_payload)


def test_schema_accepts_structured_failure_run(
    validator: Draft202012Validator, valid_failure_payload: dict[str, object]
) -> None:
    validator.validate(valid_failure_payload)


def test_schema_rejects_missing_required_field(
    validator: Draft202012Validator, valid_success_payload: dict[str, object]
) -> None:
    payload = dict(valid_success_payload)
    payload.pop("run_id")

    with pytest.raises(ValidationError, match="run_id"):
        validator.validate(payload)


def test_schema_rejects_invalid_phase(
    validator: Draft202012Validator, valid_success_payload: dict[str, object]
) -> None:
    payload = dict(valid_success_payload)
    payload["phase"] = "fresh"

    with pytest.raises(ValidationError, match="restart-warm"):
        validator.validate(payload)


def test_schema_rejects_success_run_with_null_error_metrics_contract(
    validator: Draft202012Validator, valid_success_payload: dict[str, object]
) -> None:
    payload = dict(valid_success_payload)
    payload["ttft_ms"] = None

    with pytest.raises(ValidationError):
        validator.validate(payload)


def test_schema_rejects_failure_run_without_error_type(
    validator: Draft202012Validator, valid_failure_payload: dict[str, object]
) -> None:
    payload = dict(valid_failure_payload)
    payload["error_type"] = None

    with pytest.raises(ValidationError):
        validator.validate(payload)
