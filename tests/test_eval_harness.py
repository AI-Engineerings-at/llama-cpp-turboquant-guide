from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
PROMPT_PACKS = [
    ROOT / "scripts" / "eval" / "prompts" / "runtime-regression.jsonl",
    ROOT / "scripts" / "eval" / "prompts" / "long-context.jsonl",
    ROOT / "scripts" / "eval" / "prompts" / "fallback-regression.jsonl",
]
RUN_MATRIX = ROOT / "scripts" / "eval" / "run_matrix.py"
COLLECT_RESULTS = ROOT / "scripts" / "eval" / "collect_results.py"
SCHEMA_PATH = ROOT / "results" / "schema" / "run-result.schema.json"


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return Draft202012Validator(
        schema, format_checker=Draft202012Validator.FORMAT_CHECKER
    )


@pytest.mark.parametrize("path", PROMPT_PACKS)
def test_prompt_pack_rows_have_required_fields(path: Path) -> None:
    rows = _load_rows(path)

    assert rows, f"prompt pack is empty: {path}"
    for row in rows:
        assert isinstance(row["id"], str) and row["id"].strip(), row
        assert isinstance(row["category"], str) and row["category"].strip(), row
        assert isinstance(row["input"], str) and row["input"].strip(), row


@pytest.mark.parametrize("path", PROMPT_PACKS)
def test_prompt_pack_ids_are_unique_within_pack(path: Path) -> None:
    rows = _load_rows(path)
    ids = [row["id"] for row in rows]

    assert len(ids) == len(set(ids)), f"duplicate ids in {path}: {ids}"


@pytest.mark.parametrize("path", PROMPT_PACKS)
def test_prompt_pack_expected_markers_are_optional_but_well_typed(path: Path) -> None:
    rows = _load_rows(path)

    for row in rows:
        if "expected_markers" not in row:
            continue
        assert isinstance(row["expected_markers"], list), row
        assert all(
            isinstance(item, str) and item for item in row["expected_markers"]
        ), row


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        assert isinstance(payload, dict), payload
        rows.append(payload)
    return rows


def test_run_matrix_help_succeeds() -> None:
    result = subprocess.run(
        [sys.executable, str(RUN_MATRIX), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--profile" in result.stdout
    assert "--dry-run" in result.stdout


def test_collect_results_help_succeeds() -> None:
    result = subprocess.run(
        [sys.executable, str(COLLECT_RESULTS), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--validate" in result.stdout


def test_run_matrix_dry_run_writes_schema_valid_jsonl(
    tmp_path: Path, validator: Draft202012Validator
) -> None:
    output_path = tmp_path / "dry-run-results.json"
    result = subprocess.run(
        [
            sys.executable,
            str(RUN_MATRIX),
            "--profile",
            "profiles/baseline.yaml",
            "--model",
            "mistral-small3.2:24b",
            "--mode",
            "f16",
            "--ctx",
            "8192",
            "--prompt-pack",
            str(PROMPT_PACKS[0]),
            "--repeats",
            "2",
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    rows = _load_rows(output_path)
    assert len(rows) == 6
    assert {row["phase"] for row in rows} == {"cold", "warm", "restart-warm"}
    for row in rows:
        validator.validate(row)
        assert row["success"] is True
        assert row["selected_backend"] == "primary"
        assert row["prompt_pack"] == PROMPT_PACKS[0].as_posix()


def test_collect_results_validates_dry_run_output(tmp_path: Path) -> None:
    output_path = tmp_path / "dry-run-results.json"
    run_result = subprocess.run(
        [
            sys.executable,
            str(RUN_MATRIX),
            "--profile",
            "profiles/turbo3-100k.yaml",
            "--model",
            "mistral-small3.2:24b",
            "--mode",
            "turbo3",
            "--ctx",
            "100000",
            "--prompt-pack",
            str(PROMPT_PACKS[1]),
            "--output",
            str(output_path),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr

    validate_result = subprocess.run(
        [
            sys.executable,
            str(COLLECT_RESULTS),
            "--validate",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert validate_result.returncode == 0, validate_result.stderr
    assert "validated 3 records across 1 files" in validate_result.stdout


def test_collect_results_rejects_invalid_record_file(tmp_path: Path) -> None:
    invalid_path = tmp_path / "invalid-results.json"
    invalid_path.write_text('{"model":"broken"}\n', encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(COLLECT_RESULTS),
            "--validate",
            str(invalid_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "ERROR:" in result.stderr
