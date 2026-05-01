from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCRIPT = ROOT / "scripts" / "eval" / "report.py"


def test_report_help_succeeds() -> None:
    result = subprocess.run(
        [sys.executable, str(REPORT_SCRIPT), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--input-glob" in result.stdout
    assert "--out" in result.stdout


def test_report_generation_from_valid_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "sample-results.json"
    report_path = tmp_path / "benchmark-summary.md"
    summary_path = tmp_path / "benchmark-summary.json"

    records = [
        {
            "model": "mistral-small3.2:24b",
            "backend_host": "http://10.40.10.90:8010",
            "profile": "profiles/mistral24b-turbo3-100k.yaml",
            "mode": "turbo3",
            "context_target": 100000,
            "phase": "cold",
            "prompt_pack": "scripts/eval/prompts/long-context.jsonl",
            "run_id": "run-001",
            "ttft_ms": 900.0,
            "tokens_per_sec": 47.0,
            "latency_ms": 12000.0,
            "vram_gb": 17.1,
            "rss_gb": 5.2,
            "success": True,
            "error_type": None,
            "fallback_triggered": False,
            "selected_backend": "primary",
            "git_rev": "abc1234",
            "timestamp": "2026-04-11T03:00:00Z",
        },
        {
            "model": "mistral-small3.2:24b",
            "backend_host": "http://10.40.10.90:8010",
            "profile": "profiles/mistral24b-turbo3-100k.yaml",
            "mode": "turbo3",
            "context_target": 100000,
            "phase": "warm",
            "prompt_pack": "scripts/eval/prompts/long-context.jsonl",
            "run_id": "run-002",
            "ttft_ms": 850.0,
            "tokens_per_sec": 48.0,
            "latency_ms": 11800.0,
            "vram_gb": 17.3,
            "rss_gb": 5.3,
            "success": True,
            "error_type": None,
            "fallback_triggered": False,
            "selected_backend": "primary",
            "git_rev": "abc1234",
            "timestamp": "2026-04-11T03:01:00Z",
        },
        {
            "model": "gemma4:26b-a4b-it-q4_K_M",
            "backend_host": "http://10.40.10.99:11434",
            "profile": "profiles/gemma4-64k-3090.yaml",
            "mode": "f16",
            "context_target": 65536,
            "phase": "restart-warm",
            "prompt_pack": "scripts/eval/prompts/runtime-regression.jsonl",
            "run_id": "run-003",
            "ttft_ms": None,
            "tokens_per_sec": None,
            "latency_ms": None,
            "vram_gb": None,
            "rss_gb": None,
            "success": False,
            "error_type": "restart_command_missing",
            "fallback_triggered": True,
            "selected_backend": None,
            "git_rev": "def5678",
            "timestamp": "2026-04-11T03:02:00Z",
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--input-glob",
            str(input_path),
            "--out",
            str(report_path),
            "--gates-out",
            str(summary_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    markdown = report_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "# Benchmark Summary" in markdown
    assert "## Overall" in markdown
    assert "Success rate: 66.7%" in markdown
    assert "Fallback rate: 33.3%" in markdown
    assert "mistral-small3.2:24b" in markdown
    assert "gemma4:26b-a4b-it-q4_K_M" in markdown
    assert summary["overall"]["record_count"] == 3
    assert summary["overall"]["success_count"] == 2
    assert summary["overall"]["error_types"] == {"restart_command_missing": 1}


def test_report_generation_rejects_empty_input_file(tmp_path: Path) -> None:
    input_path = tmp_path / "empty-results.json"
    report_path = tmp_path / "benchmark-summary.md"
    input_path.write_text("", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            "--input-glob",
            str(input_path),
            "--out",
            str(report_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "ERROR:" in result.stderr
