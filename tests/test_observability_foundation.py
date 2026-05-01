from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

from qllama.config import (
    DEFAULT_REQUEST_DURATION_BUCKETS,
    DEFAULT_UPSTREAM_HEALTH_BUCKETS,
    AppConfig,
    load_config,
)
from qllama.main import create_app
from qllama.observability.logging import configure_logging, get_logger
from qllama.observability.metrics import MetricsCatalog, metrics, normalize_endpoint
from qllama.observability.state import (
    RUNTIME_STATE_CODES,
    RuntimeState,
    RuntimeStateTracker,
)
from qllama.runtime import llama_server as runtime_module
from qllama.runtime.llama_server import LlamaServerRuntime


class FakeStdout:
    async def readline(self) -> bytes:
        return b""


class FakeProcess:
    def __init__(self, *, pid: int, exit_code: int | None = None) -> None:
        self.pid = pid
        self.returncode = exit_code
        self.stdout = FakeStdout()

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


class FakeProfile:
    def __init__(self, model_path: Path) -> None:
        self.name = "baseline"
        self.model_path = model_path.name
        self.alias = "local"
        self.cache_type_k = "f16"
        self.cache_type_v = "f16"
        self.context_size = 4096
        self.gpu_layers = 99
        self.extra_args: list[str] = []
        self.is_turboquant = False
        self.verification = SimpleNamespace(
            required_cache_capability=None,
            expected_context_length=4096,
            startup_timeout_seconds=1,
        )
        self._resolved_model_path = model_path

    def resolved_model_path(self, _: Path) -> Path:
        return self._resolved_model_path


async def _noop(*_: object, **__: object) -> None:
    return None


async def _mark_upstream_ready(runtime: LlamaServerRuntime) -> None:
    runtime.upstream_status = 200


def test_load_config_includes_observability_settings(monkeypatch) -> None:
    monkeypatch.setenv("QLLAMA_LOG_FORMAT", "json")
    monkeypatch.setenv("QLLAMA_CORRELATION_ID_HEADER", "X-Correlation-ID")
    monkeypatch.setenv("QLLAMA_METRICS_ENABLED", "false")
    monkeypatch.setenv("QLLAMA_INCLUDE_UPSTREAM_METRICS", "false")
    monkeypatch.setenv("QLLAMA_REQUEST_DURATION_BUCKETS", "0.1,0.2,0.5")
    monkeypatch.setenv("QLLAMA_UPSTREAM_HEALTH_DURATION_BUCKETS", "0.01,0.05")
    monkeypatch.setenv("QLLAMA_AUTH_REQUIRED", "true")

    config = load_config()

    assert config.log_format == "json"
    assert config.correlation_id_header == "X-Correlation-ID"
    assert config.metrics_enabled is False
    assert config.include_upstream_metrics is False
    assert config.request_duration_buckets == (0.1, 0.2, 0.5)
    assert config.upstream_health_duration_buckets == (0.01, 0.05)
    assert config.auth_required is True


def test_default_bucket_constants_are_stable() -> None:
    assert DEFAULT_REQUEST_DURATION_BUCKETS[0] == 0.005
    assert DEFAULT_REQUEST_DURATION_BUCKETS[-1] == 10.0
    assert DEFAULT_UPSTREAM_HEALTH_BUCKETS[0] == 0.001
    assert DEFAULT_UPSTREAM_HEALTH_BUCKETS[-1] == 1.0


def test_metrics_catalog_exposes_expected_metric_families() -> None:
    family_names = {metric.name for metric in metrics.registry.collect()}

    assert {
        "qllama_http_requests",
        "qllama_http_request_duration_seconds",
        "qllama_http_requests_in_flight",
        "qllama_auth_failures",
        "qllama_upstream_failures",
        "qllama_upstream_health_check_duration_seconds",
        "qllama_startup_attempts",
        "qllama_startup_failures",
        "qllama_child_process_exits",
        "qllama_child_process_restarts",
        "qllama_runtime_state",
        "qllama_ready_status",
    }.issubset(family_names)

    rendered = metrics.render_latest().decode("utf-8")
    assert (
        "# HELP qllama_http_requests_total Total HTTP requests to qllama." in rendered
    )
    assert (
        "# HELP qllama_runtime_state Current qllama runtime state as a one-hot gauge."
        in rendered
    )
    assert 'qllama_runtime_state{state_name="initial"} 0.0' in rendered
    assert "qllama_ready_status 0.0" in rendered
    assert (
        normalize_endpoint("/v1/chat/completions?stream=true") == "/v1/chat/completions"
    )
    assert normalize_endpoint("/unknown/123") == "other"


def test_runtime_state_tracker_records_transitions() -> None:
    tracker = RuntimeStateTracker()

    assert tracker.state == RuntimeState.INITIAL
    assert tracker.ready is False

    tracker.transition(RuntimeState.STARTING, "booting app")
    transition = tracker.transition(RuntimeState.READY, "upstream became healthy")
    snapshot = tracker.snapshot()

    assert transition.from_state == RuntimeState.STARTING
    assert transition.to_state == RuntimeState.READY
    assert snapshot["state"] == "ready"
    assert snapshot["state_code"] == RUNTIME_STATE_CODES[RuntimeState.READY]
    assert snapshot["ready"] is True
    assert snapshot["accepts_traffic"] is True
    assert snapshot["history"][-1]["reason"] == "upstream became healthy"


def test_child_restart_counter_does_not_increment_on_cold_start(
    monkeypatch, tmp_path: Path
) -> None:
    catalog = MetricsCatalog()
    monkeypatch.setattr(runtime_module, "metrics", catalog)

    model_path = tmp_path / "model.gguf"
    model_path.write_text("fake model")
    processes = [FakeProcess(pid=1001)]

    async def fake_create_subprocess_exec(*args, **kwargs):
        return processes.pop(0)

    async def verify_context_length(*args, **kwargs):
        return 4096

    monkeypatch.setattr(
        runtime_module, "load_profile", lambda *args, **kwargs: FakeProfile(model_path)
    )
    monkeypatch.setattr(
        runtime_module, "verify_cache_capability", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_module, "verify_upstream_context_length", verify_context_length
    )
    monkeypatch.setattr(
        runtime_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )
    monkeypatch.setattr(LlamaServerRuntime, "_consume_logs", _noop)
    monkeypatch.setattr(LlamaServerRuntime, "_watch_process_exit", _noop)
    monkeypatch.setattr(
        LlamaServerRuntime, "_wait_for_upstream_ready", _mark_upstream_ready
    )

    runtime = LlamaServerRuntime(AppConfig())
    asyncio.run(runtime.start())

    rendered = catalog.render_latest().decode("utf-8")

    assert runtime.process is not None
    assert runtime.process.pid == 1001
    assert "qllama_child_process_restarts_total{" not in rendered


def test_child_restart_counter_increments_only_on_real_restart(
    monkeypatch, tmp_path: Path
) -> None:
    catalog = MetricsCatalog()
    monkeypatch.setattr(runtime_module, "metrics", catalog)

    model_path = tmp_path / "model.gguf"
    model_path.write_text("fake model")
    processes = [FakeProcess(pid=1001), FakeProcess(pid=1002)]

    async def fake_create_subprocess_exec(*args, **kwargs):
        return processes.pop(0)

    async def verify_context_length(*args, **kwargs):
        return 4096

    monkeypatch.setattr(
        runtime_module, "load_profile", lambda *args, **kwargs: FakeProfile(model_path)
    )
    monkeypatch.setattr(
        runtime_module, "verify_cache_capability", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_module, "verify_upstream_context_length", verify_context_length
    )
    monkeypatch.setattr(
        runtime_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )
    monkeypatch.setattr(LlamaServerRuntime, "_consume_logs", _noop)
    monkeypatch.setattr(LlamaServerRuntime, "_watch_process_exit", _noop)
    monkeypatch.setattr(
        LlamaServerRuntime, "_wait_for_upstream_ready", _mark_upstream_ready
    )

    runtime = LlamaServerRuntime(AppConfig())
    asyncio.run(runtime.start())
    first_pid = runtime.process.pid if runtime.process else None

    asyncio.run(runtime.restart_child_process("unexpected_exit"))
    rendered = catalog.render_latest().decode("utf-8")

    assert first_pid == 1001
    assert runtime.process is not None
    assert runtime.process.pid == 1002
    assert (
        'qllama_child_process_restarts_total{reason="unexpected_exit"} 1.0' in rendered
    )


def test_repeated_child_restarts_accumulate_counter(
    monkeypatch, tmp_path: Path
) -> None:
    catalog = MetricsCatalog()
    monkeypatch.setattr(runtime_module, "metrics", catalog)

    model_path = tmp_path / "model.gguf"
    model_path.write_text("fake model")
    processes = [FakeProcess(pid=1001), FakeProcess(pid=1002), FakeProcess(pid=1003)]

    async def fake_create_subprocess_exec(*args, **kwargs):
        return processes.pop(0)

    async def verify_context_length(*args, **kwargs):
        return 4096

    monkeypatch.setattr(
        runtime_module, "load_profile", lambda *args, **kwargs: FakeProfile(model_path)
    )
    monkeypatch.setattr(
        runtime_module, "verify_cache_capability", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_module, "verify_upstream_context_length", verify_context_length
    )
    monkeypatch.setattr(
        runtime_module.asyncio, "create_subprocess_exec", fake_create_subprocess_exec
    )
    monkeypatch.setattr(LlamaServerRuntime, "_consume_logs", _noop)
    monkeypatch.setattr(LlamaServerRuntime, "_watch_process_exit", _noop)
    monkeypatch.setattr(
        LlamaServerRuntime, "_wait_for_upstream_ready", _mark_upstream_ready
    )

    runtime = LlamaServerRuntime(AppConfig())
    asyncio.run(runtime.start())
    asyncio.run(runtime.restart_child_process("crash_loop"))
    asyncio.run(runtime.restart_child_process("crash_loop"))
    rendered = catalog.render_latest().decode("utf-8")

    assert runtime.process is not None
    assert runtime.process.pid == 1003
    assert 'qllama_child_process_restarts_total{reason="crash_loop"} 2.0' in rendered


def test_create_app_uses_extended_config(monkeypatch) -> None:
    monkeypatch.setenv("QLLAMA_LOG_FORMAT", "json")
    monkeypatch.setenv("QLLAMA_INCLUDE_UPSTREAM_METRICS", "false")
    monkeypatch.setenv("QLLAMA_DEGRADED_THRESHOLD", "5")

    app = create_app()
    config = app.state.config

    assert config.log_format == "json"
    assert config.include_upstream_metrics is False
    assert config.degraded_threshold == 5
    assert app.state.runtime.config is config


def test_logging_bootstrap_supports_json_renderer() -> None:
    configure_logging(level="debug", fmt="json", force=True)
    logger = get_logger("tests.observability")
    logger.info("observability_foundation_test", check=True)

    assert logging.getLogger().handlers
