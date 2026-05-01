from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qllama.config import AppConfig
from qllama.observability.metrics import MetricsCatalog
from qllama.routes.system import router as system_router
from qllama.runtime import llama_server as runtime_module
from qllama.runtime.llama_server import (
    ContextVerificationError,
    LlamaServerRuntime,
    RuntimeStartupError,
)


class FakeStdout:
    async def readline(self) -> bytes:
        return b""


class FakeProcess:
    def __init__(self, *, pid: int = 4321, exit_code: int | None = None) -> None:
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


class ExitProcess(FakeProcess):
    def __init__(self, exit_code: int) -> None:
        super().__init__(pid=8765, exit_code=None)
        self._exit_code = exit_code

    async def wait(self) -> int:
        self.returncode = self._exit_code
        return self.returncode


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


class DummyRuntime:
    def __init__(self, *, ready: bool, snapshot: dict[str, object]) -> None:
        self.ready = ready
        self._snapshot = snapshot

    def snapshot(self) -> dict[str, object]:
        return self._snapshot


async def _noop(*_: object, **__: object) -> None:
    return None


async def _mark_upstream_ready(runtime: LlamaServerRuntime) -> None:
    runtime.upstream_status = 200


async def _run_start(runtime: LlamaServerRuntime) -> None:
    await runtime.start()
    await runtime.stop()


def test_runtime_start_success_transitions_to_ready(
    monkeypatch, tmp_path: Path
) -> None:
    catalog = MetricsCatalog()
    monkeypatch.setattr(runtime_module, "metrics", catalog)

    model_path = tmp_path / "model.gguf"
    model_path.write_text("fake model")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    monkeypatch.setattr(
        runtime_module, "load_profile", lambda *args, **kwargs: FakeProfile(model_path)
    )
    monkeypatch.setattr(
        runtime_module, "verify_cache_capability", lambda *args, **kwargs: None
    )

    async def verify_context_length(*args, **kwargs):
        return 4096

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
    asyncio.run(_run_start(runtime))

    snapshot = runtime.snapshot()
    rendered = catalog.render_latest().decode("utf-8")

    assert snapshot["phase"] == "stopped"
    assert snapshot["transition_history"][-2]["to_state"] == "ready"
    assert snapshot["expected_context_length"] == 4096
    assert snapshot["actual_context_length"] == 4096
    assert snapshot["context_verification_status"] == "verified"
    assert snapshot["context_verification_error"] is None
    assert snapshot["verified_cache_capability"] is None
    assert "qllama_startup_attempts_total 1.0" in rendered
    assert 'qllama_runtime_state{state_name="stopped"} 1.0' in rendered


def test_runtime_context_verification_failure_is_visible_in_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    catalog = MetricsCatalog()
    monkeypatch.setattr(runtime_module, "metrics", catalog)

    model_path = tmp_path / "model.gguf"
    model_path.write_text("fake model")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProcess()

    async def raise_context_mismatch(*args, **kwargs):
        raise ContextVerificationError(
            "Profile expected context length 4096, but upstream reported 2048",
            expected_context_length=4096,
            actual_context_length=2048,
        )

    monkeypatch.setattr(
        runtime_module, "load_profile", lambda *args, **kwargs: FakeProfile(model_path)
    )
    monkeypatch.setattr(
        runtime_module, "verify_cache_capability", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_module, "verify_upstream_context_length", raise_context_mismatch
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

    snapshot = runtime.snapshot()

    assert snapshot["phase"] == "failed"
    assert snapshot["expected_context_length"] == 4096
    assert snapshot["actual_context_length"] == 2048
    assert snapshot["context_verification_status"] == "failed"
    assert (
        snapshot["context_verification_error"]
        == "Profile expected context length 4096, but upstream reported 2048"
    )


def test_runtime_start_failure_transitions_to_failed(monkeypatch) -> None:
    catalog = MetricsCatalog()
    monkeypatch.setattr(runtime_module, "metrics", catalog)
    monkeypatch.setattr(
        runtime_module,
        "load_profile",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeStartupError("Unknown qllama profile 'broken'")
        ),
    )

    runtime = LlamaServerRuntime(AppConfig(profile_name="broken"))
    asyncio.run(runtime.start())

    snapshot = runtime.snapshot()
    rendered = catalog.render_latest().decode("utf-8")

    assert snapshot["phase"] == "failed"
    assert snapshot["ready"] is False
    assert "Unknown qllama profile" in snapshot["last_error"]
    assert 'qllama_startup_failures_total{reason="invalid_profile"} 1.0' in rendered
    assert 'qllama_runtime_state{state_name="failed"} 1.0' in rendered


def test_child_exit_after_ready_marks_runtime_failed(monkeypatch) -> None:
    catalog = MetricsCatalog()
    monkeypatch.setattr(runtime_module, "metrics", catalog)

    runtime = LlamaServerRuntime(AppConfig())
    runtime.process = ExitProcess(exit_code=9)
    runtime._transition(runtime_module.RuntimeState.READY, "runtime ready for traffic")

    asyncio.run(runtime._watch_process_exit())

    snapshot = runtime.snapshot()
    rendered = catalog.render_latest().decode("utf-8")

    assert snapshot["phase"] == "failed"
    assert "exited with code 9" in snapshot["last_error"]
    assert 'qllama_child_process_exits_total{exit_code="9"} 1.0' in rendered
    assert 'qllama_runtime_state{state_name="failed"} 1.0' in rendered


def test_system_routes_keep_liveness_and_readiness_split() -> None:
    app = FastAPI()
    app.state.runtime = DummyRuntime(
        ready=False,
        snapshot={
            "phase": "failed",
            "state": "failed",
            "ready": False,
            "detail": "invalid profile",
            "last_error": "invalid profile",
            "expected_context_length": 4096,
            "actual_context_length": 2048,
            "context_verification_status": "failed",
            "context_verification_error": "Profile expected context length 4096, but upstream reported 2048",
        },
    )
    app.include_router(system_router)

    with TestClient(app) as client:
        system = client.get("/system")
        health = client.get("/health")
        ready = client.get("/ready")

    assert system.status_code == 200
    assert system.json()["runtime"]["expected_context_length"] == 4096
    assert system.json()["runtime"]["actual_context_length"] == 2048
    assert system.json()["runtime"]["context_verification_status"] == "failed"
    assert health.status_code == 200
    assert health.json()["runtime"]["state"] == "failed"
    assert ready.status_code == 503
    assert ready.json()["status"] == "not_ready"
