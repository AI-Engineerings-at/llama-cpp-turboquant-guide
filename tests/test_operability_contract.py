from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import qllama.runtime.llama_server as llama_server_runtime
from qllama.runtime.llama_server import (
    RuntimeStartupError,
    extract_context_length,
    verify_upstream_context_length,
)

ROOT = Path(__file__).resolve().parents[1]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_dockerfile_exposes_healthcheck_and_observability_defaults() -> None:
    dockerfile = read("Dockerfile")

    assert "QLLAMA_LOG_FORMAT=json" in dockerfile
    assert "QLLAMA_METRICS_ENABLED=true" in dockerfile
    assert "QLLAMA_INCLUDE_UPSTREAM_METRICS=true" in dockerfile
    assert (
        "HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3"
        in dockerfile
    )
    assert "curl -fsS http://localhost:8000/health" in dockerfile


def test_runtime_core_smoke_sets_auth_for_secure_by_default_api() -> None:
    script = read("scripts/smoke/runtime-core.sh")

    assert 'API_KEY="${QLLAMA_SMOKE_API_KEY:-qllama-smoke-key}"' in script
    assert 'API_KEY="${QLLAMA_SMOKE_API_KEY:-qllama-smoke-key}"' in script
    assert '-e "QLLAMA_API_KEYS=${API_KEY}"' in script
    assert 'AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")' in script
    assert "http://localhost:${PORT}/v1/models" in script
    assert "http://localhost:${PORT}/v1/chat/completions" in script


def test_operability_smoke_covers_metrics_logs_and_degraded_recovery() -> None:
    script = read("scripts/smoke/operability.sh")

    assert "http://localhost:${PORT}/metrics" in script
    assert 'qllama_auth_failures_total{reason=\\"missing\\"}' in script
    assert 'qllama_runtime_state{state_name=\\"degraded\\"} 1.0' in script
    assert 'qllama_runtime_state{state_name=\\"ready\\"} 1.0' in script
    assert "X-Request-ID: ${REQUEST_ID}" in script
    assert "structured request logs ok" in script
    assert "kill -STOP ${CHILD_PID}" in script
    assert "kill -CONT ${CHILD_PID}" in script


def test_readme_documents_operability_surface_and_verification_commands() -> None:
    readme = read("README.md")

    assert "scripts/smoke/operability.sh" in readme
    assert "GET /metrics" in readme
    assert "secure by default" in readme
    assert "QLLAMA_API_KEYS" in readme
    assert "bash scripts/smoke/runtime-core.sh" in readme
    assert "bash scripts/smoke/operability.sh" in readme


def test_extract_context_length_reads_nested_props_shape() -> None:
    payload = {
        "model_path": "/models/example.gguf",
        "default_generation_settings": {"n_ctx": 100000},
    }

    assert extract_context_length(payload) == 100000


class _FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise llama_server_runtime.httpx.HTTPStatusError(
                "boom",
                request=llama_server_runtime.httpx.Request(
                    "GET", "http://localhost:8010/props"
                ),
                response=llama_server_runtime.httpx.Response(self.status_code),
            )

    def json(self) -> object:
        return self._payload


class _FakeAsyncClient:
    def __init__(self, responses: list[_FakeResponse], timeout: float) -> None:
        self._responses = responses
        self.timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str) -> _FakeResponse:
        assert url.endswith("/props")
        return self._responses.pop(0)


def test_verify_upstream_context_length_accepts_matching_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [_FakeResponse({"default_generation_settings": {"n_ctx": 8192}})]
    monkeypatch.setattr(
        llama_server_runtime.httpx,
        "AsyncClient",
        lambda timeout=2.0: _FakeAsyncClient(responses, timeout),
    )

    actual = asyncio.run(verify_upstream_context_length("http://localhost:8010", 8192))

    assert actual == 8192


def test_verify_upstream_context_length_rejects_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [_FakeResponse({"default_generation_settings": {"n_ctx": 4096}})]
    monkeypatch.setattr(
        llama_server_runtime.httpx,
        "AsyncClient",
        lambda timeout=2.0: _FakeAsyncClient(responses, timeout),
    )

    with pytest.raises(
        RuntimeStartupError,
        match="expected context length 8192, but upstream reported 4096",
    ):
        asyncio.run(verify_upstream_context_length("http://localhost:8010", 8192))
