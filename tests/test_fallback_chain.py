from __future__ import annotations

import httpx
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI
from fastapi.testclient import TestClient

import qllama.auth as auth_module
import qllama.middleware.http_observability as http_observability_module
from qllama.config import AppConfig
from qllama.middleware import HTTPObservabilityMiddleware
from qllama.observability.metrics import MetricsCatalog
from qllama.routes import openai_proxy as openai_proxy_module
from qllama.routes import system as system_module
from qllama.routes.openai_proxy import router as openai_router
from qllama.routes.system import router as system_router


class DummyRuntime:
    def __init__(self) -> None:
        self.ready = True
        self.state = "ready"
        self.failures: list[tuple[str, str]] = []
        self.successes = 0
        self.process = object()
        self.selected_backend: str | None = None
        self.fallback_depth = 0
        self.fallback_reason: str | None = None
        self.last_backend_error: str | None = None

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state,
            "phase": self.state,
            "ready": self.ready,
            "detail": self.failures[-1][1] if self.failures else "ready",
            "last_error": self.failures[-1][1] if self.failures else None,
            "selected_backend": self.selected_backend,
            "fallback_depth": self.fallback_depth,
            "fallback_reason": self.fallback_reason,
            "last_backend_error": self.last_backend_error,
        }

    def set_backend_selection(
        self,
        *,
        selected_backend: str | None,
        fallback_depth: int,
        fallback_reason: str | None = None,
        last_backend_error: str | None = None,
    ) -> None:
        self.selected_backend = selected_backend
        self.fallback_depth = fallback_depth
        self.fallback_reason = fallback_reason
        self.last_backend_error = last_backend_error

    def record_proxy_failure(self, failure_type: str, detail: str) -> None:
        self.failures.append((failure_type, detail))
        self.ready = False
        self.state = "degraded"

    def record_proxy_success(self) -> None:
        self.successes += 1
        self.ready = True
        self.state = "ready"


class FakeHTTPResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        content: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.content = content if content is not None else text.encode("utf-8")
        self.headers = headers or {"content-type": "application/json"}


class FakeAsyncClient:
    def __init__(
        self,
        *,
        request_sequence: list[FakeHTTPResponse | Exception],
    ) -> None:
        self._request_sequence = list(request_sequence)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(
        self, method: str, url: str, *, content=None, headers=None
    ) -> FakeHTTPResponse:
        item = self._request_sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class DummyAsyncClientFactory:
    def __init__(self, client: FakeAsyncClient) -> None:
        self._client = client

    def __call__(self, *args, **kwargs) -> FakeAsyncClient:
        return self._client


def patch_catalog(monkeypatch, catalog: MetricsCatalog) -> None:
    monkeypatch.setattr(system_module, "metrics", catalog)
    monkeypatch.setattr(openai_proxy_module, "metrics", catalog)
    monkeypatch.setattr(auth_module, "metrics", catalog)
    monkeypatch.setattr(http_observability_module, "metrics", catalog)


def build_app(config: AppConfig, runtime: DummyRuntime) -> FastAPI:
    app = FastAPI()
    app.state.config = config
    app.state.runtime = runtime
    app.add_middleware(HTTPObservabilityMiddleware)
    app.add_middleware(
        CorrelationIdMiddleware,
        header_name=config.correlation_id_header,
        validator=None,
    )
    app.include_router(system_router)
    app.include_router(openai_router)
    return app


def test_primary_success_records_primary_backend(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_FALLBACK_URL", raising=False)
    monkeypatch.delenv("OLLAMA_TERTIARY_URL", raising=False)

    catalog = MetricsCatalog()
    patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        openai_proxy_module.httpx,
        "AsyncClient",
        DummyAsyncClientFactory(
            FakeAsyncClient(
                request_sequence=[
                    FakeHTTPResponse(status_code=200, text='{"data":[{"id":"local"}]}')
                ]
            )
        ),
    )

    runtime = DummyRuntime()
    app = build_app(
        AppConfig(
            auth_required=True, api_keys=("secret",), include_upstream_metrics=False
        ),
        runtime,
    )

    with TestClient(app) as client:
        response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        system = client.get("/system")
        metrics_response = client.get("/metrics")

    assert response.status_code == 200
    assert runtime.selected_backend == "primary"
    assert runtime.fallback_depth == 0
    assert runtime.fallback_reason is None
    assert runtime.last_backend_error is None
    assert system.json()["runtime"]["selected_backend"] == "primary"
    assert system.json()["runtime"]["fallback_depth"] == 0
    assert (
        'qllama_backend_requests_total{backend="primary",result="success"} 1.0'
        in metrics_response.text
    )
    assert 'qllama_selected_backend{backend="primary"} 1.0' in metrics_response.text
    assert "qllama_fallback_depth 0.0" in metrics_response.text


def test_primary_transport_error_falls_back_to_fallback1(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_FALLBACK_URL", "http://10.40.10.99:11434")
    monkeypatch.delenv("OLLAMA_TERTIARY_URL", raising=False)

    catalog = MetricsCatalog()
    patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        openai_proxy_module.httpx,
        "AsyncClient",
        DummyAsyncClientFactory(
            FakeAsyncClient(
                request_sequence=[
                    httpx.ConnectError("down"),
                    FakeHTTPResponse(
                        status_code=200, text='{"data":[{"id":"fallback1"}]}'
                    ),
                ]
            )
        ),
    )

    runtime = DummyRuntime()
    app = build_app(
        AppConfig(
            auth_required=True, api_keys=("secret",), include_upstream_metrics=False
        ),
        runtime,
    )

    with TestClient(app) as client:
        response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        system = client.get("/system")
        metrics_response = client.get("/metrics")

    assert response.status_code == 200
    assert runtime.successes == 1
    assert runtime.selected_backend == "fallback1"
    assert runtime.fallback_depth == 1
    assert runtime.fallback_reason == "transport"
    assert (
        runtime.last_backend_error
        == "upstream transport error while requesting /v1/models via primary"
    )
    assert system.json()["runtime"]["selected_backend"] == "fallback1"
    assert system.json()["runtime"]["fallback_reason"] == "transport"
    assert (
        'qllama_backend_requests_total{backend="primary",result="transport"} 1.0'
        in metrics_response.text
    )
    assert (
        'qllama_backend_requests_total{backend="fallback1",result="success"} 1.0'
        in metrics_response.text
    )
    assert (
        'qllama_fallback_activations_total{from_backend="primary",reason="transport",to_backend="fallback1"} 1.0'
        in metrics_response.text
    )
    assert 'qllama_selected_backend{backend="fallback1"} 1.0' in metrics_response.text
    assert "qllama_fallback_depth 1.0" in metrics_response.text


def test_primary_and_fallback1_fail_before_fallback2_success(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_FALLBACK_URL", "http://10.40.10.99:11434")
    monkeypatch.setenv("OLLAMA_TERTIARY_URL", "http://10.40.10.80:11434")

    catalog = MetricsCatalog()
    patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        openai_proxy_module.httpx,
        "AsyncClient",
        DummyAsyncClientFactory(
            FakeAsyncClient(
                request_sequence=[
                    FakeHTTPResponse(status_code=503, text='{"error":"loading"}'),
                    httpx.ReadTimeout("slow"),
                    FakeHTTPResponse(
                        status_code=200, text='{"data":[{"id":"fallback2"}]}'
                    ),
                ]
            )
        ),
    )

    runtime = DummyRuntime()
    app = build_app(
        AppConfig(
            auth_required=True, api_keys=("secret",), include_upstream_metrics=False
        ),
        runtime,
    )

    with TestClient(app) as client:
        response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        system = client.get("/system")
        metrics_response = client.get("/metrics")

    assert response.status_code == 200
    assert runtime.successes == 1
    assert runtime.selected_backend == "fallback2"
    assert runtime.fallback_depth == 2
    assert runtime.fallback_reason == "timeout"
    assert (
        runtime.last_backend_error
        == "upstream timeout while requesting /v1/models via fallback1"
    )
    assert system.json()["runtime"]["selected_backend"] == "fallback2"
    assert system.json()["runtime"]["fallback_depth"] == 2
    assert system.json()["runtime"]["fallback_reason"] == "timeout"
    assert (
        'qllama_backend_requests_total{backend="primary",result="http_5xx"} 1.0'
        in metrics_response.text
    )
    assert (
        'qllama_backend_requests_total{backend="fallback1",result="timeout"} 1.0'
        in metrics_response.text
    )
    assert (
        'qllama_backend_requests_total{backend="fallback2",result="success"} 1.0'
        in metrics_response.text
    )
    assert (
        'qllama_fallback_activations_total{from_backend="primary",reason="http_5xx",to_backend="fallback1"} 1.0'
        in metrics_response.text
    )
    assert (
        'qllama_fallback_activations_total{from_backend="fallback1",reason="timeout",to_backend="fallback2"} 1.0'
        in metrics_response.text
    )
    assert 'qllama_selected_backend{backend="fallback2"} 1.0' in metrics_response.text
    assert "qllama_fallback_depth 2.0" in metrics_response.text


def test_auth_failure_does_not_trigger_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_FALLBACK_URL", "http://10.40.10.99:11434")
    monkeypatch.setenv("OLLAMA_TERTIARY_URL", "http://10.40.10.80:11434")

    catalog = MetricsCatalog()
    patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        openai_proxy_module.httpx,
        "AsyncClient",
        DummyAsyncClientFactory(
            FakeAsyncClient(
                request_sequence=[FakeHTTPResponse(status_code=200, text='{"data":[]}')]
            )
        ),
    )

    runtime = DummyRuntime()
    app = build_app(
        AppConfig(
            auth_required=True, api_keys=("secret",), include_upstream_metrics=False
        ),
        runtime,
    )

    with TestClient(app) as client:
        response = client.get("/v1/models")
        system = client.get("/system")
        metrics_response = client.get("/metrics")

    assert response.status_code == 401
    assert runtime.selected_backend is None
    assert runtime.fallback_depth == 0
    assert runtime.successes == 0
    assert system.json()["runtime"]["selected_backend"] is None
    assert "qllama_fallback_activations_total{" not in metrics_response.text
    assert "qllama_backend_requests_total{" not in metrics_response.text
