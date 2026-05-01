from __future__ import annotations

from uuid import uuid4

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
        get_response: FakeHTTPResponse | Exception | None = None,
        request_response: FakeHTTPResponse | None = None,
        request_error: Exception | None = None,
        request_sequence: list[FakeHTTPResponse | Exception] | None = None,
    ) -> None:
        self._get_response = get_response
        self._request_response = request_response
        self._request_error = request_error
        self._request_sequence = list(request_sequence or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url: str) -> FakeHTTPResponse:
        if isinstance(self._get_response, Exception):
            raise self._get_response
        assert self._get_response is not None
        return self._get_response

    async def request(
        self, method: str, url: str, *, content=None, headers=None
    ) -> FakeHTTPResponse:
        if self._request_sequence:
            item = self._request_sequence.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if self._request_error is not None:
            raise self._request_error
        assert self._request_response is not None
        return self._request_response


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


def test_metrics_endpoint_survives_missing_upstream_metrics(monkeypatch) -> None:
    catalog = MetricsCatalog()
    patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        system_module.httpx,
        "AsyncClient",
        DummyAsyncClientFactory(
            FakeAsyncClient(get_response=httpx.ConnectError("boom"))
        ),
    )

    app = build_app(
        AppConfig(
            metrics_enabled=True, include_upstream_metrics=True, auth_required=False
        ),
        DummyRuntime(),
    )

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "qllama_ready_status" in response.text
    assert "llamacpp:" not in response.text


def test_metrics_endpoint_merges_upstream_payload(monkeypatch) -> None:
    catalog = MetricsCatalog()
    patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        system_module.httpx,
        "AsyncClient",
        DummyAsyncClientFactory(
            FakeAsyncClient(
                get_response=FakeHTTPResponse(
                    status_code=200,
                    text="# HELP llamacpp:test help\nllamacpp:test 1\n",
                )
            )
        ),
    )

    app = build_app(
        AppConfig(
            metrics_enabled=True, include_upstream_metrics=True, auth_required=False
        ),
        DummyRuntime(),
    )

    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "qllama_ready_status" in response.text
    assert "llamacpp:test 1" in response.text


def test_auth_failure_is_counted_and_request_id_is_echoed(monkeypatch) -> None:
    catalog = MetricsCatalog()
    patch_catalog(monkeypatch, catalog)

    config = AppConfig(
        auth_required=True, api_keys=("secret",), include_upstream_metrics=False
    )
    app = build_app(config, DummyRuntime())
    request_id = str(uuid4())

    with TestClient(app) as client:
        response = client.get("/v1/models", headers={"X-Request-ID": request_id})
        metrics_response = client.get("/metrics")

    assert response.status_code == 401
    assert response.headers["X-Request-ID"] == request_id
    assert 'qllama_auth_failures_total{reason="missing"} 1.0' in metrics_response.text
    assert (
        'qllama_http_requests_total{endpoint="/v1/models",method="GET",status_code="401"} 1.0'
        in metrics_response.text
    )


def test_proxy_transport_error_marks_runtime_degraded(monkeypatch) -> None:
    catalog = MetricsCatalog()
    patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        openai_proxy_module.httpx,
        "AsyncClient",
        DummyAsyncClientFactory(
            FakeAsyncClient(request_error=httpx.ConnectError("down"))
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
        ready = client.get("/ready")
        metrics_response = client.get("/metrics")

    assert response.status_code == 502
    assert runtime.failures[-1][0] == "transport"
    assert runtime.selected_backend == "primary"
    assert runtime.fallback_depth == 0
    assert runtime.fallback_reason == "transport"
    assert ready.status_code == 503
    assert (
        'qllama_upstream_failures_total{type="transport"} 1.0' in metrics_response.text
    )
    assert (
        'qllama_backend_requests_total{backend="primary",result="transport"} 1.0'
        in metrics_response.text
    )


def test_proxy_http_5xx_marks_runtime_degraded(monkeypatch) -> None:
    catalog = MetricsCatalog()
    patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        openai_proxy_module.httpx,
        "AsyncClient",
        DummyAsyncClientFactory(
            FakeAsyncClient(
                request_response=FakeHTTPResponse(
                    status_code=503, text='{"error":"loading"}'
                )
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
        response = client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
            json={"model": "local", "messages": [{"role": "user", "content": "hi"}]},
        )
        metrics_response = client.get("/metrics")

    assert response.status_code == 503
    assert runtime.failures[-1][0] == "http_5xx"
    assert runtime.selected_backend == "primary"
    assert runtime.fallback_depth == 0
    assert runtime.fallback_reason == "http_5xx"
    assert (
        'qllama_upstream_failures_total{type="http_5xx"} 1.0' in metrics_response.text
    )
    assert (
        'qllama_backend_requests_total{backend="primary",result="http_5xx"} 1.0'
        in metrics_response.text
    )


def test_proxy_success_recovers_degraded_runtime(monkeypatch) -> None:
    catalog = MetricsCatalog()
    patch_catalog(monkeypatch, catalog)
    monkeypatch.setattr(
        openai_proxy_module.httpx,
        "AsyncClient",
        DummyAsyncClientFactory(
            FakeAsyncClient(
                request_response=FakeHTTPResponse(
                    status_code=200, text='{"data":[{"id":"local"}]}'
                )
            )
        ),
    )

    runtime = DummyRuntime()
    runtime.ready = False
    runtime.state = "degraded"
    app = build_app(
        AppConfig(
            auth_required=True, api_keys=("secret",), include_upstream_metrics=False
        ),
        runtime,
    )

    with TestClient(app) as client:
        response = client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        ready = client.get("/ready")

    assert response.status_code == 200
    assert runtime.successes == 1
    assert runtime.state == "ready"
    assert runtime.selected_backend == "primary"
    assert runtime.fallback_depth == 0
    assert ready.status_code == 200
