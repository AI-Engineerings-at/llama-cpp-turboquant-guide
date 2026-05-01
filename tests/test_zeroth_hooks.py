from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from qllama.routes.openai_proxy import router

REAL_ASYNC_CLIENT = httpx.AsyncClient


class _RuntimeStub:
    def __init__(self, ready: bool = True, state: str = "ready") -> None:
        self.ready = ready
        self._state = state
        self.failures: list[tuple[str, str]] = []
        self.success_count = 0

    def snapshot(self) -> dict[str, object]:
        return {"state": self._state, "ready": self.ready}

    def record_proxy_failure(self, failure_type: str, detail: str) -> None:
        self.failures.append((failure_type, detail))

    def record_proxy_success(self) -> None:
        self.success_count += 1


class _FakeHooks:
    def __init__(self) -> None:
        self.before_calls = []
        self.after_calls = []

    async def before_inference(self, context):
        self.before_calls.append(context)

    async def after_inference(self, context):
        self.after_calls.append(context)


class _StubAsyncClient:
    def __init__(self, responder, *args, **kwargs) -> None:
        self._responder = responder

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method, url, content=None, headers=None):
        return await self._responder(method=method, url=url, content=content, headers=headers or {})



def _build_app(*, hooks_enabled: bool = True, runtime: _RuntimeStub | None = None, hooks: _FakeHooks | None = None):
    app = FastAPI()
    app.include_router(router)
    app.state.runtime = runtime or _RuntimeStub()
    app.state.config = SimpleNamespace(
        auth_required=False,
        api_keys=(),
        request_timeout_seconds=5.0,
        internal_base_url="http://internal-qllama",
        correlation_id_header="X-Request-ID",
        profile_name="baseline",
        zeroth_hooks_enabled=hooks_enabled,
    )
    app.state.zeroth_hooks = hooks or _FakeHooks()
    return app


@pytest.mark.asyncio
async def test_zeroth_hooks_wrap_successful_proxy_request(monkeypatch):
    hooks = _FakeHooks()
    runtime = _RuntimeStub(ready=True, state="ready")
    app = _build_app(runtime=runtime, hooks=hooks)
    transport = httpx.ASGITransport(app=app)

    async def responder(**kwargs):
        return httpx.Response(
            200,
            json={"id": "chatcmpl-1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]},
            headers={"content-type": "application/json"},
        )

    monkeypatch.setattr(
        "qllama.routes.openai_proxy.httpx.AsyncClient",
        lambda *args, **kwargs: _StubAsyncClient(responder, *args, **kwargs),
    )

    async with REAL_ASYNC_CLIENT(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={
                "Authorization": "Bearer secret-token",
                "X-Request-ID": "hook-req-1",
                "Content-Type": "application/json",
            },
            json={"model": "qllama/baseline", "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 200
    assert runtime.success_count == 1
    assert len(hooks.before_calls) == 1
    assert len(hooks.after_calls) == 1

    before = hooks.before_calls[0]
    after = hooks.after_calls[0]
    assert before.request_id == "hook-req-1"
    assert before.profile == "baseline"
    assert before.upstream_path == "/v1/chat/completions"
    assert before.headers == {"content-type": "application/json", "X-Request-ID": "hook-req-1"}
    assert b'"model":"qllama/baseline"' in before.body.replace(b" ", b"")
    assert after.status_code == 200
    assert after.error_type is None
    assert after.duration_ms >= 0


@pytest.mark.asyncio
async def test_zeroth_hooks_receive_timeout_failure_context(monkeypatch):
    hooks = _FakeHooks()
    runtime = _RuntimeStub(ready=True, state="ready")
    app = _build_app(runtime=runtime, hooks=hooks)
    transport = httpx.ASGITransport(app=app)

    async def responder(**kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(
        "qllama.routes.openai_proxy.httpx.AsyncClient",
        lambda *args, **kwargs: _StubAsyncClient(responder, *args, **kwargs),
    )

    async with REAL_ASYNC_CLIENT(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "hook-req-timeout", "Content-Type": "application/json"},
            json={"model": "qllama/baseline", "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 502
    body = response.json()
    assert body["error"]["type"] == "upstream_timeout"
    assert runtime.failures and runtime.failures[0][0] == "timeout"
    assert len(hooks.before_calls) == 1
    assert len(hooks.after_calls) == 1
    assert hooks.after_calls[0].status_code == 502
    assert hooks.after_calls[0].error_type == "upstream_timeout"


@pytest.mark.asyncio
async def test_hooks_can_be_disabled_without_breaking_proxy(monkeypatch):
    hooks = _FakeHooks()
    runtime = _RuntimeStub(ready=True, state="ready")
    app = _build_app(runtime=runtime, hooks=hooks, hooks_enabled=False)
    transport = httpx.ASGITransport(app=app)

    async def responder(**kwargs):
        return httpx.Response(200, json={"data": []}, headers={"content-type": "application/json"})

    monkeypatch.setattr(
        "qllama.routes.openai_proxy.httpx.AsyncClient",
        lambda *args, **kwargs: _StubAsyncClient(responder, *args, **kwargs),
    )

    async with REAL_ASYNC_CLIENT(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert hooks.before_calls == []
    assert hooks.after_calls == []
    assert runtime.success_count == 1
