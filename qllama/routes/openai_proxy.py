from __future__ import annotations

import os
import time

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from qllama.auth import require_api_key
from qllama.observability.logging import get_logger
from qllama.observability.metrics import metrics
from qllama.zeroth_hooks import PostInferenceHookContext, PreInferenceHookContext

logger = get_logger(__name__)
router = APIRouter(prefix="/v1", tags=["openai"])


BACKEND_SNAPSHOT_KEYS = (
    "selected_backend",
    "fallback_depth",
    "fallback_reason",
    "last_backend_error",
)


def _runtime_snapshot(runtime: object) -> dict[str, object]:
    snapshot = runtime.snapshot()
    state_tracker = getattr(runtime, "state_tracker", None)
    if state_tracker is not None and hasattr(state_tracker, "snapshot"):
        state_snapshot = state_tracker.snapshot()
        for key in BACKEND_SNAPSHOT_KEYS:
            if key not in snapshot:
                snapshot[key] = state_snapshot.get(key)
    return snapshot


def _set_backend_observability(
    runtime: object,
    *,
    selected_backend: str | None,
    fallback_depth: int,
    fallback_reason: str | None = None,
    last_backend_error: str | None = None,
) -> None:
    state_tracker = getattr(runtime, "state_tracker", None)
    if state_tracker is not None and hasattr(state_tracker, "set_backend_selection"):
        state_tracker.set_backend_selection(
            selected_backend=selected_backend,
            fallback_depth=fallback_depth,
            fallback_reason=fallback_reason,
            last_backend_error=last_backend_error,
        )
    elif hasattr(runtime, "set_backend_selection"):
        runtime.set_backend_selection(
            selected_backend=selected_backend,
            fallback_depth=fallback_depth,
            fallback_reason=fallback_reason,
            last_backend_error=last_backend_error,
        )
    metrics.set_selected_backend(selected_backend, fallback_depth)


def _configured_backends(config: object) -> list[tuple[str, str]]:
    backends = [("primary", str(getattr(config, "internal_base_url")))]
    candidates = (
        (
            "fallback1",
            getattr(config, "fallback_url", None) or os.getenv("OLLAMA_FALLBACK_URL"),
        ),
        (
            "fallback2",
            getattr(config, "tertiary_url", None) or os.getenv("OLLAMA_TERTIARY_URL"),
        ),
    )
    for name, raw_url in candidates:
        if not raw_url:
            continue
        url = str(raw_url).rstrip("/")
        if url and all(existing_url != url for _, existing_url in backends):
            backends.append((name, url))
    return backends


def error_response(
    request: Request, *, status_code: int, message: str, error_type: str
) -> JSONResponse:
    runtime = request.app.state.runtime
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": status_code,
                "message": message,
                "type": error_type,
            },
            "runtime": _runtime_snapshot(runtime),
        },
    )


def not_ready_response(request: Request) -> JSONResponse:
    return error_response(
        request,
        status_code=503,
        message="qllama is not ready",
        error_type="unavailable_error",
    )


def _hook_headers(request: Request) -> dict[str, str]:
    config = request.app.state.config
    headers: dict[str, str] = {}
    if content_type := request.headers.get("content-type"):
        headers["content-type"] = content_type
    if request_id := request.headers.get(config.correlation_id_header):
        headers[config.correlation_id_header] = request_id
    return headers


def _build_pre_hook_context(
    request: Request,
    *,
    endpoint: str,
    upstream_path: str,
    request_body: bytes,
) -> PreInferenceHookContext:
    runtime = request.app.state.runtime
    config = request.app.state.config
    snapshot = _runtime_snapshot(runtime)
    return PreInferenceHookContext(
        endpoint=endpoint,
        upstream_path=upstream_path,
        method=request.method,
        request_id=request.headers.get(config.correlation_id_header),
        profile=config.profile_name,
        runtime_state=snapshot.get("state"),
        headers=_hook_headers(request),
        body=request_body,
    )


async def _run_before_hook(request: Request, context: PreInferenceHookContext) -> None:
    hooks = getattr(request.app.state, "zeroth_hooks", None)
    config = request.app.state.config
    if not config.zeroth_hooks_enabled or hooks is None:
        return
    await hooks.before_inference(context)


async def _run_after_hook(
    request: Request,
    pre_context: PreInferenceHookContext,
    *,
    status_code: int,
    error_type: str | None,
    started_at: float,
) -> None:
    hooks = getattr(request.app.state, "zeroth_hooks", None)
    config = request.app.state.config
    if not config.zeroth_hooks_enabled or hooks is None:
        return

    post_context = PostInferenceHookContext(
        endpoint=pre_context.endpoint,
        upstream_path=pre_context.upstream_path,
        method=pre_context.method,
        request_id=pre_context.request_id,
        profile=pre_context.profile,
        runtime_state=pre_context.runtime_state,
        headers=pre_context.headers,
        body=pre_context.body,
        status_code=status_code,
        error_type=error_type,
        duration_ms=int((time.time() - started_at) * 1000),
    )
    await hooks.after_inference(post_context)


async def proxy_request(request: Request, upstream_path: str) -> Response:
    runtime = request.app.state.runtime
    config = request.app.state.config
    endpoint = getattr(request.state, "normalized_endpoint", request.url.path)

    runtime_state = _runtime_snapshot(runtime).get("state")

    if not runtime.ready and runtime_state != "degraded":
        logger.warning("proxy_rejected_not_ready", endpoint=endpoint)
        return not_ready_response(request)

    request_body = await request.body()
    pre_context = _build_pre_hook_context(
        request,
        endpoint=endpoint,
        upstream_path=upstream_path,
        request_body=request_body,
    )
    await _run_before_hook(request, pre_context)

    headers: dict[str, str] = {}
    if content_type := request.headers.get("content-type"):
        headers["content-type"] = content_type

    started_at = time.time()
    backend_chain = _configured_backends(config)
    fallback_reason: str | None = None
    last_backend_error: str | None = None

    async with httpx.AsyncClient(timeout=config.request_timeout_seconds) as client:
        for index, (backend_name, base_url) in enumerate(backend_chain):
            upstream_url = f"{base_url.rstrip('/')}{upstream_path}"
            try:
                upstream = await client.request(
                    request.method,
                    upstream_url,
                    content=request_body or None,
                    headers=headers,
                )
            except httpx.TimeoutException as exc:
                metrics.upstream_failures_total.labels(type="timeout").inc()
                metrics.record_backend_attempt(backend_name, "timeout")
                detail = f"upstream timeout while requesting {upstream_path} via {backend_name}"
                logger.warning(
                    "proxy_upstream_timeout",
                    endpoint=endpoint,
                    upstream_path=upstream_path,
                    backend=backend_name,
                    error_type=exc.__class__.__name__,
                )
                last_backend_error = detail
                if index + 1 < len(backend_chain):
                    next_backend = backend_chain[index + 1][0]
                    fallback_reason = "timeout"
                    metrics.record_fallback_activation(
                        reason=fallback_reason,
                        from_backend=backend_name,
                        to_backend=next_backend,
                    )
                    logger.warning(
                        "proxy_backend_fallback",
                        endpoint=endpoint,
                        upstream_path=upstream_path,
                        from_backend=backend_name,
                        to_backend=next_backend,
                        reason=fallback_reason,
                    )
                    continue

                runtime.record_proxy_failure("timeout", detail)
                _set_backend_observability(
                    runtime,
                    selected_backend=backend_name,
                    fallback_depth=index,
                    fallback_reason=fallback_reason or "timeout",
                    last_backend_error=detail,
                )
                response = error_response(
                    request,
                    status_code=502,
                    message="Upstream llama-server timed out",
                    error_type="upstream_timeout",
                )
                await _run_after_hook(
                    request,
                    pre_context,
                    status_code=response.status_code,
                    error_type="upstream_timeout",
                    started_at=started_at,
                )
                return response
            except httpx.HTTPError as exc:
                metrics.upstream_failures_total.labels(type="transport").inc()
                metrics.record_backend_attempt(backend_name, "transport")
                detail = f"upstream transport error while requesting {upstream_path} via {backend_name}"
                logger.warning(
                    "proxy_upstream_transport_error",
                    endpoint=endpoint,
                    upstream_path=upstream_path,
                    backend=backend_name,
                    error_type=exc.__class__.__name__,
                )
                last_backend_error = detail
                if index + 1 < len(backend_chain):
                    next_backend = backend_chain[index + 1][0]
                    fallback_reason = "transport"
                    metrics.record_fallback_activation(
                        reason=fallback_reason,
                        from_backend=backend_name,
                        to_backend=next_backend,
                    )
                    logger.warning(
                        "proxy_backend_fallback",
                        endpoint=endpoint,
                        upstream_path=upstream_path,
                        from_backend=backend_name,
                        to_backend=next_backend,
                        reason=fallback_reason,
                    )
                    continue

                runtime.record_proxy_failure("transport", detail)
                _set_backend_observability(
                    runtime,
                    selected_backend=backend_name,
                    fallback_depth=index,
                    fallback_reason=fallback_reason or "transport",
                    last_backend_error=detail,
                )
                response = error_response(
                    request,
                    status_code=502,
                    message="Upstream llama-server transport error",
                    error_type="upstream_transport_error",
                )
                await _run_after_hook(
                    request,
                    pre_context,
                    status_code=response.status_code,
                    error_type="upstream_transport_error",
                    started_at=started_at,
                )
                return response

            response_headers: dict[str, str] = {}
            if content_type := upstream.headers.get("content-type"):
                response_headers["content-type"] = content_type

            error_type: str | None = None
            if upstream.status_code >= 500:
                error_type = "http_5xx"
                metrics.upstream_failures_total.labels(type="http_5xx").inc()
                metrics.record_backend_attempt(backend_name, "http_5xx")
                detail = f"upstream returned {upstream.status_code} for {upstream_path} via {backend_name}"
                logger.warning(
                    "proxy_upstream_http_5xx",
                    endpoint=endpoint,
                    upstream_path=upstream_path,
                    backend=backend_name,
                    status_code=upstream.status_code,
                )
                last_backend_error = detail
                if index + 1 < len(backend_chain):
                    next_backend = backend_chain[index + 1][0]
                    fallback_reason = "http_5xx"
                    metrics.record_fallback_activation(
                        reason=fallback_reason,
                        from_backend=backend_name,
                        to_backend=next_backend,
                    )
                    logger.warning(
                        "proxy_backend_fallback",
                        endpoint=endpoint,
                        upstream_path=upstream_path,
                        from_backend=backend_name,
                        to_backend=next_backend,
                        reason=fallback_reason,
                    )
                    continue

                runtime.record_proxy_failure("http_5xx", detail)
                _set_backend_observability(
                    runtime,
                    selected_backend=backend_name,
                    fallback_depth=index,
                    fallback_reason=fallback_reason or "http_5xx",
                    last_backend_error=detail,
                )
            else:
                metrics.record_backend_attempt(backend_name, "success")
                runtime.record_proxy_success()
                _set_backend_observability(
                    runtime,
                    selected_backend=backend_name,
                    fallback_depth=index,
                    fallback_reason=fallback_reason if index > 0 else None,
                    last_backend_error=last_backend_error if index > 0 else None,
                )

            response = Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=response_headers,
            )
            await _run_after_hook(
                request,
                pre_context,
                status_code=response.status_code,
                error_type=error_type,
                started_at=started_at,
            )
            return response

    response = error_response(
        request,
        status_code=502,
        message="No upstream backend was available",
        error_type="upstream_unavailable",
    )
    await _run_after_hook(
        request,
        pre_context,
        status_code=response.status_code,
        error_type="upstream_unavailable",
        started_at=started_at,
    )
    return response


@router.get("/models", dependencies=[Depends(require_api_key)])
async def models(request: Request) -> Response:
    return await proxy_request(request, "/v1/models")


@router.post("/chat/completions", dependencies=[Depends(require_api_key)])
async def chat_completions(request: Request) -> Response:
    return await proxy_request(request, "/v1/chat/completions")
