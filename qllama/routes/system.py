from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from qllama.observability.logging import get_logger
from qllama.observability.metrics import combine_metrics_payload, metrics

logger = get_logger(__name__)
router = APIRouter(tags=["system"])


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


def _system_payload(request: Request) -> dict[str, object]:
    runtime = request.app.state.runtime
    return {
        "service": "qllama",
        "runtime": _runtime_snapshot(runtime),
    }


@router.get("/system")
async def system_status(request: Request) -> JSONResponse:
    return JSONResponse(status_code=200, content=_system_payload(request))


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    payload = {
        "status": "ok",
        **_system_payload(request),
    }
    return JSONResponse(status_code=200, content=payload)


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    runtime = request.app.state.runtime
    payload = {
        "status": "ok" if runtime.ready else "not_ready",
        "service": "qllama",
        "runtime": runtime.snapshot(),
    }
    status_code = 200 if runtime.ready else 503
    return JSONResponse(status_code=status_code, content=payload)


@router.get("/metrics")
async def metrics_endpoint(request: Request) -> Response:
    config = request.app.state.config
    if not config.metrics_enabled:
        raise HTTPException(status_code=404, detail="Metrics disabled")

    wrapper_payload = metrics.render_latest()
    upstream_payload: str | None = None

    if config.include_upstream_metrics:
        try:
            async with httpx.AsyncClient(
                timeout=config.upstream_metrics_timeout_seconds
            ) as client:
                response = await client.get(f"{config.internal_base_url}/metrics")
            if response.status_code == 200:
                upstream_payload = response.text
            else:
                logger.warning(
                    "upstream_metrics_unavailable",
                    status_code=response.status_code,
                )
        except httpx.HTTPError as exc:
            logger.warning(
                "upstream_metrics_unavailable",
                error=str(exc),
                error_type=exc.__class__.__name__,
            )

    payload = combine_metrics_payload(wrapper_payload, upstream_payload)
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
