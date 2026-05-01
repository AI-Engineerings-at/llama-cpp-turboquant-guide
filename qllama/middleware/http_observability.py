from __future__ import annotations

import time

from asgi_correlation_id import correlation_id
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

import qllama.observability.logging as logging_module
from qllama.observability.metrics import metrics, normalize_endpoint


class HTTPObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        logger = logging_module.get_logger(__name__)
        endpoint = normalize_endpoint(request.url.path)
        request_id = correlation_id.get() or request.headers.get(
            request.app.state.config.correlation_id_header
        )

        request.state.normalized_endpoint = endpoint
        request.state.request_id = request_id

        logging_module.clear_log_context()
        logging_module.bind_log_context(
            request_id=request_id,
            method=request.method,
            endpoint=endpoint,
        )

        if request.app.state.config.metrics_enabled:
            metrics.http_requests_in_flight.labels(method=request.method).inc()

        logger.info("request_started")
        started_at = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            duration = time.perf_counter() - started_at
            if request.app.state.config.metrics_enabled:
                metrics.http_requests_total.labels(
                    method=request.method,
                    endpoint=endpoint,
                    status_code="500",
                ).inc()
                metrics.http_request_duration_seconds.labels(
                    method=request.method,
                    endpoint=endpoint,
                    status_code="500",
                ).observe(duration)
                metrics.http_requests_in_flight.labels(method=request.method).dec()
            logger.exception("request_failed", duration_ms=round(duration * 1000, 3))
            logging_module.clear_log_context()
            raise

        duration = time.perf_counter() - started_at
        response.headers.setdefault("X-Process-Time", f"{duration:.6f}")

        if request.app.state.config.metrics_enabled:
            metrics.http_requests_total.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(response.status_code),
            ).inc()
            metrics.http_request_duration_seconds.labels(
                method=request.method,
                endpoint=endpoint,
                status_code=str(response.status_code),
            ).observe(duration)
            metrics.http_requests_in_flight.labels(method=request.method).dec()

        logger.info(
            "request_completed",
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 3),
        )
        logging_module.clear_log_context()
        return response
