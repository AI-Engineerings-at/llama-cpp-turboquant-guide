from __future__ import annotations

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from qllama.config import (
    DEFAULT_REQUEST_DURATION_BUCKETS,
    DEFAULT_UPSTREAM_HEALTH_BUCKETS,
)
from qllama.observability.state import RUNTIME_STATES, RuntimeState

KNOWN_ENDPOINTS = {
    "/",
    "/health",
    "/ready",
    "/metrics",
    "/v1/models",
    "/v1/chat/completions",
}


class MetricsCatalog:
    def __init__(
        self,
        *,
        registry: CollectorRegistry | None = None,
        request_duration_buckets: tuple[float, ...] = DEFAULT_REQUEST_DURATION_BUCKETS,
        upstream_health_buckets: tuple[float, ...] = DEFAULT_UPSTREAM_HEALTH_BUCKETS,
    ) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)

        self.http_requests_total = Counter(
            "qllama_http_requests_total",
            "Total HTTP requests to qllama.",
            ["method", "endpoint", "status_code"],
            registry=self.registry,
        )
        self.http_request_duration_seconds = Histogram(
            "qllama_http_request_duration_seconds",
            "HTTP request latency in seconds.",
            ["method", "endpoint", "status_code"],
            buckets=request_duration_buckets,
            registry=self.registry,
        )
        self.http_requests_in_flight = Gauge(
            "qllama_http_requests_in_flight",
            "Current in-flight HTTP requests.",
            ["method"],
            registry=self.registry,
        )
        self.auth_failures_total = Counter(
            "qllama_auth_failures_total",
            "Total authentication failures.",
            ["reason"],
            registry=self.registry,
        )
        self.upstream_failures_total = Counter(
            "qllama_upstream_failures_total",
            "Total failures returned by or reaching llama-server.",
            ["type"],
            registry=self.registry,
        )
        self.upstream_health_check_duration_seconds = Histogram(
            "qllama_upstream_health_check_duration_seconds",
            "Upstream llama-server health-check latency in seconds.",
            buckets=upstream_health_buckets,
            registry=self.registry,
        )
        self.startup_attempts_total = Counter(
            "qllama_startup_attempts_total",
            "Total qllama runtime startup attempts.",
            registry=self.registry,
        )
        self.startup_failures_total = Counter(
            "qllama_startup_failures_total",
            "Total qllama runtime startup failures.",
            ["reason"],
            registry=self.registry,
        )
        self.child_process_exits_total = Counter(
            "qllama_child_process_exits_total",
            "Total llama-server child process exits.",
            ["exit_code"],
            registry=self.registry,
        )
        self.child_process_restarts_total = Counter(
            "qllama_child_process_restarts_total",
            "Total llama-server child process restarts.",
            ["reason"],
            registry=self.registry,
        )
        self.backend_requests_total = Counter(
            "qllama_backend_requests_total",
            "Total qllama upstream backend attempts by backend and result.",
            ["backend", "result"],
            registry=self.registry,
        )
        self.fallback_activations_total = Counter(
            "qllama_fallback_activations_total",
            "Total qllama fallback activations between upstream backends.",
            ["reason", "from_backend", "to_backend"],
            registry=self.registry,
        )
        self.selected_backend = Gauge(
            "qllama_selected_backend",
            "Current selected upstream backend as a one-hot gauge.",
            ["backend"],
            registry=self.registry,
        )
        self.fallback_depth = Gauge(
            "qllama_fallback_depth",
            "Current fallback depth for the selected upstream backend.",
            registry=self.registry,
        )
        self._known_backends: set[str] = set()
        self.runtime_state = Gauge(
            "qllama_runtime_state",
            "Current qllama runtime state as a one-hot gauge.",
            ["state_name"],
            registry=self.registry,
        )
        self.ready_status = Gauge(
            "qllama_ready_status",
            "qllama ready status (1=ready, 0=not ready).",
            registry=self.registry,
        )

        for state in RUNTIME_STATES:
            self.runtime_state.labels(state_name=state.value).set(0)
        self.ready_status.set(0)
        self.fallback_depth.set(0)

    def metric_names(self) -> set[str]:
        return {
            sample.name
            for metric in self.registry.collect()
            for sample in metric.samples
        }

    def render_latest(self) -> bytes:
        return generate_latest(self.registry)

    def set_runtime_state(self, state: RuntimeState) -> None:
        for known_state in RUNTIME_STATES:
            value = 1 if known_state == state else 0
            self.runtime_state.labels(state_name=known_state.value).set(value)
        self.ready_status.set(1 if state == RuntimeState.READY else 0)

    def set_selected_backend(self, backend: str | None, fallback_depth: int) -> None:
        if backend is not None:
            self._known_backends.add(backend)
        for known_backend in self._known_backends:
            value = 1 if backend is not None and known_backend == backend else 0
            self.selected_backend.labels(backend=known_backend).set(value)
        self.fallback_depth.set(fallback_depth)

    def record_backend_attempt(self, backend: str, result: str) -> None:
        self.backend_requests_total.labels(backend=backend, result=result).inc()

    def record_fallback_activation(
        self,
        *,
        reason: str,
        from_backend: str,
        to_backend: str,
    ) -> None:
        self.fallback_activations_total.labels(
            reason=reason,
            from_backend=from_backend,
            to_backend=to_backend,
        ).inc()


metrics = MetricsCatalog()


def normalize_endpoint(path: str) -> str:
    canonical = path.split("?", 1)[0]
    if canonical in KNOWN_ENDPOINTS:
        return canonical
    return "other"


def combine_metrics_payload(
    wrapper_payload: bytes,
    upstream_payload: bytes | str | None,
) -> bytes:
    parts: list[bytes] = [wrapper_payload.rstrip()]
    if upstream_payload:
        if isinstance(upstream_payload, str):
            upstream_payload = upstream_payload.encode("utf-8")
        parts.append(upstream_payload.strip())
    return b"\n".join(part for part in parts if part) + b"\n"
