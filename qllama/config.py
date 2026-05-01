from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_REQUEST_DURATION_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
DEFAULT_UPSTREAM_HEALTH_BUCKETS: tuple[float, ...] = (
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1.0,
)


class AppConfig(BaseModel):
    profile_name: str = "baseline"
    profiles_dir: Path = Path("profiles")
    model_root: Path = Path("/models")
    llama_server_bin: str = "llama-server"
    public_host: str = "0.0.0.0"
    public_port: int = 8000
    internal_host: str = "127.0.0.1"
    internal_port: int = 8010
    request_timeout_seconds: float = 600.0
    startup_timeout_seconds: int = 180
    log_level: str = "info"
    log_format: str = "text"
    correlation_id_header: str = "X-Request-ID"
    metrics_enabled: bool = True
    include_upstream_metrics: bool = True
    upstream_metrics_timeout_seconds: float = 2.0
    request_duration_buckets: tuple[float, ...] = DEFAULT_REQUEST_DURATION_BUCKETS
    upstream_health_duration_buckets: tuple[float, ...] = DEFAULT_UPSTREAM_HEALTH_BUCKETS
    upstream_health_check_interval_seconds: float = 1.0
    degraded_threshold: int = 3
    recovery_threshold: int = 3
    auth_required: bool = True
    api_keys: tuple[str, ...] = Field(default_factory=tuple)
    zeroth_hooks_enabled: bool = True

    @property
    def internal_base_url(self) -> str:
        return f"http://{self.internal_host}:{self.internal_port}"

    @classmethod
    def from_env(cls) -> "AppConfig":
        raw_keys = os.getenv("QLLAMA_API_KEYS", "")
        api_keys = tuple(part.strip() for part in raw_keys.split(",") if part.strip())

        return cls(
            profile_name=os.getenv("QLLAMA_PROFILE", "baseline"),
            profiles_dir=Path(os.getenv("QLLAMA_PROFILES_DIR", "profiles")),
            model_root=Path(os.getenv("QLLAMA_MODEL_ROOT", "/models")),
            llama_server_bin=os.getenv("QLLAMA_LLAMA_SERVER_BIN", "llama-server"),
            public_host=os.getenv("QLLAMA_HOST", "0.0.0.0"),
            public_port=int(os.getenv("QLLAMA_PORT", "8000")),
            internal_host=os.getenv("QLLAMA_INTERNAL_HOST", "127.0.0.1"),
            internal_port=int(os.getenv("QLLAMA_INTERNAL_PORT", "8010")),
            request_timeout_seconds=float(os.getenv("QLLAMA_REQUEST_TIMEOUT_SECONDS", "600")),
            startup_timeout_seconds=int(os.getenv("QLLAMA_STARTUP_TIMEOUT_SECONDS", "180")),
            log_level=os.getenv("QLLAMA_LOG_LEVEL", "info"),
            log_format=os.getenv("QLLAMA_LOG_FORMAT", "text"),
            correlation_id_header=os.getenv("QLLAMA_CORRELATION_ID_HEADER", "X-Request-ID"),
            metrics_enabled=parse_bool_env("QLLAMA_METRICS_ENABLED", True),
            include_upstream_metrics=parse_bool_env("QLLAMA_INCLUDE_UPSTREAM_METRICS", True),
            upstream_metrics_timeout_seconds=float(
                os.getenv("QLLAMA_UPSTREAM_METRICS_TIMEOUT_SECONDS", "2.0")
            ),
            request_duration_buckets=parse_float_tuple_env(
                "QLLAMA_REQUEST_DURATION_BUCKETS",
                DEFAULT_REQUEST_DURATION_BUCKETS,
            ),
            upstream_health_duration_buckets=parse_float_tuple_env(
                "QLLAMA_UPSTREAM_HEALTH_DURATION_BUCKETS",
                DEFAULT_UPSTREAM_HEALTH_BUCKETS,
            ),
            upstream_health_check_interval_seconds=float(
                os.getenv("QLLAMA_UPSTREAM_HEALTH_CHECK_INTERVAL_SECONDS", "1.0")
            ),
            degraded_threshold=int(os.getenv("QLLAMA_DEGRADED_THRESHOLD", "3")),
            recovery_threshold=int(os.getenv("QLLAMA_RECOVERY_THRESHOLD", "3")),
            auth_required=parse_bool_env("QLLAMA_AUTH_REQUIRED", True),
            api_keys=api_keys,
            zeroth_hooks_enabled=parse_bool_env("QLLAMA_ZEROTH_HOOKS_ENABLED", True),
        )


def parse_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_float_tuple_env(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default

    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    return values or default


def load_config() -> AppConfig:
    return AppConfig.from_env()
