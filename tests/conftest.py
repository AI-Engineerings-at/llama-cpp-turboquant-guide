from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_DEFAULTS = {
    "QLLAMA_PROFILE": "baseline",
    "QLLAMA_PROFILES_DIR": "profiles",
    "QLLAMA_MODEL_ROOT": "/models",
    "QLLAMA_LOG_LEVEL": "info",
    "QLLAMA_LOG_FORMAT": "text",
    "QLLAMA_METRICS_ENABLED": "true",
    "QLLAMA_INCLUDE_UPSTREAM_METRICS": "true",
    "QLLAMA_UPSTREAM_HEALTH_CHECK_INTERVAL_SECONDS": "1.0",
    "QLLAMA_DEGRADED_THRESHOLD": "3",
    "QLLAMA_RECOVERY_THRESHOLD": "3",
    "QLLAMA_AUTH_REQUIRED": "false",
}

CLEARED_KEYS = [
    "QLLAMA_API_KEYS",
    "QLLAMA_REQUEST_DURATION_BUCKETS",
    "QLLAMA_UPSTREAM_HEALTH_DURATION_BUCKETS",
    "QLLAMA_CORRELATION_ID_HEADER",
]


@pytest.fixture(autouse=True)
def qllama_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in CLEARED_KEYS:
        monkeypatch.delenv(key, raising=False)

    for key, value in TEST_DEFAULTS.items():
        monkeypatch.setenv(key, value)
