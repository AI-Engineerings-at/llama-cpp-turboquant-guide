from __future__ import annotations

import secrets

from fastapi import HTTPException, Request, status

from qllama.observability.logging import get_logger
from qllama.observability.metrics import metrics

logger = get_logger(__name__)


def extract_bearer_token(authorization_header: str | None) -> str | None:
    if not authorization_header:
        return None

    scheme, _, token = authorization_header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None

    return token.strip()


async def require_api_key(request: Request) -> None:
    config = request.app.state.config
    auth_is_required = config.auth_required or bool(config.api_keys)
    if not auth_is_required:
        return

    endpoint = getattr(request.state, "normalized_endpoint", request.url.path)

    if not config.api_keys:
        metrics.auth_failures_total.labels(reason="not_configured").inc()
        logger.error("auth_not_configured", endpoint=endpoint)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is required but no API keys are configured",
        )

    token = extract_bearer_token(request.headers.get("Authorization"))
    if token is None:
        metrics.auth_failures_total.labels(reason="missing").inc()
        logger.warning("auth_failed", endpoint=endpoint, reason="missing")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not any(secrets.compare_digest(token, key) for key in config.api_keys):
        metrics.auth_failures_total.labels(reason="invalid").inc()
        logger.warning("auth_failed", endpoint=endpoint, reason="invalid")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid bearer token",
        )
