from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class PreInferenceHookContext:
    endpoint: str
    upstream_path: str
    method: str
    request_id: str | None
    profile: str
    runtime_state: str | None
    headers: dict[str, str]
    body: bytes


@dataclass(slots=True)
class PostInferenceHookContext(PreInferenceHookContext):
    status_code: int
    error_type: str | None
    duration_ms: int


class ZerothHookDispatcher(Protocol):
    async def before_inference(self, context: PreInferenceHookContext) -> None: ...

    async def after_inference(self, context: PostInferenceHookContext) -> None: ...


class NoOpZerothHooks:
    async def before_inference(self, context: PreInferenceHookContext) -> None:
        return None

    async def after_inference(self, context: PostInferenceHookContext) -> None:
        return None
