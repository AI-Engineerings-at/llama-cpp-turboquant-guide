from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RuntimeState(str, Enum):
    INITIAL = "initial"
    STARTING = "starting"
    VALIDATING_PROFILE = "validating_profile"
    VALIDATING_MODEL = "validating_model"
    VALIDATING_BINARY = "validating_binary"
    STARTING_CHILD = "starting_child"
    WAITING_UPSTREAM = "waiting_upstream"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPED = "stopped"


RUNTIME_STATES: tuple[RuntimeState, ...] = tuple(RuntimeState)
RUNTIME_STATE_CODES: dict[RuntimeState, int] = {
    state: index for index, state in enumerate(RUNTIME_STATES)
}


@dataclass(frozen=True)
class StateTransition:
    from_state: RuntimeState
    to_state: RuntimeState
    reason: str
    timestamp: float
    error: str | None = None


class RuntimeStateTracker:
    def __init__(self, *, history_limit: int = 50) -> None:
        self._state = RuntimeState.INITIAL
        self._history: deque[StateTransition] = deque(maxlen=history_limit)
        self.last_reason = "startup not attempted"
        self.last_error: str | None = None
        self.expected_context_length: int | None = None
        self.actual_context_length: int | None = None
        self.context_verification_status = "not_requested"
        self.context_verification_error: str | None = None
        self.selected_backend: str | None = None
        self.fallback_depth = 0
        self.fallback_reason: str | None = None
        self.last_backend_error: str | None = None

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state == RuntimeState.READY

    @property
    def accepts_traffic(self) -> bool:
        return self._state == RuntimeState.READY

    def note(self, reason: str, *, error: str | None = None) -> None:
        self.last_reason = reason
        if error is not None:
            self.last_error = error

    def set_context_verification(
        self,
        *,
        expected_context_length: int | None = None,
        actual_context_length: int | None = None,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        if expected_context_length is not None:
            self.expected_context_length = expected_context_length
            if status is None and self.context_verification_status == "not_requested":
                self.context_verification_status = "pending"

        if actual_context_length is not None:
            self.actual_context_length = actual_context_length

        if status is not None:
            self.context_verification_status = status

        self.context_verification_error = error

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

    def transition(
        self,
        to_state: RuntimeState,
        reason: str,
        *,
        error: str | None = None,
    ) -> StateTransition:
        transition = StateTransition(
            from_state=self._state,
            to_state=to_state,
            reason=reason,
            timestamp=time.time(),
            error=error,
        )
        self._history.append(transition)
        self._state = to_state
        self.last_reason = reason
        self.last_error = error
        return transition

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "state_code": RUNTIME_STATE_CODES[self._state],
            "ready": self.ready,
            "accepts_traffic": self.accepts_traffic,
            "last_reason": self.last_reason,
            "last_error": self.last_error,
            "expected_context_length": self.expected_context_length,
            "actual_context_length": self.actual_context_length,
            "context_verification_status": self.context_verification_status,
            "context_verification_error": self.context_verification_error,
            "selected_backend": self.selected_backend,
            "fallback_depth": self.fallback_depth,
            "fallback_reason": self.fallback_reason,
            "last_backend_error": self.last_backend_error,
            "history": [
                {
                    "from_state": transition.from_state.value,
                    "to_state": transition.to_state.value,
                    "reason": transition.reason,
                    "timestamp": transition.timestamp,
                    "error": transition.error,
                }
                for transition in self._history
            ],
        }
