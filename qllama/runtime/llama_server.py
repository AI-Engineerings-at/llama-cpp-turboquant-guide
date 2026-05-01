from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from collections import deque
from typing import Any

import httpx

from qllama.config import AppConfig
from qllama.observability.logging import get_logger
from qllama.observability.metrics import metrics
from qllama.observability.state import RuntimeState, RuntimeStateTracker
from qllama.profiles import LlamaServerProfile, load_profile


class RuntimeStartupError(RuntimeError):
    """Raised when qllama cannot establish a verified llama-server runtime."""


class ContextVerificationError(RuntimeStartupError):
    def __init__(
        self,
        message: str,
        *,
        expected_context_length: int,
        actual_context_length: int | None = None,
    ) -> None:
        super().__init__(message)
        self.expected_context_length = expected_context_length
        self.actual_context_length = actual_context_length


class LlamaServerRuntime:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.logger = get_logger(__name__)
        self.profile: LlamaServerProfile | None = None
        self.process: asyncio.subprocess.Process | None = None
        self.upstream_status: int | None = None
        self.verified_cache_capability: str | None = None
        self.verified_context_length: int | None = None
        self.model_path: str | None = None
        self.log_tail: deque[str] = deque(maxlen=50)
        self.state_tracker = RuntimeStateTracker()
        self._log_task: asyncio.Task[None] | None = None
        self._watch_task: asyncio.Task[None] | None = None
        self._stopping = False
        self._consecutive_proxy_failures = 0
        self._consecutive_proxy_successes = 0
        metrics.set_runtime_state(self.state_tracker.state)

    @property
    def phase(self) -> str:
        return self.state_tracker.state.value

    @property
    def detail(self) -> str:
        return self.state_tracker.last_reason

    @property
    def ready(self) -> bool:
        return self.state_tracker.ready

    @property
    def last_error(self) -> str | None:
        return self.state_tracker.last_error

    def snapshot(self) -> dict[str, Any]:
        state = self.state_tracker.snapshot()
        return {
            "phase": state["state"],
            "state": state["state"],
            "state_code": state["state_code"],
            "ready": state["ready"],
            "accepts_traffic": state["accepts_traffic"],
            "detail": state["last_reason"],
            "last_error": state["last_error"],
            "expected_context_length": state["expected_context_length"],
            "actual_context_length": state["actual_context_length"],
            "context_verification_status": state["context_verification_status"],
            "context_verification_error": state["context_verification_error"],
            "selected_profile": self.profile.name
            if self.profile
            else self.config.profile_name,
            "model_path": self.model_path,
            "child_pid": self.process.pid if self.process else None,
            "upstream_status": self.upstream_status,
            "verified_cache_capability": self.verified_cache_capability,
            "verified_context_length": self.verified_context_length,
            "internal_base_url": self.config.internal_base_url,
            "transition_history": state["history"],
            "log_tail": list(self.log_tail),
        }

    async def start(self) -> None:
        self._stopping = False
        self.profile = None
        self.process = None
        self.upstream_status = None
        self.verified_cache_capability = None
        self.verified_context_length = None
        self.model_path = None
        self.log_tail.clear()
        self._consecutive_proxy_failures = 0
        self._consecutive_proxy_successes = 0
        self.state_tracker = RuntimeStateTracker()
        metrics.set_runtime_state(self.state_tracker.state)
        metrics.startup_attempts_total.inc()

        self._transition(RuntimeState.STARTING, "starting qllama runtime")

        try:
            self._transition(
                RuntimeState.VALIDATING_PROFILE,
                f"loading profile '{self.config.profile_name}'",
            )
            self.profile = load_profile(
                self.config.profile_name, self.config.profiles_dir
            )
            expected_context_length = self.profile.verification.expected_context_length
            self.state_tracker.set_context_verification(
                expected_context_length=expected_context_length
            )

            model_path = self.profile.resolved_model_path(self.config.model_root)
            self.model_path = str(model_path)
            self._transition(
                RuntimeState.VALIDATING_MODEL,
                f"validating model artifact at {model_path}",
            )
            if not model_path.exists():
                raise RuntimeStartupError(f"Model artifact not found at {model_path}")

            required_cache = self.profile.verification.required_cache_capability
            if required_cache:
                self._transition(
                    RuntimeState.VALIDATING_BINARY,
                    f"verifying cache capability '{required_cache}'",
                )
                verify_cache_capability(self.config.llama_server_bin, required_cache)
                self.verified_cache_capability = required_cache
            elif self.profile.is_turboquant:
                self.verified_cache_capability = self.profile.cache_type_k

            await self._start_child_process()

        except Exception as exc:
            failure_reason = classify_startup_failure(exc, self.state_tracker.state)
            metrics.startup_failures_total.labels(reason=failure_reason).inc()
            self._transition(
                RuntimeState.FAILED,
                str(exc),
                error=str(exc),
                log_level="error",
                failure_reason=failure_reason,
            )
            await self.stop(keep_failure_state=True)

    async def stop(self, keep_failure_state: bool = False) -> None:
        self._stopping = True

        await self._terminate_child_process()
        await self._cancel_background_tasks()

        self.process = None
        self._watch_task = None
        self._log_task = None
        self.upstream_status = None

        if not keep_failure_state:
            self._transition(RuntimeState.STOPPED, "llama-server child stopped")

    async def restart_child_process(self, reason: str) -> None:
        if self.profile is None:
            raise RuntimeStartupError(
                "Cannot restart llama-server child before profile is loaded"
            )

        metrics.child_process_restarts_total.labels(reason=reason).inc()
        await self._start_child_process(restart_reason=reason)

    async def _start_child_process(self, restart_reason: str | None = None) -> None:
        if self.profile is None:
            raise RuntimeStartupError(
                "Cannot start llama-server child before profile is loaded"
            )

        self._stopping = False
        if self.process is not None:
            self._stopping = True
            await self._cancel_background_tasks()
            await self._terminate_child_process()
            self._stopping = False
        else:
            await self._cancel_background_tasks()
        self.process = None
        self._watch_task = None
        self._log_task = None
        self.upstream_status = None
        self.verified_context_length = None

        start_reason = "starting llama-server child process"
        ready_reason = "selected profile validated and upstream is ready"
        if restart_reason is not None:
            start_reason = (
                f"restarting llama-server child process after {restart_reason}"
            )
            ready_reason = (
                f"llama-server child restarted successfully after {restart_reason}"
            )

        self._transition(RuntimeState.STARTING_CHILD, start_reason)
        command = build_llama_server_command(self.config, self.profile)
        self.logger.info(
            "llama_server_starting",
            profile=self.profile.name,
            command=command,
            model_path=self.model_path,
            restart_reason=restart_reason,
        )
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self._log_task = asyncio.create_task(self._consume_logs())
        self._watch_task = asyncio.create_task(self._watch_process_exit())

        self._transition(
            RuntimeState.WAITING_UPSTREAM,
            "waiting for llama-server health to report ready",
        )
        await self._wait_for_upstream_ready()
        await self._verify_context_contract()
        self._transition(RuntimeState.READY, ready_reason)

    async def _verify_context_contract(self) -> None:
        if self.profile is None:
            return

        expected_context_length = self.profile.verification.expected_context_length
        if expected_context_length is None:
            return

        self.state_tracker.set_context_verification(
            expected_context_length=expected_context_length,
            status="pending",
            error=None,
        )
        try:
            self.verified_context_length = await verify_upstream_context_length(
                self.config.internal_base_url,
                expected_context_length,
            )
        except ContextVerificationError as exc:
            self.state_tracker.set_context_verification(
                expected_context_length=exc.expected_context_length,
                actual_context_length=exc.actual_context_length,
                status="failed",
                error=str(exc),
            )
            raise
        except RuntimeStartupError as exc:
            self.state_tracker.set_context_verification(
                expected_context_length=expected_context_length,
                status="failed",
                error=str(exc),
            )
            raise
        else:
            self.state_tracker.set_context_verification(
                expected_context_length=expected_context_length,
                actual_context_length=self.verified_context_length,
                status="verified",
                error=None,
            )

    async def _terminate_child_process(self) -> None:
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()

    async def _cancel_background_tasks(self) -> None:
        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass

        if self._log_task and not self._log_task.done():
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                pass

    def record_proxy_failure(self, failure_type: str, detail: str) -> None:
        if self.state_tracker.state not in {RuntimeState.READY, RuntimeState.DEGRADED}:
            return

        self._consecutive_proxy_failures += 1
        self._consecutive_proxy_successes = 0

        if (
            self.state_tracker.state == RuntimeState.READY
            and self._consecutive_proxy_failures >= self.config.degraded_threshold
        ):
            self._transition(
                RuntimeState.DEGRADED,
                detail,
                error=detail,
                log_level="warning",
                failure_type=failure_type,
                consecutive_failures=self._consecutive_proxy_failures,
            )
        else:
            self.state_tracker.note(detail, error=detail)
            self.logger.warning(
                "runtime_proxy_failure_observed",
                failure_type=failure_type,
                detail=detail,
                consecutive_failures=self._consecutive_proxy_failures,
                state=self.state_tracker.state.value,
            )

    def record_proxy_success(self) -> None:
        if self.state_tracker.state == RuntimeState.DEGRADED:
            self._consecutive_proxy_successes += 1
            self._consecutive_proxy_failures = 0
            if self._consecutive_proxy_successes >= self.config.recovery_threshold:
                self._transition(
                    RuntimeState.READY,
                    f"recovered after {self._consecutive_proxy_successes} successful upstream requests",
                )
            return

        self._consecutive_proxy_failures = 0
        self._consecutive_proxy_successes = 0

    async def _consume_logs(self) -> None:
        if not self.process or not self.process.stdout:
            return

        while True:
            line = await self.process.stdout.readline()
            if not line:
                break
            self.log_tail.append(line.decode("utf-8", errors="replace").rstrip())

    async def _watch_process_exit(self) -> None:
        if not self.process:
            return

        exit_code = await self.process.wait()
        metrics.child_process_exits_total.labels(exit_code=str(exit_code)).inc()

        if self._stopping:
            self.logger.info("llama_server_stopped", exit_code=exit_code)
            return

        detail = f"llama-server child exited with code {exit_code}"
        self._transition(
            RuntimeState.FAILED,
            detail,
            error=detail,
            log_level="error",
            exit_code=exit_code,
        )

    async def _wait_for_upstream_ready(self) -> None:
        timeout_seconds = (
            self.profile.verification.startup_timeout_seconds
            if self.profile
            else self.config.startup_timeout_seconds
        )
        deadline = time.monotonic() + timeout_seconds
        last_detail = "waiting for upstream readiness"

        async with httpx.AsyncClient(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if self.process and self.process.returncode is not None:
                    raise RuntimeStartupError(
                        f"llama-server exited early with code {self.process.returncode}"
                    )

                started_at = time.perf_counter()
                try:
                    response = await client.get(
                        f"{self.config.internal_base_url}/health"
                    )
                    metrics.upstream_health_check_duration_seconds.observe(
                        time.perf_counter() - started_at
                    )
                    self.upstream_status = response.status_code
                    if response.status_code == 200:
                        return
                    if response.status_code == 503:
                        last_detail = "llama-server still loading model"
                    else:
                        last_detail = (
                            f"unexpected upstream health status {response.status_code}"
                        )
                        metrics.upstream_failures_total.labels(
                            type=classify_upstream_status(response.status_code)
                        ).inc()
                except httpx.TimeoutException as exc:
                    metrics.upstream_health_check_duration_seconds.observe(
                        time.perf_counter() - started_at
                    )
                    metrics.upstream_failures_total.labels(type="timeout").inc()
                    last_detail = f"waiting for upstream health endpoint: {exc.__class__.__name__}"
                except httpx.HTTPError as exc:
                    metrics.upstream_health_check_duration_seconds.observe(
                        time.perf_counter() - started_at
                    )
                    metrics.upstream_failures_total.labels(type="transport").inc()
                    last_detail = f"waiting for upstream health endpoint: {exc.__class__.__name__}"

                self.state_tracker.note(last_detail)
                await asyncio.sleep(self.config.upstream_health_check_interval_seconds)

        raise RuntimeStartupError(
            f"Timed out after {timeout_seconds}s waiting for llama-server readiness: {last_detail}"
        )

    def _transition(
        self,
        state: RuntimeState,
        reason: str,
        *,
        error: str | None = None,
        log_level: str = "info",
        **fields: object,
    ) -> None:
        transition = self.state_tracker.transition(state, reason, error=error)
        metrics.set_runtime_state(state)
        if state == RuntimeState.READY:
            self._consecutive_proxy_failures = 0
            self._consecutive_proxy_successes = 0
        log_method = getattr(self.logger, log_level, self.logger.info)
        log_method(
            "runtime_state_transition",
            from_state=transition.from_state.value,
            to_state=transition.to_state.value,
            reason=reason,
            error=error,
            **fields,
        )


def classify_startup_failure(exc: Exception, state: RuntimeState) -> str:
    message = str(exc)
    if "Unknown qllama profile" in message:
        return "invalid_profile"
    if "Model artifact not found" in message:
        return "missing_model_artifact"
    if state == RuntimeState.VALIDATING_BINARY:
        return "binary_capability_check_failed"
    if "exited early" in message:
        return "child_exited_early"
    if "Timed out after" in message:
        return "upstream_timeout"
    if state == RuntimeState.STARTING_CHILD:
        return "child_start_failed"
    return "startup_error"


def classify_upstream_status(status_code: int) -> str:
    if status_code >= 500:
        return "http_5xx"
    if status_code >= 400:
        return "http_4xx"
    return "http_other"


def verify_cache_capability(llama_server_bin: str, cache_type: str) -> None:
    resolved = shutil.which(llama_server_bin)
    if not resolved:
        raise RuntimeStartupError(
            f"llama-server binary '{llama_server_bin}' was not found in PATH"
        )

    result = subprocess.run(
        [resolved, "-h"],
        check=False,
        capture_output=True,
        text=True,
    )
    help_text = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        message = help_text.strip() or f"exit code {result.returncode}"
        raise RuntimeStartupError(
            f"Failed to inspect llama-server capabilities via '{resolved} -h': {message}"
        )
    if cache_type not in help_text:
        raise RuntimeStartupError(
            f"llama-server binary '{resolved}' does not advertise cache type '{cache_type}'"
        )


async def verify_upstream_context_length(
    internal_base_url: str, expected_context_length: int
) -> int:
    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            response = await client.get(f"{internal_base_url}/props")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeStartupError(
                "Unable to verify upstream context length via "
                f"'{internal_base_url}/props': {exc.__class__.__name__}"
            ) from exc

    actual_context_length = extract_context_length(response.json())
    if actual_context_length is None:
        raise ContextVerificationError(
            f"Upstream props response from '{internal_base_url}/props' did not expose a usable context length",
            expected_context_length=expected_context_length,
        )
    if actual_context_length != expected_context_length:
        raise ContextVerificationError(
            "Profile expected context length "
            f"{expected_context_length}, but upstream reported {actual_context_length}",
            expected_context_length=expected_context_length,
            actual_context_length=actual_context_length,
        )
    return actual_context_length


def extract_context_length(payload: object) -> int | None:
    if isinstance(payload, dict):
        for key in ("n_ctx", "context_length", "ctx_size"):
            value = payload.get(key)
            parsed = _parse_positive_int(value)
            if parsed is not None:
                return parsed
        for value in payload.values():
            parsed = extract_context_length(value)
            if parsed is not None:
                return parsed
        return None

    if isinstance(payload, list):
        for item in payload:
            parsed = extract_context_length(item)
            if parsed is not None:
                return parsed

    return None


def _parse_positive_int(value: object) -> int | None:
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def build_llama_server_command(
    config: AppConfig, profile: LlamaServerProfile
) -> list[str]:
    resolved = shutil.which(config.llama_server_bin) or config.llama_server_bin
    command = [
        resolved,
        "--model",
        profile.model_path,
        "--alias",
        profile.alias,
        "--cache-type-k",
        profile.cache_type_k,
        "--cache-type-v",
        profile.cache_type_v,
        "-c",
        str(profile.context_size),
        "--host",
        config.internal_host,
        "--port",
        str(config.internal_port),
        "-ngl",
        str(profile.gpu_layers),
    ]
    if config.include_upstream_metrics and "--metrics" not in profile.extra_args:
        command.append("--metrics")
    command.extend(profile.extra_args)
    return command
