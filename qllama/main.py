from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI

from qllama.config import load_config
from qllama.middleware import HTTPObservabilityMiddleware
from qllama.observability.logging import configure_logging
from qllama.profiles import list_profiles
from qllama.routes.openai_proxy import router as openai_proxy_router
from qllama.routes.system import router as system_router
from qllama.runtime.llama_server import LlamaServerRuntime
from qllama.zeroth_hooks import NoOpZerothHooks


def create_app() -> FastAPI:
    config = load_config()
    configure_logging(level=config.log_level, fmt=config.log_format)
    runtime = LlamaServerRuntime(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        startup_task = asyncio.create_task(runtime.start())
        try:
            yield
        finally:
            if not startup_task.done():
                startup_task.cancel()
                with suppress(asyncio.CancelledError):
                    await startup_task
            await runtime.stop()

    app = FastAPI(title="qllama", version="0.1.0", lifespan=lifespan)
    app.state.config = config
    app.state.runtime = runtime
    app.state.zeroth_hooks = NoOpZerothHooks() if config.zeroth_hooks_enabled else None
    app.add_middleware(HTTPObservabilityMiddleware)
    app.add_middleware(
        CorrelationIdMiddleware,
        header_name=config.correlation_id_header,
        validator=None,
    )
    app.include_router(system_router)
    app.include_router(openai_proxy_router)

    @app.get("/")
    async def root() -> dict[str, object]:
        return {
            "service": "qllama",
            "profile": config.profile_name,
            "profiles": list_profiles(config.profiles_dir),
            "runtime": runtime.snapshot(),
        }

    return app
