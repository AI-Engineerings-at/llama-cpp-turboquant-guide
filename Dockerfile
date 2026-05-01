# qllama — TurboQuant-capable llama.cpp runtime with FastAPI control plane
# Builds llama-server from the TurboQuant fork, then packages qllama as the public service.

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# CRITICAL: Must use --branch feature/turboquant-kv-cache
# Default 'master' does NOT have turbo2/turbo3/turbo4 cache types.
RUN git clone https://github.com/TheTom/llama-cpp-turboquant.git \
    --branch feature/turboquant-kv-cache \
    --depth=1

WORKDIR /build/llama-cpp-turboquant

# Fix: libcuda.so.1 is not available at build time (driver is injected at runtime only).
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j4 --target llama-server

FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QLLAMA_HOST=0.0.0.0 \
    QLLAMA_PORT=8000 \
    QLLAMA_INTERNAL_HOST=127.0.0.1 \
    QLLAMA_INTERNAL_PORT=8010 \
    QLLAMA_PROFILE=baseline \
    QLLAMA_PROFILES_DIR=/app/profiles \
    QLLAMA_MODEL_ROOT=/models \
    QLLAMA_LOG_FORMAT=json \
    QLLAMA_METRICS_ENABLED=true \
    QLLAMA_INCLUDE_UPSTREAM_METRICS=true

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/llama-cpp-turboquant/build/bin /opt/llama/bin
RUN ln -sf /opt/llama/bin/llama-server /usr/local/bin/llama-server \
    && echo "/opt/llama/bin" > /etc/ld.so.conf.d/llama-bin.conf \
    && ldconfig
COPY pyproject.toml README.md /app/
COPY qllama /app/qllama
COPY profiles /app/profiles
COPY scripts /app/scripts

RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir \
        fastapi \
        httpx \
        pydantic \
        PyYAML \
        'uvicorn[standard]' \
    && python3 -m pip install --no-cache-dir .

RUN mkdir -p /models

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health >/dev/null || exit 1

CMD ["python3", "-m", "uvicorn", "qllama.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
