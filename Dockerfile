# TurboQuant llama.cpp — CUDA Build
# Builds llama-server with TurboQuant KV-cache quantization support
# turbo2 / turbo3 / turbo4 cache types enabled
#
# CRITICAL: Use --branch feature/turboquant-kv-cache (NOT master!)
# The master branch is a standard llama.cpp without TurboQuant support.
#
# Usage:
#   docker build -t turboquant:feature .
#   docker run --rm turboquant:feature llama-server -h 2>&1 | grep -A3 "cache-type-k"
#   # Must show: turbo2, turbo3, turbo4

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

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
# Default 'master' does NOT have turbo2/turbo3/turbo4 cache types!
RUN git clone https://github.com/TheTom/llama-cpp-turboquant.git \
    --branch feature/turboquant-kv-cache \
    --depth=1

WORKDIR /build/llama-cpp-turboquant

# Fix: libcuda.so.1 is not available at build time (driver is injected at runtime only)
# The devel image provides a stub at /usr/local/cuda/lib64/stubs/libcuda.so
# Symlink to .1 so the linker finds it during cmake build
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
    && ldconfig

# IMPORTANT: Use -DGGML_CUDA=ON (not -DLLAMA_CUBLAS=ON which was renamed in ~2024)
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j4 --target llama-server

RUN cp build/bin/llama-server /usr/local/bin/llama-server

WORKDIR /models
EXPOSE 8180

# Default: show help. Override CMD in docker run to actually serve a model.
CMD ["llama-server", "--help"]
