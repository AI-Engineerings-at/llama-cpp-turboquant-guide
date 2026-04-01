#!/usr/bin/env bash
# TurboQuant Baseline — f16 KV-Cache, context=8192
# Reference measurement for comparison with TurboQuant run
#
# Usage: bash scripts/run-baseline.sh [model-path] [port]
# Default model: /models/mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf
# Default port: 8180

MODEL="${1:-/models/mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf}"
PORT="${2:-8180}"
VOLUME="${VOLUME_NAME:-turboquant-models}"
IMAGE="${IMAGE:-turboquant:feature}"

echo "=== TurboQuant Baseline Run ==="
echo "Model: $MODEL"
echo "Cache: f16 (full precision)"
echo "Context: 8192 tokens"
echo "Port: $PORT"
echo ""

# Stop any existing baseline container
docker rm -f turboquant-baseline 2>/dev/null || true

docker run --rm --gpus all \
  -v "${VOLUME}:/models" \
  -p "${PORT}:8180" \
  --name turboquant-baseline \
  "${IMAGE}" \
  llama-server \
    --model "${MODEL}" \
    --cache-type-k f16 \
    --cache-type-v f16 \
    -c 8192 \
    --host 0.0.0.0 \
    --port 8180 \
    -ngl 99

echo ""
echo "Baseline serving at: http://localhost:${PORT}"
echo "OpenAI-compatible:   http://localhost:${PORT}/v1/chat/completions"
echo ""
echo "After startup (~45s), measure VRAM: nvidia-smi --query-gpu=memory.used --format=csv,noheader"
