#!/usr/bin/env bash
# TurboQuant turbo3 — 3-bit KV-Cache, context=100000
# 12× more context than baseline, +1.8 GB VRAM only
#
# Usage: bash scripts/run-turbo.sh [model-path] [port]
# Default model: /models/mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf
# Default port: 8182
#
# NOTE: Port 8180 is used by the baseline run. Use a different port here.

MODEL="${1:-/models/mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf}"
PORT="${2:-8182}"
VOLUME="${VOLUME_NAME:-turboquant-models}"
IMAGE="${IMAGE:-turboquant:feature}"

echo "=== TurboQuant turbo3 Run ==="
echo "Model: $MODEL"
echo "Cache: turbo3 (3-bit KV quantization)"
echo "Context: 100,000 tokens"
echo "Port: $PORT"
echo ""
echo "Expected VRAM: ~17.2 GB (+1.8 GB vs baseline)"
echo "Expected TPS: ~45 (-8.5% vs baseline)"
echo ""

# Stop any existing turbo container
docker rm -f turboquant-turbo3 2>/dev/null || true

docker run --rm --gpus all \
  -v "${VOLUME}:/models" \
  -p "${PORT}:8182" \
  --name turboquant-turbo3 \
  "${IMAGE}" \
  llama-server \
    --model "${MODEL}" \
    --cache-type-k turbo3 \
    --cache-type-v turbo3 \
    -c 100000 \
    --host 0.0.0.0 \
    --port 8182 \
    -ngl 99

echo ""
echo "TurboQuant serving at: http://localhost:${PORT}"
echo "OpenAI-compatible:     http://localhost:${PORT}/v1/chat/completions"
echo ""
echo "After startup (~90s, 100K context allocation takes longer):"
echo "  VRAM: nvidia-smi --query-gpu=memory.used --format=csv,noheader"
