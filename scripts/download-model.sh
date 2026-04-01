#!/usr/bin/env bash
# Download Mistral-Small-3.2-24B-Instruct Q4_K_M GGUF model
#
# Usage: export HF_TOKEN=hf_... && bash scripts/download-model.sh
#
# Always verify the repo name via HF Search API before downloading.
# HF repo names change and compressed context can reconstruct them incorrectly.

set -e

MODEL_REPO="bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF"
MODEL_FILE="mistralai_Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"
VOLUME_NAME="${VOLUME_NAME:-turboquant-models}"

if [ -z "$HF_TOKEN" ]; then
  echo "ERROR: HF_TOKEN not set."
  echo "Get a free token at: https://huggingface.co/settings/tokens"
  echo "Then: export HF_TOKEN=hf_..."
  exit 1
fi

echo "=== Verifying repo exists ==="
HF_CHECK=$(curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/models/${MODEL_REPO}")

if [ "$HF_CHECK" != "200" ]; then
  echo "ERROR: Repo not found or unauthorized (HTTP $HF_CHECK)"
  echo "Search for available repos:"
  curl -s -H "Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/api/models?search=bartowski+mistral+small+3.2&limit=5" \
    | python3 -c "import sys,json; [print(m['modelId']) for m in json.load(sys.stdin)]"
  exit 1
fi

echo "Repo: ${MODEL_REPO} ✓"
echo "File: ${MODEL_FILE}"
echo "Volume: ${VOLUME_NAME}"
echo ""

docker volume create ${VOLUME_NAME} 2>/dev/null || true

echo "=== Downloading (~14 GB, may take 20-30 min) ==="
docker run --rm \
  -v ${VOLUME_NAME}:/models \
  -e HF_TOKEN="${HF_TOKEN}" \
  python:3.11-slim \
  bash -c "
    pip install -q huggingface_hub && \
    python -c \"
import os
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='${MODEL_REPO}',
    filename='${MODEL_FILE}',
    local_dir='/models',
    resume_download=True,
    token=os.environ.get('HF_TOKEN')
)
print('Downloaded to:', path)
print('Size: {:.1f} GB'.format(os.path.getsize(path) / 1e9))
\"
  "

echo ""
echo "=== Done ==="
echo "Model ready at: /models/${MODEL_FILE} (in Docker volume '${VOLUME_NAME}')"
echo "Run: bash scripts/run-baseline.sh"
