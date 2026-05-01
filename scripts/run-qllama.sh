#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-${QLLAMA_PROFILE:-baseline}}"
HOST="${QLLAMA_HOST:-0.0.0.0}"
PORT="${QLLAMA_PORT:-8000}"
PROFILES_DIR="${QLLAMA_PROFILES_DIR:-profiles}"

export QLLAMA_PROFILE="${PROFILE}"
export QLLAMA_HOST="${HOST}"
export QLLAMA_PORT="${PORT}"
export QLLAMA_PROFILES_DIR="${PROFILES_DIR}"

python -m uvicorn qllama.main:create_app --factory --host "${HOST}" --port "${PORT}"
