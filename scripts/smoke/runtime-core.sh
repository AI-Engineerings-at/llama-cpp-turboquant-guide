#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-qllama:dev}"
CONTAINER="${CONTAINER_NAME:-qllama-smoke}"
INVALID_CONTAINER="${INVALID_CONTAINER_NAME:-qllama-smoke-invalid}"
PORT="${PORT:-8000}"
INVALID_PORT="${INVALID_PORT:-8001}"
PROFILE="${QLLAMA_PROFILE:-baseline}"
VOLUME="${VOLUME_NAME:-turboquant-models}"
API_KEY="${QLLAMA_SMOKE_API_KEY:-qllama-smoke-key}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-180}"
STARTUP_TIMEOUT_SECONDS="${QLLAMA_STARTUP_TIMEOUT_SECONDS:-${READY_TIMEOUT_SECONDS}}"

cleanup() {
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  docker rm -f "${INVALID_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

show_container_logs() {
  local container_name="$1"
  echo "---- logs: ${container_name} ----" >&2
  docker logs "${container_name}" >&2 || true
  echo "---- end logs: ${container_name} ----" >&2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command '$1' not found" >&2
    exit 1
  }
}

wait_for_http() {
  local url="$1"
  local expected_status="$2"
  local timeout="$3"
  local container_name="${4:-}"
  local started_at
  started_at="$(date +%s)"

  while true; do
    local now
    now="$(date +%s)"
    if (( now - started_at > timeout )); then
      echo "ERROR: timed out waiting for ${url} -> ${expected_status}" >&2
      if [[ -n "${container_name}" ]]; then
        show_container_logs "${container_name}"
      fi
      return 1
    fi

    local status
    status="$(curl -s -o /tmp/qllama-smoke-response.json -w '%{http_code}' "$url" || true)"
    if [[ "$status" == "$expected_status" ]]; then
      cat /tmp/qllama-smoke-response.json
      return 0
    fi

    if [[ -n "${container_name}" ]] && ! docker ps --format '{{.Names}}' | grep -Fxq "${container_name}"; then
      echo "ERROR: container ${container_name} exited before ${url} reached ${expected_status}" >&2
      show_container_logs "${container_name}"
      return 1
    fi

    sleep 2
  done
}

require_command docker
require_command curl
require_command python3

docker info >/dev/null 2>&1 || {
  echo "ERROR: Docker daemon is not running." >&2
  exit 1
}

echo "== Negative path: invalid profile must stay fail-closed =="
docker rm -f "${INVALID_CONTAINER}" >/dev/null 2>&1 || true
docker run -d --rm \
  --name "${INVALID_CONTAINER}" \
  -p "${INVALID_PORT}:8000" \
  -e QLLAMA_PROFILE=invalid-profile \
  "${IMAGE}" >/dev/null

INVALID_HEALTH_PAYLOAD="$(wait_for_http "http://localhost:${INVALID_PORT}/health" 200 30 "${INVALID_CONTAINER}")"
python3 -c 'import json,sys; payload=json.load(sys.stdin); runtime=payload["runtime"]; assert runtime["phase"] == "failed", payload; assert "Unknown qllama profile" in runtime["detail"], payload; print("invalid profile health check ok")' <<<"${INVALID_HEALTH_PAYLOAD}"

INVALID_READY_PAYLOAD="$(wait_for_http "http://localhost:${INVALID_PORT}/ready" 503 30 "${INVALID_CONTAINER}")"
python3 -c 'import json,sys; payload=json.load(sys.stdin); assert payload["status"] == "not_ready", payload; print("invalid profile ready gate ok")' <<<"${INVALID_READY_PAYLOAD}"

docker rm -f "${INVALID_CONTAINER}" >/dev/null 2>&1 || true

echo "== Positive path: selected profile must become ready =="
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true

RUN_ARGS=(
  run -d --rm
  --name "${CONTAINER}"
  --gpus all
  -p "${PORT}:8000"
  -v "${VOLUME}:/models"
  -e "QLLAMA_PROFILE=${PROFILE}"
  -e "QLLAMA_API_KEYS=${API_KEY}"
)

RUN_ARGS+=( "${IMAGE}" )
docker "${RUN_ARGS[@]}" >/dev/null

wait_for_http "http://localhost:${PORT}/health" 200 30 "${CONTAINER}" >/dev/null
READY_PAYLOAD="$(wait_for_http "http://localhost:${PORT}/ready" 200 "${READY_TIMEOUT_SECONDS}" "${CONTAINER}")"
python3 -c 'import json,sys; payload=json.load(sys.stdin); runtime=payload["runtime"]; assert runtime["phase"] == "ready", payload; print("ready gate ok")' <<<"${READY_PAYLOAD}"

AUTH_ARGS=(-H "Authorization: Bearer ${API_KEY}")

curl -sf "${AUTH_ARGS[@]}" "http://localhost:${PORT}/v1/models" | python3 -c 'import json,sys; payload=json.load(sys.stdin); assert payload["data"], payload; print("models endpoint ok")'

curl -sf "${AUTH_ARGS[@]}" \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}' \
  "http://localhost:${PORT}/v1/chat/completions" | python3 -c 'import json,sys; payload=json.load(sys.stdin); assert payload["choices"], payload; print("chat completions endpoint ok")'

echo "Smoke verification passed for image ${IMAGE}"
