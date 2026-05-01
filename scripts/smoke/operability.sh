#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-qllama:dev}"
CONTAINER="${CONTAINER_NAME:-qllama-operability}"
INVALID_CONTAINER="${INVALID_CONTAINER_NAME:-qllama-operability-invalid}"
PORT="${PORT:-8000}"
INVALID_PORT="${INVALID_PORT:-8001}"
PROFILE="${QLLAMA_PROFILE:-baseline}"
VOLUME="${VOLUME_NAME:-turboquant-models}"
API_KEY="${QLLAMA_SMOKE_API_KEY:-qllama-operability-key}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-180}"
STARTUP_TIMEOUT_SECONDS="${QLLAMA_STARTUP_TIMEOUT_SECONDS:-${READY_TIMEOUT_SECONDS}}"
REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS:-1}"
DEGRADED_THRESHOLD="${DEGRADED_THRESHOLD:-1}"
RECOVERY_THRESHOLD="${RECOVERY_THRESHOLD:-1}"
REQUEST_ID="${REQUEST_ID:-operability-check-001}"

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
    status="$(curl -s -o /tmp/qllama-operability-response.json -w '%{http_code}' "$url" || true)"
    if [[ "$status" == "$expected_status" ]]; then
      cat /tmp/qllama-operability-response.json
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

curl_status() {
  local output_file="$1"
  shift
  curl -s -o "$output_file" -w '%{http_code}' "$@"
}

wait_for_authenticated_models() {
  local timeout="$1"
  local started_at
  started_at="$(date +%s)"

  while true; do
    local now
    now="$(date +%s)"
    if (( now - started_at > timeout )); then
      echo "ERROR: timed out waiting for authenticated /v1/models" >&2
      show_container_logs "${CONTAINER}"
      return 1
    fi

    local status
    status="$(curl_status /tmp/qllama-operability-response.json \
      -H "Authorization: Bearer ${API_KEY}" \
      "http://localhost:${PORT}/v1/models" || true)"
    if [[ "$status" == "200" ]]; then
      cat /tmp/qllama-operability-response.json
      return 0
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
  -e QLLAMA_LOG_FORMAT=json \
  "${IMAGE}" >/dev/null

INVALID_HEALTH_PAYLOAD="$(wait_for_http "http://localhost:${INVALID_PORT}/health" 200 30 "${INVALID_CONTAINER}")"
python3 -c 'import json,sys; payload=json.load(sys.stdin); runtime=payload["runtime"]; assert runtime["state"] == "failed", payload; assert "Unknown qllama profile" in runtime["detail"], payload; print("invalid profile health check ok")' <<<"${INVALID_HEALTH_PAYLOAD}"

INVALID_READY_PAYLOAD="$(wait_for_http "http://localhost:${INVALID_PORT}/ready" 503 30 "${INVALID_CONTAINER}")"
python3 -c 'import json,sys; payload=json.load(sys.stdin); assert payload["status"] == "not_ready", payload; print("invalid profile ready gate ok")' <<<"${INVALID_READY_PAYLOAD}"

docker rm -f "${INVALID_CONTAINER}" >/dev/null 2>&1 || true

echo "== Positive path: operability surfaces must be observable =="
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run -d --rm \
  --name "${CONTAINER}" \
  --gpus all \
  -p "${PORT}:8000" \
  -v "${VOLUME}:/models" \
  -e "QLLAMA_PROFILE=${PROFILE}" \
  -e "QLLAMA_API_KEYS=${API_KEY}" \
  -e "QLLAMA_STARTUP_TIMEOUT_SECONDS=${STARTUP_TIMEOUT_SECONDS}" \
  -e QLLAMA_LOG_FORMAT=json \
  -e QLLAMA_DEGRADED_THRESHOLD="${DEGRADED_THRESHOLD}" \
  -e QLLAMA_RECOVERY_THRESHOLD="${RECOVERY_THRESHOLD}" \
  -e QLLAMA_REQUEST_TIMEOUT_SECONDS="${REQUEST_TIMEOUT_SECONDS}" \
  "${IMAGE}" >/dev/null

wait_for_http "http://localhost:${PORT}/health" 200 30 "${CONTAINER}" >/dev/null
READY_PAYLOAD="$(wait_for_http "http://localhost:${PORT}/ready" 200 "${READY_TIMEOUT_SECONDS}" "${CONTAINER}")"
CHILD_PID="$(python3 -c 'import json,sys; payload=json.load(sys.stdin); runtime=payload["runtime"]; assert runtime["state"] == "ready", payload; print(runtime["child_pid"])' <<<"${READY_PAYLOAD}")"

echo "== Metrics scrape must expose qllama metrics =="
METRICS_PAYLOAD="$(curl -sf "http://localhost:${PORT}/metrics")"
python3 -c 'import sys; text=sys.stdin.read(); assert "qllama_ready_status" in text, text; assert "qllama_runtime_state" in text, text; assert "qllama_http_requests_total" in text, text; print("wrapper metrics ok")' <<<"${METRICS_PAYLOAD}"
if grep -q 'llamacpp:' <<<"${METRICS_PAYLOAD}"; then
  echo "upstream metrics appended"
else
  echo "upstream metrics unavailable; wrapper metrics still served"
fi

echo "== Missing auth must be visible =="
UNAUTH_STATUS="$(curl_status /tmp/qllama-operability-response.json "http://localhost:${PORT}/v1/models")"
[[ "${UNAUTH_STATUS}" == "401" ]] || {
  echo "ERROR: expected unauthenticated /v1/models to return 401, got ${UNAUTH_STATUS}" >&2
  show_container_logs "${CONTAINER}"
  exit 1
}
AUTH_METRICS_PAYLOAD="$(curl -sf "http://localhost:${PORT}/metrics")"
python3 -c 'import sys; text=sys.stdin.read(); assert "qllama_auth_failures_total{reason=\"missing\"}" in text, text; print("auth failure metric ok")' <<<"${AUTH_METRICS_PAYLOAD}"

echo "== Structured logs must carry the request ID =="
REQ_STATUS="$(curl -s -D /tmp/qllama-operability-headers.txt -o /tmp/qllama-operability-response.json -w '%{http_code}' \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "X-Request-ID: ${REQUEST_ID}" \
  "http://localhost:${PORT}/v1/models")"
[[ "${REQ_STATUS}" == "200" ]] || {
  echo "ERROR: expected authenticated /v1/models to return 200, got ${REQ_STATUS}" >&2
  show_container_logs "${CONTAINER}"
  exit 1
}
HEADERS_CONTENT="$(tr '[:upper:]' '[:lower:]' </tmp/qllama-operability-headers.txt)"
if [[ "${HEADERS_CONTENT}" == *"x-request-id: ${REQUEST_ID}"* ]]; then
  echo "request id header ok"
else
  echo "ERROR: request id header missing from response" >&2
  cat /tmp/qllama-operability-headers.txt >&2 || true
  show_container_logs "${CONTAINER}"
  exit 1
fi
docker logs "${CONTAINER}" 2>&1 | python3 -c 'import json, sys; request_id="'"${REQUEST_ID}"'"; matches=[]; lines=sys.stdin.read().splitlines()
for line in lines:
    try:
        payload=json.loads(line)
    except json.JSONDecodeError:
        continue
    if payload.get("request_id") == request_id and payload.get("event") in {"request_started", "request_completed"}:
        matches.append(payload)
assert matches, lines[-20:]
assert all("timestamp" in item and "level" in item for item in matches), matches
print("structured request logs ok")'

echo "== Degraded transition and recovery must be visible =="
docker exec "${CONTAINER}" sh -lc "kill -STOP ${CHILD_PID}"
DEGRADED_STATUS="$(curl_status /tmp/qllama-operability-response.json \
  -H "Authorization: Bearer ${API_KEY}" \
  "http://localhost:${PORT}/v1/models" || true)"
if [[ "${DEGRADED_STATUS}" != "502" ]]; then
  echo "ERROR: expected degraded probe request to return 502, got ${DEGRADED_STATUS}" >&2
  show_container_logs "${CONTAINER}"
  exit 1
fi
wait_for_http "http://localhost:${PORT}/health" 200 20 "${CONTAINER}" >/dev/null
DEGRADED_READY_PAYLOAD="$(wait_for_http "http://localhost:${PORT}/ready" 503 20 "${CONTAINER}")"
python3 -c 'import json,sys; payload=json.load(sys.stdin); runtime=payload["runtime"]; assert runtime["state"] == "degraded", payload; print("degraded ready gate ok")' <<<"${DEGRADED_READY_PAYLOAD}"
DEGRADED_METRICS_PAYLOAD="$(curl -sf "http://localhost:${PORT}/metrics")"
python3 -c 'import sys; text=sys.stdin.read(); assert "qllama_runtime_state{state_name=\"degraded\"} 1.0" in text, text; assert ("qllama_upstream_failures_total{type=\"timeout\"}" in text) or ("qllama_upstream_failures_total{type=\"transport\"}" in text), text; print("degraded metrics ok")' <<<"${DEGRADED_METRICS_PAYLOAD}"

docker exec "${CONTAINER}" sh -lc "kill -CONT ${CHILD_PID}"
wait_for_authenticated_models 30 >/dev/null
RECOVERED_READY_PAYLOAD="$(wait_for_http "http://localhost:${PORT}/ready" 200 30 "${CONTAINER}")"
python3 -c 'import json,sys; payload=json.load(sys.stdin); runtime=payload["runtime"]; assert runtime["state"] == "ready", payload; print("recovery ready gate ok")' <<<"${RECOVERED_READY_PAYLOAD}"
RECOVERED_METRICS_PAYLOAD="$(curl -sf "http://localhost:${PORT}/metrics")"
python3 -c 'import sys; text=sys.stdin.read(); assert "qllama_runtime_state{state_name=\"ready\"} 1.0" in text, text; print("recovery metrics ok")' <<<"${RECOVERED_METRICS_PAYLOAD}"

echo "Operability verification passed for image ${IMAGE}"