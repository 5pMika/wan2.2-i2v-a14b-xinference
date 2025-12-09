#!/usr/bin/env bash
set -euo pipefail

XINFERENCE_PORT="${XINFERENCE_PORT:-9997}"
XINFERENCE_HOST="${XINFERENCE_HOST:-0.0.0.0}"
export XINFERENCE_HOME="${XINFERENCE_HOME:-/data}"

mkdir -p "${XINFERENCE_HOME}"

LOG_LEVEL="${XINFERENCE_LOG_LEVEL:-info}"

xinference-local -H "${XINFERENCE_HOST}" -p "${XINFERENCE_PORT}" --log-level "${LOG_LEVEL}" &
SERVER_PID=$!

trap "kill ${SERVER_PID} 2>/dev/null" TERM INT

if [ "${AUTO_LAUNCH_MODEL:-1}" != "0" ]; then
  python /app/bootstrap.py || { kill "${SERVER_PID}" 2>/dev/null; exit 1; }
fi

wait "${SERVER_PID}"

