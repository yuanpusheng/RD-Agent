#!/usr/bin/env bash
set -euo pipefail

PORT="${MONITOR_DASHBOARD_PORT:-19555}"
APP_PATH="rdagent_china/dashboard/a_share_monitor/app.py"

APP_ARGS=()
if [[ -n "${MONITOR_LOG_DIR:-}" ]]; then
  APP_ARGS+=("--log-dir" "${MONITOR_LOG_DIR}")
fi
if [[ -n "${MONITOR_SESSION:-}" ]]; then
  APP_ARGS+=("--session" "${MONITOR_SESSION}")
fi

exec streamlit run "${APP_PATH}" \
  --server.address=0.0.0.0 \
  --server.port="${PORT}" \
  --browser.gatherUsageStats=false \
  -- \
  "${APP_ARGS[@]}"
