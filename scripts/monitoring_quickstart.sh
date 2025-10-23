#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${1:-$ROOT_DIR/.env.monitoring}"

if [[ -f "$ENV_FILE" ]]; then
  echo "Loading environment overrides from $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

resolve_path() {
  local value="$1"
  if [[ -z "$value" ]]; then
    return
  fi
  if [[ "$value" == /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$ROOT_DIR/$value"
  fi
}

if ! command -v rdc >/dev/null 2>&1; then
  echo "Installing RD-Agent China with monitoring extras (one-off)..."
  (cd "$ROOT_DIR" && python3 -m pip install --upgrade pip && python3 -m pip install -e ".[monitor]")
fi

CONFIG_DIR="$ROOT_DIR/monitor-config"
SAMPLE_DIR="$ROOT_DIR/examples/ashare_monitoring"
mkdir -p "$CONFIG_DIR"
for sample in monitor_rules.yaml alert_subscriptions.yaml; do
  if [[ -f "$SAMPLE_DIR/$sample" && ! -f "$CONFIG_DIR/$sample" ]]; then
    cp "$SAMPLE_DIR/$sample" "$CONFIG_DIR/$sample"
  fi
done

: "${RDC_MONITOR_RULES_PATH:=$CONFIG_DIR/monitor_rules.yaml}"
: "${RDC_MONITOR_ALERT_SUBSCRIPTIONS_PATH:=$CONFIG_DIR/alert_subscriptions.yaml}"

DATA_PATH="${RDC_DUCKDB_PATH:-rdagent_china/data/market.duckdb}"
CACHE_DIR="${RDC_DATA_CACHE_DIR:-rdagent_china/cache}"
DATA_PATH="$(resolve_path "$DATA_PATH")"
CACHE_DIR="$(resolve_path "$CACHE_DIR")"
export RDC_DUCKDB_PATH="$DATA_PATH"
export RDC_DATA_CACHE_DIR="$CACHE_DIR"
mkdir -p "$(dirname "$DATA_PATH")" "$CACHE_DIR"

export RDC_MONITOR_RULES_PATH="$(resolve_path "$RDC_MONITOR_RULES_PATH")"
export RDC_MONITOR_ALERT_SUBSCRIPTIONS_PATH="$(resolve_path "$RDC_MONITOR_ALERT_SUBSCRIPTIONS_PATH")"

APP_PATH="$ROOT_DIR/rdagent_china/dashboard/a_share_monitor/app.py"
PORT="${MONITOR_DASHBOARD_PORT:-19555}"

MONITOR_ARGS=()
if [[ "${MONITOR_ENABLE_INTRADAY:-false}" == "true" ]]; then
  MONITOR_ARGS+=("--intraday")
fi
if [[ -n "${MONITOR_UNIVERSE:-}" ]]; then
  MONITOR_ARGS+=("--universe" "${MONITOR_UNIVERSE}")
fi
if [[ -n "${MONITOR_WATCHLIST:-}" ]]; then
  MONITOR_ARGS+=("--watchlist" "${MONITOR_WATCHLIST}")
fi
if [[ "${MONITOR_RUN_ONCE:-false}" == "true" ]]; then
  MONITOR_ARGS+=("--run-once")
fi
if [[ -n "${MONITOR_START_DATE:-}" ]]; then
  MONITOR_ARGS+=("--start" "${MONITOR_START_DATE}")
fi
if [[ -n "${MONITOR_END_DATE:-}" ]]; then
  MONITOR_ARGS+=("--end" "${MONITOR_END_DATE}")
fi

cleanup() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    echo
    echo "Stopping monitoring scheduler (pid=${MONITOR_PID})"
    kill "$MONITOR_PID" >/dev/null 2>&1 || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "Starting monitoring scheduler..."
rdc monitor "${MONITOR_ARGS[@]}" &
MONITOR_PID=$!
sleep 2 || true
if ! kill -0 "$MONITOR_PID" >/dev/null 2>&1; then
  echo "Monitor loop exited unexpectedly." >&2
  wait "$MONITOR_PID"
  exit 1
fi

echo "Launching Streamlit dashboard on port $PORT (Ctrl+C to exit)"
DASHBOARD_ARGS=()
if [[ -n "${MONITOR_LOG_DIR:-}" ]]; then
  DASHBOARD_ARGS+=("--log-dir" "${MONITOR_LOG_DIR}")
fi
if [[ -n "${MONITOR_SESSION:-}" ]]; then
  DASHBOARD_ARGS+=("--session" "${MONITOR_SESSION}")
fi

STREAMLIT_CMD=(
  streamlit run "$APP_PATH" \
    --server.address=0.0.0.0 \
    --server.port="$PORT" \
    --browser.gatherUsageStats=false
)
if [[ ${#DASHBOARD_ARGS[@]} -gt 0 ]]; then
  STREAMLIT_CMD+=(--)
  STREAMLIT_CMD+=("${DASHBOARD_ARGS[@]}")
fi

"${STREAMLIT_CMD[@]}"
