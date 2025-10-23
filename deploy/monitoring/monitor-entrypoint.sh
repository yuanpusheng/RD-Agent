#!/usr/bin/env bash
set -euo pipefail

ARGS=()

if [[ "${MONITOR_ENABLE_INTRADAY:-false}" == "true" ]]; then
  ARGS+=("--intraday")
fi

if [[ -n "${MONITOR_UNIVERSE:-}" ]]; then
  ARGS+=("--universe" "${MONITOR_UNIVERSE}")
fi

if [[ -n "${MONITOR_WATCHLIST:-}" ]]; then
  ARGS+=("--watchlist" "${MONITOR_WATCHLIST}")
fi

if [[ "${MONITOR_RUN_ONCE:-false}" == "true" ]]; then
  ARGS+=("--run-once")
fi

if [[ -n "${MONITOR_START_DATE:-}" ]]; then
  ARGS+=("--start" "${MONITOR_START_DATE}")
fi

if [[ -n "${MONITOR_END_DATE:-}" ]]; then
  ARGS+=("--end" "${MONITOR_END_DATE}")
fi

exec rdc monitor "${ARGS[@]}"
