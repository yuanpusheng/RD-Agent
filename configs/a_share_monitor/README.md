# A-share monitor configuration templates

This folder contains starter configuration snippets for the A-share monitoring loop.

## Environment variables

Copy the following into your `.env` (or export them) to customise a run:

```
ASHARE_MONITOR_SYMBOLS=000300.SH,600519.SS
ASHARE_MONITOR_LOOKBACK_DAYS=30
ASHARE_MONITOR_REFRESH_MINUTES=15
ASHARE_MONITOR_MODE=live
ASHARE_MONITOR_BACKTEST_START=2024-01-01
ASHARE_MONITOR_BACKTEST_END=2024-03-31
```

## TOML template

The accompanying [`config.toml`](./config.toml) mirrors the same options for use with
third-party configuration loaders.
