from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from rdagent_china.backtest.config import SignalBacktestConfig
from rdagent_china.db import Database


@dataclass
class RuleWindowSummary:
    rule: str
    window: int
    total_signals: int
    true_positives: int
    false_positives: int
    false_negatives: int
    ignored_signals: int
    precision: float | None
    recall: float | None
    hit_rate: float | None
    avg_return: float | None
    avg_benchmark_return: float | None
    avg_excess_return: float | None
    max_drawdown: float | None
    equity_curve: pd.Series

    def as_dict(self) -> dict[str, Any]:
        equity_points: list[dict[str, Any]] = []
        if isinstance(self.equity_curve, pd.Series) and not self.equity_curve.empty:
            for idx, value in self.equity_curve.items():
                equity_points.append({
                    "date": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
                    "equity": float(value),
                })
        return {
            "rule": self.rule,
            "window": self.window,
            "total_signals": self.total_signals,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "ignored_signals": self.ignored_signals,
            "precision": self.precision,
            "recall": self.recall,
            "hit_rate": self.hit_rate,
            "avg_return": self.avg_return,
            "avg_benchmark_return": self.avg_benchmark_return,
            "avg_excess_return": self.avg_excess_return,
            "max_drawdown": self.max_drawdown,
            "equity_curve": equity_points,
        }


@dataclass
class SignalBacktestResult:
    config: SignalBacktestConfig
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    metrics: list[RuleWindowSummary]
    trades: pd.DataFrame
    windows: list[int]
    window_positive_counts: dict[int, int]
    benchmark_symbol: str | None = None

    @classmethod
    def empty(
        cls,
        config: SignalBacktestConfig,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
        windows: Sequence[int],
    ) -> "SignalBacktestResult":
        empty_cols = [
            "symbol",
            "date",
            "rule",
            "window",
            "return",
            "benchmark_return",
            "excess_return",
            "label",
            "hit",
            "exit_date",
        ]
        trades = pd.DataFrame(columns=empty_cols)
        return cls(
            config=config,
            start=start,
            end=end,
            metrics=[],
            trades=trades,
            windows=list(windows),
            window_positive_counts={window: 0 for window in windows},
            benchmark_symbol=config.benchmark.symbol if config.benchmark else None,
        )

    def is_empty(self) -> bool:
        return not self.metrics and self.trades.empty

    def overall_metrics(self) -> dict[int, dict[str, float | int | None]]:
        summary: dict[int, dict[str, float | int | None]] = {}
        for window in self.windows:
            subset = self.trades[self.trades["window"] == window]
            total = int(len(subset))
            hits = int(subset["hit"].sum()) if total else 0
            positives = self.window_positive_counts.get(window, 0)
            precision = hits / total if total else None
            recall = hits / positives if positives else None
            avg_return = subset["return"].mean() if total else None
            avg_benchmark = subset["benchmark_return"].dropna().mean() if total else None
            avg_excess = subset["excess_return"].dropna().mean() if total else None
            summary[window] = {
                "total_signals": total,
                "hits": hits,
                "precision": precision,
                "recall": recall,
                "avg_return": avg_return,
                "avg_benchmark_return": avg_benchmark,
                "avg_excess_return": avg_excess,
            }
        return summary

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start.isoformat() if isinstance(self.start, pd.Timestamp) else None,
            "end": self.end.isoformat() if isinstance(self.end, pd.Timestamp) else None,
            "metrics": [metric.as_dict() for metric in self.metrics],
            "overall": self.overall_metrics(),
            "benchmark": self.benchmark_symbol,
            "trades": [self._serialize_trade(row) for row in self.trades.to_dict(orient="records")],
        }

    @staticmethod
    def _serialize_trade(row: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                result[key] = value.isoformat()
            elif isinstance(value, (np.floating, float)):
                result[key] = None if pd.isna(value) else float(value)
            else:
                result[key] = value
        return result

    def _plot_html(self) -> str:
        if not self.metrics:
            return "<p>No equity curves available.</p>"
        try:
            import plotly.graph_objs as go
        except Exception as exc:  # pragma: no cover - optional dependency guard
            logger.warning("Plotly unavailable for report generation: %s", exc)
            return "<p>Plot generation unavailable.</p>"
        fig = go.Figure()
        for metric in self.metrics:
            curve = metric.equity_curve
            if isinstance(curve, pd.Series) and not curve.empty:
                fig.add_trace(
                    go.Scatter(
                        x=list(curve.index),
                        y=list(curve.values),
                        mode="lines",
                        name=f"{metric.rule} ({metric.window}d)",
                    )
                )
        if not fig.data:
            return "<p>No equity curves available.</p>"
        fig.update_layout(
            title="Signal Backtest Equity Curves",
            xaxis_title="Exit Date",
            yaxis_title="Equity",
            template="plotly_white",
        )
        return fig.to_html(include_plotlyjs="cdn", full_html=False)

    def render_report(self) -> str:
        from jinja2 import Template

        template = Template(
            """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Signal Backtest Summary</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 24px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: right; }
    th { background-color: #f2f2f2; text-transform: uppercase; }
    td.label { text-align: left; }
  </style>
</head>
<body>
  <h1>Signal Backtest Summary</h1>
  <p><strong>Start:</strong> {{ start or "N/A" }} &nbsp; <strong>End:</strong> {{ end or "N/A" }}</p>
  {% if benchmark %}<p><strong>Benchmark:</strong> {{ benchmark }}</p>{% endif %}
  <h2>Overall Metrics</h2>
  <table>
    <tr>
      <th>Window</th>
      <th>Total Signals</th>
      <th>Hits</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Avg Return</th>
      <th>Avg Benchmark</th>
      <th>Avg Excess</th>
    </tr>
    {% for window, stats in overall.items() %}
    <tr>
      <td class="label">{{ window }}d</td>
      <td>{{ stats.total_signals }}</td>
      <td>{{ stats.hits }}</td>
      <td>{{ '%.3f'|format(stats.precision) if stats.precision is not none else 'N/A' }}</td>
      <td>{{ '%.3f'|format(stats.recall) if stats.recall is not none else 'N/A' }}</td>
      <td>{{ '%.3f'|format(stats.avg_return) if stats.avg_return is not none else 'N/A' }}</td>
      <td>{{ '%.3f'|format(stats.avg_benchmark_return) if stats.avg_benchmark_return is not none else 'N/A' }}</td>
      <td>{{ '%.3f'|format(stats.avg_excess_return) if stats.avg_excess_return is not none else 'N/A' }}</td>
    </tr>
    {% endfor %}
  </table>
  <h2>Rule Breakdown</h2>
  <table>
    <tr>
      <th>Rule</th>
      <th>Window</th>
      <th>Total</th>
      <th>Hits</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>Hit Rate</th>
      <th>Drawdown</th>
      <th>Avg Return</th>
      <th>Avg Benchmark</th>
      <th>Avg Excess</th>
      <th>Ignored</th>
    </tr>
    {% for metric in metrics %}
    <tr>
      <td class="label">{{ metric.rule }}</td>
      <td>{{ metric.window }}d</td>
      <td>{{ metric.total_signals }}</td>
      <td>{{ metric.true_positives }}</td>
      <td>{{ '%.3f'|format(metric.precision) if metric.precision is not none else 'N/A' }}</td>
      <td>{{ '%.3f'|format(metric.recall) if metric.recall is not none else 'N/A' }}</td>
      <td>{{ '%.3f'|format(metric.hit_rate) if metric.hit_rate is not none else 'N/A' }}</td>
      <td>{{ '%.3f'|format(metric.max_drawdown) if metric.max_drawdown is not none else 'N/A' }}</td>
      <td>{{ '%.3f'|format(metric.avg_return) if metric.avg_return is not none else 'N/A' }}</td>
      <td>{{ '%.3f'|format(metric.avg_benchmark_return) if metric.avg_benchmark_return is not none else 'N/A' }}</td>
      <td>{{ '%.3f'|format(metric.avg_excess_return) if metric.avg_excess_return is not none else 'N/A' }}</td>
      <td>{{ metric.ignored_signals }}</td>
    </tr>
    {% endfor %}
  </table>
  <h2>Equity Curves</h2>
  {{ plot_html | safe }}
</body>
</html>
            """
        )
        return template.render(
            start=self.start.isoformat() if isinstance(self.start, pd.Timestamp) else None,
            end=self.end.isoformat() if isinstance(self.end, pd.Timestamp) else None,
            metrics=[metric.as_dict() for metric in self.metrics],
            overall=self.overall_metrics(),
            plot_html=self._plot_html(),
            benchmark=self.benchmark_symbol,
        )

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        summary_path = output_dir / "metrics.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
        report_path = output_dir / "report.html"
        report_path.write_text(self.render_report(), encoding="utf-8")
        trades_path = output_dir / "trades.csv"
        if not self.trades.empty:
            frame = self.trades.copy()
            for column in ("date", "exit_date"):
                if column in frame.columns:
                    frame[column] = pd.to_datetime(frame[column]).dt.strftime("%Y-%m-%d")
            frame.to_csv(trades_path, index=False)
        else:
            trades_path.write_text("symbol,date,rule,window,return,benchmark_return,excess_return,label,hit,exit_date\n", encoding="utf-8")


class SignalBacktester:
    def __init__(self, db: Database, config: SignalBacktestConfig) -> None:
        self.db = db
        self.config = config
        self.windows = config.resolved_windows()
        self._qlib_available: bool | None = None
        self._qlib_module: Any | None = None
        self._qlib_initialized = False

    def run(self) -> SignalBacktestResult:
        signals = self._load_signals()
        if signals.empty:
            return SignalBacktestResult.empty(self.config, None, None, self.windows)
        symbols = self._resolve_symbols(signals)
        if not symbols:
            return SignalBacktestResult.empty(self.config, None, None, self.windows)
        start, end = self._resolve_time_bounds(signals)
        price_data = self._fetch_price_data(symbols, start, end)
        if price_data.empty:
            logger.warning("Price data unavailable for symbols: %s", symbols)
            return SignalBacktestResult.empty(self.config, start, end, self.windows)
        label_frame = self._compute_labels(price_data)
        if label_frame.empty:
            return SignalBacktestResult.empty(self.config, start, end, self.windows)
        benchmark_frame = self._load_benchmark_frame(start, end)
        metrics, trades, positive_counts = self._compute_metrics(
            signals=signals,
            labels=label_frame,
            benchmark=benchmark_frame,
            symbols=symbols,
            start=start,
            end=end,
        )
        return SignalBacktestResult(
            config=self.config,
            start=start,
            end=end,
            metrics=metrics,
            trades=trades,
            windows=self.windows,
            window_positive_counts=positive_counts,
            benchmark_symbol=self.config.benchmark.symbol if self.config.benchmark else None,
        )

    def _load_signals(self) -> pd.DataFrame:
        frame = self.db.read_signals(
            universe=self.config.universe,
            symbols=self.config.symbols,
            start=self.config.start,
            end=self.config.end,
            rules=self.config.rules,
            run_version=self.config.run_version,
        )
        if frame.empty:
            return frame
        working = frame.copy()
        working["as_of_date"] = pd.to_datetime(working["as_of_date"]).dt.normalize()
        working["timestamp"] = pd.to_datetime(working["timestamp"])
        if "rule" in working.columns:
            working["rule"] = working["rule"].astype(str)
        if "symbol" in working.columns:
            working["symbol"] = working["symbol"].astype(str)
        if "triggered" in working.columns:
            working = working[working["triggered"].astype(bool)]
        return working.reset_index(drop=True)

    def _resolve_symbols(self, signals: pd.DataFrame) -> list[str]:
        if self.config.symbols:
            return sorted({str(sym) for sym in self.config.symbols})
        if "symbol" not in signals.columns:
            return []
        return sorted({str(sym) for sym in signals["symbol"].unique().tolist()})

    def _resolve_time_bounds(self, signals: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
        if self.config.start:
            start = pd.to_datetime(self.config.start).normalize()
        else:
            start = pd.to_datetime(signals["as_of_date"].min()).normalize()
        if self.config.end:
            end = pd.to_datetime(self.config.end).normalize()
        else:
            end = pd.to_datetime(signals["as_of_date"].max()).normalize()
        return start, end

    def _fetch_price_data(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        use_qlib = self._can_use_qlib()
        try:
            if use_qlib:
                data = self._load_prices_via_qlib(symbols, start, end)
                if not data.empty:
                    return data
        except Exception as exc:  # pragma: no cover - qlib optional failure path
            logger.warning("QLib data loading failed (%s); falling back to DuckDB", exc)
        return self._load_prices_via_duckdb(symbols, start, end)

    def _load_prices_via_duckdb(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        margin_days = max(self.windows) + 5
        start_query = (start - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end_query = (end + pd.Timedelta(days=margin_days)).strftime("%Y-%m-%d")
        if self.config.data.table == "prices":
            frame = self.db.read_prices(symbols=symbols, start=start_query, end=end_query)
        else:
            frame = self.db.read_price_daily(symbols=symbols, start=start_query, end=end_query)
        if frame.empty:
            return frame
        working = frame.copy()
        if "date" in working.columns:
            working["date"] = pd.to_datetime(working["date"]).dt.normalize()
        elif "timestamp" in working.columns:
            working["date"] = pd.to_datetime(working["timestamp"]).dt.normalize()
        else:
            raise ValueError("Price data must include a date or timestamp column")
        working["symbol"] = working["symbol"].astype(str)
        keep_columns = {"symbol", "date", *self.config.data.fields}
        existing = [col for col in working.columns if col in keep_columns]
        return working[existing]

    def _can_use_qlib(self) -> bool:
        if self.config.data.source == "duckdb":
            return False
        if self._qlib_available is not None:
            return self._qlib_available
        try:
            import qlib  # type: ignore

            self._qlib_module = qlib
            self._qlib_available = True
        except Exception:
            self._qlib_module = None
            self._qlib_available = False
        return bool(self._qlib_available)

    def _load_prices_via_qlib(self, symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if not self._qlib_module:
            return pd.DataFrame()
        qlib = self._qlib_module
        if not self._qlib_initialized:
            init_kwargs: dict[str, Any] = {"region": self.config.data.qlib_region, "skip_if_reg": True}
            if self.config.data.qlib_provider_uri:
                init_kwargs["provider_uri"] = self.config.data.qlib_provider_uri
            qlib.init(**init_kwargs)
            self._qlib_initialized = True
        from qlib.data import D  # type: ignore

        start_str = start.strftime("%Y-%m-%d")
        end_str = (end + pd.Timedelta(days=max(self.windows) + 5)).strftime("%Y-%m-%d")
        q_fields: list[str] = []
        rename_map: dict[str, str] = {}
        for field in self.config.data.fields:
            name = field if field.startswith("$") else f"${field}"
            q_fields.append(name)
            rename_map[name] = field if not field.startswith("$") else field[1:]
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            try:
                qdf = D.features(  # type: ignore[attr-defined]
                    instruments=symbol,
                    fields=q_fields,
                    freq=self.config.data.freq,
                    start=start_str,
                    end=end_str,
                )
            except Exception:
                continue
            if qdf is None or qdf.empty:
                continue
            if isinstance(qdf.columns, pd.MultiIndex):
                qdf.columns = [col[-1] if isinstance(col, tuple) else col for col in qdf.columns]
            qdf = qdf.reset_index()
            if "datetime" in qdf.columns:
                qdf = qdf.rename(columns={"datetime": "date"})
            elif "index" in qdf.columns:
                qdf = qdf.rename(columns={"index": "date"})
            qdf["symbol"] = symbol
            for src, target in rename_map.items():
                if src in qdf.columns:
                    qdf = qdf.rename(columns={src: target})
            frames.append(qdf[["symbol", "date", *rename_map.values()]])
        if not frames:
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        return result

    def _compute_labels(self, prices: pd.DataFrame) -> pd.DataFrame:
        price_field = self.config.labeling.price_field
        if price_field not in prices.columns:
            raise KeyError(f"Price field '{price_field}' not found in price data")
        frames: list[pd.DataFrame] = []
        for symbol, frame in prices.groupby("symbol"):
            sorted_frame = frame.sort_values("date")
            dates = pd.to_datetime(sorted_frame["date"]).dt.normalize().reset_index(drop=True)
            price_series = pd.Series(sorted_frame[price_field].astype(float).values)
            base = pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    price_field: price_series,
                }
            )
            for window in self.windows:
                future_price = price_series.shift(-window)
                returns = future_price / price_series - 1
                labels = self._apply_labeling(returns)
                exit_dates = dates.shift(-window)
                base[f"return_{window}"] = returns
                base[f"label_{window}"] = labels
                base[f"exit_date_{window}"] = exit_dates
            frames.append(base)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _apply_labeling(self, returns: pd.Series) -> pd.Series:
        threshold = self.config.labeling.threshold
        direction = self.config.labeling.direction
        if direction == "long":
            result = returns >= threshold
        else:
            result = returns <= -threshold
        return result.astype(bool)

    def _load_benchmark_frame(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
        if not self.config.benchmark:
            return None
        symbol = self.config.benchmark.symbol
        price_field = self.config.benchmark.price_field
        frame = self._fetch_price_data([symbol], start, end)
        if frame.empty:
            logger.warning("Benchmark data unavailable for %s", symbol)
            return None
        if price_field not in frame.columns:
            raise KeyError(f"Benchmark price field '{price_field}' not present in data")
        frames: list[pd.DataFrame] = []
        for _, sym_frame in frame.groupby("symbol"):
            sorted_frame = sym_frame.sort_values("date")
            dates = pd.to_datetime(sorted_frame["date"]).dt.normalize().reset_index(drop=True)
            price_series = pd.Series(sorted_frame[price_field].astype(float).values)
            base = pd.DataFrame({"symbol": symbol, "date": dates})
            for window in self.windows:
                future_price = price_series.shift(-window)
                returns = future_price / price_series - 1
                exit_dates = dates.shift(-window)
                base[f"return_{window}"] = returns
                base[f"exit_date_{window}"] = exit_dates
            frames.append(base)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def _compute_metrics(
        self,
        *,
        signals: pd.DataFrame,
        labels: pd.DataFrame,
        benchmark: pd.DataFrame | None,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[list[RuleWindowSummary], pd.DataFrame, dict[int, int]]:
        symbol_set = {str(sym) for sym in symbols}
        label_scope = labels[(labels["symbol"].isin(symbol_set))]
        label_scope = label_scope[(label_scope["date"] >= start) & (label_scope["date"] <= end)]
        positive_counts: dict[int, int] = {}
        for window in self.windows:
            label_column = f"label_{window}"
            if label_column not in label_scope.columns:
                positive_counts[window] = 0
                continue
            series = label_scope[label_column].dropna().astype(bool)
            positive_counts[window] = int(series.sum())
        benchmark_scope: pd.DataFrame | None = None
        if benchmark is not None:
            benchmark_scope = benchmark.copy()
            benchmark_scope = benchmark_scope[
                (benchmark_scope["date"] >= start) & (benchmark_scope["date"] <= end)
            ]
        metrics: list[RuleWindowSummary] = []
        trade_rows: list[dict[str, Any]] = []
        rules = sorted({str(rule) for rule in signals["rule"].unique().tolist()})
        for rule in rules:
            rule_frame = signals[signals["rule"] == rule].copy()
            rule_frame = rule_frame[(rule_frame["symbol"].isin(symbol_set))]
            rule_frame = rule_frame[(rule_frame["as_of_date"] >= start) & (rule_frame["as_of_date"] <= end)]
            if rule_frame.empty:
                for window in self.windows:
                    metrics.append(
                        RuleWindowSummary(
                            rule=rule,
                            window=window,
                            total_signals=0,
                            true_positives=0,
                            false_positives=0,
                            false_negatives=positive_counts.get(window, 0),
                            ignored_signals=0,
                            precision=None,
                            recall=None,
                            hit_rate=None,
                            avg_return=None,
                            avg_benchmark_return=None,
                            avg_excess_return=None,
                            max_drawdown=None,
                            equity_curve=pd.Series(dtype=float),
                        )
                    )
                continue
            rule_frame = rule_frame.drop_duplicates(subset=["symbol", "as_of_date"])
            for window in self.windows:
                valid_events: list[dict[str, Any]] = []
                ignored = 0
                for _, event in rule_frame.iterrows():
                    symbol = str(event["symbol"])
                    date = pd.to_datetime(event["as_of_date"]).normalize()
                    match = label_scope[(label_scope["symbol"] == symbol) & (label_scope["date"] == date)]
                    if match.empty:
                        ignored += 1
                        continue
                    record = match.iloc[0]
                    event_return = record.get(f"return_{window}")
                    if pd.isna(event_return):
                        ignored += 1
                        continue
                    label_value = record.get(f"label_{window}")
                    if pd.isna(label_value):
                        ignored += 1
                        continue
                    exit_date = record.get(f"exit_date_{window}")
                    benchmark_return = None
                    if benchmark_scope is not None:
                        bench_row = benchmark_scope[
                            (benchmark_scope["date"] == date)
                        ]
                        if not bench_row.empty:
                            value = bench_row.iloc[0].get(f"return_{window}")
                            benchmark_return = None if pd.isna(value) else float(value)
                    hit = bool(label_value)
                    excess_return = None
                    if benchmark_return is not None and not pd.isna(event_return):
                        excess_return = float(event_return) - float(benchmark_return)
                    valid_events.append(
                        {
                            "symbol": symbol,
                            "date": date,
                            "rule": rule,
                            "window": window,
                            "return": float(event_return),
                            "label": bool(label_value),
                            "hit": hit,
                            "benchmark_return": benchmark_return,
                            "excess_return": excess_return,
                            "exit_date": pd.to_datetime(exit_date) if not pd.isna(exit_date) else date,
                        }
                    )
                total_considered = len(valid_events)
                hits = sum(1 for event in valid_events if event["hit"])
                false_positives = total_considered - hits
                symbols_in_rule = rule_frame["symbol"].unique().tolist()
                positives_rule = label_scope[label_scope["symbol"].isin(symbols_in_rule)]
                label_column = f"label_{window}"
                if label_column in positives_rule.columns:
                    positives_series = positives_rule[label_column].dropna().astype(bool)
                    total_positive_events = int(positives_series.sum())
                else:
                    total_positive_events = 0
                false_negatives = max(total_positive_events - hits, 0)
                precision = hits / total_considered if total_considered else None
                recall = hits / total_positive_events if total_positive_events else None
                hit_rate = precision
                returns_array = [event["return"] for event in valid_events]
                avg_return = float(np.mean(returns_array)) if returns_array else None
                bench_vals = [event["benchmark_return"] for event in valid_events if event["benchmark_return"] is not None]
                avg_benchmark = float(np.mean(bench_vals)) if bench_vals else None
                excess_vals = [event["excess_return"] for event in valid_events if event["excess_return"] is not None]
                avg_excess = float(np.mean(excess_vals)) if excess_vals else None
                equity_curve, max_drawdown = self._build_equity_curve(valid_events)
                metrics.append(
                    RuleWindowSummary(
                        rule=rule,
                        window=window,
                        total_signals=total_considered,
                        true_positives=hits,
                        false_positives=false_positives,
                        false_negatives=false_negatives,
                        ignored_signals=ignored,
                        precision=precision,
                        recall=recall,
                        hit_rate=hit_rate,
                        avg_return=avg_return,
                        avg_benchmark_return=avg_benchmark,
                        avg_excess_return=avg_excess,
                        max_drawdown=max_drawdown,
                        equity_curve=equity_curve,
                    )
                )
                trade_rows.extend(valid_events)
        trade_columns = [
            "symbol",
            "date",
            "rule",
            "window",
            "return",
            "benchmark_return",
            "excess_return",
            "label",
            "hit",
            "exit_date",
        ]
        trades = pd.DataFrame(trade_rows, columns=trade_columns) if trade_rows else pd.DataFrame(columns=trade_columns)
        return metrics, trades, positive_counts

    def _build_equity_curve(self, events: list[dict[str, Any]]) -> tuple[pd.Series, float | None]:
        if not events:
            return pd.Series(dtype=float), None
        sorted_events = sorted(events, key=lambda item: (item["exit_date"], item["date"]))
        equity_values: list[float] = []
        equity_dates: list[pd.Timestamp] = []
        equity = 1.0
        for item in sorted_events:
            equity *= 1.0 + float(item["return"])
            exit_date = pd.to_datetime(item["exit_date"]) if not isinstance(item["exit_date"], pd.Timestamp) else item["exit_date"]
            equity_dates.append(exit_date.normalize())
            equity_values.append(equity)
        series = pd.Series(equity_values, index=equity_dates)
        if series.empty:
            return series, None
        running_max = series.cummax()
        drawdown = series / running_max - 1
        max_drawdown = float(drawdown.min()) if not drawdown.empty else None
        return series, max_drawdown


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, pd.Series):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


__all__ = [
    "RuleWindowSummary",
    "SignalBacktester",
    "SignalBacktestResult",
]
