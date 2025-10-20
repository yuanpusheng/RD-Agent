from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

from rdagent_china.db import Database, BacktestResult


def _to_bt_feed(df: pd.DataFrame):
    import backtrader as bt  # lazy import

    # expects columns: date, open, high, low, close, volume
    pdf = df.copy()
    pdf = pdf[["date", "open", "high", "low", "close", "volume"]]
    pdf["date"] = pd.to_datetime(pdf["date"])  # ensure
    pdf.set_index("date", inplace=True)
    data = bt.feeds.PandasData(dataname=pdf)
    return data


def run_backtest(
    db: Database, symbols: Iterable[str], start: Optional[str], end: Optional[str], strategy: str = "sma"
) -> BacktestResult:
    import backtrader as bt  # lazy import

    class SMABaseStrategy(bt.Strategy):
        params = dict(short=10, long=30)

        def __init__(self):
            self.sma_s = bt.indicators.SMA(self.data, period=self.params.short)
            self.sma_l = bt.indicators.SMA(self.data, period=self.params.long)

        def next(self):
            if not self.position:
                if self.sma_s[0] > self.sma_l[0]:
                    self.buy()
            else:
                if self.sma_s[0] < self.sma_l[0]:
                    self.sell()

    cerebro = bt.Cerebro()
    # Use first symbol as demo portfolio baseline
    for sym in symbols:
        df = db.read_prices(symbols=[sym], start=start, end=end)
        if df.empty:
            logger.warning(f"No data for {sym}; skip")
            continue
        data = _to_bt_feed(df)
        cerebro.adddata(data, name=sym)
        break

    cerebro.addstrategy(SMABaseStrategy)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

    results = cerebro.run()
    strat = results[0]
    timeret = strat.analyzers.timereturn.get_analysis()
    dq = strat.analyzers.drawdown.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()

    eq = pd.Series(timeret)
    eq.index = pd.to_datetime(eq.index)
    equity = (1 + eq).cumprod()
    stats = {
        "final_value": cerebro.broker.getvalue(),
        "sharpe": sharpe.get("sharperatio"),
        "max_drawdown": dq.get("max"),
    }

    html = _render_report(equity, stats)
    return BacktestResult(stats=stats, equity_curve=equity.to_frame("equity"), html_report=html)


def _render_report(equity: pd.DataFrame, stats: dict) -> str:
    import plotly.graph_objs as go
    from jinja2 import Template

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity["equity"], mode="lines", name="Equity"))
    fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity (normalized)")
    plot_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    template = Template(
        """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Backtest Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    .stats { margin-bottom: 20px; }
    .stats table { border-collapse: collapse; }
    .stats th, .stats td { border: 1px solid #ddd; padding: 8px; }
  </style>
</head>
<body>
  <h1>Baseline SMA Backtest Report</h1>
  <div class="stats">
    <h2>Stats</h2>
    <table>
      <tr><th>Final Value</th><td>{{ stats.final_value }}</td></tr>
      <tr><th>Sharpe</th><td>{{ stats.sharpe }}</td></tr>
      <tr><th>Max Drawdown</th><td>{{ stats.max_drawdown }}</td></tr>
    </table>
  </div>
  <div class="plot">{{ plot_html|safe }}</div>
</body>
</html>
        """
    )
    return template.render(plot_html=plot_html, stats=stats)
