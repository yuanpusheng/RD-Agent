from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import pandas as pd
import typer
from loguru import logger

from rdagent_china.config import settings
from rdagent_china.data.provider import UnifiedDataProvider
from rdagent_china.data.universe import resolve_universe
from rdagent_china.db import Database, get_db
from rdagent_china.strategy import sma
from rdagent_china.strategy.interface import Signal

DEFAULT_UNIVERSE = settings.monitor_default_universe
DEFAULT_LOOKBACK_DAYS = max(30, settings.monitor_lookback_days)

app = typer.Typer(help="End-to-end daily signal pipeline for RD-Agent China.")


@dataclass(frozen=True)
class AgentInsights:
    commentary: str
    data_start: Optional[pd.Timestamp]
    data_end: Optional[pd.Timestamp]
    symbol_comments: Dict[str, str]


@dataclass(frozen=True)
class StrategyConfig:
    slug: str
    version: str
    generator: Callable[..., pd.Series]
    parameters: Dict[str, Any]
    description: str


@dataclass(frozen=True)
class SelectedStrategy:
    config: StrategyConfig
    score: float
    rationale: str

    @property
    def slug(self) -> str:
        return self.config.slug

    @property
    def version(self) -> str:
        return self.config.version


@dataclass
class PipelineResult:
    universe: str
    signals: pd.DataFrame
    insights: AgentInsights
    strategy: SelectedStrategy
    export_path: Optional[Path] = None


STRATEGY_REGISTRY: Dict[str, StrategyConfig] = {
    "sma": StrategyConfig(
        slug="sma",
        version="1.0",
        generator=sma.generate_signals,
        parameters={"short": 10, "long": 30},
        description="Simple moving average crossover using 10 and 30 day windows.",
    ),
}


def run_pipeline(
    universe: str = DEFAULT_UNIVERSE,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    limit: Optional[int] = None,
    strategy: Optional[str] = None,
    export_path: Optional[Path] = None,
    export_format: Optional[str] = None,
    *,
    db: Optional[Database] = None,
    provider: Optional[UnifiedDataProvider] = None,
) -> PipelineResult:
    db_instance = db or get_db()
    db_instance.init()
    provider_instance = provider or UnifiedDataProvider(db=db_instance)

    universe_label = _normalize_universe_label(universe)
    symbols = _resolve_symbols(universe)
    if limit is not None and limit > 0:
        symbols = symbols[:limit]

    if not symbols:
        raise RuntimeError("No symbols resolved for the requested universe")

    start_str, end_str = _determine_window(start_date, end_date, lookback_days)
    market_data = _refresh_market_data(provider_instance, symbols, start=start_str, end=end_str)
    if market_data.empty:
        raise RuntimeError("No market data returned for the requested period")

    insights = _generate_agent_insights(market_data, universe_label)
    selected_strategy = _select_strategy(strategy, market_data)
    signals = _generate_signal_frame(market_data, selected_strategy, insights, universe_label)
    if signals.empty:
        raise RuntimeError("Signal generation produced no rows")

    db_instance.write_strategy_signals(signals)
    export_target = _export_signals(signals, export_path, export_format)

    result = PipelineResult(
        universe=universe_label,
        signals=signals,
        insights=insights,
        strategy=selected_strategy,
        export_path=export_target,
    )
    return result


@app.command()
def run(
    universe: str = typer.Option(DEFAULT_UNIVERSE, help="Universe identifier or comma-separated symbol list."),
    start_date: Optional[str] = typer.Option(None, help="Override start date (YYYY-MM-DD)."),
    end_date: Optional[str] = typer.Option(None, help="Override end date (YYYY-MM-DD)."),
    lookback_days: int = typer.Option(DEFAULT_LOOKBACK_DAYS, help="Lookback window when start date not provided."),
    limit: Optional[int] = typer.Option(None, help="Limit number of symbols to process."),
    strategy: Optional[str] = typer.Option(None, help="Force a specific strategy id (e.g. 'sma')."),
    export_path: Optional[Path] = typer.Option(None, help="Optional file path to export the signals."),
    export_format: Optional[str] = typer.Option(
        None, help="Export format (csv or parquet). Defaults to suffix derived from export_path or csv."
    ),
) -> None:
    try:
        result = run_pipeline(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            limit=limit,
            strategy=strategy,
            export_path=export_path,
            export_format=export_format,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    except RuntimeError as exc:
        typer.secho(f"Daily signal pipeline failed: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    display_summary(result)


def display_summary(result: PipelineResult) -> None:
    typer.secho(
        f"Generated {len(result.signals)} signals using {result.strategy.slug} v{result.strategy.version}",
        fg=typer.colors.GREEN,
    )
    typer.echo(result.strategy.rationale)
    typer.echo(result.insights.commentary)

    preview_cols = ["symbol", "signal", "confidence", "explanation"]
    preview = result.signals[preview_cols].copy()
    preview["signal"] = preview["signal"].astype(int)
    typer.echo("\n" + preview.to_string(index=False, max_colwidth=80))
    if result.export_path:
        typer.echo(f"Signals exported to {result.export_path}")


def _normalize_universe_label(universe: str) -> str:
    text = universe.strip() if universe else DEFAULT_UNIVERSE
    if not text:
        return DEFAULT_UNIVERSE
    return text


def _resolve_symbols(universe: str) -> list[str]:
    resolved = resolve_universe(universe)
    deduped: list[str] = []
    seen = set()
    for sym in resolved:
        if sym not in seen:
            deduped.append(sym)
            seen.add(sym)
    return deduped


def _determine_window(
    start_date: Optional[str], end_date: Optional[str], lookback_days: int
) -> tuple[Optional[str], Optional[str]]:
    end_dt = pd.to_datetime(end_date).normalize() if end_date else pd.Timestamp.utcnow().normalize()
    start_dt = pd.to_datetime(start_date).normalize() if start_date else end_dt - pd.Timedelta(days=max(1, lookback_days))
    start_str = start_dt.strftime("%Y-%m-%d") if start_dt is not None else None
    end_str = end_dt.strftime("%Y-%m-%d") if end_dt is not None else None
    return start_str, end_str


def _refresh_market_data(
    provider: UnifiedDataProvider,
    symbols: Sequence[str],
    *,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    frame = provider.get_price_daily(symbols, start=start, end=end)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    return frame


def _generate_agent_insights(data: pd.DataFrame, universe: str) -> AgentInsights:
    if data.empty:
        return AgentInsights(
            commentary="No market data available for analysis.",
            data_start=None,
            data_end=None,
            symbol_comments={},
        )
    ordered = data.sort_values("date")
    latest = ordered.groupby("symbol").tail(1)
    symbol_comments: Dict[str, str] = {}
    snippets: list[str] = []

    for row in latest.itertuples():
        change = 0.0
        if getattr(row, "open", None):
            open_px = float(row.open)
            if open_px:
                change = (float(row.close) - open_px) / open_px * 100
        note = f"{row.symbol}: close {float(row.close):.2f} ({change:+.2f}%)"
        symbol_comments[row.symbol] = note
        snippets.append(note)

    data_start = pd.to_datetime(ordered["date"].min())
    data_end = pd.to_datetime(ordered["date"].max())
    commentary = (
        f"Agent review for {universe} covering {data_start:%Y-%m-%d} to {data_end:%Y-%m-%d}: "
        + "; ".join(snippets)
    )
    return AgentInsights(
        commentary=commentary,
        data_start=data_start,
        data_end=data_end,
        symbol_comments=symbol_comments,
    )


def _select_strategy(strategy: Optional[str], data: pd.DataFrame) -> SelectedStrategy:
    registry = STRATEGY_REGISTRY
    if strategy:
        key = strategy.lower()
        if key not in registry:
            raise ValueError(f"Unknown strategy id '{strategy}'")
        config = registry[key]
        score = _score_candidate(config, data)
        rationale = f"Strategy '{config.slug}' selected via CLI override."
        return SelectedStrategy(config=config, score=score, rationale=rationale)

    if not registry:
        raise RuntimeError("No strategies are configured for selection")

    scored: list[tuple[float, StrategyConfig]] = []
    for config in registry.values():
        score = _score_candidate(config, data)
        scored.append((score, config))

    best_score, best_config = max(scored, key=lambda item: item[0])
    rationale = f"Strategy '{best_config.slug}' selected with average SMA spread score {best_score:.4f}."
    return SelectedStrategy(config=best_config, score=best_score, rationale=rationale)


def _score_candidate(config: StrategyConfig, data: pd.DataFrame) -> float:
    spreads: list[float] = []
    short_window = int(config.parameters.get("short", 10))
    long_window = int(config.parameters.get("long", 30))
    for _, frame in data.groupby("symbol"):
        if frame.empty:
            continue
        ordered = frame.sort_values("date")
        closes = ordered["close"].astype(float)
        short_ma = closes.rolling(short_window, min_periods=1).mean()
        long_ma = closes.rolling(long_window, min_periods=1).mean()
        spread = (short_ma - long_ma).abs().tail(5).mean()
        spreads.append(float(spread) if pd.notna(spread) else 0.0)
    if not spreads:
        return 0.0
    return float(sum(spreads) / len(spreads))


def _generate_signal_frame(
    data: pd.DataFrame,
    selected_strategy: SelectedStrategy,
    insights: AgentInsights,
    universe: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    config = selected_strategy.config
    short_window = int(config.parameters.get("short", 10))
    long_window = int(config.parameters.get("long", 30))
    timestamp_now = pd.Timestamp.utcnow()

    for symbol, frame in data.groupby("symbol"):
        ordered = frame.sort_values("date").reset_index(drop=True)
        if ordered.empty:
            continue
        signals = config.generator(ordered, **config.parameters).astype(int)
        latest_signal = int(signals.iloc[-1])
        closes = ordered["close"].astype(float)
        short_ma = closes.rolling(short_window, min_periods=1).mean().iloc[-1]
        long_ma = closes.rolling(long_window, min_periods=1).mean().iloc[-1]
        latest_close = float(closes.iloc[-1])
        as_of_date = pd.to_datetime(ordered["date"].iloc[-1])

        explanation = _build_explanation(
            symbol=symbol,
            signal_value=latest_signal,
            latest_close=latest_close,
            short_ma=short_ma,
            long_ma=long_ma,
            selected_strategy=selected_strategy,
            insights=insights,
        )
        confidence = _confidence_from_signal(latest_signal, short_ma, long_ma)

        records.append(
            {
                "universe": universe,
                "symbol": symbol,
                "as_of_date": as_of_date,
                "timestamp": timestamp_now,
                "signal": latest_signal,
                "strategy_id": config.slug,
                "strategy_version": config.version,
                "confidence": confidence,
                "explanation": explanation,
            }
        )

    columns = [
        "universe",
        "symbol",
        "as_of_date",
        "timestamp",
        "signal",
        "strategy_id",
        "strategy_version",
        "confidence",
        "explanation",
    ]
    return pd.DataFrame(records, columns=columns) if records else pd.DataFrame(columns=columns)


def _build_explanation(
    *,
    symbol: str,
    signal_value: int,
    latest_close: float,
    short_ma: float,
    long_ma: float,
    selected_strategy: SelectedStrategy,
    insights: AgentInsights,
) -> str:
    if signal_value == Signal.BUY:
        direction = "Bullish crossover detected"
    elif signal_value == Signal.SELL:
        direction = "Bearish crossover detected"
    else:
        direction = "Neutral trend"

    base = (
        f"{symbol}: {direction}. Last close {latest_close:.2f}; short SMA {short_ma:.2f} vs long SMA {long_ma:.2f}. "
        f"Strategy {selected_strategy.slug} v{selected_strategy.version}."
    )
    symbol_comment = insights.symbol_comments.get(symbol)
    if symbol_comment:
        return f"{base} Agent note: {symbol_comment}."
    return f"{base} Agent summary: {insights.commentary}."


def _confidence_from_signal(signal_value: int, short_ma: float, long_ma: float) -> float:
    base_map = {Signal.BUY: 0.72, Signal.SELL: 0.68, Signal.HOLD: 0.55}
    base = base_map.get(signal_value, 0.6)
    denominator = abs(long_ma) if long_ma else 1.0
    spread = abs(short_ma - long_ma) / denominator
    confidence = min(0.95, base + 0.2 * min(spread, 1.0))
    return round(float(confidence), 4)


def _export_signals(
    signals: pd.DataFrame, export_path: Optional[Path], export_format: Optional[str]
) -> Optional[Path]:
    if export_path is None:
        return None

    fmt = export_format.lower() if export_format else export_path.suffix.lstrip(".").lower()
    fmt = fmt or "csv"
    export_path = export_path.expanduser().resolve()
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        signals.to_csv(export_path, index=False)
    elif fmt in {"parquet", "pq"}:
        try:
            signals.to_parquet(export_path, index=False)
        except (ImportError, ValueError) as exc:  # pragma: no cover - optional dependencies
            raise RuntimeError("Parquet export requires 'pyarrow' or 'fastparquet'") from exc
    else:
        raise RuntimeError(f"Unsupported export format '{fmt}'")

    logger.info("Signals exported to %s", export_path)
    return export_path


if __name__ == "__main__":
    app()
