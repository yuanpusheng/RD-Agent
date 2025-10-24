"""Lightweight exploratory data analysis utilities for daily price data.

This module provides helpers for generating candlestick (K-line) charts, volume
plots, simple moving-average indicator plots, and summary statistics that are
useful for manual data-quality inspections. Artifacts are persisted to disk so
that analysts can quickly review the health of cleaned datasets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

# Force a non-interactive backend so plotting works in headless environments
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402  (needs Agg backend set first)
import pandas as pd
from loguru import logger

try:  # pragma: no cover - optional dependency guard
    import mplfinance as mpf

    MPF_AVAILABLE = True
except Exception:  # pragma: no cover - mplfinance optional fallback
    mpf = None  # type: ignore
    MPF_AVAILABLE = False

from rdagent_china.data.provider import UnifiedDataProvider

DEFAULT_MOVING_AVERAGES: tuple[int, ...] = (5, 20)


@dataclass
class SymbolEDAResult:
    """Container describing EDA artifacts generated for a symbol."""

    symbol: str
    rows: int
    output_dir: Path
    summary_path: Path
    plots: dict[str, Path] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _safe_symbol_dir_name(symbol: str) -> str:
    """Convert a ticker into a filesystem-safe directory name."""

    safe = symbol.replace("/", "_").replace("\\", "_")
    safe = safe.replace(":", "_")
    return safe


def _prepare_frame(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalise the raw frame so downstream plotting code can rely on schema."""

    if frame.empty:
        return frame.copy()

    df = frame.copy()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol] if symbol else df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df = df.sort_index()

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _format_date(value: pd.Timestamp | None) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def generate_symbol_eda(
    symbol: str,
    frame: pd.DataFrame,
    base_output_dir: Path | str,
    moving_averages: Sequence[int] = DEFAULT_MOVING_AVERAGES,
    extra_notes: Optional[Iterable[str]] = None,
) -> SymbolEDAResult:
    """Generate plots and summary stats for a single symbol.

    Parameters
    ----------
    symbol:
        Ticker identifier whose EDA report should be created.
    frame:
        Daily price data containing at minimum the OHLCV columns. Extra columns
        are preserved when computing summary statistics but ignored by plots.
    base_output_dir:
        Directory where symbol-specific folders and artifacts are written.
    moving_averages:
        Moving-average windows (in trading days) to overlay on indicator plots
        and candlestick charts.
    extra_notes:
        Optional messages to append to the summary notes – useful when callers
        need to expose upstream warnings (e.g. data provider failures).
    """

    output_root = Path(base_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    symbol_dir = output_root / _safe_symbol_dir_name(symbol)
    symbol_dir.mkdir(parents=True, exist_ok=True)

    prepared = _prepare_frame(frame, symbol)
    result = SymbolEDAResult(
        symbol=symbol,
        rows=int(len(prepared)),
        output_dir=symbol_dir,
        summary_path=symbol_dir / "summary.json",
    )
    if extra_notes:
        result.notes.extend(str(note) for note in extra_notes if note)

    numeric_cols = [
        col for col in ("open", "high", "low", "close", "volume") if col in prepared.columns
    ]

    close_available = "close" in prepared.columns and not prepared["close"].dropna().empty
    indicator_status = {}
    if close_available:
        valid_close_len = len(prepared["close"].dropna())
        for window in moving_averages:
            key = f"sma_{window}"
            if valid_close_len >= window:
                indicator_status[key] = "generated"
            else:
                indicator_status[key] = "insufficient_data"
    else:
        indicator_status = {f"sma_{window}": "missing_close" for window in moving_averages}

    if prepared.empty:
        if "No data available for symbol" not in result.notes:
            result.notes.append("No data available for symbol")
        _write_summary(result, prepared, numeric_cols, indicator_status)
        return result

    # Generate K-line (candlestick) plot
    if MPF_AVAILABLE and all(col in prepared.columns for col in ("open", "high", "low", "close")):
        rename_map = {col: col.capitalize() for col in ("open", "high", "low", "close") if col in prepared.columns}
        if "volume" in prepared.columns:
            rename_map["volume"] = "Volume"
        ohlc = prepared.rename(columns=rename_map)
        mav_windows = [window for window in moving_averages if indicator_status.get(f"sma_{window}") == "generated"]
        has_volume = "volume" in prepared.columns and not prepared["volume"].dropna().empty
        try:
            fig, _ = mpf.plot(  # type: ignore[attr-defined]
                ohlc,
                type="candle",
                style="yahoo",
                volume=has_volume,
                mav=mav_windows if mav_windows else None,
                returnfig=True,
            )
        except Exception as exc:  # pragma: no cover - plotting backend guard
            logger.warning("K-line plot failed for %s: %s", symbol, exc)
            result.notes.append(f"Failed to render K-line plot: {exc}")
        else:
            kline_path = symbol_dir / "kline.png"
            fig.savefig(kline_path)
            plt.close(fig)
            result.plots["kline"] = kline_path
    else:
        if not MPF_AVAILABLE:
            result.notes.append("mplfinance not available; skipping K-line plot")
        else:
            result.notes.append("Skipping K-line plot: insufficient OHLC data")

    # Volume bar chart
    if "volume" in prepared.columns and not prepared["volume"].dropna().empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.bar(prepared.index, prepared["volume"].fillna(0), width=0.8, color="#ff7f0e")
        ax.set_title(f"{symbol} Volume")
        ax.set_ylabel("Shares")
        ax.set_xlabel("Date")
        fig.autofmt_xdate()
        fig.tight_layout()
        volume_path = symbol_dir / "volume.png"
        fig.savefig(volume_path)
        plt.close(fig)
        result.plots["volume"] = volume_path
    else:
        result.notes.append("Skipping volume plot: volume data unavailable")

    # Indicator plot (close price + selected moving averages)
    if close_available:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(prepared.index, prepared["close"], label="Close", color="#1f77b4")
        for window in moving_averages:
            status = indicator_status.get(f"sma_{window}")
            if status != "generated":
                continue
            ma_series = prepared["close"].rolling(window=window).mean()
            ax.plot(ma_series.index, ma_series, label=f"SMA{window}")
        ax.set_title(f"{symbol} Close & Moving Averages")
        ax.set_ylabel("Price")
        ax.set_xlabel("Date")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        indicator_path = symbol_dir / "indicators.png"
        fig.savefig(indicator_path)
        plt.close(fig)
        result.plots["indicators"] = indicator_path
    else:
        result.notes.append("Skipping indicator plot: close price unavailable")

    _write_summary(result, prepared, numeric_cols, indicator_status)
    return result


def _write_summary(
    result: SymbolEDAResult,
    prepared: pd.DataFrame,
    numeric_cols: Sequence[str],
    indicator_status: dict[str, str],
) -> None:
    """Serialize summary information for downstream manual inspection."""

    if prepared.empty:
        start_date = end_date = None
    else:
        start_date = _format_date(prepared.index.min())
        end_date = _format_date(prepared.index.max())

    missing_counts = {
        col: int(prepared[col].isna().sum()) if col in prepared.columns else 0 for col in numeric_cols
    }

    statistics = None
    if numeric_cols and not prepared.empty:
        describe_df = prepared[list(numeric_cols)].describe().transpose()
        describe_df = describe_df.applymap(lambda value: float(value) if pd.notna(value) else None)
        statistics = describe_df.to_dict()

    summary_payload = {
        "symbol": result.symbol,
        "rows": result.rows,
        "start_date": start_date,
        "end_date": end_date,
        "missing_counts": missing_counts,
        "has_volume_data": bool(
            "volume" in prepared.columns and not prepared["volume"].dropna().empty
        ),
        "plots": {name: str(path.name) for name, path in result.plots.items()},
        "indicator_status": indicator_status,
        "notes": result.notes,
    }
    if statistics is not None:
        summary_payload["statistics"] = statistics

    result.summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False))


def run_eda(
    symbols: Sequence[str],
    output_dir: Path | str,
    provider: Optional[UnifiedDataProvider] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    moving_averages: Sequence[int] = DEFAULT_MOVING_AVERAGES,
) -> dict[str, SymbolEDAResult]:
    """Fetch data for the requested symbols and materialise EDA assets."""

    provider_instance = provider or UnifiedDataProvider()

    try:
        data = provider_instance.get_price_daily(symbols, start=start, end=end)
    except Exception as exc:  # pragma: no cover - guard against optional deps
        logger.warning("Failed to load price data for EDA: %s", exc)
        data = pd.DataFrame()
        error_note = f"Failed to load data from provider: {exc}"
    else:
        error_note = None

    grouped_frames: dict[str, pd.DataFrame] = {}
    if not data.empty and "symbol" in data.columns:
        grouped_frames = {
            symbol: group.copy()
            for symbol, group in data.groupby("symbol", sort=False)
        }

    results: dict[str, SymbolEDAResult] = {}
    for symbol in symbols:
        frame = grouped_frames.get(symbol, pd.DataFrame(columns=data.columns))
        notes = [error_note] if error_note else None
        results[symbol] = generate_symbol_eda(
            symbol=symbol,
            frame=frame,
            base_output_dir=output_dir,
            moving_averages=moving_averages,
            extra_notes=notes,
        )

    return results


__all__ = [
    "DEFAULT_MOVING_AVERAGES",
    "MPF_AVAILABLE",
    "SymbolEDAResult",
    "generate_symbol_eda",
    "run_eda",
]
