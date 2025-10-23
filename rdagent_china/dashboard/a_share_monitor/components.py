from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SignalSummary:
    """Compact summary of a slice of persisted monitoring signals."""

    total: int = 0
    triggered: int = 0
    triggered_pct: float = 0.0
    unique_symbols: int = 0
    unique_rules: int = 0
    earliest: pd.Timestamp | None = None
    latest: pd.Timestamp | None = None


def summarize_signals(frame: pd.DataFrame) -> SignalSummary:
    """Return key metrics for a computed_signals dataframe."""

    if frame is None or frame.empty:
        return SignalSummary()

    df = frame.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])

    total = int(len(df))
    triggered = int(df.get("triggered", pd.Series(dtype=bool)).astype(bool).sum()) if total else 0
    unique_symbols = int(df.get("symbol", pd.Series(dtype=str)).nunique()) if total else 0
    unique_rules = int(df.get("rule", pd.Series(dtype=str)).nunique()) if total else 0
    earliest = df["timestamp"].min() if "timestamp" in df.columns and not df.empty else None
    latest = df["timestamp"].max() if "timestamp" in df.columns and not df.empty else None

    triggered_pct = float(triggered / total) if total else 0.0

    return SignalSummary(
        total=total,
        triggered=triggered,
        triggered_pct=triggered_pct,
        unique_symbols=unique_symbols,
        unique_rules=unique_rules,
        earliest=earliest,
        latest=latest,
    )


def compute_market_breadth(price_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute advancing/declining counts from a price frame.

    The input dataframe is expected to contain columns ``symbol`` and ``date`` along with a
    ``close`` price field.  The output dataframe contains the daily breadth summary:
    ``date``, ``advancing``, ``declining``, ``unchanged``, and ``breadth`` (net advancing ratio).
    """

    if price_frame is None or price_frame.empty:
        return pd.DataFrame(columns=["date", "advancing", "declining", "unchanged", "breadth"]).astype(
            {"advancing": int, "declining": int, "unchanged": int, "breadth": float}
        )

    required = {"symbol", "date", "close"}
    missing = required - set(price_frame.columns)
    if missing:
        raise ValueError(f"price_frame missing required columns: {sorted(missing)}")

    frame = price_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["date", "symbol"])

    pivot = frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    pivot = pivot.sort_index()
    if pivot.empty:
        return pd.DataFrame(columns=["date", "advancing", "declining", "unchanged", "breadth"])

    returns = pivot.pct_change().replace([pd.NA, pd.NaT], pd.NA)
    returns = returns.dropna(how="all")

    rows: list[dict[str, object]] = []
    for idx, row in returns.iterrows():
        valid = row.dropna()
        if valid.empty:
            continue
        advancing = int((valid > 0).sum())
        declining = int((valid < 0).sum())
        unchanged = int((valid == 0).sum())
        total = advancing + declining + unchanged
        breadth = float((advancing - declining) / total) if total else 0.0
        rows.append(
            {
                "date": idx,
                "advancing": advancing,
                "declining": declining,
                "unchanged": unchanged,
                "breadth": breadth,
            }
        )

    return pd.DataFrame(rows)


def make_return_heatmap(price_frame: pd.DataFrame, periods: int = 5) -> pd.DataFrame:
    """Create a pivoted percentage change matrix suitable for heatmap rendering."""

    if price_frame is None or price_frame.empty:
        return pd.DataFrame()

    frame = price_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["date", "symbol"])
    pivot = frame.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    pivot = pivot.sort_index()
    if pivot.empty:
        return pd.DataFrame()

    heatmap = pivot.pct_change().fillna(0.0)
    if periods > 0:
        heatmap = heatmap.tail(periods)
    return heatmap


def build_watchlist_view(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Merge latest signal metadata with most recent price snapshot for a watchlist table."""

    signal_cols = [
        "symbol",
        "timestamp",
        "rule",
        "label",
        "severity",
        "triggered",
        "value",
        "universe",
    ]
    price_cols = ["symbol", "date", "close", "volume"]

    if signals is None or signals.empty:
        latest_signals = pd.DataFrame(columns=signal_cols)
    else:
        latest_signals = signals.copy()
        latest_signals["timestamp"] = pd.to_datetime(latest_signals["timestamp"], utc=True, errors="coerce")
        latest_signals = latest_signals.sort_values("timestamp")
        latest_signals = latest_signals.dropna(subset=["timestamp"])
        latest_signals = latest_signals.groupby("symbol", as_index=False).tail(1)
        latest_signals = latest_signals[signal_cols]

    if prices is None or prices.empty:
        latest_prices = pd.DataFrame(columns=price_cols + ["prev_close", "change_pct"])
    else:
        latest_prices = prices.copy()
        latest_prices["date"] = pd.to_datetime(latest_prices["date"], utc=True, errors="coerce")
        latest_prices = latest_prices.dropna(subset=["date"]).sort_values(["symbol", "date"])
        latest_prices["prev_close"] = latest_prices.groupby("symbol")["close"].shift(1)
        latest_prices["change_pct"] = (latest_prices["close"] - latest_prices["prev_close"]) / latest_prices[
            "prev_close"
        ]
        latest_prices["change_pct"] = latest_prices["change_pct"].replace([pd.NA, pd.NaT], 0.0)
        latest_prices["change_pct"] = latest_prices["change_pct"].replace([float("inf"), float("-inf")], 0.0)
        latest_prices = latest_prices.groupby("symbol", as_index=False).tail(1)
        latest_prices = latest_prices[price_cols + ["change_pct"]]

    merged = pd.merge(latest_signals, latest_prices, on="symbol", how="outer")
    merged = merged.sort_values("symbol").reset_index(drop=True)
    merged = merged.rename(columns={"date": "price_as_of"})

    # Ensure stable column order
    ordered_cols: list[str] = [
        "symbol",
        "timestamp",
        "price_as_of",
        "close",
        "change_pct",
        "rule",
        "label",
        "severity",
        "triggered",
        "value",
        "volume",
        "universe",
    ]
    for col in ordered_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    merged = merged[ordered_cols]
    return merged


def filter_by_view_mode(frame: pd.DataFrame, view_mode: str) -> pd.DataFrame:
    """Filter a signals dataframe by intraday/EOD mode based on run_version naming."""

    if frame is None or frame.empty or view_mode.lower() == "both":
        return frame
    df = frame.copy()
    view_mode = view_mode.lower()
    if "run_version" not in df.columns:
        return df
    if view_mode == "intraday":
        mask = df["run_version"].fillna("").str.contains("intraday", case=False, na=False)
        return df[mask]
    if view_mode == "eod":
        mask = df["run_version"].fillna("").str.contains("eod", case=False, na=False)
        return df[mask]
    return df


def filter_by_rules(frame: pd.DataFrame, rules: Iterable[str] | None) -> pd.DataFrame:
    """Filter a signals dataframe by rule identifiers."""

    if frame is None or frame.empty or not rules:
        return frame
    rule_set = {r for r in rules if r}
    if not rule_set:
        return frame
    df = frame.copy()
    return df[df["rule"].isin(rule_set)]


__all__ = [
    "SignalSummary",
    "build_watchlist_view",
    "compute_market_breadth",
    "filter_by_rules",
    "filter_by_view_mode",
    "make_return_heatmap",
    "summarize_signals",
]
