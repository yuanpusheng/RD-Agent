from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from rdagent.log.base import Message
from rdagent.log.storage import FileStorage


def discover_sessions(log_root: Path | str | None, max_sessions: int = 25) -> list[Path]:
    """Return candidate log session directories under ``log_root``.

    The function looks for directories containing pickled RD-Agent trace files.  It
    prefers direct children of ``log_root`` but will fall back to a depth-2 search if
    nothing is found at the first level.
    """

    if log_root is None or (isinstance(log_root, str) and not log_root.strip()):
        return []
    base = Path(log_root).expanduser().resolve()
    if not base.exists():
        return []

    def _finder(paths: Iterable[Path]) -> list[Path]:
        results: list[Path] = []
        for entry in sorted(paths):
            if not entry.is_dir():
                continue
            if any(entry.glob("**/*.pkl")):
                results.append(entry)
            if len(results) >= max_sessions:
                break
        return results

    sessions = _finder(base.glob("*"))
    if sessions:
        return sessions[:max_sessions]

    # fall back to nested search (depth=2) to catch log/<scenario>/<run-id>
    return _finder(base.glob("*/*"))[:max_sessions]


def _message_to_text(message: object) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    return repr(message)


def collect_trace_rows(
    session_path: Path | str | None,
    *,
    symbol_filter: str | None = None,
    rule_filter: Iterable[str] | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Collect monitoring trace messages into a dataframe filtered by symbol / rule."""

    if session_path is None:
        return pd.DataFrame(columns=["timestamp", "tag", "symbol_match", "rule_match", "summary"])
    session = Path(session_path).expanduser().resolve()
    if not session.exists() or not session.is_dir():
        return pd.DataFrame(columns=["timestamp", "tag", "symbol_match", "rule_match", "summary"])

    storage = FileStorage(session)
    rule_set = {rule for rule in (rule_filter or []) if rule}
    symbol_filter = symbol_filter or ""

    rows: list[dict[str, object]] = []
    for msg in storage.iter_msg():
        if not isinstance(msg, Message):  # defensive guard
            continue
        tag = msg.tag or ""
        if "a_share_monitor" not in tag.lower():
            continue
        summary = _message_to_text(msg.content)
        symbol_match = bool(symbol_filter) and symbol_filter in (tag + " " + summary)
        rule_match = bool(rule_set) and any(rule in (tag + " " + summary) for rule in rule_set)
        if symbol_filter and not symbol_match:
            continue
        if rule_set and not rule_match:
            continue
        rows.append(
            {
                "timestamp": msg.timestamp,
                "tag": tag,
                "symbol_match": symbol_match,
                "rule_match": rule_match,
                "summary": summary,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["timestamp", "tag", "symbol_match", "rule_match", "summary"])

    if limit > 0:
        rows = rows[-limit:]

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return frame


__all__ = ["collect_trace_rows", "discover_sessions"]
