from __future__ import annotations

from typing import Sequence

import pandas as pd

from .base import SignalRecord
from rdagent_china.db import Database


def records_to_dataframe(records: Sequence[SignalRecord]) -> pd.DataFrame:
    """Convert signal records into a pandas DataFrame."""

    return SignalRecord.to_frame(records)


def persist_signal_records(db: Database, records: Sequence[SignalRecord]) -> pd.DataFrame:
    """Persist signal records via the configured :class:`~rdagent_china.db.Database`."""

    frame = records_to_dataframe(records)
    if frame.empty:
        return frame
    db.write_signals(frame)
    return frame


__all__ = ["records_to_dataframe", "persist_signal_records"]
