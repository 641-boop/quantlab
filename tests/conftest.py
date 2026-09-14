"""Shared synthetic fixtures: a tiny, deterministic long panel."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quantlab.data.base import DATE, INSTRUMENT

_CODES = ["000001", "000002", "000003"]
_CLOSE = {
    "000001": [1.0, 2.0, 3.0, 4.0, 5.0],
    "000002": [10.0, 12.0, 14.0, 16.0, 18.0],
    "000003": [100.0, 90.0, 80.0, 70.0, 60.0],
}


@pytest.fixture
def panel() -> pl.DataFrame:
    """A 3-instrument x 5-day panel with known close/volume series."""
    dates = [date(2020, 1, d) for d in range(1, 6)]
    rows = []
    for code in _CODES:
        for i, d in enumerate(dates):
            c = _CLOSE[code][i]
            rows.append(
                {
                    DATE: d,
                    INSTRUMENT: code,
                    "open": c - 0.5,
                    "high": c + 1.0,
                    "low": c - 1.0,
                    "close": c,
                    "volume": c * 100.0,
                }
            )
    return pl.DataFrame(rows).sort([DATE, INSTRUMENT])


def backtest_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    """OHLCV panel + predictions for a small smoke backtest (7 days)."""
    codes = ["000001", "000002", "000003"]
    dates = [date(2020, 1, d) for d in range(1, 8)]
    base = {"000001": 10.0, "000002": 20.0, "000003": 30.0}
    data_rows, pred_rows = [], []
    for code in codes:
        for i, d in enumerate(dates):
            close = base[code] + i
            data_rows.append({DATE: d, INSTRUMENT: code, "close": close, "volume": 1000.0})
            pred_rows.append({DATE: d, INSTRUMENT: code, "predict": close})
    data = pl.DataFrame(data_rows).sort([DATE, INSTRUMENT])
    pred = pl.DataFrame(pred_rows).sort([DATE, INSTRUMENT])
    return data, pred
