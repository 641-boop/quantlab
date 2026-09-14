"""Metric formula tests against hand-computed values."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from quantlab.metrics import (
    annualized_return,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summary,
)


def test_annualized_return():
    # one day, nav doubles -> annualized = 2^252 - 1
    assert annualized_return(np.array([1.0, 2.0])) == pytest.approx(2.0**252 - 1.0)


def test_max_drawdown():
    nav = np.array([1.0, 2.0, 1.0, 0.5])
    assert max_drawdown(nav) == pytest.approx(0.75)


def test_sharpe_ratio():
    r = np.array([0.01, 0.02, 0.03])
    expected = 0.02 / r.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected)


def test_sortino_ratio_no_downsides_is_nan():
    r = np.array([0.01, 0.02, 0.03])  # all positive -> no downside
    assert np.isnan(sortino_ratio(r))


def test_summary_keys():
    nav = pl.DataFrame(
        {
            "date": [date(2020, 1, d) for d in range(1, 6)],
            "nav": [1.0, 1.02, 1.05, 1.03, 1.08],
            "benchmark": [1.0, 1.01, 1.02, 1.015, 1.04],
        }
    )
    m = summary(nav)
    for key in (
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "information_ratio",
        "beta",
        "alpha",
    ):
        assert key in m
    assert m["total_return"] == pytest.approx(1.08 / 1.0 - 1.0)
