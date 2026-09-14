"""Risk metric tests: VaR / CVaR conventions, underwater days, scenario stress."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from quantlab.metrics.risk import (
    expected_shortfall,
    longest_drawdown_days,
    risk_metrics,
    scenario_stress,
    value_at_risk,
)


def test_value_at_risk_known_values():
    # conservative "higher" quantile: 95% VaR is the 2nd-worst daily return
    r = np.array([-0.01, -0.02, -0.03, 0.01, 0.02])
    assert value_at_risk(r, confidence=0.95) == pytest.approx(0.02)
    assert value_at_risk(r, confidence=0.99) == pytest.approx(0.02)


def test_value_at_risk_positive_and_monotonic():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, 1000)
    v95 = value_at_risk(r, confidence=0.95)
    v99 = value_at_risk(r, confidence=0.99)
    assert v95 > 0
    assert v99 >= v95


def test_expected_shortfall_known_and_bounds():
    r = np.array([-0.01, -0.02, -0.03, 0.01, 0.02])
    assert expected_shortfall(r, confidence=0.95) == pytest.approx(0.025)
    assert expected_shortfall(r, confidence=0.95) >= value_at_risk(r, confidence=0.95)


def test_longest_drawdown_days():
    nav = np.array([1.0, 1.2, 1.1, 0.9, 1.0, 1.3])
    assert longest_drawdown_days(nav) == 3
    # recovery exactly at the old peak resets the run
    nav_recovered = np.array([1.0, 1.1, 0.9, 1.1, 1.0])
    assert longest_drawdown_days(nav_recovered) == 1
    assert longest_drawdown_days(np.linspace(1.0, 2.0, 10)) == 0


def _nav_from_returns(strategy_ret: np.ndarray, bench_ret: np.ndarray) -> pl.DataFrame:
    """NAV / benchmark frames whose daily returns equal the input series."""
    start = date(2020, 1, 1)
    dates = [start + timedelta(days=int(i)) for i in range(len(strategy_ret) + 1)]
    nav = np.concatenate([[1.0], np.cumprod(1.0 + strategy_ret)])
    bench = np.concatenate([[1.0], np.cumprod(1.0 + bench_ret)])
    return pl.DataFrame({"date": dates, "nav": nav, "benchmark": bench})


def test_scenario_stress_maps_through_beta():
    rng = np.random.default_rng(1)
    bench = rng.normal(0.0, 0.005, 300)
    strategy = 2.0 * bench  # true beta = 2
    nav = _nav_from_returns(strategy, bench)
    out = scenario_stress(nav)
    # single-day -10% market shock -> strategy -20%
    assert out["单日 -10%"] == pytest.approx(-0.20, rel=1e-6)
    # three consecutive days of -5% -> (1 - 0.10)^3 - 1
    assert out["连续 3 日 -5%"] == pytest.approx((0.9) ** 3 - 1.0, rel=1e-6)


def test_risk_metrics_keys():
    rng = np.random.default_rng(2)
    bench = rng.normal(0.0, 0.004, 200)
    nav = _nav_from_returns(0.5 * bench, bench)
    m = risk_metrics(nav)
    for key in ("var_95", "var_99", "cvar_95", "cvar_99", "max_drawdown_days"):
        assert key in m
    assert m["var_95"] >= 0
    assert m["max_drawdown_days"] >= 0
