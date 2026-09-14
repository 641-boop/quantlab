"""Risk metrics: daily VaR / CVaR, underwater duration and scenario stress.

These answer the "how much can this strategy lose" questions that the pure
performance metrics in :mod:`quantlab.metrics.performance` do not cover.

Conventions
-----------
- Returns are daily decimals unless stated otherwise.
- ``value_at_risk`` / ``expected_shortfall`` return *positive* loss magnitudes
  (``0.021`` means a 2.1% one-day loss at the given confidence).
- VaR uses historical simulation with the conservative ``higher`` quantile
  method, so the reported value always coincides with an observed daily return
  instead of an interpolation.
- ``scenario_stress`` maps market shocks onto the strategy through the market
  beta fitted on the backtest period (``r_strategy ≈ beta * r_market``);
  results are signed, negative meaning a loss.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from quantlab.metrics.performance import daily_returns

_DEFAULT_CONFIDENCES = (0.95, 0.99)

#: (scenario label, single-day market shock, number of consecutive days)
_DEFAULT_SCENARIOS: list[tuple[str, float, int]] = [
    ("单日 -5%", -0.05, 1),
    ("单日 -10%", -0.10, 1),
    ("连续 3 日 -5%", -0.05, 3),
    ("连续 5 日 -3%", -0.03, 5),
]


def _clean(r: pl.Series | np.ndarray) -> np.ndarray:
    a = np.asarray(r, dtype=float)
    return a[~np.isnan(a)]


def _returns(nav: pl.DataFrame) -> pl.DataFrame:
    """NAV frame with a ``ret`` (and ``bench_ret`` when available) column."""
    df = daily_returns(nav)
    if "benchmark" in df.columns:
        df = df.with_columns((pl.col("benchmark") / pl.col("benchmark").shift(1) - 1).alias("bench_ret"))
    return df


def value_at_risk(ret: pl.Series | np.ndarray, confidence: float = 0.95) -> float:
    """Historical-simulation daily VaR as a positive loss magnitude."""
    r = _clean(ret)
    if len(r) < 2:
        return float("nan")
    return float(-np.quantile(r, 1.0 - confidence, method="higher"))


def expected_shortfall(ret: pl.Series | np.ndarray, confidence: float = 0.95) -> float:
    """Daily CVaR / expected shortfall of the worst tail (positive loss magnitude)."""
    r = _clean(ret)
    if len(r) < 2:
        return float("nan")
    cutoff = np.quantile(r, 1.0 - confidence, method="higher")
    tail = r[r <= cutoff]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


def longest_drawdown_days(nav: pl.Series | np.ndarray) -> int:
    """Longest uninterrupted run of days whose NAV stayed below its running peak.

    A day that equals or exceeds the running peak resets the run: the
    drawdown is considered recovered that day.
    """
    v = _clean(nav)
    if len(v) == 0:
        return 0
    longest = 0
    current = 0
    peak = float(v[0])
    for x in v[1:]:
        if x >= peak:
            peak = float(x)
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def rolling_value_at_risk(
    ret: pl.Series | np.ndarray,
    window: int = 60,
    confidence: float = 0.95,
) -> np.ndarray:
    """Rolling historical VaR over trailing ``window`` returns (NaN lead-in)."""
    r = _clean(ret)
    out = np.full(len(r), np.nan)
    if window < 2:
        return out
    for i in range(window - 1, len(r)):
        out[i] = value_at_risk(r[i - window + 1 : i + 1], confidence)
    return out


def risk_metrics(nav: pl.DataFrame, confidences: tuple[float, float] = _DEFAULT_CONFIDENCES) -> dict[str, float]:
    """Flat dict of daily VaR / CVaR levels plus the longest underwater period."""
    df = _returns(nav)
    ret = df["ret"]
    out: dict[str, float] = {"max_drawdown_days": float(longest_drawdown_days(nav["nav"]))}
    for conf in confidences:
        level = int(round(conf * 100))
        out[f"var_{level}"] = value_at_risk(ret, confidence=conf)
        out[f"cvar_{level}"] = expected_shortfall(ret, confidence=conf)
    return out


def scenario_stress(
    nav: pl.DataFrame,
    scenarios: list[tuple[str, float, int]] | None = None,
) -> dict[str, float]:
    """Signed predicted strategy impact under each market shock scenario.

    The strategy's market beta is fitted by OLS on the backtest daily returns.
    A scenario that moves the benchmark by ``shock`` for ``days`` consecutive
    days is then expected to move the strategy by roughly ``beta * shock`` per
    day; the reported value is the compounded impact (negative = loss). When
    the NAV frame carries no benchmark, beta is assumed to be 1.
    """
    if scenarios is None:
        scenarios = _DEFAULT_SCENARIOS
    df = _returns(nav)
    strat = _clean(df["ret"])
    beta = 1.0
    if "bench_ret" in df.columns:
        bench = _clean(df["bench_ret"])
        m = ~(np.isnan(np.asarray(df["ret"], dtype=float)) | np.isnan(np.asarray(df["bench_ret"], dtype=float)))
        if m.sum() >= 2 and bench.std(ddof=1) > 0:
            b, _ = np.polyfit(np.asarray(df["bench_ret"], dtype=float)[m], np.asarray(df["ret"], dtype=float)[m], 1)
            beta = float(b)
    out: dict[str, float] = {}
    for label, shock, days in scenarios:
        daily = beta * shock
        out[label] = float((1.0 + daily) ** days - 1.0)
    if len(strat) < 2:
        return {k: float("nan") for k in out}
    return out
