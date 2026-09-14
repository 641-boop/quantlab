"""Performance metrics.

All formulas mirror the original ``report.py`` (Sharpe / Sortino / information
ratio / max drawdown / annualized return & volatility / beta-alpha), re-expressed
against polars columns.

Conventions
-----------
- Returns are daily and not annualized unless a ``periods_per_year`` is applied.
- The risk-free rate ``rf`` is an annualized decimal (e.g. ``0.02``).
"""

from __future__ import annotations

import numpy as np
import polars as pl

PERIODS_PER_YEAR = 252


def daily_returns(nav: pl.DataFrame, value_col: str = "nav") -> pl.DataFrame:
    """Add a ``ret`` column of daily simple returns to the NAV frame."""
    return nav.with_columns((pl.col(value_col) / pl.col(value_col).shift(1) - 1).alias("ret"))


def annualized_return(nav: pl.Series | np.ndarray, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Compound annualized return from an equity (NAV) series."""
    v = np.asarray(nav, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2 or v[0] <= 0:
        return float("nan")
    n = len(v) - 1
    return float((v[-1] / v[0]) ** (periods_per_year / n) - 1)


def annualized_volatility(ret: pl.Series | np.ndarray, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized volatility of a daily return series."""
    r = np.asarray(ret, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(ret: pl.Series | np.ndarray, rf: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe ratio against a risk-free rate."""
    r = np.asarray(ret, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    rf_daily = rf / periods_per_year
    excess = r - rf_daily
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(ret: pl.Series | np.ndarray, rf: float = 0.0, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    r = np.asarray(ret, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    rf_daily = rf / periods_per_year
    excess = r - rf_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("nan")
    dd = np.sqrt((downside**2).mean())
    if dd == 0:
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(nav: pl.Series | np.ndarray) -> float:
    """Maximum peak-to-trough drawdown (positive magnitude, e.g. ``0.23`` = 23%)."""
    v = np.asarray(nav, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    peak = np.maximum.accumulate(v)
    dd = (v - peak) / peak
    return float(-dd.min())


def drawdown_series(nav: pl.Series | np.ndarray) -> np.ndarray:
    """Peak-to-trough drawdown at every point (negative values)."""
    v = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(v)
    return (v - peak) / peak


def information_ratio(strategy_ret: pl.Series | np.ndarray, benchmark_ret: pl.Series | np.ndarray) -> float:
    """Annualized information ratio of active returns vs. a benchmark."""
    s = np.asarray(strategy_ret, dtype=float)
    b = np.asarray(benchmark_ret, dtype=float)
    m = ~(np.isnan(s) | np.isnan(b))
    s, b = s[m], b[m]
    if len(s) < 2:
        return float("nan")
    active = s - b
    std = active.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(active.mean() / std * np.sqrt(PERIODS_PER_YEAR))


def calmar_ratio(nav: pl.Series | np.ndarray, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized return over max drawdown."""
    ann = annualized_return(nav, periods_per_year)
    mdd = max_drawdown(nav)
    if mdd == 0 or np.isnan(mdd) or np.isnan(ann):
        return float("nan")
    return float(ann / mdd)


def win_rate(ret: pl.Series | np.ndarray) -> float:
    """Fraction of trading days with a positive return."""
    r = np.asarray(ret, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def beta_alpha(strategy_ret: pl.Series | np.ndarray, benchmark_ret: pl.Series | np.ndarray) -> tuple[float, float]:
    """OLS beta and annualized alpha of strategy returns against a benchmark."""
    s = np.asarray(strategy_ret, dtype=float)
    b = np.asarray(benchmark_ret, dtype=float)
    m = ~(np.isnan(s) | np.isnan(b))
    s, b = s[m], b[m]
    if len(s) < 2 or b.std(ddof=1) == 0:
        return float("nan"), float("nan")
    beta, alpha_daily = np.polyfit(b, s, 1)
    alpha = (1.0 + alpha_daily) ** PERIODS_PER_YEAR - 1.0
    return float(beta), float(alpha)


def summary(nav: pl.DataFrame, rf: float = 0.0) -> dict[str, float]:
    """Compute a full metrics dict from a ``[date, nav, benchmark]`` frame.

    Missing-benchmark frames degrade gracefully (benchmark metrics become NaN).
    """
    nav = daily_returns(nav)
    if "benchmark" in nav.columns:
        nav = nav.with_columns((pl.col("benchmark") / pl.col("benchmark").shift(1) - 1).alias("bench_ret"))

    ret = nav["ret"]
    nav_first = float(nav["nav"].drop_nulls().head(1).item())
    nav_last = float(nav["nav"].drop_nulls().tail(1).item())
    out: dict[str, float] = {
        "total_return": float(nav_last / nav_first - 1.0),
        "annualized_return": annualized_return(nav["nav"]),
        "annualized_volatility": annualized_volatility(ret),
        "sharpe_ratio": sharpe_ratio(ret, rf=rf),
        "sortino_ratio": sortino_ratio(ret, rf=rf),
        "max_drawdown": max_drawdown(nav["nav"]),
        "calmar_ratio": calmar_ratio(nav["nav"]),
        "win_rate": win_rate(ret),
    }
    if "bench_ret" in nav.columns:
        out["information_ratio"] = information_ratio(ret, nav["bench_ret"])
        beta, alpha = beta_alpha(ret, nav["bench_ret"])
        out["beta"] = beta
        out["alpha"] = alpha
    return out
