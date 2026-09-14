"""Time-series and cross-sectional operators over a long polars panel.

Operators take a column name (``str``) or a ``pl.Expr`` and return a ``pl.Expr``
that already carries the right grouping:

* ``ts_*`` operate **per instrument** over time (``.over(INSTRUMENT)``).
* ``cs_*`` operate **per date** across instruments (``.over(DATE)``).

They are meant to be composed inside :meth:`pl.DataFrame.with_columns`::

    df.with_columns(ts_zscore("close", 20).alias("close_zscore20"))

The panel must be sorted by ``(date, instrument)`` (see
:func:`quantlab.data.base.ensure_schema`).

Semantics follow the original ``scutquant/operators.py``; a few known bugs in the
original (``ts_product``, ``bigger``/``smaller``) are fixed here.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from quantlab.data.base import DATE, INSTRUMENT

Expr = pl.Expr


def _e(x: str | Expr) -> Expr:
    """Coerce a column name or an expression into a ``pl.Expr``."""
    return pl.col(x) if isinstance(x, str) else x


# --------------------------------------------------------------------------- #
# Time-series operators (per instrument)
# --------------------------------------------------------------------------- #


def ts_delay(x: str | Expr, n: int) -> Expr:
    """Value ``n`` periods ago."""
    return _e(x).shift(n).over(INSTRUMENT)


def ts_delta(x: str | Expr, n: int) -> Expr:
    """``x - ts_delay(x, n)``."""
    x = _e(x)
    return x - ts_delay(x, n)


def ts_returns(x: str | Expr, n: int) -> Expr:
    """Relative change ``ts_delta(x, n) / ts_delay(x, n)``."""
    return ts_delta(x, n) / ts_delay(x, n)


def ts_sum(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_sum(window_size=n).over(INSTRUMENT)


def ts_product(x: str | Expr, n: int) -> Expr:
    """Product of the past ``n`` values (null until ``n`` observations)."""
    x = _e(x)
    cp = x.cum_prod().over(INSTRUMENT)
    lag = cp.shift(n).over(INSTRUMENT)
    divisor = lag.fill_null(1.0)  # empty product == 1 for the first full window
    prod = cp / divisor
    enough = cp.shift(n - 1).over(INSTRUMENT).is_not_null()
    return pl.when(enough).then(prod).otherwise(None)


def ts_max(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_max(window_size=n).over(INSTRUMENT)


def ts_min(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_min(window_size=n).over(INSTRUMENT)


def ts_mean(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_mean(window_size=n).over(INSTRUMENT)


def ts_median(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_median(window_size=n).over(INSTRUMENT)


def ts_std(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_std(window_size=n).over(INSTRUMENT)


def ts_variance(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_var(window_size=n).over(INSTRUMENT)


def ts_skew(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_skew(window_size=n).over(INSTRUMENT)


def ts_kurt(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_kurtosis(window_size=n).over(INSTRUMENT)


def ts_ewma(x: str | Expr, alpha: float) -> Expr:
    return _e(x).ewm_mean(alpha=alpha).over(INSTRUMENT)


def ts_quantile_up(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_quantile(quantile=0.75, window_size=n).over(INSTRUMENT)


def ts_quantile_down(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_quantile(quantile=0.25, window_size=n).over(INSTRUMENT)


def ts_rank(x: str | Expr, n: int) -> Expr:
    """Percentile rank of the current value within the past ``n`` values."""
    return _e(x).rolling_map(lambda w: _pct_rank(w[-1], w.to_numpy()), window_size=n).over(INSTRUMENT)


def ts_argmax(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_map(lambda w: float(np.argmax(w.to_numpy())), window_size=n).over(INSTRUMENT)


def ts_argmin(x: str | Expr, n: int) -> Expr:
    return _e(x).rolling_map(lambda w: float(np.argmin(w.to_numpy())), window_size=n).over(INSTRUMENT)


def ts_pos_count(x: str | Expr, n: int) -> Expr:
    """Number of positive values in the past ``n`` periods."""
    return (_e(x) > 0).cast(pl.Float64).rolling_sum(window_size=n).over(INSTRUMENT)


def ts_neg_count(x: str | Expr, n: int) -> Expr:
    """Number of negative values in the past ``n`` periods."""
    return (_e(x) < 0).cast(pl.Float64).rolling_sum(window_size=n).over(INSTRUMENT)


def ts_decay(x: str | Expr, n: int) -> Expr:
    """Linearly-decaying weighted mean over the past ``n`` values."""
    weights = np.arange(1, n + 1, dtype=np.float64)
    weights /= weights.sum()
    return _e(x).rolling_mean(window_size=n, weights=weights.tolist()).over(INSTRUMENT)


def ts_dstd(x: str | Expr, n: int) -> Expr:
    """Downside (only positive) standard deviation over the past ``n`` values."""

    def _downside_std(w: np.ndarray) -> float:
        w = w[w > 0]
        return float(np.std(w)) if len(w) > 1 else 0.0

    return _e(x).rolling_map(lambda w: _downside_std(w.to_numpy()), window_size=n, min_samples=2).over(INSTRUMENT)


def ts_zscore(x: str | Expr, n: int) -> Expr:
    x = _e(x)
    return (x - ts_mean(x, n)) / ts_std(x, n)


def ts_robust_zscore(x: str | Expr, n: int) -> Expr:
    x = _e(x)
    med = ts_median(x, n)
    mad = (x - med).abs().rolling_median(window_size=n).over(INSTRUMENT)
    return (x - med) / (mad * 1.4826)


def ts_scale(x: str | Expr, n: int) -> Expr:
    x = _e(x)
    return (x - ts_min(x, n)) / (ts_max(x, n) - ts_min(x, n))


def ts_sharpe(x: str | Expr, n: int) -> Expr:
    return ts_mean(x, n) / ts_std(x, n)


def ts_av_diff(x: str | Expr, n: int) -> Expr:
    return _e(x) - ts_mean(x, n)


def ts_max_diff(x: str | Expr, n: int) -> Expr:
    return _e(x) - ts_max(x, n)


def ts_min_diff(x: str | Expr, n: int) -> Expr:
    return _e(x) - ts_min(x, n)


def ts_corr(x1: str | Expr, x2: str | Expr, n: int) -> Expr:
    return pl.rolling_corr(_e(x1), _e(x2), window_size=n).over(INSTRUMENT)


def ts_cov(x1: str | Expr, x2: str | Expr, n: int) -> Expr:
    return pl.rolling_cov(_e(x1), _e(x2), window_size=n).over(INSTRUMENT)


def ts_beta(x1: str | Expr, x2: str | Expr, n: int) -> Expr:
    """Rolling beta of ``x2`` regressed on ``x1``."""
    return ts_cov(x1, x2, n) / ts_variance(x1, n)


def ts_regression(x1: str | Expr, x2: str | Expr, n: int, rettype: int = 0) -> Expr:
    """Rolling linear model ``x2 = beta * x1 + alpha + resid``.

    ``rettype``: 0=resid, 1=beta, 2=alpha, 3=y_hat, 4=R^2.
    """
    beta = ts_beta(x1, x2, n)
    alpha = ts_mean(x2, n) - beta * ts_mean(x1, n)
    predict = beta * _e(x1) + alpha
    if rettype == 0:
        return _e(x2) - predict
    if rettype == 1:
        return beta
    if rettype == 2:
        return alpha
    if rettype == 3:
        return predict
    return ts_corr(predict, x2, n) ** 2


def ts_ffill(x: str | Expr) -> Expr:
    return _e(x).forward_fill().over(INSTRUMENT)


def ts_backfill(x: str | Expr) -> Expr:
    return _e(x).backward_fill().over(INSTRUMENT)


# --------------------------------------------------------------------------- #
# Cross-sectional operators (per date)
# --------------------------------------------------------------------------- #


def cs_rank(x: str | Expr) -> Expr:
    """Rank within the date cross-section, mapped to ``(0, 1]``."""
    return (_e(x).rank(method="average") / pl.len()).over(DATE)


def cs_zscore(x: str | Expr) -> Expr:
    x = _e(x)
    return (x - x.mean().over(DATE)) / x.std().over(DATE)


def cs_robust_zscore(x: str | Expr) -> Expr:
    x = _e(x)
    med = x.median().over(DATE)
    mad = (x - med).abs().median().over(DATE)
    return (x - med) / (mad * 1.4826)


def cs_scale(x: str | Expr) -> Expr:
    x = _e(x)
    return (x - x.min().over(DATE)) / (x.max().over(DATE) - x.min().over(DATE))


def cs_mean(x: str | Expr) -> Expr:
    return _e(x).mean().over(DATE)


def cs_std(x: str | Expr) -> Expr:
    return _e(x).std().over(DATE)


def cs_variance(x: str | Expr) -> Expr:
    return _e(x).var().over(DATE)


def cs_cov(x1: str | Expr, x2: str | Expr) -> Expr:
    return pl.cov(_e(x1), _e(x2)).over(DATE)


def cs_corr(x1: str | Expr, x2: str | Expr) -> Expr:
    return pl.corr(_e(x1), _e(x2)).over(DATE)


def cs_beta(x1: str | Expr, x2: str | Expr) -> Expr:
    """Cross-sectional beta of ``x2`` regressed on ``x1``."""
    return cs_cov(x1, x2) / cs_variance(x1)


def cs_alpha(x1: str | Expr, x2: str | Expr) -> Expr:
    return cs_mean(x2) - cs_mean(x1) * cs_beta(x1, x2)


def cs_resid(x1: str | Expr, x2: str | Expr) -> Expr:
    """Cross-sectional residual of ``x2`` regressed on ``x1``."""
    beta = cs_beta(x1, x2)
    alpha = cs_mean(x2) - cs_mean(x1) * beta
    return _e(x2) - _e(x1) * beta - alpha


def cs_shrink(x: str | Expr) -> Expr:
    """Winsorize values outside ``[-3, 3]`` toward the extremes (soft shrink)."""
    x = _e(x)
    return x.clip(-3.0, 3.0)


# --------------------------------------------------------------------------- #
# Element-wise helpers
# --------------------------------------------------------------------------- #


def demean(x: str | Expr) -> Expr:
    return _e(x) - cs_mean(x)


def mean(x1: str | Expr, x2: str | Expr) -> Expr:
    return (_e(x1) + _e(x2)) / 2


def sign(x: str | Expr) -> Expr:
    return _e(x).sign()


def abs_(x: str | Expr) -> Expr:
    return _e(x).abs()


def sign_power(x: str | Expr, p: float) -> Expr:
    x = _e(x)
    return x.sign() * x.abs().pow(p)


def log(x: str | Expr) -> Expr:
    return _e(x).log()


def tanh(x: str | Expr) -> Expr:
    return _e(x).tanh()


def sigmoid(x: str | Expr) -> Expr:
    return 1.0 / (1.0 + (-_e(x)).exp())


def bigger(x1: str | Expr, x2: str | Expr) -> Expr:
    """Element-wise maximum."""
    return pl.max_horizontal(_e(x1), _e(x2))


def smaller(x1: str | Expr, x2: str | Expr) -> Expr:
    """Element-wise minimum."""
    return pl.min_horizontal(_e(x1), _e(x2))


def inf_mask(x: str | Expr) -> Expr:
    """Replace ``+/-inf`` with null."""
    return pl.when(_e(x).is_infinite()).then(None).otherwise(_e(x))


def mad_winsor(x: str | Expr, n: float = 3.0) -> Expr:
    """Winsorize per date cross-section at ``median ± n * 1.4826 * MAD``."""
    x = _e(x)
    med = x.median().over(DATE)
    mad = (x - med).abs().median().over(DATE)
    up = med + n * mad * 1.4826
    down = med - n * mad * 1.4826
    return pl.when(x > up).then(up).when(x < down).then(down).otherwise(x)


def _pct_rank(value: float, window: np.ndarray) -> float:
    """Fraction of the window ``<= value`` (simple percentile rank)."""
    if np.isnan(value):
        return float("nan")
    return float(np.mean(window <= value))
