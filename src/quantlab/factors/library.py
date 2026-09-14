"""Built-in factor library.

Each factor is a ``pl.Expr`` over a long panel with ``[date, instrument]`` and
OHLCV columns. The formulas reuse the classic ``alpha158`` factor set from the
original project (momentum, reversal, volatility, volume-price correlation,
RSI / MACD / RSV, K-bar shape, WVMA, ...).

``compute_factors`` evaluates a subset of ``FACTOR_REGISTRY`` against a panel.
"""

from __future__ import annotations

import polars as pl

from quantlab.data.base import DATE, INSTRUMENT
from quantlab.factors import operators as op


def _pos(x: pl.Expr) -> pl.Expr:
    """Element-wise ``max(x, 0)``."""
    return pl.max_horizontal(x, pl.lit(0.0))


def _neg(x: pl.Expr) -> pl.Expr:
    """Element-wise ``abs(min(x, 0))``."""
    return pl.min_horizontal(x, pl.lit(0.0)).abs()


def _rsi(n: int) -> pl.Expr:
    """Wilder's relative strength index."""
    delta = op.ts_delta("close", 1)
    avg_gain = _pos(delta).ewm_mean(alpha=1 / n).over(INSTRUMENT)
    avg_loss = _neg(delta).ewm_mean(alpha=1 / n).over(INSTRUMENT)
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _macd(fast: int = 12, slow: int = 26, signal: int = 9) -> pl.Expr:
    """MACD histogram ``2 * (DIF - DEA)``."""
    ema_fast = pl.col("close").ewm_mean(span=fast).over(INSTRUMENT)
    ema_slow = pl.col("close").ewm_mean(span=slow).over(INSTRUMENT)
    dif = ema_fast - ema_slow
    dea = dif.ewm_mean(span=signal).over(INSTRUMENT)
    return (dif - dea) * 2


def _wvma(n: int) -> pl.Expr:
    """Volume-weighted return volatility / mean."""
    w = op.ts_returns("close", 1).abs() * pl.col("volume")
    return op.ts_std(w, n) / op.ts_mean(w, n)


FACTOR_REGISTRY: dict[str, pl.Expr] = {
    # Price momentum & reversal
    "momentum_20": op.ts_returns("close", 20),
    "momentum_60": op.ts_returns("close", 60),
    "reversal_5": -op.ts_returns("close", 5),
    # Volatility & volume
    "volatility_20": op.ts_std(op.ts_returns("close", 1), 20),
    "volume_ratio_20": op.ts_mean("volume", 20) / op.ts_mean("volume", 60),
    "turnover_20": op.ts_mean("turnover_rate", 20),
    # Price-volume interaction
    "corr_cv_20": op.ts_corr("close", op.log("volume"), 20),
    "beta_oc_20": op.ts_beta("open", "close", 20),
    # Oscillators
    "rsi_14": _rsi(14),
    "macd": _macd(),
    "rsv_20": (pl.col("close") - op.ts_min("low", 20)) / (op.ts_max("high", 20) - op.ts_min("low", 20)),
    # Up/down statistics
    "cntp_20": op.ts_pos_count(op.ts_delta("close", 1), 20) / 20,
    "sump_20": op.ts_sum(_pos(op.ts_delta("close", 1)), 20) / op.ts_sum(op.ts_delta("close", 1).abs(), 20),
    # K-bar shape
    "kbar_kmid": pl.col("close") / pl.col("open") - 1,
    "kbar_klen": (pl.col("high") - pl.col("low")) / pl.col("open"),
    # Volume-weighted volatility
    "wvma_20": _wvma(20),
}


def get_factor(name: str) -> pl.Expr:
    """Return the expression for a single registered factor."""
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"unknown factor '{name}'; available: {sorted(FACTOR_REGISTRY)}")
    return FACTOR_REGISTRY[name]


def compute_factors(df: pl.DataFrame, names: list[str] | None = None) -> pl.DataFrame:
    """Evaluate factors over a panel, returning ``[date, instrument, *factors]``."""
    if names is None:
        names = list(FACTOR_REGISTRY)
    unknown = [n for n in names if n not in FACTOR_REGISTRY]
    if unknown:
        raise KeyError(f"unknown factors: {unknown}; available: {sorted(FACTOR_REGISTRY)}")
    exprs = [FACTOR_REGISTRY[n].alias(n) for n in names]
    return df.select([pl.col(DATE), pl.col(INSTRUMENT), *exprs])
