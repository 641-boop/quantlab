"""Neutralization helpers.

* :func:`market_neutralize` builds a dollar-neutral weight vector per date (the
  WorldQuant-style portfolio), reusing the original ``alpha.market_neutralize``.
* :func:`neutralize` orthogonalizes factors against given targets (e.g. market
  cap) by cross-sectional residual regression, reusing ``operators.cs_resid``.
"""

from __future__ import annotations

import polars as pl

from quantlab.data.base import DATE, INSTRUMENT
from quantlab.factors import operators as op


def _e(x: str | pl.Expr) -> pl.Expr:
    return pl.col(x) if isinstance(x, str) else x


def market_neutralize(x: str | pl.Expr, long_only: bool = False) -> pl.Expr:
    """Cross-sectionally demean ``x`` then scale by its absolute sum.

    Returns a weight vector whose per-date sum is 0 and absolute sum is 1
    (so ``0.5`` means half a position long, ``-0.25`` a quarter short).
    """
    x = _e(x)
    x = x - x.mean().over(DATE)
    w = x / x.abs().sum().over(DATE)
    if long_only:
        w = pl.when(w < 0).then(0.0).otherwise(w) * 2
    return w


def neutralize(df: pl.DataFrame, factor_cols: list[str], target_cols: list[str]) -> pl.DataFrame:
    """Regress each factor on ``target_cols`` cross-sectionally and keep residuals.

    Targets are orthogonalized sequentially. Returns ``[date, instrument, *factor_cols]``.
    """
    out = df.select([pl.col(DATE), pl.col(INSTRUMENT)])
    for f in factor_cols:
        resid: pl.Expr = pl.col(f)
        for t in target_cols:
            resid = op.cs_resid(t, resid)
        out = out.with_columns(resid.alias(f))
    return out
