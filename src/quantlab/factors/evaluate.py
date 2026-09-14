"""Factor evaluation: IC / RankIC / layered returns / turnover.

These metrics quantify signal quality before any trading-constraint backtest.
Formulas reuse the original ``alpha.get_factor_metrics`` and
``report.group_return_ana``.
"""

from __future__ import annotations

from typing import cast

import polars as pl

from quantlab.data.base import DATE, INSTRUMENT
from quantlab.factors import neutral
from quantlab.factors import operators as op


def forward_return(df: pl.DataFrame, horizon: int = 1) -> pl.DataFrame:
    """Label = next-``horizon``-day return, ``close[t+h] / close[t] - 1`` per instrument."""
    ret = (pl.col("close").shift(-horizon).over(INSTRUMENT) / pl.col("close") - 1).alias("label")
    return df.select([pl.col(DATE), pl.col(INSTRUMENT), ret])


def standardize(df: pl.DataFrame, cols: list[str], method: str = "zscore") -> pl.DataFrame:
    """Cross-sectionally standardize ``cols`` in place (zscore / rank / scale / robust_zscore)."""
    out = df
    for c in cols:
        if method == "zscore":
            e = op.cs_zscore(c)
        elif method == "robust_zscore":
            e = op.cs_robust_zscore(c)
        elif method == "rank":
            e = op.cs_rank(c)
        elif method == "scale":
            e = op.cs_scale(c)
        else:
            raise ValueError(f"unknown normalize method '{method}'")
        out = out.with_columns(e.alias(c))
    return out


def ic_series(df: pl.DataFrame, factor_col: str, label_col: str = "label") -> pl.DataFrame:
    """Per-date Pearson IC between a factor and the label."""
    return df.group_by(DATE).agg(pl.corr(pl.col(factor_col), pl.col(label_col)).alias("ic")).sort(DATE)


def rank_ic_series(df: pl.DataFrame, factor_col: str, label_col: str = "label") -> pl.DataFrame:
    """Per-date Spearman (rank) IC."""
    r = df.with_columns(
        op.cs_rank(factor_col).alias("_fr"),
        op.cs_rank(label_col).alias("_lr"),
    )
    return r.group_by(DATE).agg(pl.corr("_fr", "_lr").alias("rank_ic")).sort(DATE)


def ic_summary(df: pl.DataFrame, factor_col: str, label_col: str = "label") -> dict[str, float]:
    """Mean IC / ICIR / t-stat plus their rank-IC counterparts."""
    # Drop the warm-up days (factor lookback -> NaN/None IC) before aggregating,
    # otherwise ``mean`` / ``std`` would propagate NaN through the whole series.
    ic = ic_series(df, factor_col, label_col).filter(pl.col("ic").is_not_nan())["ic"]
    ric = rank_ic_series(df, factor_col, label_col).filter(pl.col("rank_ic").is_not_nan())["rank_ic"]
    ic_mean = cast(float, ic.mean())
    ic_std = cast(float, ic.std())
    icir = ic_mean / ic_std if ic_std > 0 else float("nan")
    ric_mean = cast(float, ric.mean())
    ric_std = cast(float, ric.std())
    ricir = ric_mean / ric_std if ric_std > 0 else float("nan")
    return {
        "ic_mean": ic_mean,
        "icir": icir,
        "t_stat": icir * (len(ic) ** 0.5),
        "rank_ic_mean": ric_mean,
        "rank_icir": ricir,
    }


def group_return(df: pl.DataFrame, factor_col: str, label_col: str = "label", n: int = 10) -> pl.DataFrame:
    """Layered returns: split each date into ``n`` groups by factor value.

    Group 1 holds the highest factor values. Adds a ``long-short`` column.
    """
    g = df.with_columns((op.cs_rank(factor_col) * n).ceil().cast(pl.Int64).alias("_grp"))
    grp_mean = g.group_by([DATE, "_grp"]).agg(pl.col(label_col).mean().alias("ret"))
    wide = grp_mean.pivot(index=DATE, on="_grp", values="ret", aggregate_function="first").sort(DATE)
    # normalize group column names and pad any empty group (fewer names than groups).
    for i in range(1, n + 1):
        src, dst = str(i), f"Group{i}"
        if src in wide.columns:
            wide = wide.rename({src: dst})
        else:
            wide = wide.with_columns(pl.lit(None, dtype=pl.Float64).alias(dst))
    wide = wide.with_columns((pl.col("Group1") - pl.col(f"Group{n}")).alias("long-short"))
    return wide.select([DATE, *[f"Group{i}" for i in range(1, n + 1)], "long-short"])


def factor_turnover(df: pl.DataFrame, factor_col: str) -> pl.DataFrame:
    """Daily turnover of the dollar-neutral factor portfolio."""
    w = neutral.market_neutralize(factor_col)
    t = df.with_columns(w.alias("_w"))
    t = t.with_columns(pl.col("_w").shift(1).over(INSTRUMENT).alias("_w_prev"))
    t = t.with_columns((pl.col("_w") - pl.col("_w_prev")).abs().fill_null(0).alias("_to"))
    return t.group_by(DATE).agg(pl.col("_to").sum().alias("turnover")).sort(DATE)


def factor_portfolio(
    df: pl.DataFrame, factor_col: str, label_col: str = "label", long_only: bool = False
) -> pl.DataFrame:
    """Cumulative NAV of the market-neutral (or long-only) factor portfolio."""
    w = neutral.market_neutralize(factor_col, long_only=long_only)
    t = df.with_columns(w.alias("_w"))
    t = t.with_columns((pl.col("_w") * pl.col(label_col)).alias("_fr"))
    daily = t.group_by(DATE).agg(pl.col("_fr").sum().alias("ret")).sort(DATE)
    return daily.with_columns((1 + pl.col("ret")).cum_prod().alias("nav"))
