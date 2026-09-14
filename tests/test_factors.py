"""Factor library and evaluation tests on synthetic data."""

from __future__ import annotations

import polars as pl
import pytest

from quantlab.data.base import DATE, INSTRUMENT
from quantlab.factors.evaluate import forward_return, group_return, ic_summary
from quantlab.factors.library import compute_factors, get_factor


def test_compute_factors_shape(panel):
    names = ["momentum_20", "reversal_5", "kbar_kmid"]
    out = compute_factors(panel, names)
    assert out.columns == [DATE, INSTRUMENT, *names]
    assert out.shape[0] == panel.shape[0]


def test_unknown_factor_raises():
    with pytest.raises(KeyError):
        get_factor("does_not_exist")


def test_forward_return(panel):
    lab = forward_return(panel, 1)
    row = lab.filter(pl.col(INSTRUMENT) == "000001").sort(DATE)
    labels = row["label"].to_list()
    assert labels[0] == pytest.approx(1.0)  # 2/1 - 1
    assert labels[-1] is None  # no future on the last date


def test_ic_summary_perfect_signal(panel):
    df = panel.with_columns(pl.col("close").alias("factor"), pl.col("close").alias("label"))
    s = ic_summary(df, "factor", "label")
    assert s["ic_mean"] == pytest.approx(1.0)
    assert s["rank_ic_mean"] == pytest.approx(1.0)


def test_group_return_has_long_short(panel):
    df = panel.with_columns(pl.col("close").alias("factor"), pl.col("close").alias("label"))
    g = group_return(df, "factor", "label", n=10)
    assert "long-short" in g.columns
    assert "Group1" in g.columns and "Group10" in g.columns


def test_ic_summary_ignores_warmup_nan(panel):
    # ts_* factors are null during their lookback window; those dates must be
    # dropped before mean/std, otherwise NaN propagates through the whole IC series.
    df = panel.with_columns(pl.col("close").alias("factor"), pl.col("close").alias("label"))
    first_date = df[DATE].min()
    df = df.with_columns(
        pl.when(pl.col(DATE) == first_date)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("factor"))
        .alias("factor")
    )
    s = ic_summary(df, "factor", "label")
    assert s["ic_mean"] == pytest.approx(1.0)
    assert s["rank_ic_mean"] == pytest.approx(1.0)
