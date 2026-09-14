"""Operator tests against hand-computed values on a tiny panel."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from quantlab.data.base import DATE, INSTRUMENT
from quantlab.factors import operators as op


def _series_for(panel: pl.DataFrame, code: str, col: str) -> list:
    return panel.filter(pl.col(INSTRUMENT) == code).sort(DATE)[col].to_list()


def test_ts_mean(panel):
    out = panel.with_columns(op.ts_mean("close", 3).alias("m"))
    assert _series_for(out, "000001", "m") == [None, None, 2.0, 3.0, 4.0]


def test_ts_delay_and_returns(panel):
    out = panel.with_columns(op.ts_returns("close", 1).alias("r"))
    r = _series_for(out, "000001", "r")
    assert r[0] is None
    assert r[1] == pytest.approx(1.0)  # (2-1)/1
    assert r[2] == pytest.approx(0.5)  # (3-2)/2
    assert r[4] == pytest.approx(0.25)  # (5-4)/4


def test_ts_product(panel):
    out = panel.with_columns(op.ts_product("close", 3).alias("p"))
    p = _series_for(out, "000001", "p")
    assert p == [None, None, 6.0, 24.0, 60.0]  # 1*2*3, 2*3*4, 3*4*5


def test_ts_corr_identity_is_one(panel):
    out = panel.with_columns(op.ts_corr("close", "close", 3).alias("c"))
    c = _series_for(out, "000001", "c")
    assert c[:2] == [None, None]
    assert all(x == pytest.approx(1.0) for x in c[2:])


def test_cs_rank_on_last_date(panel):
    out = panel.with_columns(op.cs_rank("close").alias("r"))
    last = out.filter(pl.col(DATE) == date(2020, 1, 5)).sort(INSTRUMENT)
    assert last["r"].to_list() == [pytest.approx(1 / 3), pytest.approx(2 / 3), pytest.approx(1.0)]


def test_cs_zscore_mean_zero(panel):
    out = panel.with_columns(op.cs_zscore("close").alias("z"))
    means = out.group_by(DATE).agg(pl.col("z").mean().alias("m")).sort(DATE)
    for m in means["m"].to_list():
        assert m == pytest.approx(0.0, abs=1e-9)


def test_ts_scale_bounds(panel):
    out = panel.with_columns(op.ts_scale("close", 3).alias("s"))
    s = _series_for(out, "000001", "s")
    assert s[2] == pytest.approx(1.0)  # last of window [1,2,3] is max
    assert s[3] == pytest.approx(1.0)  # [2,3,4] max
    assert s[4] == pytest.approx(1.0)


def test_bigger_smaller_elementwise(panel):
    out = panel.with_columns(
        op.bigger("close", pl.lit(3.0)).alias("b"),
        op.smaller("close", pl.lit(3.0)).alias("s"),
    )
    row = out.filter((pl.col(INSTRUMENT) == "000001") & (pl.col("close") == 1.0))
    assert row["b"][0] == 3.0
    assert row["s"][0] == 1.0
