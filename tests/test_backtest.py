"""Account and engine tests: fee arithmetic, lot rounding, and invariants."""

from __future__ import annotations

import polars as pl
import pytest
from conftest import backtest_inputs

from quantlab.backtest import Account, TopKStrategy, TradingRules, run_backtest


def test_buy_floors_to_lot_and_charges_fee():
    rules = TradingRules()
    acct = Account(cash=1_000_000.0, rules=rules)
    shares = acct.buy("000001", 150, 10.0)  # 150 -> 100 (lot)
    assert shares == 100
    assert acct.position["000001"] == 100
    value = 1000.0
    fee = 5.0 + value * rules.transfer_fee  # commission floor 5 + transfer
    assert acct.cash == pytest.approx(1_000_000.0 - value - fee)


def test_sell_charges_stamp_tax():
    rules = TradingRules()
    acct = Account(cash=1_000_000.0, rules=rules)
    acct.position["000001"] = 100
    shares = acct.sell("000001", 100, 10.0)
    assert shares == 100
    assert "000001" not in acct.position
    value = 1000.0
    fee = 5.0 + value * rules.transfer_fee + value * rules.stamp_tax
    assert acct.cash == pytest.approx(1_000_000.0 + value - fee)


def test_buy_cannot_overdraw_cash():
    rules = TradingRules()
    acct = Account(cash=500.0, rules=rules)
    shares = acct.buy("000001", 100, 10.0)  # needs 1000 + fee, cash 500
    assert shares == 0
    assert acct.position.get("000001", 0) == 0


def test_run_backtest_invariants():
    data, pred = backtest_inputs()
    result = run_backtest(pred, data, TopKStrategy(top_k=0.2), TradingRules())
    assert result.nav.columns == ["date", "nav", "benchmark"]
    assert result.nav.shape[0] == 7
    # no negative cash and every position is an integer lot
    assert result.account.cash >= -1e-6
    for shares in result.account.position.values():
        assert shares % 100 == 0
    # a stock is never both bought and sold on the same day
    if len(result.trades):
        sides = result.trades.group_by(["date", "code"]).agg(pl.col("side").n_unique().alias("n_side"))
        assert sides["n_side"].max() <= 1
