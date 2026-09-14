"""Backtest account: cash, positions, costs, and mark-to-market NAV.

Reuses the semantics of the original ``account.py`` (cash-constrained buys,
sell-first-then-buy ordering, cumulative cost tracking) but fixes the old
buy/sell fee-swap bug and enforces T+1 through the daily-rebalance structure.

T+1 note: because we rebalance at most once per day and never round-trip a
stock within a single day (a target weight is a single number), every share is
held for at least one full day. This makes ``t_plus_1`` automatically
satisfied, so no separate "available" ledger is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantlab.backtest.rules import TradingRules, buy_fee, round_lot, sell_fee


@dataclass
class Account:
    """A single cash account holding a long-only stock book."""

    cash: float
    rules: TradingRules
    position: dict[str, int] = field(default_factory=dict)
    cost: float = 0.0  # cumulative realized trading cost (fees + taxes)
    trade_value: float = 0.0  # gross traded value since the last reset (for turnover)
    nav_history: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)

    def total_value(self, prices: dict[str, float]) -> float:
        """Mark-to-market: cash + sum(shares * close)."""
        holdings = sum(shares * prices.get(code, 0.0) for code, shares in self.position.items())
        return self.cash + holdings

    def sell(self, code: str, shares: int, price: float, date: str | None = None) -> int:
        """Sell at most ``shares`` (capped by held shares, floored to lot)."""
        if shares <= 0 or code not in self.position:
            return 0
        shares = round_lot(min(shares, self.position[code]), self.rules.lot_size)
        if shares <= 0:
            return 0
        value = shares * price
        fee = sell_fee(value, self.rules)
        self.cash += value - fee
        self.cost += fee
        self.trade_value += value
        self.position[code] -= shares
        if self.position[code] <= 0:
            del self.position[code]
        self.trade_log.append(
            {"date": date, "code": code, "side": "sell", "shares": shares, "price": price, "fee": fee}
        )
        return shares

    def buy(self, code: str, shares: int, price: float, date: str | None = None) -> int:
        """Buy up to ``shares``, reduced to what the remaining cash can afford."""
        if shares <= 0 or price <= 0:
            return 0
        shares = round_lot(shares, self.rules.lot_size)
        if shares <= 0:
            return 0
        # Gross cash limit ignoring the min-commission floor, then correct.
        unit_rate = self.rules.commission_rate + self.rules.transfer_fee
        max_value = self.cash / (1.0 + unit_rate)
        shares = min(shares, int(max_value / price // self.rules.lot_size) * self.rules.lot_size)
        # Handle the min-commission floor exactly (usually a couple of iterations).
        while shares > 0:
            value = shares * price
            if value + buy_fee(value, self.rules) <= self.cash:
                break
            shares -= self.rules.lot_size
        if shares <= 0:
            return 0
        value = shares * price
        fee = buy_fee(value, self.rules)
        self.cash -= value + fee
        self.cost += fee
        self.trade_value += value
        self.position[code] = self.position.get(code, 0) + shares
        self.trade_log.append({"date": date, "code": code, "side": "buy", "shares": shares, "price": price, "fee": fee})
        return shares
