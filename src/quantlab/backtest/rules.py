"""A-share trading rules: fees, lots, price limits, and T+1.

Every rule here is parameterized so the backtest stays transparent and can be
reconfigured from :class:`quantlab.config.BacktestConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradingRules:
    """Concrete trading-cost / microstructure model for the A-share market."""

    commission_rate: float = 2.5e-4  # 佣金 万2.5
    min_commission: float = 5.0  # 佣金下限 5 元
    stamp_tax: float = 1e-3  # 印花税 卖出 0.1%
    transfer_fee: float = 1e-5  # 过户费 万0.1（双边）
    slippage: float = 5e-4  # 滑点 万5（成交价相对收盘价）
    lot_size: int = 100  # 100 股为一手，买入必须为整数手
    t_plus_1: bool = True  # 当日买入次日方可卖出


def limit_ratio(instrument: str) -> float:
    """Daily price-limit band for a given instrument code.

    - 创业板 (300xxx) / 科创板 (688xxx): 20%
    - 北交所 (8xxxxx / 4xxxxx): 30%
    - 主板 / 其余: 10%
    """
    code = str(instrument)
    if code.startswith(("300", "688")):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def commission(value: float, rules: TradingRules) -> float:
    """Commission, floored at ``min_commission``."""
    return max(rules.min_commission, value * rules.commission_rate)


def buy_fee(value: float, rules: TradingRules) -> float:
    """Buy-side cost: commission + transfer fee (no stamp tax on buys)."""
    return commission(value, rules) + value * rules.transfer_fee


def sell_fee(value: float, rules: TradingRules) -> float:
    """Sell-side cost: commission + transfer fee + stamp tax."""
    return commission(value, rules) + value * rules.transfer_fee + value * rules.stamp_tax


def round_lot(shares: float, lot_size: int = 100) -> int:
    """Floor a share count to an integer multiple of the lot size."""
    return int(shares // lot_size) * lot_size
