"""Daily-loop backtest engine.

The loop follows the skeleton of the original ``Executor.execute``:

    for each trading day:
        target weights  ->  target shares
        sells first (respect T+1 / limit-down)  ->  frees cash
        buys in priority order (respect cash / limit-up)
        mark-to-market  ->  NAV + benchmark

Trades execute at the *close* of the signal day using a signal computed only
from that day's information, so there is no lookahead: the position earns the
return from today's close to tomorrow's close.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from quantlab.backtest.account import Account
from quantlab.backtest.rules import TradingRules, limit_ratio, round_lot
from quantlab.backtest.strategy import TopKStrategy
from quantlab.data.base import DATE, INSTRUMENT


@dataclass
class BacktestResult:
    """Everything the engine produces for downstream metrics and reporting."""

    nav: pl.DataFrame  # [date, nav, benchmark]
    account: Account
    trades: pl.DataFrame  # [date, code, side, shares, price, fee]
    turnover: pl.DataFrame  # [date, turnover] daily gross turnover

    def __post_init__(self) -> None:
        self.trades = self.trades.sort([DATE, "code"])


def _limit_sets(day: pl.DataFrame) -> tuple[set[str], set[str]]:
    """Return (limit_up_codes, limit_down_codes) for one day, using prev close."""
    up: set[str] = set()
    down: set[str] = set()
    for code, close, prev in day.select([INSTRUMENT, "close", "prev_close"]).iter_rows():
        if prev is None or prev <= 0:
            continue
        band = limit_ratio(code)
        if close >= prev * (1.0 + band) - 1e-6:
            up.add(code)
        elif close <= prev * (1.0 - band) + 1e-6:
            down.add(code)
    return up, down


def _equal_weight_benchmark(data: pl.DataFrame) -> pl.DataFrame:
    """Equal-weight buy-and-hold NAV over the whole universe (rebalanced daily)."""
    ret = (
        data.sort([INSTRUMENT, DATE])
        .with_columns((pl.col("close") / pl.col("close").shift(1).over(INSTRUMENT) - 1).alias("ret"))
        .group_by(DATE)
        .agg(pl.col("ret").mean().alias("bench_ret"))
        .sort(DATE)
    )
    return ret.select(DATE, (1.0 + pl.col("bench_ret").fill_null(0.0)).cum_prod().alias("benchmark"))


def run_backtest(
    pred: pl.DataFrame,
    data: pl.DataFrame,
    strategy: TopKStrategy | None = None,
    rules: TradingRules | None = None,
    init_cash: float = 1e8,
    benchmark: str = "equal",
) -> BacktestResult:
    """Run a daily close-to-close backtest.

    Parameters
    ----------
    pred: ``[date, instrument, predict]`` out-of-sample predictions.
    data: OHLCV panel (at least ``[date, instrument, close]``).
    """
    strategy = strategy or TopKStrategy()
    rules = rules or TradingRules()

    price = data.select([DATE, INSTRUMENT, "close"]).sort([DATE, INSTRUMENT])
    price = price.with_columns(pl.col("close").shift(1).over(INSTRUMENT).alias("prev_close"))
    merged = pred.join(price, on=[DATE, INSTRUMENT], how="inner").sort([DATE, INSTRUMENT])

    account = Account(cash=init_cash, rules=rules)
    dates = merged[DATE].unique().sort().to_list()

    nav_rows: list[tuple] = []
    turnover_rows: list[tuple] = []

    for date in dates:
        day = merged.filter(pl.col(DATE) == date)
        prices = dict(day.select([INSTRUMENT, "close"]).iter_rows())
        limit_up, limit_down = _limit_sets(day)

        account.trade_value = 0.0
        pre_value = account.total_value(prices)

        # 1. target shares from target weights.
        target_w = strategy.target_weights(day)
        target_shares: dict[str, int] = {}
        for code, w in target_w.items():
            px = prices.get(code)
            if px and px > 0:
                target_shares[code] = round_lot(w * pre_value / px, rules.lot_size)

        # 2. sells first (free cash). Skip names locked at limit-down.
        for code in list(account.position):
            target = target_shares.get(code, 0)
            held = account.position[code]
            if held > target and code not in limit_down:
                sell_px = prices[code] * (1.0 - rules.slippage)
                account.sell(code, held - target, sell_px, date=str(date))

        # 3. buys in descending prediction order (respect cash, skip limit-up).
        buy_list = [
            (code, target_shares[code] - account.position.get(code, 0))
            for code in target_shares
            if account.position.get(code, 0) < target_shares[code]
        ]
        pred_rank = dict(day.select([INSTRUMENT, "predict"]).iter_rows())
        buy_list.sort(key=lambda item: -float(pred_rank.get(item[0], -1e9)))
        for code, need in buy_list:
            if code in limit_up:
                continue
            buy_px = prices[code] * (1.0 + rules.slippage)
            account.buy(code, need, buy_px, date=str(date))

        # 4. mark to market.
        nav = account.total_value(prices)
        account.nav_history.append(nav)
        nav_rows.append((date, nav))
        turnover_rows.append((date, account.trade_value / pre_value if pre_value else 0.0))

    nav_df = pl.DataFrame(nav_rows, schema=[DATE, "nav"], orient="row").with_columns(pl.col(DATE).cast(pl.Date))

    benchmark_df = _equal_weight_benchmark(data)
    result_nav = nav_df.join(benchmark_df, on=DATE, how="left").sort(DATE)

    # Normalize both curves to 1.0 at the first backtest date for comparison.
    nav0 = float(result_nav["nav"].head(1).item())
    bench0 = float(result_nav["benchmark"].drop_nulls().head(1).item())
    if nav0 > 0 and bench0 > 0:
        result_nav = result_nav.with_columns(
            (pl.col("nav") / nav0).alias("nav"),
            (pl.col("benchmark") / bench0).alias("benchmark"),
        )

    trades = (
        pl.DataFrame(account.trade_log)
        if account.trade_log
        else pl.DataFrame(
            schema={
                DATE: pl.Utf8,
                "code": pl.Utf8,
                "side": pl.Utf8,
                "shares": pl.Int64,
                "price": pl.Float64,
                "fee": pl.Float64,
            }
        )
    )
    if len(trades):
        trades = trades.with_columns(pl.col(DATE).cast(pl.Utf8))

    turnover_df = pl.DataFrame(turnover_rows, schema=[DATE, "turnover"], orient="row").with_columns(
        pl.col(DATE).cast(pl.Date)
    )

    return BacktestResult(nav=result_nav, account=account, trades=trades, turnover=turnover_df)
