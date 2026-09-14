"""Generate a synthetic A-share style panel for offline demos and CI.

The panel embeds a mild momentum signal: every instrument carries a persistent
daily drift ``mu_i`` (its own "alpha"), so past cumulative returns (the momentum
factors) are genuinely informative about the next day's return. This lets the
whole research pipeline — factors → model → backtest → report — run end-to-end
without any network access, which is handy when the akshare upstream is
unreachable (e.g. a broken proxy) and keeps CI hermetic.

Usage:

    uv run python examples/make_synthetic.py --n-stocks 30
    uv run python examples/run_research.py --config configs/synthetic.yaml
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import typer

from quantlab.data import DATE, INSTRUMENT, cache_key, save_cache
from quantlab.data.base import ensure_schema

app = typer.Typer(help="生成合成行情面板（离线演示 / CI 用）")


def _business_days(start: str, end: str) -> list:
    """Weekdays between ``start`` and ``end`` (inclusive), ignoring holidays."""
    days = pl.date_range(date.fromisoformat(start), date.fromisoformat(end), interval="1d", eager=True)
    return [d for d in days.to_list() if d.weekday() < 5]


@app.command()
def make(
    start: str = typer.Option("2019-01-01", help="起始日期 YYYY-MM-DD"),
    end: str = typer.Option("2023-12-31", help="结束日期 YYYY-MM-DD"),
    n_stocks: int = typer.Option(30, help="合成股票数量"),
    seed: int = typer.Option(42, help="随机种子"),
    out_dir: str = typer.Option("data", help="输出缓存目录"),
) -> None:
    rng = np.random.default_rng(seed)
    codes = [f"{i:06d}" for i in range(1, n_stocks + 1)]
    days = _business_days(start, end)
    t = len(days)

    frames: list[pl.DataFrame] = []
    for code in codes:
        mu = float(rng.normal(0.0, 0.004))  # persistent daily drift -> the momentum signal
        sigma = float(rng.uniform(0.015, 0.03))  # per-stock daily volatility
        log_ret = mu + sigma * rng.standard_normal(t)
        close = 100.0 * np.exp(np.cumsum(log_ret))

        open_ = np.empty(t)
        open_[0] = close[0]
        open_[1:] = close[:-1] * (1.0 + rng.normal(0.0, 0.002, t - 1))

        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, t)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, t)))
        volume = 1e7 * (1.0 + 20.0 * np.abs(log_ret) + rng.normal(0.0, 0.1, t))
        turnover_rate = np.clip(volume / 2e9 * 100.0, 0.05, 15.0)  # percent
        amount = close * volume

        frames.append(
            pl.DataFrame(
                {
                    DATE: days,
                    INSTRUMENT: [code] * t,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": amount,
                    "turnover_rate": turnover_rate,
                }
            )
        )

    panel = ensure_schema(pl.concat(frames))
    key = cache_key("synthetic", start, end, "hfq")
    path = save_cache(panel, out_dir, key)
    typer.echo(f"已生成 {path}：{panel.shape[0]:,} 行 x {panel.shape[1]} 列，{len(codes)} 只 / {t} 天")


if __name__ == "__main__":
    app()
