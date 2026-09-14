"""End-to-end research demo.

Pipeline:

    data (cache -> akshare) -> factors -> label -> model -> backtest -> metrics -> report

Usage:

    uv run python examples/run_research.py --config configs/baseline.yaml
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from quantlab.backtest import TopKStrategy, TradingRules, run_backtest
from quantlab.config import Config, load_config
from quantlab.data import DATE, INSTRUMENT, AkshareSource, cache_key, load_cached, save_cache
from quantlab.factors.evaluate import (
    forward_return,
    group_return,
    ic_series,
    ic_summary,
    rank_ic_series,
    standardize,
)
from quantlab.factors.library import compute_factors
from quantlab.metrics import risk_metrics, scenario_stress
from quantlab.metrics import summary as metrics_summary
from quantlab.models import LGBMModel, fit_predict
from quantlab.report import render_report

app = typer.Typer(help="QuantLab 研究流水线")


def acquire_data(cfg: Config):
    """Load the panel from cache or download it via akshare."""
    key = cache_key(cfg.data.universe, cfg.data.start_date, cfg.data.end_date, cfg.data.adjust)
    df = load_cached(cfg.data.data_dir, key)
    if df is None:
        typer.echo(f"  未命中缓存，正在下载 {cfg.data.universe} ...")
        source = AkshareSource(adjust=cfg.data.adjust)
        df = source.get_daily(cfg.data.universe, cfg.data.start_date, cfg.data.end_date)
        save_cache(df, cfg.data.data_dir, key)
    else:
        typer.echo(f"  命中缓存 {cfg.data.data_dir / f'{key}.parquet'}")
    return df


def build_model(cfg: Config):
    name = cfg.model.name.lower()
    if name == "lgbm":
        return LGBMModel(params=cfg.model.params)
    # PyTorch models are imported lazily so the LGBM path never needs torch.
    if name == "mlp":
        from quantlab.models.torch_model import MLPModel

        return MLPModel(**cfg.model.params)
    if name == "transformer":
        from quantlab.models.torch_model import TransformerModel

        return TransformerModel(**cfg.model.params)
    raise ValueError(f"unknown model '{cfg.model.name}' (expected lgbm/mlp/transformer)")


def default_bounds(cfg: Config, df) -> tuple[str, str]:
    """Derive train/valid cutoffs at the 60% / 80% date quantiles when unset."""
    dates = df[DATE].unique().sort()
    n = len(dates)
    train_end = cfg.model.train_end or str(dates[int(n * 0.6)])
    valid_end = cfg.model.valid_end or str(dates[int(n * 0.8)])
    return train_end, valid_end


@app.command()
def run(
    config_path: Path = typer.Option(Path("configs/baseline.yaml"), "--config", "-c", help="YAML 配置文件"),
) -> None:
    cfg = load_config(config_path)

    typer.echo("== 1/6 数据 ==")
    data = acquire_data(cfg)
    typer.echo(
        f"    面板 {data.shape[0]:,} 行 / {data[INSTRUMENT].n_unique()} 只 / {data[DATE].min()} ~ {data[DATE].max()}"
    )

    typer.echo("== 2/6 因子 ==")
    factor_names = cfg.factors.names
    features = compute_factors(data, factor_names)
    if cfg.factors.normalize and cfg.factors.normalize.lower() != "none":
        features = standardize(features, factor_names, method=cfg.factors.normalize)
    label = forward_return(data, cfg.model.label_horizon)
    panel = features.join(label, on=[DATE, INSTRUMENT], how="inner")
    typer.echo(f"    因子数 {len(factor_names)}，训练面板 {panel.shape[0]:,} 行")

    typer.echo("== 3/6 模型 ==")
    if cfg.model.name == "signal":
        # 单因子信号：跳过模型训练，直接把某个因子当预测值（离线/CI 免原生 ML 运行库）。
        signal_col = str(cfg.model.params.get("factor", factor_names[0]))
        typer.echo(f"    单因子信号 {signal_col}（免训练）")
        pred = features.select([DATE, INSTRUMENT, signal_col]).rename({signal_col: "predict"})
        importance: dict[str, float] | None = None
    else:
        train_end, valid_end = default_bounds(cfg, data)
        typer.echo(f"    切分 train<= {train_end} < valid<= {valid_end} < test")
        model = build_model(cfg)
        model, pred = fit_predict(model, panel, factor_names, "label", train_end, valid_end)
        typer.echo(f"    样本外预测 {pred.shape[0]:,} 行")
        importance = model.feature_importances(factor_names)

    typer.echo("== 4/6 因子/信号评估 ==")
    pred_label = pred.join(label, on=[DATE, INSTRUMENT], how="inner")
    ic = ic_series(pred_label, "predict", "label")
    ric = rank_ic_series(pred_label, "predict", "label")
    ic_df = ic.join(ric, on=DATE, how="inner")
    ic_stat = ic_summary(pred_label, "predict", "label")
    typer.echo(
        "    IC 均值 {:.4f} / ICIR {:.3f} / RankIC 均值 {:.4f} / RankICIR {:.3f}".format(
            ic_stat["ic_mean"], ic_stat["icir"], ic_stat["rank_ic_mean"], ic_stat["rank_icir"]
        )
    )
    group = group_return(pred_label, "predict", "label", n=10)

    typer.echo("== 5/6 回测 ==")
    strategy = TopKStrategy(top_k=cfg.backtest.top_k)
    rules = TradingRules(
        commission_rate=cfg.backtest.commission_rate,
        min_commission=cfg.backtest.min_commission,
        stamp_tax=cfg.backtest.stamp_tax,
        transfer_fee=cfg.backtest.transfer_fee,
        slippage=cfg.backtest.slippage,
    )
    result = run_backtest(pred, data, strategy, rules, init_cash=cfg.backtest.init_cash)
    typer.echo(f"    回测 {result.nav.shape[0]} 天 / 成交 {result.trades.shape[0]} 笔")

    typer.echo("== 6/6 指标 + 风险 + 报告 ==")
    m = metrics_summary(result.nav)
    m.update(risk_metrics(result.nav))
    scenarios = scenario_stress(result.nav)
    typer.echo(
        "    年化 {:.2%} / 夏普 {:.2f} / 最大回撤 {:.2%} / VaR95(1日) {:.2%} / 最长水下 {:.0f} 天".format(
            m["annualized_return"],
            m["sharpe_ratio"],
            m["max_drawdown"],
            m["var_95"],
            m["max_drawdown_days"],
        )
    )
    typer.echo(
        "    压力测试 单日-10%: {:.2%} / 连续3日-5%: {:.2%}".format(scenarios["单日 -10%"], scenarios["连续 3 日 -5%"])
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = cfg.report.output_dir / ts
    report_path = render_report(
        out_dir, m, result.nav, ic=ic_df, group=group, importance=importance, scenarios=scenarios
    )
    typer.echo(f"    报告已生成：{report_path}")


if __name__ == "__main__":
    app()
