"""Typed configuration models for a research run.

A single :class:`Config` describes the whole ``data -> factors -> models ->
backtest -> report`` pipeline and can be loaded from a YAML file via
:func:`load_config`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Where and how market data is acquired."""

    universe: str = Field(default="000300", description="指数代码，如沪深300为'000300'，中证500为'000905'")
    start_date: str = Field(description="数据开始日期，YYYYMMDD 或 YYYY-MM-DD")
    end_date: str = Field(description="数据结束日期，YYYYMMDD 或 YYYY-MM-DD")
    data_dir: Path = Field(default=Path("data"), description="本地 parquet 缓存目录")
    adjust: str = Field(default="hfq", description="复权方式：hfq 后复权 / qfq 前复权 / 空为不复权")


class FactorConfig(BaseModel):
    """Which factors to compute and how to normalize them."""

    names: list[str] = Field(
        default_factory=lambda: ["momentum_20", "reversal_5", "volatility_20", "turnover_20", "rsi_14"],
        description="内置因子名列表，见 factors.library 的 FACTOR_REGISTRY",
    )
    normalize: str = Field(default="zscore", description="因子标准化：zscore/rank/scale/robust_zscore/None")
    neutralize: bool = Field(default=False, description="是否做行业/市值中性化")
    neutralize_targets: list[str] = Field(default_factory=lambda: ["market_cap"], description="中性化目标因子")


class ModelConfig(BaseModel):
    """Model choice and the label definition."""

    name: str = Field(default="lgbm", description="模型：lgbm / mlp / transformer / signal（单因子信号，免训练）")
    label_horizon: int = Field(default=1, description="标签：未来 h 日收益率 close.shift(-h)/close - 1")
    train_start: str | None = Field(default=None, description="训练集起点（缺省为数据起点）")
    train_end: str | None = Field(default=None, description="训练集终点")
    valid_start: str | None = Field(default=None, description="验证集起点")
    valid_end: str | None = Field(default=None, description="验证集终点")
    params: dict[str, Any] = Field(default_factory=dict, description="模型超参数")


class BacktestConfig(BaseModel):
    """Backtest strategy and A-share trading-cost assumptions."""

    strategy: str = Field(default="topk", description="策略：topk / neutral")
    top_k: float = Field(default=0.2, description="TopK 持股比例（0~1）")
    init_cash: float = Field(default=1e8, description="初始资金")
    commission_rate: float = Field(default=2.5e-4, description="佣金费率（双边）")
    min_commission: float = Field(default=5.0, description="最低佣金")
    stamp_tax: float = Field(default=1e-3, description="印花税（仅卖出）")
    transfer_fee: float = Field(default=1e-5, description="过户费（双边）")
    slippage: float = Field(default=5e-4, description="滑点（成交价偏离，双边）")
    risk_degree: float = Field(default=0.95, description="最大风险度（仓位上限）")
    max_position_weight: float = Field(default=1.0, description="单票最大权重")
    benchmark: str = Field(default="equal", description="基准：equal 等权 / 指数代码")


class ReportConfig(BaseModel):
    """Report output location."""

    output_dir: Path = Field(default=Path("reports"), description="报告输出目录")


class Config(BaseModel):
    """Root config aggregating all sub-configs."""

    data: DataConfig
    factors: FactorConfig
    model: ModelConfig
    backtest: BacktestConfig
    report: ReportConfig


def load_config(path: str | Path) -> Config:
    """Load a :class:`Config` from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)
    return Config.model_validate(raw)
