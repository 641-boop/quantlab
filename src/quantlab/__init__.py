"""QuantLab: a modern A-share quantitative research platform.

The pipeline is:

    data  ->  factors  ->  models  ->  backtest  ->  metrics/report

Subpackages:
    data       data acquisition and local parquet caching (akshare-based).
    factors    time-series / cross-sectional operators, a factor library,
               factor evaluation and neutralization.
    models     a unified model interface with LightGBM and PyTorch backends.
    backtest   a daily-rebalance backtest engine with A-share trading rules.
    metrics    performance metrics (Sharpe, Sortino, IR, drawdown, ...).
    report     markdown + chart report generation.
"""

__version__ = "0.1.0"
