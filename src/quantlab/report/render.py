"""Report rendering: Markdown summary plus matplotlib PNG charts.

Produces a self-contained report directory:

    reports/<timestamp>/
        report.md
        nav.png
        drawdown.png
        risk.png        (VaR / CVaR analysis)
        ic.png            (if IC data given)
        group_return.png  (if layered-return data given)
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from quantlab.metrics.performance import drawdown_series  # noqa: E402
from quantlab.metrics.risk import rolling_value_at_risk, value_at_risk  # noqa: E402

# Prefer a CJK-capable font so the Chinese axis labels / titles render.
_CJK_FONT_CANDIDATES = ("Microsoft YaHei", "SimHei", "Noto Sans SC", "PingFang SC", "WenQuanYi Zen Hei")


def _pick_cjk_font() -> str | None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((name for name in _CJK_FONT_CANDIDATES if name in installed), None)


def _style() -> None:
    font = _pick_cjk_font()
    rc: dict = {
        "figure.dpi": 120,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
    if font is not None:
        rc["font.sans-serif"] = [font, "DejaVu Sans"]
        rc["axes.unicode_minus"] = False
    plt.rcParams.update(rc)


def _dates(d: pl.Series) -> list:
    return d.to_list()


def plot_nav(nav: pl.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(_dates(nav["date"]), nav["nav"].to_list(), label="策略 (strategy)", lw=1.5)
    if "benchmark" in nav.columns:
        ax.plot(_dates(nav["date"]), nav["benchmark"].to_list(), label="基准 (benchmark)", lw=1.2, alpha=0.8)
    ax.set_title("净值曲线 (NAV)")
    ax.set_ylabel("净值")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_drawdown(nav: pl.DataFrame, path: Path) -> None:
    _style()
    dd = drawdown_series(nav["nav"].to_numpy())
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.fill_between(_dates(nav["date"]), dd, 0, color="tab:red", alpha=0.4)
    ax.plot(_dates(nav["date"]), dd, color="tab:red", lw=1.0)
    ax.set_title("回撤 (Drawdown)")
    ax.set_ylabel("回撤")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_risk(nav: pl.DataFrame, path: Path) -> None:
    """Daily-return histogram with historical VaR lines + rolling 60d VaR95."""
    _style()
    v = np.asarray(nav["nav"].to_numpy(), dtype=float)
    ret = v[1:] / v[:-1] - 1.0  # simple daily returns, aligned with dates[1:]
    dates = _dates(nav["date"])[1:]
    colors = {0.95: "tab:orange", 0.99: "tab:red"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1.5]})
    ax1.hist(ret * 100.0, bins=60, color="tab:blue", alpha=0.6)
    for conf, color in colors.items():
        var = value_at_risk(ret, confidence=conf)
        if var == var:  # not NaN
            ax1.axvline(-var * 100.0, color=color, ls="--", lw=1.2, label=f"VaR {conf:.0%} = {var * 100:.2f}%")
    ax1.set_title("日收益分布与 VaR")
    ax1.set_xlabel("日收益 (%)")
    ax1.legend(fontsize=8)

    roll = rolling_value_at_risk(ret, window=60, confidence=0.95)
    ax2.plot(dates, roll * 100.0, color="tab:orange", lw=1.2, label="VaR95 (60 日滚动)")
    ax2.axhline(-value_at_risk(ret, confidence=0.95) * 100.0, color="k", ls=":", lw=1.0, label="全程 VaR95")
    ax2.set_title("滚动 VaR95")
    ax2.set_xlabel("日期")
    ax2.set_ylabel("单日损失上限 (%)")
    ax2.legend(fontsize=8)
    fig.suptitle("风险分析 (VaR / 历史模拟)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)


def plot_ic_series(ic: pl.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(9, 3))
    cols = [c for c in ic.columns if c != "date"]
    for c in cols:
        ax.plot(_dates(ic["date"]), ic[c].to_list(), label=c, lw=1.0)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("IC 序列")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_group_returns(group: pl.DataFrame, path: Path) -> None:
    _style()
    cols = [c for c in group.columns if c != "date" and c != "long-short"]
    means = {c: cast(float, group[c].mean()) for c in cols}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(means.keys()), list(means.values()), color="tab:blue", alpha=0.75)
    ax.set_title("分层收益 (分组均值)")
    ax.set_ylabel("平均收益")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fmt(v: float, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and (v != v)):
        return "N/A"
    return f"{v:.{digits}f}"


_METRIC_LABELS: list[tuple[str, str, bool]] = [
    ("total_return", "累计收益", True),
    ("annualized_return", "年化收益", True),
    ("annualized_volatility", "年化波动率", True),
    ("sharpe_ratio", "夏普比率", False),
    ("sortino_ratio", "索提诺比率", False),
    ("calmar_ratio", "卡玛比率", False),
    ("max_drawdown", "最大回撤", True),
    ("win_rate", "日胜率", True),
    ("information_ratio", "信息比率", False),
    ("beta", "Beta", False),
    ("alpha", "Alpha(年化)", True),
]


def render_report(
    output_dir: Path,
    metrics: dict[str, float],
    nav: pl.DataFrame,
    ic: pl.DataFrame | None = None,
    group: pl.DataFrame | None = None,
    importance: dict[str, float] | None = None,
    title: str = "QuantLab 研究简报",
    scenarios: dict[str, float] | None = None,
) -> Path:
    """Render charts and a Markdown report into ``output_dir`` (created if needed)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nav_png = output_dir / "nav.png"
    dd_png = output_dir / "drawdown.png"
    plot_nav(nav, nav_png)
    plot_drawdown(nav, dd_png)

    lines: list[str] = [f"# {title}", ""]

    lines.append("## 回测表现")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    for key, label, is_pct in _METRIC_LABELS:
        if key in metrics:
            val = metrics[key]
            text = _fmt(val * 100 if is_pct else val, 2) + ("%" if is_pct else "")
            lines.append(f"| {label} | {text} |")
    lines.append("")
    lines.append("![净值曲线](nav.png)")
    lines.append("")
    lines.append("![回撤](drawdown.png)")
    lines.append("")

    risk_keys = [k for k in ("var_95", "var_99", "cvar_95", "cvar_99", "max_drawdown_days") if k in metrics]
    if risk_keys:
        risk_png = output_dir / "risk.png"
        plot_risk(nav, risk_png)
        lines.append("## 风险指标")
        lines.append("")
        lines.append("VaR / CVaR 基于历史模拟，数值为日度损失幅度（占组合净值百分比）。")
        lines.append("")
        lines.append("| 指标 | 数值 |")
        lines.append("| --- | --- |")
        risk_rows: list[tuple[str, str, bool]] = [
            ("var_95", "VaR (95%, 1 日)", True),
            ("var_99", "VaR (99%, 1 日)", True),
            ("cvar_95", "CVaR / ES (95%)", True),
            ("cvar_99", "CVaR / ES (99%)", True),
            ("max_drawdown_days", "最长水下时间（交易日）", False),
        ]
        for key, label, is_pct in risk_rows:
            if key not in metrics:
                continue
            text = _fmt(metrics[key] * 100, 2) + "%" if is_pct else f"{int(metrics[key])} 天"
            lines.append(f"| {label} | {text} |")
        lines.append("")
        lines.append("![风险分析](risk.png)")
        lines.append("")

    if scenarios:
        lines.append("## 压力测试")
        lines.append("")
        lines.append("按回测期拟合的市场 Beta，将基准情景冲击映射为组合收益（负数为损失）。")
        lines.append("")
        lines.append("| 情景 | 组合预估冲击 |")
        lines.append("| --- | --- |")
        for label, impact in scenarios.items():
            impact = float(impact)
            text = "N/A" if impact != impact else f"{_fmt(impact * 100, 2)}%"
            lines.append(f"| {label} | {text} |")
        lines.append("")

    if ic is not None and len(ic):
        ic_png = output_dir / "ic.png"
        plot_ic_series(ic, ic_png)
        lines.append("## 因子 IC")
        lines.append("")
        lines.append("![IC 序列](ic.png)")
        lines.append("")

    if group is not None and len(group):
        grp_png = output_dir / "group_return.png"
        plot_group_returns(group, grp_png)
        lines.append("## 分层收益")
        lines.append("")
        lines.append("![分层收益](group_return.png)")
        lines.append("")

    if importance:
        lines.append("## 因子重要性 (Top 20)")
        lines.append("")
        lines.append("| 因子 | 重要性 |")
        lines.append("| --- | --- |")
        for name, imp in sorted(importance.items(), key=lambda kv: -kv[1])[:20]:
            lines.append(f"| {name} | {_fmt(imp, 4)} |")
        lines.append("")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
