"""绩效指标。全部可手算复现，不引入第三方绩效库。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(nav: pd.Series) -> float:
    """区间总收益。"""
    if len(nav) < 2:
        return 0.0
    return float(nav.iloc[-1] / nav.iloc[0] - 1)


def annual_return(nav: pd.Series, trading_days: int = 252) -> float:
    """年化收益（几何）。"""
    if len(nav) < 2 or nav.iloc[0] <= 0:
        return 0.0
    periods = len(nav) - 1
    growth = nav.iloc[-1] / nav.iloc[0]
    if growth <= 0:
        return -1.0
    return float(growth ** (trading_days / periods) - 1)


def annual_vol(daily_ret: pd.Series, trading_days: int = 252) -> float:
    clean = daily_ret.dropna()
    if len(clean) < 2:
        return 0.0
    return float(clean.std(ddof=1) * np.sqrt(trading_days))


def sharpe(daily_ret: pd.Series, rf_annual: float = 0.0, trading_days: int = 252) -> float:
    clean = daily_ret.dropna()
    if len(clean) < 2:
        return 0.0
    excess = clean - rf_annual / trading_days
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(trading_days))


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤，返回负数（-0.25 表示回撤 25%）。"""
    if len(nav) < 2:
        return 0.0
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    return float(drawdown.min())


def drawdown_series(nav: pd.Series) -> pd.Series:
    if len(nav) < 1:
        return nav
    return nav / nav.cummax() - 1.0


def calmar(nav: pd.Series, trading_days: int = 252) -> float:
    mdd = abs(max_drawdown(nav))
    if mdd < 1e-9:
        return 0.0
    return float(annual_return(nav, trading_days) / mdd)


def win_rate(daily_ret: pd.Series) -> float:
    clean = daily_ret.dropna()
    if len(clean) == 0:
        return 0.0
    return float((clean > 0).mean())


def t_stat(daily_ret: pd.Series) -> float:
    """日收益均值的 t 统计量（检验是否显著异于 0）。"""
    clean = daily_ret.dropna()
    if len(clean) < 3:
        return 0.0
    std = clean.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(clean.mean() / (std / np.sqrt(len(clean))))


def information_ratio(excess_ret: pd.Series, trading_days: int = 252) -> float:
    clean = excess_ret.dropna()
    if len(clean) < 2:
        return 0.0
    std = clean.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(clean.mean() / std * np.sqrt(trading_days))


def summarize(
    nav: pd.Series,
    daily_ret: pd.Series,
    *,
    benchmark_nav: pd.Series | None = None,
    turnover: pd.Series | None = None,
    trading_days: int = 252,
) -> dict:
    """把所有指标折成一个 dict（前端/报告直接用）。"""
    ann = annual_return(nav, trading_days)
    vol = annual_vol(daily_ret, trading_days)
    bench_ann = annual_return(benchmark_nav, trading_days) if benchmark_nav is not None else None
    excess = None
    if benchmark_nav is not None:
        bench_ret = benchmark_nav.pct_change(fill_method=None).reindex(daily_ret.index).fillna(0.0)
        excess = daily_ret - bench_ret

    result = {
        "total_return": round(total_return(nav), 4),
        "annual_return": round(ann, 4),
        "annual_vol": round(vol, 4),
        "sharpe": round(sharpe(daily_ret, trading_days=trading_days), 3),
        "max_drawdown": round(max_drawdown(nav), 4),
        "calmar": round(calmar(nav, trading_days), 3),
        "win_rate": round(win_rate(daily_ret), 4),
        "t_stat": round(t_stat(daily_ret), 3),
        "days": int(len(daily_ret.dropna())),
        "start": str(nav.index[0].date()) if len(nav) else None,
        "end": str(nav.index[-1].date()) if len(nav) else None,
    }
    if turnover is not None and len(turnover.dropna()) > 0:
        # 平均单边换手率（按调仓日计）与年化成本拖累
        result["avg_turnover"] = round(float(turnover.dropna().mean()), 4)
        result["annual_turnover"] = round(
            float(turnover.dropna().mean() * trading_days / max(1, len(turnover.dropna())))
            * len(turnover.dropna()) / trading_days * trading_days, 4
        )
    if bench_ann is not None:
        result["benchmark_annual"] = round(bench_ann, 4)
        result["excess_annual"] = round(ann - bench_ann, 4)
    if excess is not None:
        result["information_ratio"] = round(information_ratio(excess, trading_days), 3)
    return result
