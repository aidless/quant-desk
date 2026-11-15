"""因子评价：IC / RankIC / ICIR / 分层收益 / 衰减曲线。

判读口径（统一写在返回里，避免下游各说各话）：
    IC       —— 当日截面因子值与未来 h 日收益的 Pearson 相关
    RankIC   —— 同上但用 Spearman（A 股用这个更稳，抗极端值）
    ICIR     —— IC 均值 / IC 标准差（年化前）
    IC_t     —— IC 均值的 t 统计量 = ICIR * sqrt(有效天数)
    正 IC 比例 —— IC > 0 的天数占比
分层收益 —— 按因子值分 Q 组，各组未来 h 日平均收益，检验单调性
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.store import MarketPanel
from . import engine


def _pair_series(factor: pd.Series, forward: pd.Series, min_cross: int = 10):
    """取同一交易日、同一批股票的配对样本，剔除缺失。"""
    joined = pd.concat([factor.rename("f"), forward.rename("r")], axis=1).dropna()
    if len(joined) < min_cross:
        return None
    return joined


def ic_series(
    factor: pd.DataFrame,
    forward: pd.DataFrame,
    method: str = "spearman",
    min_cross: int = 10,
) -> pd.Series:
    """逐日截面相关，返回按日期的 IC 序列。"""
    records: dict[pd.Timestamp, float] = {}
    common_dates = factor.index.intersection(forward.index)
    for date in common_dates:
        pair = _pair_series(factor.loc[date], forward.loc[date], min_cross=min_cross)
        if pair is None:
            continue
        value = pair["f"].corr(pair["r"], method=method)
        if pd.notna(value):
            records[date] = float(value)
    series = pd.Series(records).sort_index()
    series.name = f"{method}_ic"
    return series


def ic_summary(ic: pd.Series) -> dict:
    """把 IC 序列折成统计量。"""
    clean = ic.dropna()
    n = int(len(clean))
    if n == 0:
        return {"n": 0, "ic_mean": None, "ic_std": None, "icir": None, "ic_t": None,
                "ic_pos_ratio": None, "ic_abs_mean": None}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if n > 1 else 0.0
    icir = mean / std if std > 0 else None
    return {
        "n": n,
        "ic_mean": round(mean, 5),
        "ic_std": round(std, 5),
        "icir": round(icir, 4) if icir is not None else None,
        "ic_t": round(icir * np.sqrt(n), 3) if icir is not None else None,
        "ic_pos_ratio": round(float((clean > 0).mean()), 4),
        "ic_abs_mean": round(float(clean.abs().mean()), 5),
        "ic_max": round(float(clean.max()), 5),
        "ic_min": round(float(clean.min()), 5),
    }


def quantile_returns(
    factor: pd.DataFrame,
    forward: pd.DataFrame,
    quantiles: int = 5,
    min_cross: int = 10,
) -> pd.DataFrame:
    """分层平均收益。返回 index=日期, columns=Q1..Qn（Q1=因子值最低）。"""
    out: dict[pd.Timestamp, dict[str, float]] = {}
    labels = [f"Q{i + 1}" for i in range(quantiles)]
    for date in factor.index.intersection(forward.index):
        pair = _pair_series(factor.loc[date], forward.loc[date], min_cross=min_cross)
        if pair is None or len(pair) < quantiles * 3:
            continue
        try:
            groups = pd.qcut(pair["f"].rank(method="first"), quantiles, labels=labels)
        except ValueError:
            continue
        means = pair.groupby(groups, observed=False)["r"].mean()
        out[date] = {k: float(means.get(k, np.nan)) for k in labels}
    frame = pd.DataFrame.from_dict(out, orient="index").sort_index()
    return frame


def quantile_summary(qret: pd.DataFrame) -> dict:
    """分层收益的均值与单调性。"""
    if qret.empty:
        return {"groups": {}, "monotonic": None, "long_short": None}
    means = qret.mean() * 100  # 转成百分数
    groups = {col: round(float(means[col]), 4) for col in means.index}
    ordered = [float(means[col]) for col in means.index]
    # 单调性：相邻组差值的符号一致比例
    diffs = np.diff(ordered)
    mono = float((np.sign(diffs) == np.sign(diffs[0])).mean()) if len(diffs) else 0.0
    long_short = round(ordered[-1] - ordered[0], 4) if len(ordered) >= 2 else None
    return {
        "groups": groups,
        "monotonic": round(mono, 3),
        "long_short_pct": long_short,
        "qret_days": int(len(qret)),
    }


def decay_table(
    factor: pd.DataFrame,
    panel: MarketPanel,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 60),
    method: str = "spearman",
) -> pd.DataFrame:
    """因子在不同持有期的 RankIC 均值，用于判断信息衰减速度。"""
    rows = []
    for h in horizons:
        forward = panel.fwd_ret(h)
        ic = ic_series(factor, forward, method=method)
        summary = ic_summary(ic)
        rows.append({"horizon": h, "ic_mean": summary.get("ic_mean"),
                     "icir": summary.get("icir"), "t": summary.get("ic_t"),
                     "n": summary.get("n")})
    return pd.DataFrame(rows)


def evaluate_factor(
    expression: str,
    panel: MarketPanel,
    *,
    horizon: int = 20,
    sign: int = 1,
    quantiles: int = 5,
    method: str = "spearman",
    industry_neutral: bool = False,
) -> dict:
    """一站式：算因子 → 算 IC → 分层 → 衰减。返回可 JSON 序列化的 dict。"""
    result = engine.compute(expression, panel, industry_neutral=industry_neutral)
    values = result.values * sign
    forward = panel.fwd_ret(horizon)

    ic = ic_series(values, forward, method=method)
    summary = ic_summary(ic)
    qret = quantile_returns(values, forward, quantiles=quantiles)
    qsum = quantile_summary(qret)

    # 每日 IC 的累计曲线，给前端画图
    ic_curve = [
        {"date": d.strftime("%Y-%m-%d"), "ic": round(float(v), 5)}
        for d, v in ic.items()
    ]

    return {
        "expression": expression,
        "sign": sign,
        "horizon": horizon,
        "method": method,
        "coverage": round(result.coverage(), 4),
        "ic": summary,
        "quantiles": qsum,
        "ic_curve": ic_curve[-250:],
        "stats": result.stats,
    }
