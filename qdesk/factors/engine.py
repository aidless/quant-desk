"""因子计算与预处理。

链路：DSL 表达式 → 原始因子值 → 缩尾 → 截面标准化 →（可选）行业中性
→ 输出可直接喂给评估/回测的因子矩阵。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.store import MarketPanel
from . import dsl


@dataclass
class FactorResult:
    """一次因子计算的完整产物。"""

    expression: str
    values: pd.DataFrame                     # date × code，已预处理
    raw: pd.DataFrame = field(repr=False)    # 预处理前
    stats: dict = field(default_factory=dict)

    def coverage(self) -> float:
        """非空比例，衡量因子可用度。"""
        total = self.values.shape[0] * self.values.shape[1]
        if total == 0:
            return 0.0
        return float(self.values.notna().sum().sum() / total)


def _winsorize_row(frame: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    lo = frame.quantile(q, axis=1)
    hi = frame.quantile(1 - q, axis=1)
    return frame.clip(lower=lo, upper=hi, axis=0)


def _zscore_row(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1)
    return frame.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


def _industry_neutralize(frame: pd.DataFrame, industry: pd.Series) -> pd.DataFrame:
    """行业内去均值：每个交易日在每个行业内减去行业均值。"""
    codes = [c for c in frame.columns if c in industry.index]
    if not codes:
        return frame
    groups = industry.reindex(codes)
    out = frame.copy()
    # 按行业分组，逐组减去组内均值（axis=1 表示对每个交易日横截面）
    for _, sub_codes in groups.groupby(groups).groups.items():
        cols = list(sub_codes)
        block = frame[cols]
        out[cols] = block.sub(block.mean(axis=1), axis=0)
    return out


def compute(
    expression: str,
    panel: MarketPanel,
    *,
    winsorize: bool = True,
    standardize: bool = True,
    industry_neutral: bool = False,
    clip_quantile: float = 0.01,
) -> FactorResult:
    """把 DSL 表达式算成因子矩阵。"""
    ctx = dsl.build_context(panel)
    index = panel.dates
    columns = pd.Index(panel.codes)

    raw = dsl.evaluate(expression, ctx, index=index, columns=columns)

    values = raw
    if winsorize:
        values = _winsorize_row(values, q=clip_quantile)
    if industry_neutral and "industry" in panel.long.columns:
        industry = panel.meta["industry"] if "industry" in panel.meta.columns else None
        if industry is not None:
            values = _industry_neutralize(values, industry)
    if standardize:
        values = _zscore_row(values)

    stats = {
        "expression": expression,
        "shape": list(values.shape),
        "coverage": float(values.notna().sum().sum() / max(1, values.size)),
        "mean": float(np.nanmean(values.to_numpy(dtype=float))),
        "std": float(np.nanstd(values.to_numpy(dtype=float))),
        "start": str(index[0].date()) if len(index) else None,
        "end": str(index[-1].date()) if len(index) else None,
    }
    return FactorResult(expression=expression, values=values, raw=raw, stats=stats)


def top_bottom(frame: pd.DataFrame, top_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按截面分位切出多空两端，返回布尔掩码。"""
    ranks = frame.rank(axis=1, pct=True)
    upper = ranks > (1 - top_frac)
    lower = ranks <= top_frac
    return upper, lower
