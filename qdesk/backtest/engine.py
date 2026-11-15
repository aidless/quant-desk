"""向量化回测引擎（A 股日频、T+1、含成本）。

执行约定（写死在代码里，避免口径漂移）：
    1. 因子在 T 日收盘后计算，使用 T 日及之前的全部数据；
    2. 下单在 T+1 日执行，因此 T 日的目标权重从 T+1 日才开始吃收益；
       实现上是 weights.shift(lag)，lag 默认 1；
    3. T+1 日涨停/停牌/成交额不足的标的无法买入，权重置 0 后归一再分配；
    4. 收益按收盘对收盘计（close→close）。这个简化会略微高估可交易性
       （真实成交价在次日开盘附近），README 里已标注为已知偏差。
    5. 成本按买入/卖出腿分别计：佣金双边、滑点双边、印花税仅卖出腿。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CONFIG, BacktestConfig
from ..data.store import MarketPanel
from ..factors import engine as factor_engine
from . import metrics


@dataclass
class BacktestResult:
    nav: pd.Series
    daily_ret: pd.Series
    gross_ret: pd.Series
    cost: pd.Series
    turnover: pd.Series
    benchmark_nav: pd.Series
    weights: pd.DataFrame
    metrics: dict
    config: dict = field(default_factory=dict)
    holdings_log: list[dict] = field(default_factory=list)

    def equity_curve(self, max_points: int = 400) -> list[dict]:
        """给前端的下采样净值曲线。"""
        frame = pd.concat([self.nav.rename("nav"), self.benchmark_nav.rename("bench")], axis=1)
        frame = frame.dropna(how="all")
        if len(frame) > max_points:
            step = max(1, len(frame) // max_points)
            frame = frame.iloc[::step]
        return [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "nav": round(float(row["nav"]), 6) if pd.notna(row["nav"]) else None,
                "bench": round(float(row["bench"]), 6) if pd.notna(row["bench"]) else None,
            }
            for idx, row in frame.iterrows()
        ]


def _target_weights(
    factor: pd.DataFrame,
    panel: MarketPanel,
    cfg: BacktestConfig,
    selection_sign: int = 1,
) -> pd.DataFrame:
    """按因子值选股并构造等权目标权重（仅在调仓日有值，其余为 NaN）。"""
    dates = factor.index
    codes = factor.columns
    # 0/1 选择掩码（调仓日选中的票=1），避免 NaN-ffill 把上期持仓带进本期
    weights = pd.DataFrame(np.nan, index=dates, columns=codes)

    ranked = (factor * selection_sign).rank(axis=1, pct=True)
    rebalance_dates = dates[:: max(1, cfg.rebalance_days)]

    for date in rebalance_dates:
        row = ranked.loc[date].dropna()
        if len(row) < cfg.min_holdings:
            continue
        # 只保留可交易标的（当日口径；执行日会再校验一次）
        tradable = panel.tradable(cfg.min_amount).reindex(index=[date], columns=codes).iloc[0]
        row = row[tradable.fillna(False).astype(bool)]
        if len(row) < cfg.min_holdings:
            continue
        n_pick = int(np.clip(round(len(row) * cfg.top_frac), cfg.min_holdings, cfg.max_holdings))
        n_pick = min(n_pick, len(row))
        picked = row.nlargest(n_pick).index
        weights.loc[date, picked] = 1.0 / n_pick
    return weights


def _execution_mask(panel: MarketPanel, cfg: BacktestConfig, dates: pd.DatetimeIndex, codes) -> pd.DataFrame:
    """执行日可买掩码：非涨停、非停牌、成交额达标。"""
    mask = pd.DataFrame(True, index=dates, columns=codes)
    if cfg.exclude_limit_up:
        mask &= ~panel.limit_up.reindex(index=dates, columns=codes).fillna(False)
    if cfg.exclude_suspended:
        mask &= ~panel.suspended.reindex(index=dates, columns=codes).fillna(False)
    amount = panel.wide("amount").reindex(index=dates, columns=codes)
    mask &= (amount.fillna(0) >= cfg.min_amount)
    return mask


def run(
    factor: pd.DataFrame,
    panel: MarketPanel,
    cfg: BacktestConfig | None = None,
    *,
    lag: int = 1,
    selection_sign: int = 1,
    benchmark: str = "equal_weight",
) -> BacktestResult:
    """跑一次多空/多头回测。

    参数
    ----
    factor : 已预处理的因子矩阵（date × code）
    panel  : 行情面板
    cfg    : 成本与选股参数
    lag    : 信号到持仓的滞后交易日数（A 股 T+1，默认 1）
    """
    cfg = cfg or CONFIG.backtest
    codes = pd.Index(panel.codes)
    dates = panel.dates

    factor = factor.reindex(index=dates, columns=codes)
    target = _target_weights(factor, panel, cfg, selection_sign=selection_sign)

    # 关键修复：调仓日把全行显式清零再写入新目标权重。
    # 否则 target 在调仓日的「非新选票」位置是 NaN，ffill 会把上一期持仓
    # 一路带过来与新票叠加，持仓数越滚越多（实测滚到 87 只 vs 目标 20 只）。
    rb_dates = dates[:: max(1, cfg.rebalance_days)]
    target.loc[rb_dates] = target.loc[rb_dates].fillna(0.0)
    held = target.ffill()
    # 信号滞后：T 日算出的权重从 T+lag 日生效
    effective = held.shift(lag)

    # 执行日不可买的标的不给权重，然后重新归一
    exec_mask = _execution_mask(panel, cfg, dates, codes)
    effective = effective.where(exec_mask, 0.0)
    row_sum = effective.sum(axis=1)
    effective = effective.div(row_sum.replace(0, np.nan), axis=0).fillna(0.0)

    ret = panel.ret.reindex(index=dates, columns=codes).fillna(0.0)
    gross = (effective * ret).sum(axis=1)

    # 换手与成本
    delta = effective.diff()
    delta.iloc[0] = effective.iloc[0]
    buys = delta.clip(lower=0).sum(axis=1)
    sells = (-delta.clip(upper=0)).sum(axis=1)
    one_way = 0.5 * (buys + sells)

    comm = (buys + sells) * cfg.commission_bp / 1e4
    slip = (buys + sells) * cfg.slippage_bp / 1e4
    stamp = sells * cfg.stamp_bp / 1e4
    cost = comm + slip + stamp

    net = gross - cost
    nav = (1 + net).cumprod()
    # 首日净值基准归一
    nav = nav / nav.iloc[0] if len(nav) and nav.iloc[0] else nav

    # 基准：全池等权（不做任何筛选）
    if benchmark == "equal_weight":
        bench_ret = ret.mean(axis=1)
    else:
        bench_ret = ret.mean(axis=1)
    bench_nav = (1 + bench_ret).cumprod()

    summary = metrics.summarize(
        nav,
        net,
        benchmark_nav=bench_nav,
        turnover=one_way,
        trading_days=cfg.trading_days,
    )
    summary["cost_drag_annual"] = round(float(cost.mean() * cfg.trading_days), 4)
    summary["gross_annual"] = round(
        float((1 + gross.mean()) ** cfg.trading_days - 1) if abs(gross.mean()) < 0.1 else np.nan, 4
    )
    summary["avg_holdings"] = round(float((effective > 0).sum(axis=1).replace(0, np.nan).mean()), 1)

    holdings_log = []
    for date in dates[:: max(1, cfg.rebalance_days)][-20:]:
        row = effective.loc[date]
        picked = [c for c in row.index if row[c] > 0]
        if picked:
            holdings_log.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "count": len(picked),
                    "codes": picked[:20],
                }
            )

    return BacktestResult(
        nav=nav,
        daily_ret=net,
        gross_ret=gross,
        cost=cost,
        turnover=one_way,
        benchmark_nav=bench_nav,
        weights=effective,
        metrics=summary,
        config={
            "top_frac": cfg.top_frac,
            "rebalance_days": cfg.rebalance_days,
            "commission_bp": cfg.commission_bp,
            "stamp_bp": cfg.stamp_bp,
            "slippage_bp": cfg.slippage_bp,
            "min_amount": cfg.min_amount,
            "lag": lag,
        },
        holdings_log=holdings_log,
    )


def run_expression(
    expression: str,
    panel: MarketPanel,
    cfg: BacktestConfig | None = None,
    *,
    sign: int = 1,
    industry_neutral: bool = False,
    lag: int = 1,
) -> BacktestResult:
    """从 DSL 表达式直接回测。"""
    cfg = cfg or CONFIG.backtest
    result = factor_engine.compute(
        expression, panel, industry_neutral=industry_neutral, standardize=True
    )
    return run(result.values, panel, cfg, lag=lag, selection_sign=sign)
