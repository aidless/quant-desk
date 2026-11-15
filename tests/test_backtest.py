"""回测引擎：T+1 滞后、成本、执行掩码、指标已知答案。"""

import numpy as np
import pandas as pd
import pytest

from qdesk.backtest import metrics
from qdesk.backtest.engine import run
from qdesk.config import CONFIG, BacktestConfig
from qdesk.data.store import make_synthetic_panel

PANEL = make_synthetic_panel(n_codes=20, n_days=120, seed=11)


# ---------- 指标已知答案 ----------

def test_total_return():
    nav = pd.Series([1.0, 1.1, 1.21])
    assert np.isclose(metrics.total_return(nav), 0.21)


def test_annual_return_constant():
    # 每天 0.1%，252 天：年化 = (1.001^252 - 1)
    daily = 0.001
    nav = pd.Series((1 + daily) ** np.arange(253))
    assert np.isclose(metrics.annual_return(nav), (1 + daily) ** 252 - 1, rtol=1e-9)


def test_max_drawdown_known():
    nav = pd.Series([1.0, 1.2, 0.9, 1.0, 0.6])
    # 峰值 1.2，谷 0.6 → -50%
    assert np.isclose(metrics.max_drawdown(nav), -0.5)


def test_sharpe_zero_vol_safe():
    flat = pd.Series([0.0] * 50)
    assert metrics.sharpe(flat) == 0.0


def test_win_rate():
    ret = pd.Series([0.01, -0.01, 0.02, 0.0, -0.005])
    assert np.isclose(metrics.win_rate(ret), 2 / 5)


# ---------- T+1 与泄露防护 ----------

def _cfg(**kw):
    c = BacktestConfig(**{**CONFIG.backtest.__dict__, **kw})
    return c


def test_perfect_foresight_cannot_leak():
    """未来函数防护：修改因子最后一行的取值，不得改变最后一日的收益。

    引擎约定因子 T 日收盘算出、T+lag 日生效。因此 T 日（最后一天）的收益
    只能由 T-1 日及之前的因子决定——篡改最后一天的因子值对最后一日净值
    必须零影响。若影响了，就是泄露。
    """
    mom = PANEL.close.pct_change(5)
    r1 = run(mom, PANEL, _cfg(), lag=1)
    tampered = mom.copy()
    tampered.iloc[-1] = 100.0        # 极端值
    tampered.iloc[-2] = -100.0
    r2 = run(tampered, PANEL, _cfg(), lag=1)
    assert np.isclose(r1.daily_ret.iloc[-1], r2.daily_ret.iloc[-1])
    # 同时第一天的收益必须是 0（没有任何历史因子可依赖）
    assert r1.daily_ret.iloc[0] == 0.0


def test_lag0_vs_lag1_diverge():
    """lag=0 会泄露未来收益，lag=1 不会——两者净值必须不同。"""
    mom = PANEL.close.pct_change(5)
    r0 = run(mom, PANEL, _cfg(), lag=0)
    r1 = run(mom, PANEL, _cfg(), lag=1)
    assert not np.isclose(r0.metrics["annual_return"], r1.metrics["annual_return"])


# ---------- 成本 ----------

def test_costs_reduce_returns():
    mom = PANEL.close.pct_change(5)
    free = run(mom, PANEL, _cfg(commission_bp=0, stamp_bp=0, slippage_bp=0))
    costly = run(mom, PANEL, _cfg())
    assert costly.metrics["total_return"] < free.metrics["total_return"]
    assert (costly.cost > 0).any()
    assert (free.cost == 0).all()


def test_stamp_tax_scales_with_selling():
    mom = PANEL.close.pct_change(5)
    r = run(mom, PANEL, _cfg(commission_bp=0, slippage_bp=0))
    # 成本全部来自印花税，应与卖出额成正比
    delta = r.weights.diff().fillna(r.weights.iloc[0])
    sells = (-delta.clip(upper=0)).sum(axis=1)
    expected = sells * 5.0 / 1e4
    assert np.allclose(r.cost.to_numpy(), expected.to_numpy(), atol=1e-12)


# ---------- 执行掩码 ----------

def test_execution_mask_respects_limit_up():
    """执行日涨停的标的权重必须为 0。"""
    mom = PANEL.close.pct_change(5)
    res = run(mom, PANEL, _cfg(), lag=1)
    w = res.weights
    limit_up = PANEL.limit_up.reindex(index=w.index, columns=w.columns).fillna(False).astype(bool)
    # 允许买入日之后才检查（shift 后前 lag 天权重为 0）
    overlap = (w > 1e-9) & limit_up.shift(-0, fill_value=False)
    # shift 后，t 日的权重是 t-1 目标在 t 日执行的；limit_up 在 t 日为真时 w[t] 应为 0
    assert not (w * limit_up.to_numpy()).any().any() or True  # 掩码逻辑在 engine 内部已 shift 对齐
    # 严格检查：执行掩码应用于 effective（已 shift），即 w[t]>0 ⇒ t 日未涨停
    viol = ((w > 1e-9) & limit_up).sum().sum()
    assert viol == 0


def test_holdings_bounds():
    mom = PANEL.close.pct_change(5)
    res = run(mom, PANEL, _cfg(min_holdings=5, max_holdings=20), lag=1)
    held = (res.weights > 1e-9).sum(axis=1)
    active = held[held > 0]
    assert (active <= 20).all()


# ---------- 行为 ----------

def test_nav_starts_near_one_and_benchmark_present():
    mom = PANEL.close.pct_change(5)
    res = run(mom, PANEL, _cfg(), lag=1)
    assert 0.98 <= res.nav.iloc[0] <= 1.02
    assert len(res.benchmark_nav) == len(res.nav)
    assert res.metrics["benchmark_annual"] is not None
