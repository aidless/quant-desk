"""DSL 安全层与正确性测试。"""

import numpy as np
import pandas as pd
import pytest

from qdesk.data.store import make_synthetic_panel
from qdesk.factors.dsl import (
    DSLEvalError,
    DSLSyntaxError,
    build_context,
    evaluate,
    validate,
)

PANEL = make_synthetic_panel(n_codes=6, n_days=40, seed=7)
CTX = build_context(PANEL)


def _mini_panel(close: pd.DataFrame):
    """用一列 close 构造 build_context 需要的最小面板 stub。"""
    _dates, _codes = close.index, list(close.columns)

    class _Mini:
        codes = _codes
        dates = _dates

        def wide(self, field):
            base = {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": pd.DataFrame(1e6, index=_dates, columns=_codes),
                "amount": pd.DataFrame(1e8, index=_dates, columns=_codes),
                "pct_chg": close.pct_change(fill_method=None) * 100,
                "turnover": pd.DataFrame(1.0, index=_dates, columns=_codes),
                "amplitude": pd.DataFrame(2.0, index=_dates, columns=_codes),
            }
            return base[field]

    return _Mini()



# ---------- 安全性：必须被拒 ----------

@pytest.mark.parametrize("bad", [
    "__import__('os').system('ls')",
    "close.__class__",
    "close[0]",
    "(lambda x: x)(close)",
    "[c for c in close]",
    "import pandas",
    "close + ;",
    "nonexistent_field",
    "ts_mean()",
    "open(x)",
    "eval('close')",
    "ts_mean(close, 'x')",
    "a" * 4001,
])
def test_dsl_rejects_unsafe(bad):
    with pytest.raises(DSLSyntaxError):
        validate(bad)


@pytest.mark.parametrize("bad", [
    "ts_mean(close, 0)",
    "ts_mean(close, -3)",
    "ts_mean(close, 1.5)",
])
def test_dsl_rejects_bad_window_at_eval(bad):
    """窗口参数语义校验在 eval 期（validate 只做结构层）。"""
    validate(bad)  # 结构上合法
    with pytest.raises((DSLEvalError, DSLSyntaxError)):
        evaluate(bad, CTX)


def test_dsl_rejects_huge_expression():
    expr = " + ".join(["close"] * 700)  # 超过 MAX_NODES=600
    with pytest.raises(DSLSyntaxError):
        validate(expr)


# ---------- 正确性：已知答案 ----------

def test_ts_mean_known_answer():
    # 单只票手工构造：close = [1,2,3,4]，ts_mean(close,2) 在第2个值为 1.5
    dates = pd.date_range("2021-01-04", periods=4)
    close = pd.DataFrame({"000001": [1.0, 2.0, 3.0, 4.0]}, index=dates)
    out = evaluate("ts_mean(close, 2)", build_context(_mini_panel(close)))
    assert np.isclose(out.iloc[1, 0], 1.5)
    assert np.isnan(out.iloc[0, 0])


def test_delay_known_answer():
    dates = pd.date_range("2021-01-04", periods=3)
    close = pd.DataFrame({"A": [10.0, 20.0, 30.0]}, index=dates)
    out = evaluate("delay(close, 1)", build_context(_mini_panel(close)))
    assert np.isclose(out.iloc[1, 0], 10.0)
    assert np.isclose(out.iloc[2, 0], 20.0)


def test_cs_rank_cross_sectional():
    dates = pd.date_range("2021-01-04", periods=2)
    close = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]}, index=dates)
    out = evaluate("cs_rank(close)", build_context(_mini_panel(close)))
    # A<B，A 的 rank=0，B=1（pct rank 归一）
    assert out.iloc[0, 0] < out.iloc[0, 1]


def test_safe_div_zero_protected():
    dates = pd.date_range("2021-01-04", periods=2)
    close = pd.DataFrame({"A": [0.0, 5.0], "B": [1.0, 0.0]}, index=dates)
    out = evaluate("close / close", build_context(_mini_panel(close)))
    # 0/0 → nan（被保护），非 0/0 → 1
    assert np.isclose(out.iloc[0, 1], 1.0)
    assert np.isnan(out.iloc[0, 0]) or out.iloc[0, 0] == 0.0


def test_builtin_factor_expressions_all_compile():
    from qdesk.factors.library import BUILTIN_FACTORS
    for f in BUILTIN_FACTORS:
        validate(f.expression)  # 不抛即通过


def test_builtin_factors_all_compute():
    from qdesk.factors.library import BUILTIN_FACTORS
    from qdesk.factors.engine import compute
    long_panel = make_synthetic_panel(n_codes=6, n_days=150, seed=7)
    for f in BUILTIN_FACTORS:
        res = compute(f.expression, long_panel)
        assert res.stats["coverage"] > 0, f"{f.name} 覆盖率为 0"
        assert res.values.notna().any().any()


def test_evaluate_inf_cleaned():
    out = evaluate("close / close - 1", CTX)
    assert not np.isinf(out.to_numpy()).any()
