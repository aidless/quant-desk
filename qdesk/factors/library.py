"""内置因子库。

每个因子都是一行 DSL 表达式 + 经济学解释。这套清单参照量化选股研报里
反复出现的量价因子族：动量、反转、波动、换手、量价背离、遗憾规避、
斜率凸性、非流动性（Amihud）、彩票效应。

因子方向（sign）说明：+1 表示「因子值越大预期收益越高」，
-1 表示反向。评估时统一乘以 sign 后再算 IC，便于横向比较。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorDef:
    name: str
    cn_name: str
    expression: str
    sign: int
    horizon: int
    category: str
    logic: str


BUILTIN_FACTORS: list[FactorDef] = [
    # ---------------- 动量 / 反转 ----------------
    FactorDef(
        name="mom_20",
        cn_name="20 日动量",
        expression="close / delay(close, 20) - 1",
        sign=1,
        horizon=20,
        category="动量",
        logic="中期趋势延续。A 股 20 日动量常呈弱正向，但拥挤后易反转。",
    ),
    FactorDef(
        name="mom_60",
        cn_name="60 日动量",
        expression="close / delay(close, 60) - 1",
        sign=1,
        horizon=20,
        category="动量",
        logic="季度级趋势。与短期反转方向相反，二者组合可分离趋势与噪声。",
    ),
    FactorDef(
        name="rev_5",
        cn_name="5 日反转",
        expression="-(close / delay(close, 5) - 1)",
        sign=1,
        horizon=5,
        category="反转",
        logic="短期过度反应后的均值回复，A 股最稳定的量价因子族之一。",
    ),
    FactorDef(
        name="rev_1",
        cn_name="1 日反转",
        expression="-ret",
        sign=1,
        horizon=1,
        category="反转",
        logic="隔日反转，换手极高，扣成本后通常不剩收益——用来做成本压力测试的对照。",
    ),
    # ---------------- 波动 ----------------
    FactorDef(
        name="vol_20",
        cn_name="20 日波动率",
        expression="ts_std(ret, 20)",
        sign=-1,
        horizon=20,
        category="波动",
        logic="低波动异象：高波动股票长期跑输，做多低波动端。",
    ),
    FactorDef(
        name="idio_vol_20",
        cn_name="20 日特质波动（去均值版）",
        expression="ts_std(ret - ts_mean(ret, 20), 20)",
        sign=-1,
        horizon=20,
        category="波动",
        logic="剥离趋势后的残差波动，比原始波动更接近「特质风险」。",
    ),
    FactorDef(
        name="max_ret_20",
        cn_name="20 日最大单日涨幅",
        expression="ts_max(ret, 20)",
        sign=-1,
        horizon=20,
        category="彩票",
        logic="彩票偏好：近期出现极端单日暴涨的股票，后续收益偏低。",
    ),
    FactorDef(
        name="skew_20",
        cn_name="20 日收益偏度",
        expression="ts_skew(ret, 20)",
        sign=-1,
        horizon=20,
        category="彩票",
        logic="正偏度=右尾风险，投资者为彩票型收益付费，预期收益被压低。",
    ),
    # ---------------- 量价 ----------------
    FactorDef(
        name="vp_div_20",
        cn_name="20 日量价背离",
        expression="-ts_corr(close, volume, 20)",
        sign=1,
        horizon=20,
        category="量价",
        logic="量价同向（价涨量增）说明资金推动，同向度高时后续易衰减；"
              "取负号即做多「量价背离」端。",
    ),
    FactorDef(
        name="turn_20",
        cn_name="20 日平均换手率",
        expression="ts_mean(turnover, 20)",
        sign=-1,
        horizon=20,
        category="流动性",
        logic="高换手=高关注度=高估值，长期负向；A 股该因子单调性较强。",
    ),
    FactorDef(
        name="turn_surge",
        cn_name="换手率突增",
        expression="turnover / ts_mean(turnover, 20)",
        sign=-1,
        horizon=5,
        category="流动性",
        logic="异常放量常对应情绪顶点，短期反向。",
    ),
    FactorDef(
        name="amihud_20",
        cn_name="20 日 Amihud 非流动性",
        expression="ts_mean(abs(ret) / (amount / 1e8), 20)",
        sign=1,
        horizon=20,
        category="流动性",
        logic="单位成交额引起的价格冲击。非流动性溢价：越不流动预期收益越高。",
    ),
    # ---------------- 形态 ----------------
    FactorDef(
        name="regret_20",
        cn_name="20 日遗憾规避",
        expression="ts_max(close, 20) / close - 1",
        sign=-1,
        horizon=20,
        category="形态",
        logic="距 20 日高点的回撤幅度。遗憾规避理论下接近高点的股票被追捧"
              "（因子值小=离高点近），后续反而承压，故取负向。",
    ),
    FactorDef(
        name="slope_convex_20",
        cn_name="斜率凸性",
        expression="(ts_mean(close, 5) / ts_mean(close, 20) - 1)"
                   " - delay(ts_mean(close, 5) / ts_mean(close, 20) - 1, 5)",
        sign=1,
        horizon=10,
        category="形态",
        logic="短均线相对长均线的位置变化率，刻画趋势的加速度（凸性）。",
    ),
    FactorDef(
        name="intraday_strength",
        cn_name="日内强度",
        expression="(close - low) / (high - low)",
        sign=1,
        horizon=5,
        category="形态",
        logic="收盘在当日区间的位置，衡量尾盘承接力。",
    ),
    FactorDef(
        name="overnight_gap",
        cn_name="隔夜跳空",
        expression="ts_mean(open / delay(close, 1) - 1, 10)",
        sign=-1,
        horizon=5,
        category="形态",
        logic="隔夜溢价长期为负（A 股日内收益强于隔夜），取反向。",
    ),
]


FACTOR_MAP: dict[str, FactorDef] = {f.name: f for f in BUILTIN_FACTORS}


def get(name: str) -> FactorDef:
    if name not in FACTOR_MAP:
        raise KeyError(f"未知内置因子 {name}，可用：{sorted(FACTOR_MAP)}")
    return FACTOR_MAP[name]


def catalog() -> list[dict]:
    """给平台/LLM 用的因子清单。"""
    return [
        {
            "name": f.name,
            "cn_name": f.cn_name,
            "expression": f.expression,
            "sign": f.sign,
            "horizon": f.horizon,
            "category": f.category,
            "logic": f.logic,
        }
        for f in BUILTIN_FACTORS
    ]
