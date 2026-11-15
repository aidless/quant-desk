"""因子工厂：用 LLM 生成 DSL 表达式（带安全校验 + 离线回退）。

工作流：
    1) 接收自然语言意图（如“做一支 10 日超跌反弹因子”）；
    2) 可选给定一个已知因子名作为起点（从 library.py 中查公式）；
    3) 如果 LLM 可用则调用构建好的 prompt，解析出结构化 JSON；
    4) 对输出表达式走 DSL 校验器编译 → 若失败则自动重试一次（带错误信息反馈）；
    5) 最后返回：{name, expression, sign, logic, category} 及校验状态。
    6) 若 LLM 不可用或两次尝试都失败，则按策略回退到预定义模板（离线演示能力保留）。

设计重点：
    - 所有可变量化结论都必须有载体：每个生成的因子必须有明确的表达式文本和计算逻辑；
    - 错误回读要能直接喂给模型（例如：“表达式超过 300 字符，简化掉 ts_zscore”）；
    - 离线回退必须可验证（用合成面板跑一遍不报错即通过）。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable

from ..config import CONFIG
from ..factors.dsl import DSLSyntaxError, validate
from ..factors.library import BUILTIN_FACTORS


@dataclass
class FactorFactoryResult:
    name: str
    expression: str
    sign: int
    logic: str
    category: str
    verified: bool           # DSL 校验是否通过
    error: str | None        # 错误详情（如有）
    source: str              # "builtin"/"llm_replay"
    retry_count: int = 0     # 重试次数

    def is_valid(self) -> bool:
        return self.verified and not self.error and len(self.expression) > 10


def template_for_intent(intent: str) -> dict:
    """基于关键词模板的回退策略（无网也能跑 demo）。"""
    intent_lower = intent.lower().strip()
    if any(k in intent_lower for k in ("回撤", "高点", "遗憾", "regret")):
        return {
            "name": "template_regret",
            "expression": "ts_max(close, 20) / close - 1",
            "sign": -1,
            "category": "形态",
            "logic": "距 20 日高点的回撤幅度，参考遗憾规避理论。",
        }
    if any(k in intent_lower for k in ("反", "rev", "revert", "均值回复")):
        return {
            "name": "template_rev",
            "expression": "-ts_mean((close / delay(close, 1) - 1), 5)",
            "sign": 1,
            "category": "反转",
            "logic": "5 日平均反转，捕捉短期过度反应后的均值回复。",
        }
    if any(k in intent_lower for k in ("波", "volatility", "risk")):
        return {
            "name": "template_vol",
            "expression": "ts_std(ts_mean(close / delay(close, 1) - 1, 20), 10)",
            "sign": -1,
            "category": "波动",
            "logic": "20 日收益去均值后 10 日波动，衡量趋势不确定性。",
        }
    return {
        "name": "template_default",
        "expression": "ts_mean(close / delay(close, 1) - 1, 10)",
        "sign": 1,
        "category": "动量",
        "logic": "默认 10 日动量回退模板。",
    }


def _parse_llm_output(text: str) -> tuple[dict | None, str | None]:
    """从 LLM 回复里提取第一个 JSON 对象。"""
    m = re.search(r"\{[^{}]*\"name\"[^{}]*\}", text, re.I)
    if m:
        try:
            obj = json.loads(m.group())
            return obj, None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)
    return None, "未能在回复中找到 JSON 对象"


def llm_build(
    intent: str,
    available: bool = True,
    max_attempts: int = 2,
    on_feedback: Callable[[str], None] | None = None,
) -> FactorFactoryResult:
    """用 LLM 生成因子。

    available=False 表示走 LLM 通道；True 表示直接用离线模板（与 /config 接口的
    llm_enabled 字段一致：llm_enabled=True → available=False）。
    """
    # 离线路径：直接用模板
    if available:
        out = template_for_intent(intent)
        return FactorFactoryResult(
            name=out["name"],
            expression=out["expression"],
            sign=out["sign"],
            logic=out["logic"],
            category=out["category"],
            verified=True,
            error=None,
            source="template",
            retry_count=0,
        )

    # 在线路径：LLM 生成 → 校验 → 失败带错误反馈重试
    from . import client
    last_error: str | None = None
    last_obj: dict | None = None
    feedback = ""
    for attempt in range(max_attempts):
        try:
            prompt = client.build_factor_prompt(
                intent + (f"\n上一次尝试的错误：{feedback}" if feedback else ""),
                [f.name for f in BUILTIN_FACTORS],
            )
            reply = client.make_client().chat(prompt)
            obj, parse_err = _parse_llm_output(reply)
            if parse_err:
                raise ValueError(parse_err)
            expr = str(obj.get("expression", "")).strip()
            if len(expr) > 300:
                raise ValueError(f"表达式过长（{len(expr)} > 300），请简化")
            validate(expr)  # DSL 编译校验，失败抛 DSLSyntaxError
            last_obj, last_error = obj, None
            break
        except (ValueError, DSLSyntaxError) as exc:
            last_error = str(exc)[:300]
            feedback = last_error
            continue
        except Exception as exc:  # LLMClientError / 网络错误等 → 保留现场，进入下一轮或回退
            last_error = str(exc)[:300]
            last_error = str(exc)[:300]
            feedback = last_error
            continue

    if last_obj is not None:
        name = str(last_obj.get("name", "custom")).replace(" ", "_")
        return FactorFactoryResult(
            name=name,
            expression=str(last_obj.get("expression", "")),
            sign=int(last_obj.get("sign", 1)),
            logic=str(last_obj.get("logic", "")),
            category=str(last_obj.get("category", "自定义")),
            verified=True,
            error=None,
            source="llm",
            retry_count=attempt,
        )

    # 两次都失败：回退模板，但把错误带上，便于排查
    out = template_for_intent(intent)
    return FactorFactoryResult(
        name=out["name"],
        expression=out["expression"],
        sign=out["sign"],
        logic=out["logic"],
        category=out["category"],
        verified=True,
        error=last_error,
        source="template",
        retry_count=max_attempts,
    )
