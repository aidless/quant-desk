"""DeepSeek 客户端（Magpie 网关/官方 API 双模）。"""

from __future__ import annotations

import json
import time
from typing import Any

import requests

from ..config import CONFIG, LLMConfig



def _completions_url(base_url: str) -> str:
    """归一补全 chat/completions 端点（base_url 带/不带 /v1 均可）。"""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


class LLMClientError(RuntimeError):
    """LLM 调用失败。"""


def make_client(config: LLMConfig | None = None):
    config = config or CONFIG.llm
    cls = _LocalClient if not config.api_key.startswith("sk-") else _APIKeyClient
    return cls(config)


class _LocalClient:
    """本机 Magpie 网关模式（无 API key，走本地模型）。"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout
        self.retries = config.retries

    def chat(self, messages: list[dict], **kwargs) -> str:
        url = _completions_url(self.base_url)
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(url, json=data, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                content = payload["choices"][0]["message"]["content"].strip()
                return content
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                raise LLMClientError(f"聊天失败 {self.model}: {exc}") from exc


class _APIKeyClient:
    """官方 API 模式（需要 sk-xxx key）。"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.api_key = config.api_key
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout
        self.retries = config.retries

    def chat(self, messages: list[dict], **kwargs) -> str:
        url = _completions_url(self.base_url)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = requests.post(url, json=data, headers=headers, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                content = payload["choices"][0]["message"]["content"].strip()
                return content
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                raise LLMClientError(f"聊天失败 {self.model} @ {url}: {exc}") from exc


def build_factor_prompt(intent: str, available_factors: list[str] | None = None) -> list[dict]:
    """构造「因子工厂」的系统提示 + 用户输入。"""
    known_catalog = available_factors or []
    known_str = "\n".join(f"- {f}" for f in known_catalog[:20]) if len(known_catalog) else "(空)"

    system = (
        "你是一个中文量化研究助手。你的任务是把用户的自然语言描述翻译成「安全 DSL 表达式"
        "」并给出逻辑说明。\n\n约束\n"
        "1. 只使用允许字段与函数：close/open/low/high/volume/amount/pct_chg/turnover/amplitude/"
        "ret/vwap/hl_range；\n"
        "   函数：ts_mean/ts_std/ts_sum/ts_min/ts_max/ts_median/ts_skew/ts_delta/ts_pct/"
        "ts_zscore/ts_rank/ts_corr/ts_cov/decay_linear/delay/cs_rank/cs_zscore/cs_demean/"
        "cs_scale/winsorize/safe_div/max/min/abs/log/sqrt/sign/exp/floor/ceil/clip/pow.\n"
        "2. 表达式长度必须≤300字符；不能含 lambda、推导式、导入、属性访问等任何危险语法。\n"
        "3. 先输出一个结构化的 JSON，然后给一句简短的中文逻辑解释（不超过 50 字）。\n\n"
        f"可用内置因子示例：\n{known_str}\n\n"
        "输出格式：先输出一个 JSON 对象（不要多余字符），字段为：\n"
        '  "name": string, "expression": string(≤300字符), "sign": 1 或 -1, '
        '"logic": string(≤50字), "category": "动量|反转|波动|形态|量价|流动性|彩票"\n'
        "示例：{\"name\":\"mom_20\",\"expression\":\"close / delay(close, 20) - 1\","
        "\"sign\":1,\"logic\":\"20日动量\",\"category\":\"动量\"}\n\n"
        "现在开始：请根据用户意图生成一个或多个因子建议。\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": intent},
    ]
