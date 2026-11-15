"""全局配置。

设计原则：所有可变量走环境变量，代码里只放默认值。
这样同一份代码既能跑本机（本地网关），也能跑云端（DeepSeek 官方 API）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# 项目根目录 = 本文件所在目录的上两级（qdesk/config.py -> qdesk -> quant-desk）
ROOT = Path(__file__).resolve().parent.parent


def _env_str(key: str, default: str) -> str:
    value = os.environ.get(key)
    return value if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class DataConfig:
    """行情数据配置。"""

    # 本地缓存目录（parquet）。第一次拉取后所有分析都读这里，不再打网络。
    cache_dir: Path = field(default_factory=lambda: Path(_env_str("QD_CACHE_DIR", str(ROOT / "cache"))))
    # 股票池规模：按流通市值取前 N 只（None 表示全市场）
    universe_size: int = _env_int("QD_UNIVERSE_SIZE", 300)
    # 行情起止日期（YYYYMMDD 字符串，东财接口格式）
    start_date: str = _env_str("QD_START_DATE", "20200101")
    end_date: str = _env_str("QD_END_DATE", "20991231")
    # 抓取并发与限速：东财对高频请求会 403，默认保守
    fetch_workers: int = _env_int("QD_FETCH_WORKERS", 6)
    fetch_sleep: float = _env_float("QD_FETCH_SLEEP", 0.06)
    # 缓存有效期（小时）。超过则重新拉取。
    cache_ttl_hours: float = _env_float("QD_CACHE_TTL_HOURS", 24.0)


@dataclass
class LLMConfig:
    """DeepSeek 通道配置。

    默认指向本机 Magpie 网关（OpenAI 兼容）。要直连官方 API 只需：
        export QD_LLM_BASE_URL=https://api.deepseek.com/v1
        export QD_LLM_MODEL=deepseek-chat
        export QD_LLM_API_KEY=sk-xxxx
    """

    base_url: str = _env_str("QD_LLM_BASE_URL", "http://127.0.0.1:3425/v1")
    model: str = _env_str("QD_LLM_MODEL", "deepseek-v4.1-flash")
    api_key: str = _env_str("QD_LLM_API_KEY", "local")
    temperature: float = _env_float("QD_LLM_TEMPERATURE", 0.3)
    max_tokens: int = _env_int("QD_LLM_MAX_TOKENS", 1600)
    timeout: float = _env_float("QD_LLM_TIMEOUT", 120.0)
    retries: int = _env_int("QD_LLM_RETRIES", 2)
    # 关掉后所有 LLM 调用走本地确定性回退（离线也能演示全流程）
    enabled: bool = _env_str("QD_LLM_ENABLED", "1") not in ("0", "false", "False")


@dataclass
class BacktestConfig:
    """回测默认参数（A 股规则）。"""

    # 选股：取因子值最高的 top_frac 比例
    top_frac: float = _env_float("QD_TOP_FRAC", 0.2)
    # 最小持仓只数 / 最大持仓只数
    min_holdings: int = _env_int("QD_MIN_HOLDINGS", 10)
    max_holdings: int = _env_int("QD_MAX_HOLDINGS", 60)
    # 换仓间隔（交易日），1 = 每日调仓
    rebalance_days: int = _env_int("QD_REBALANCE_DAYS", 5)
    # 成本（基点，1bp = 0.01%）
    commission_bp: float = _env_float("QD_COMMISSION_BP", 2.5)
    stamp_bp: float = _env_float("QD_STAMP_BP", 5.0)      # 印花税，仅卖出
    slippage_bp: float = _env_float("QD_SLIPPAGE_BP", 5.0)
    # 流动性过滤：日成交额下限（元），低于此不买
    min_amount: float = _env_float("QD_MIN_AMOUNT", 3e7)
    # 涨跌停不买（无法成交）
    exclude_limit_up: bool = True
    # 停牌（成交量为 0）不买
    exclude_suspended: bool = True
    # 年化交易日
    trading_days: int = 252


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


CONFIG = Config()
