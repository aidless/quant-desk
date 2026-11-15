"""量化交易平台入口。"""

from .config import CONFIG, Config, BacktestConfig, DataConfig, LLMConfig
from .data import MarketPanel, PanelStore, build_panel_from_long, make_synthetic_panel
from .factors.dsl import evaluate as dsl_evaluate, validate, build_context, DSLSyntaxError, DSLEvalError
from .factors.engine import compute, FactorResult
from .factors.library import BUILTIN_FACTORS, FACTOR_MAP, get, catalog
from .factors.evaluate import evaluate_factor, ic_series, quantile_returns, decay_table
from .backtest.metrics import (
    total_return, annual_return, annual_vol, sharpe, max_drawdown,
    drawdown_series, calmar, win_rate, t_stat, information_ratio, summarize
)
from .backtest.engine import run, run_expression, BacktestResult
from .llm.client import make_client, build_factor_prompt, LLMClientError
from .llm.factor_factory import llm_build, FactorFactoryResult
from .app import app

__all__ = [
    "CONFIG", "Config", "DataConfig", "LLMConfig", "BacktestConfig",
    "MarketPanel", "PanelStore", "build_panel_from_long", "make_synthetic_panel",
    "validate", "DSLSyntaxError", "DSLEvalError", "build_context", "dsl_evaluate",
    "compute", "FactorResult",
    "BUILTIN_FACTORS", "FACTOR_MAP", "get", "catalog",
    "evaluate_factor", "ic_series", "quantile_returns", "decay_table",
    "total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown",
    "drawdown_series", "calmar", "win_rate", "t_stat", "information_ratio", "summarize",
    "run", "run_expression", "BacktestResult",
    "make_client", "build_factor_prompt", "LLMClientError",
    "llm_build", "FactorFactoryResult",
    "app",
]
