"""因子表达式 DSL —— 安全编译层。

为什么要自造 DSL 而不是直接 exec(LLM 生成的代码)：
    1) 安全：LLM 的输出是不可信文本。DSL 只允许白名单里的算子与字段，
       任何属性访问、下标、lambda、推导式、导入都在 AST 校验阶段被拒。
    2) 可审计：一个因子就是一行字符串，能进 git、能 diff、能复现。
    3) 可修复：报错信息能精确回喂给模型（"未知字段 volume_5d，可用字段 …"），
       形成 生成 → 编译 → 运行 → 报错 → 修复 的闭环。

表达式示例：
    close / delay(close, 20) - 1                     # 20 日动量
    -ts_mean(close / delay(close, 1) - 1, 5)         # 5 日反转
    ts_corr(close, volume, 20)                       # 量价背离
    cs_rank(ts_std(close / delay(close, 1) - 1, 20)) # 截面排序后的波动率
"""

from __future__ import annotations

import ast
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 常量

# 允许直接引用的行情字段（由 MarketPanel 提供）
ALLOWED_FIELDS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_chg",
    "turnover",
    "amplitude",
    "vwap",       # 派生：amount / (volume*100)
    "ret",        # 派生：close / delay(close,1) - 1
    "hl_range",   # 派生：(high - low) / close
}

MAX_EXPR_LEN = 4000
MAX_NODES = 600


class DSLSyntaxError(ValueError):
    """表达式语法/白名单校验失败。"""


class DSLEvalError(RuntimeError):
    """表达式运行期失败（字段缺失、长度不足等）。"""


# ---------------------------------------------------------------- 工具

def _align(value, index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    """把标量 / Series / DataFrame 统一成与面板同形的 DataFrame。"""
    if isinstance(value, pd.DataFrame):
        return value.reindex(index=index, columns=columns)
    if isinstance(value, pd.Series):
        # 一行（按 code）或一列（按 date）都尝试对齐
        if set(value.index) <= set(columns):
            return pd.DataFrame(
                np.tile(value.reindex(columns).to_numpy(dtype=float), (len(index), 1)),
                index=index,
                columns=columns,
            )
        return pd.DataFrame(
            np.tile(value.reindex(index).to_numpy(dtype=float)[:, None], (1, len(columns))),
            index=index,
            columns=columns,
        )
    return pd.DataFrame(float(value), index=index, columns=columns)


def _min_periods(n: int) -> int:
    """滚动窗口的最小样本数：窗口的 60%，至少 2。"""
    return max(2, int(n * 0.6))


# ---------------------------------------------------------------- 时序算子

def _ts_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_min_periods(n)).mean()


def _ts_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_min_periods(n)).std()


def _ts_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_min_periods(n)).sum()


def _ts_min(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_min_periods(n)).min()


def _ts_max(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_min_periods(n)).max()


def _ts_median(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_min_periods(n)).median()


def _ts_skew(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=max(5, _min_periods(n))).skew()


def _ts_delta(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x - x.shift(n)


def _ts_pct(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x / x.shift(n).replace(0, np.nan) - 1.0


def _ts_zscore(x: pd.DataFrame, n: int) -> pd.DataFrame:
    mean = _ts_mean(x, n)
    std = _ts_std(x, n)
    return (x - mean) / std.replace(0, np.nan)


def _ts_rank(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """当前值在过去 n 日窗口内的分位（0–1），越大越接近窗口高点。"""
    def _last_rank(arr: np.ndarray) -> float:
        if np.isnan(arr[-1]):
            return np.nan
        valid = arr[~np.isnan(arr)]
        if len(valid) < 2:
            return np.nan
        return float((valid < valid[-1]).mean())

    return x.rolling(n, min_periods=_min_periods(n)).apply(_last_rank, raw=True)


def _ts_corr(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_min_periods(n)).corr(y)


def _ts_cov(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_min_periods(n)).cov(y)


def _decay_linear(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """线性衰减加权均值：越近的样本权重越大。"""
    weights = np.arange(1, n + 1, dtype=float)
    weights /= weights.sum()

    def _wavg(arr: np.ndarray) -> float:
        if np.isnan(arr).any():
            return np.nan
        return float(np.dot(arr, weights))

    return x.rolling(n, min_periods=n).apply(_wavg, raw=True)


def _delay(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.shift(n)


# ---------------------------------------------------------------- 截面算子

def _cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)


def _cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    return x.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)


def _cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0)


def _cs_scale(x: pd.DataFrame) -> pd.DataFrame:
    """截面绝对值之和归一到 1（组合权重常用）。"""
    denom = x.abs().sum(axis=1).replace(0, np.nan)
    return x.div(denom, axis=0)


# ---------------------------------------------------------------- 通用数学

def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a / b.replace(0, np.nan)


def _clip(x: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    return x.clip(lower=lo, upper=hi)


def _where(cond: pd.DataFrame, a, b) -> pd.DataFrame:
    return cond.where(cond.astype(bool), b) if isinstance(a, pd.DataFrame) else cond


def _winsorize(x: pd.DataFrame, q: float = 0.01) -> pd.DataFrame:
    """按行（截面）缩尾，把极端值压到分位点。"""
    lo = x.quantile(q, axis=1)
    hi = x.quantile(1 - q, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)


# 名字 -> (最小参数, 最大参数, 实现)
FUNCTIONS: dict[str, tuple[int, int, Callable]] = {
    # 时序
    "ts_mean": (2, 2, _ts_mean),
    "ts_std": (2, 2, _ts_std),
    "ts_sum": (2, 2, _ts_sum),
    "ts_min": (2, 2, _ts_min),
    "ts_max": (2, 2, _ts_max),
    "ts_median": (2, 2, _ts_median),
    "ts_skew": (2, 2, _ts_skew),
    "ts_delta": (2, 2, _ts_delta),
    "ts_pct": (2, 2, _ts_pct),
    "ts_zscore": (2, 2, _ts_zscore),
    "ts_rank": (2, 2, _ts_rank),
    "ts_corr": (3, 3, _ts_corr),
    "ts_cov": (3, 3, _ts_cov),
    "decay_linear": (2, 2, _decay_linear),
    "delay": (2, 2, _delay),
    # 截面
    "cs_rank": (1, 1, _cs_rank),
    "cs_zscore": (1, 1, _cs_zscore),
    "cs_demean": (1, 1, _cs_demean),
    "cs_scale": (1, 1, _cs_scale),
    "winsorize": (1, 2, _winsorize),
    # 数学
    "abs": (1, 1, lambda x: x.abs()),
    "log": (1, 1, lambda x: np.log(x.where(x > 0))),
    "sqrt": (1, 1, lambda x: np.sqrt(x.where(x >= 0))),
    "sign": (1, 1, lambda x: np.sign(x)),
    "exp": (1, 1, lambda x: np.exp(x.clip(-50, 50))),
    "floor": (1, 1, lambda x: np.floor(x)),
    "ceil": (1, 1, lambda x: np.ceil(x)),
    "clip": (3, 3, _clip),
    "safe_div": (2, 2, _safe_div),
    "max": (2, 2, lambda a, b: np.maximum(a, b)),
    "min": (2, 2, lambda a, b: np.minimum(a, b)),
    "pow": (2, 2, lambda a, b: np.power(a.clip(-1e6, 1e6), b)),
    # 别名
    "rank": (1, 1, _cs_rank),
    "zscore": (1, 1, _cs_zscore),
    "mean": (2, 2, _ts_mean),
    "std": (2, 2, _ts_std),
    "corr": (3, 3, _ts_corr),
    "delta": (2, 2, _ts_delta),
}

# 允许出现的 AST 节点（白名单，其余一律拒）
ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.IfExp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


# ---------------------------------------------------------------- 校验

def validate(expr: str, extra_fields: set[str] | None = None) -> ast.Expression:
    """校验表达式，返回 AST。任何不合规都抛 DSLSyntaxError。"""
    if not isinstance(expr, str) or not expr.strip():
        raise DSLSyntaxError("表达式为空")
    if len(expr) > MAX_EXPR_LEN:
        raise DSLSyntaxError(f"表达式过长（{len(expr)} > {MAX_EXPR_LEN}）")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise DSLSyntaxError(f"语法错误：{exc.msg}（第 {exc.offset} 字符）") from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise DSLSyntaxError(f"表达式节点过多（{len(nodes)} > {MAX_NODES}）")

    allowed_fields = set(ALLOWED_FIELDS) | set(extra_fields or set())

    for node in nodes:
        if not isinstance(node, ALLOWED_NODES):
            raise DSLSyntaxError(f"不允许的语法结构：{type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id not in allowed_fields and node.id not in FUNCTIONS:
                hint = "、".join(sorted(list(allowed_fields))[:12])
                raise DSLSyntaxError(f"未知标识符 '{node.id}'。可用字段示例：{hint}…")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise DSLSyntaxError("函数调用只允许直接用函数名，不允许属性调用")
            fname = node.func.id
            if fname not in FUNCTIONS:
                raise DSLSyntaxError(f"未知函数 '{fname}'。可用函数：{'、'.join(sorted(FUNCTIONS))}")
            lo, hi, _ = FUNCTIONS[fname]
            if not (lo <= len(node.args) <= hi):
                raise DSLSyntaxError(f"函数 {fname} 需要 {lo}–{hi} 个参数，实际 {len(node.args)} 个")
            if node.keywords:
                raise DSLSyntaxError(f"函数 {fname} 不支持关键字参数")
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float, bool)):
                raise DSLSyntaxError(f"不支持的常量类型：{type(node.value).__name__}")
    return tree


# ---------------------------------------------------------------- 求值

class _Evaluator:
    def __init__(self, ctx: dict[str, pd.DataFrame], index: pd.Index, columns: pd.Index):
        self.ctx = ctx
        self.index = index
        self.columns = columns

    def run(self, node: ast.AST):
        method = getattr(self, f"_eval_{type(node).__name__}", None)
        if method is None:
            raise DSLEvalError(f"无法求值节点 {type(node).__name__}")
        return method(node)

    # --- 叶子 ---
    def _eval_Expression(self, node: ast.Expression):
        return self.run(node.body)

    def _eval_Constant(self, node: ast.Constant):
        return float(node.value)

    def _eval_Name(self, node: ast.Name):
        if node.id not in self.ctx:
            raise DSLEvalError(f"字段 '{node.id}' 未提供")
        return self.ctx[node.id]

    # --- 运算 ---
    def _eval_BinOp(self, node: ast.BinOp):
        left = _align(self.run(node.left), self.index, self.columns)
        right = _align(self.run(node.right), self.index, self.columns)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right.replace(0, np.nan)
        if isinstance(op, ast.Pow):
            return np.power(left.clip(-1e6, 1e6), right.clip(-8, 8))
        if isinstance(op, ast.Mod):
            return left % right.replace(0, np.nan)
        raise DSLEvalError(f"不支持的运算符 {type(op).__name__}")

    def _eval_UnaryOp(self, node: ast.UnaryOp):
        value = _align(self.run(node.operand), self.index, self.columns)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.Not):
            return ~value.astype(bool)
        raise DSLEvalError(f"不支持的一元运算 {type(node.op).__name__}")

    def _eval_BoolOp(self, node: ast.BoolOp):
        values = [_align(self.run(v), self.index, self.columns).astype(bool) for v in node.values]
        result = values[0]
        for value in values[1:]:
            result = (result & value) if isinstance(node.op, ast.And) else (result | value)
        return result

    def _eval_Compare(self, node: ast.Compare):
        left = _align(self.run(node.left), self.index, self.columns)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            right = _align(self.run(comparator), self.index, self.columns)
            if isinstance(op, ast.Gt):
                piece = left > right
            elif isinstance(op, ast.GtE):
                piece = left >= right
            elif isinstance(op, ast.Lt):
                piece = left < right
            elif isinstance(op, ast.LtE):
                piece = left <= right
            elif isinstance(op, ast.Eq):
                piece = left == right
            elif isinstance(op, ast.NotEq):
                piece = left != right
            else:
                raise DSLEvalError(f"不支持的比较 {type(op).__name__}")
            result = piece if result is None else (result & piece)
            left = right
        return result

    def _eval_IfExp(self, node: ast.IfExp):
        cond = _align(self.run(node.test), self.index, self.columns).astype(bool)
        a = _align(self.run(node.body), self.index, self.columns)
        b = _align(self.run(node.orelse), self.index, self.columns)
        return a.where(cond, b)

    def _eval_Call(self, node: ast.Call):
        name = node.func.id
        _, _, impl = FUNCTIONS[name]
        args = [self.run(arg) for arg in node.args]

        # 时序算子的窗口参数必须是正整数常量
        ts_funcs = {
            "ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max", "ts_median", "ts_skew",
            "ts_delta", "ts_pct", "ts_zscore", "ts_rank", "ts_corr", "ts_cov",
            "decay_linear", "delay", "mean", "std", "corr", "delta",
        }
        if name in ts_funcs:
            window = args[-1]
            if not isinstance(window, (int, float)) or float(window) != int(window) or window < 1:
                raise DSLEvalError(f"{name} 的窗口参数必须是正整数，实际 {window!r}")
            args[-1] = int(window)

        # 第一个参数（及 ts_corr 的第二个）必须是面板数据
        if name not in ("cs_rank", "cs_zscore", "cs_demean", "cs_scale", "winsorize"):
            for i in range(min(2, len(args)) if name in ("ts_corr", "ts_cov", "corr") else 1):
                args[i] = _align(args[i], self.index, self.columns)
        else:
            args[0] = _align(args[0], self.index, self.columns)

        try:
            out = impl(*args)
        except DSLEvalError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise DSLEvalError(f"函数 {name} 执行失败：{exc}") from exc

        if isinstance(out, (pd.DataFrame, pd.Series)):
            return _align(out, self.index, self.columns)
        return float(out)


def build_context(panel) -> dict[str, pd.DataFrame]:
    """从 MarketPanel 构造 DSL 上下文（宽表字段字典）。"""
    ctx: dict[str, pd.DataFrame] = {}
    for field_name in ("open", "high", "low", "close", "volume", "amount",
                       "pct_chg", "turnover", "amplitude"):
        try:
            ctx[field_name] = panel.wide(field_name)
        except KeyError:
            continue

    close = ctx["close"]
    if "amount" in ctx and "volume" in ctx:
        # 成交量单位是「手」，1 手 = 100 股
        ctx["vwap"] = ctx["amount"] / (ctx["volume"].replace(0, np.nan) * 100)
    ctx["ret"] = close.pct_change(fill_method=None)
    ctx["hl_range"] = (ctx["high"] - ctx["low"]) / close.replace(0, np.nan)
    return ctx


def evaluate(expr: str, ctx: dict[str, pd.DataFrame], index=None, columns=None) -> pd.DataFrame:
    """编译 + 求值，一步到位。"""
    tree = validate(expr, extra_fields=set(ctx) - ALLOWED_FIELDS)
    if index is None or columns is None:
        ref = next(iter(ctx.values()))
        index = ref.index if index is None else index
        columns = ref.columns if columns is None else columns
    result = _Evaluator(ctx, index, columns).run(tree)
    if not isinstance(result, pd.DataFrame):
        # 纯常数表达式：广播成同形表
        result = _align(result, index, columns)
    return result.replace([np.inf, -np.inf], np.nan)
