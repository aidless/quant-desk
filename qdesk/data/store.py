"""行情面板：缓存、宽表化、派生字段。

所有下游模块（因子、回测、平台）都只跟 MarketPanel 打交道，
不直接碰网络或文件格式。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CONFIG, DataConfig
from . import eastmoney


# 面板里保留的原始字段
BASE_FIELDS = ["open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"]


def _limit_pct(code: str, name: str = "") -> float:
    """按板块返回涨跌停幅度（百分比）。

    - 科创板 688/689、创业板 300/301：20%
    - ST/*ST：5%
    - 其余主板：10%
    """
    if "ST" in str(name).upper():
        return 5.0
    if code.startswith(("688", "689", "300", "301")):
        return 20.0
    return 10.0


@dataclass
class MarketPanel:
    """一只股票池的完整行情面板。"""

    long: pd.DataFrame                     # 长表：date, code, 各字段
    config: DataConfig = field(default_factory=lambda: CONFIG.data)
    _wide_cache: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _meta: pd.DataFrame | None = field(default=None, repr=False)

    # ---------- 基本属性 ----------

    @property
    def codes(self) -> list[str]:
        return sorted(self.long["code"].unique().tolist())

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.long["date"].unique()))

    @property
    def meta(self) -> pd.DataFrame:
        """每只股票的静态信息：code, name, industry, circ_mv。"""
        if self._meta is None:
            cols = [c for c in ("code", "name", "industry") if c in self.long.columns]
            self._meta = self.long[cols].drop_duplicates("code").set_index("code")
        return self._meta

    def name_of(self, code: str) -> str:
        try:
            return str(self.meta.loc[code, "name"])
        except Exception:  # noqa: BLE001
            return code

    # ---------- 宽表 ----------

    def wide(self, field_name: str) -> pd.DataFrame:
        """把长表某个字段转成 date × code 宽表（带缓存）。"""
        if field_name in self._wide_cache:
            return self._wide_cache[field_name]
        if field_name not in self.long.columns:
            raise KeyError(f"面板没有字段 {field_name}，可用：{sorted(self.long.columns)}")
        wide = self.long.pivot(index="date", columns="code", values=field_name).sort_index()
        self._wide_cache[field_name] = wide
        return wide

    # ---------- 派生字段 ----------

    @property
    def close(self) -> pd.DataFrame:
        return self.wide("close")

    @property
    def ret(self) -> pd.DataFrame:
        """日收益率（收盘对收盘）。"""
        if "ret" not in self._wide_cache:
            self._wide_cache["ret"] = self.close.pct_change(fill_method=None)
        return self._wide_cache["ret"]

    def fwd_ret(self, horizon: int = 1) -> pd.DataFrame:
        """未来 horizon 日收益：close[t+h]/close[t] - 1。用于因子评价，不用于交易。"""
        key = f"fwd_ret_{horizon}"
        if key not in self._wide_cache:
            close = self.close
            self._wide_cache[key] = close.shift(-horizon) / close - 1.0
        return self._wide_cache[key]

    @property
    def limit_up(self) -> pd.DataFrame:
        """当日是否涨停（无法买入）。"""
        if "limit_up" not in self._wide_cache:
            pct = self.wide("pct_chg")
            names = self.meta["name"].to_dict() if "name" in self.meta.columns else {}
            thresholds = pd.Series(
                {code: _limit_pct(code, names.get(code, "")) for code in pct.columns}
            )
            # 留 0.3 个百分点容差：9.8% 以上即视为封板
            self._wide_cache["limit_up"] = pct >= (thresholds - 0.3)
        return self._wide_cache["limit_up"]

    @property
    def limit_down(self) -> pd.DataFrame:
        if "limit_down" not in self._wide_cache:
            pct = self.wide("pct_chg")
            names = self.meta["name"].to_dict() if "name" in self.meta.columns else {}
            thresholds = pd.Series(
                {code: _limit_pct(code, names.get(code, "")) for code in pct.columns}
            )
            self._wide_cache["limit_down"] = pct <= -(thresholds - 0.3)
        return self._wide_cache["limit_down"]

    @property
    def suspended(self) -> pd.DataFrame:
        """停牌 / 无成交。"""
        if "suspended" not in self._wide_cache:
            vol = self.wide("volume")
            self._wide_cache["suspended"] = (vol.fillna(0) <= 0)
        return self._wide_cache["suspended"]

    def tradable(self, min_amount: float = 3e7) -> pd.DataFrame:
        """可买掩码：有成交、成交额达标、非涨停。

        注意：涨跌停/停牌状态用「当日」数据判断，实际下单在 T+1，
        回测引擎里会再对 T+1 当日的状态做一次校验（见 backtest/engine.py）。
        """
        key = f"tradable_{min_amount}"
        if key not in self._wide_cache:
            amount = self.wide("amount")
            mask = amount.fillna(0) >= min_amount
            mask &= ~self.suspended
            mask &= ~self.limit_up.fillna(True)
            self._wide_cache[key] = mask
        return self._wide_cache[key]

    # ---------- 统计 ----------

    def summary(self) -> dict:
        return {
            "codes": len(self.codes),
            "dates": len(self.dates),
            "start": str(self.dates[0].date()) if len(self.dates) else None,
            "end": str(self.dates[-1].date()) if len(self.dates) else None,
            "rows": int(len(self.long)),
        }


class PanelStore:
    """负责「有缓存读缓存、没缓存拉网络」。"""

    def __init__(self, config: DataConfig | None = None):
        self.config = config or CONFIG.data
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 路径 ----------

    @property
    def panel_path(self) -> Path:
        size = self.config.universe_size
        return self.cache_dir / f"panel_{size}_{self.config.start_date}_{self.config.end_date}.parquet"

    @property
    def universe_path(self) -> Path:
        return self.cache_dir / f"universe_{self.config.universe_size}.parquet"

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        return age_hours <= self.config.cache_ttl_hours

    # ---------- 股票池 ----------

    def universe(self, force: bool = False) -> pd.DataFrame:
        if not force and self._is_fresh(self.universe_path):
            return pd.read_parquet(self.universe_path)
        df = eastmoney.fetch_universe(self.config.universe_size)
        df.to_parquet(self.universe_path, index=False)
        return df

    # ---------- 面板 ----------

    def panel(self, force: bool = False, verbose: bool = True) -> MarketPanel:
        if not force and self._is_fresh(self.panel_path):
            long = pd.read_parquet(self.panel_path)
            long["date"] = pd.to_datetime(long["date"])
            if verbose:
                print(f"  [store] 命中缓存 {self.panel_path.name}（{len(long)} 行）")
            return MarketPanel(long=long, config=self.config)

        uni = self.universe(force=force)
        if verbose:
            print(f"  [store] 拉取 {len(uni)} 只股票日线，区间 {self.config.start_date}–{self.config.end_date}")
        long = eastmoney.fetch_panel(
            uni["code"].tolist(),
            start=self.config.start_date,
            end=self.config.end_date,
            sleep=self.config.fetch_sleep,
            verbose=verbose,
        )
        # 把行业信息并进来，方便后面做行业中性
        if "industry" in uni.columns:
            industry = uni.set_index("code")["industry"].to_dict()
            long["industry"] = long["code"].map(industry)
        long.to_parquet(self.panel_path, index=False)
        if verbose:
            print(f"  [store] 已缓存 {self.panel_path.name}（{len(long)} 行）")
        return MarketPanel(long=long, config=self.config)

    # ---------- 状态 ----------

    def status(self) -> dict:
        return {
            "cache_dir": str(self.cache_dir),
            "panel_cached": self.panel_path.exists(),
            "panel_fresh": self._is_fresh(self.panel_path),
            "panel_size_mb": round(self.panel_path.stat().st_size / 1e6, 2) if self.panel_path.exists() else 0,
            "universe_cached": self.universe_path.exists(),
            "universe_size": self.config.universe_size,
            "start": self.config.start_date,
            "end": self.config.end_date,
        }


def build_panel_from_long(long: pd.DataFrame) -> MarketPanel:
    """从内存长表直接构造面板（测试用，绕过网络）。"""
    return MarketPanel(long=long.copy())


def make_synthetic_panel(n_codes: int = 40, n_days: int = 300, seed: int = 7) -> MarketPanel:
    """构造合成面板，用于单元测试与离线演示。

    生成带趋势/反转结构的随机游走，使因子评价有非平凡结果。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    codes = [f"{600000 + i:06d}" for i in range(n_codes)]
    rows = []
    for j, code in enumerate(codes):
        drift = rng.normal(0, 0.0004)
        # 给每只股票一个可被动量因子捕捉的自相关成分
        ar = rng.normal(0, 0.004, n_days)
        eps = rng.normal(0, 0.02, n_days)
        rets = np.zeros(n_days)
        for t in range(1, n_days):
            rets[t] = drift + 0.08 * ar[t - 1] + eps[t]
        close = 10 * np.cumprod(1 + rets)
        high = close * (1 + np.abs(rng.normal(0, 0.01, n_days)))
        low = close * (1 - np.abs(rng.normal(0, 0.01, n_days)))
        open_ = close * (1 + rng.normal(0, 0.005, n_days))
        volume = np.abs(rng.normal(2e6, 5e5, n_days))
        # 成交额量级对齐真实 A 股（亿元级，>= BacktestConfig.min_amount=3e7），
        # 否则合成面板会被 tradable 掩码整体过滤，回测/演示权重全为 0。
        volume = np.abs(rng.normal(6e6, 1.5e6, n_days))  # 手 → 股数口径见下方注释
        amount = volume * close  # 约 6e6 股 × ~10 元 ≈ 6e7 元，跨过 3e7 门槛
        rows.append(
            pd.DataFrame(
                {
                    "code": code,
                    "name": f"SYN{j:03d}",
                    "market": 1,
                    "date": dates,
                    "open": open_,
                    "close": close,
                    "high": np.maximum.reduce([high, open_, close]),
                    "low": np.minimum.reduce([low, open_, close]),
                    "volume": volume,
                    "amount": amount,
                    "pct_chg": np.r_[0.0, (close[1:] / close[:-1] - 1) * 100],
                    "turnover": volume / 1e8 * 100,
                    "amplitude": (high - low) / close * 100,
                    "change": np.r_[0.0, np.diff(close)],
                    "industry": f"IND{j % 5}",
                }
            )
        )
    long = pd.concat(rows, ignore_index=True)
    return MarketPanel(long=long)
