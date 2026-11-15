"""东方财富行情抓取（免费公开接口，无需 token）。

两个端点：
  1) 股票池：/api/qt/clist/get        —— 全市场快照，按流通市值排序
  2) 日线：  /api/qt/stock/kline/get  —— 单只股票历史 K 线（前复权）

接口返回的是 "a,b,c" 逗号串，字段靠 fields2 的顺序决定，这里显式映射，
不做位置猜测。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests

CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"

# 沪深主板 + 创业板 + 科创板。北交所（m:0 t:81 等）波动规则不同，先不纳入。
MARKET_FILTER = "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23"

# clist 字段含义
CLIST_FIELDS = {
    "f12": "code",
    "f14": "name",
    "f2": "price",
    "f3": "pct_chg",
    "f5": "volume",
    "f6": "amount",
    "f8": "turnover",
    "f9": "pe",
    "f13": "market",
    "f20": "total_mv",
    "f21": "circ_mv",
    "f100": "industry",
}

# kline 字段含义（fields2 的顺序必须与此一致）
KLINE_FIELDS2 = "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
KLINE_COLUMNS = [
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",      # 手
    "amount",      # 元
    "amplitude",   # 振幅 %
    "pct_chg",     # 涨跌幅 %
    "change",      # 涨跌额
    "turnover",    # 换手率 %
]


class FetchError(RuntimeError):
    """抓取失败（网络、限流、字段缺失）。"""


def _get_json(url: str, params: dict, timeout: float = 15.0, retries: int = 3) -> dict:
    """带重试的 GET，返回解析后的 JSON。"""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("data") is None:
                raise FetchError(f"接口返回空数据：{url} params={params}")
            return payload
        except Exception as exc:  # noqa: BLE001 —— 统一转成 FetchError 交给上层
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    raise FetchError(f"请求失败 {url}: {last_error}")


def secid(code: str, market: int | None = None) -> str:
    """把 6 位代码转成东财的 secid（1=沪市，0=深市）。"""
    if market is None:
        market = 1 if code[0] in ("6", "9") else 0
    return f"{market}.{code}"


def fetch_universe(size: int | None = 300, page_size: int = 100) -> pd.DataFrame:
    """抓股票池，按流通市值降序取前 size 只。

    返回列：code, name, price, pct_chg, amount, turnover, pe, market, total_mv,
            circ_mv, industry
    """
    rows: list[dict] = []
    page = 1
    while True:
        params = {
            "pn": page,
            "pz": page_size,
            "po": 1,          # 降序
            "np": 1,
            "fltt": 2,
            "invt": 2,
            "fid": "f21",     # 按流通市值排序
            "fs": MARKET_FILTER,
            "fields": ",".join(CLIST_FIELDS.keys()),
        }
        payload = _get_json(CLIST_URL, params)
        data = payload["data"]
        diff = data.get("diff") or []
        if not diff:
            break
        for item in diff:
            row = {CLIST_FIELDS[k]: item.get(k) for k in CLIST_FIELDS}
            rows.append(row)
        if size is not None and len(rows) >= size:
            break
        if len(rows) >= data.get("total", 0):
            break
        page += 1
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if df.empty:
        raise FetchError("股票池为空，检查网络或接口变更")
    # 过滤掉退市/停牌无价标的
    df = df[df["price"].notna() & (df["price"] > 0)]
    df = df.sort_values("circ_mv", ascending=False).reset_index(drop=True)
    if size is not None:
        df = df.head(size)
    return df


def fetch_kline(
    code: str,
    start: str = "20200101",
    end: str = "20991231",
    market: int | None = None,
    adjust: int = 1,
) -> pd.DataFrame:
    """抓单只股票日线。adjust: 0=不复权 1=前复权 2=后复权。"""
    params = {
        "secid": secid(code, market),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": KLINE_FIELDS2,
        "klt": 101,     # 101=日线
        "fqt": adjust,
        # 东财接口只认紧凑日期（20230103），带横线会静默返回空 klines
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
    }
    payload = _get_json(KLINE_URL, params)
    data = payload["data"]
    klines = data.get("klines") or []
    if not klines:
        return pd.DataFrame(columns=["code"] + KLINE_COLUMNS)

    records = []
    for line in klines:
        parts = line.split(",")
        if len(parts) != len(KLINE_COLUMNS):
            continue
        records.append(parts)

    df = pd.DataFrame(records, columns=KLINE_COLUMNS)
    df["date"] = pd.to_datetime(df["date"])
    for col in KLINE_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["code"] = code
    df["name"] = data.get("name", code)
    df["market"] = data.get("market", market if market is not None else 0)
    return df[["code", "name", "market"] + KLINE_COLUMNS]


def fetch_panel(
    codes: Iterable[str],
    start: str,
    end: str,
    sleep: float = 0.06,
    verbose: bool = False,
) -> pd.DataFrame:
    """批量抓日线，返回长表（date, code, ...）。单只失败不影响整体。"""
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    codes = list(codes)
    for idx, code in enumerate(codes, 1):
        try:
            one = fetch_kline(code, start=start, end=end)
            if not one.empty:
                frames.append(one)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{code}:{exc}")
        if sleep:
            time.sleep(sleep)
        if verbose and idx % 50 == 0:
            print(f"  [data] {idx}/{len(codes)} 已抓取，失败 {len(failed)}")

    if not frames:
        raise FetchError("所有股票都抓取失败，检查网络")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(subset=["code", "date"]).sort_values(["code", "date"])
    if failed:
        print(f"  [data] 抓取失败 {len(failed)} 只，示例：{failed[:3]}")
    return panel.reset_index(drop=True)
