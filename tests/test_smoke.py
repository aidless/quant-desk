"""端到端 smoke：合成面板 → 因子 → 评测 → 回测 → FastAPI 路由。"""

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import qdesk  # noqa: E402
from qdesk.app import app  # noqa: E402
from qdesk.data.store import make_synthetic_panel  # noqa: E402


@pytest.fixture(scope="module")
def client():
    # 不进 lifespan（TestClient 非 with 用法），直接注入合成面板，
    # 避免真实 PanelStore 拉取覆盖注入值
    app.state.panel = make_synthetic_panel(n_codes=30, n_days=200, seed=3)
    yield TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["panel"]["codes"] == 30


def test_factor_catalog(client):
    r = client.get("/factor-catalog")
    assert r.status_code == 200
    cats = r.json()
    assert len(cats) == 16
    assert all("expression" in c for c in cats)


def test_dsl_validate_route(client):
    r = client.post("/dsl/validate", params={"expression": "ts_rank(close, 10)"})
    assert r.status_code == 200
    assert r.json()["valid"] is True
    r2 = client.post("/dsl/validate", params={"expression": "__import__('os')"})
    assert r2.json()["valid"] is False


def test_evaluate_route(client):
    # 未注册名（DSL 只认表达式，不认因子名）→ 400
    r = client.post("/evaluate", json={
        "expression": "mom_20",
        "horizon": 10,
        "sign": 1,
        "quantiles": 5,
    })
    assert r.status_code == 400
    # 真实 DSL 表达式 → 200
    r2 = client.post("/evaluate", json={
        "expression": "close / delay(close, 20) - 1",
        "horizon": 10,
        "sign": 1,
    })
    assert r2.status_code == 200
    body = r2.json()
    assert "ic" in body and "quantiles" in body


def test_evaluate_rejects_bad_expression(client):
    r = client.post("/evaluate", json={"expression": "import os"})
    assert r.status_code == 400


def test_backtest_route(client):
    r = client.post("/backtest", json={
        "expression": "close / delay(close, 20) - 1",
        "top_frac": 0.3,
        "rebalance_days": 5,
    })
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body and "equity_curve" in body
    m = body["metrics"]
    for key in ("total_return", "annual_return", "sharpe", "max_drawdown"):
        assert key in m


def test_llm_factor_offline_fallback(client):
    """LLM 关闭通道下走模板回退，必须返回可用表达式。"""
    r = client.post("/llm/factor", json={"intent": "做一支20日动量因子"})
    assert r.status_code == 200
    body = r.json()
    assert body["verified"] is True
    assert len(body["expression"]) > 5


def test_synthetic_demo(client):
    r = client.get("/synthetic")
    assert r.status_code == 200
    body = r.json()
    assert body["data_status"] == "synthetic"
    assert "sharpe" in body["backtest_summary"]


def test_library_api():
    f = qdesk.get("mom_20")
    assert f.name == "mom_20"
    assert len(qdesk.catalog()) == 16


def test_evaluate_factor_end_to_end():
    panel = make_synthetic_panel(20, 150, seed=5)
    out = qdesk.evaluate_factor("close / delay(close, 10) - 1", panel, horizon=5, sign=1)
    assert out["ic"]["n"] > 50
    assert -1.0 <= out["ic"]["ic_mean"] <= 1.0
    assert len(out["ic_curve"]) > 0
