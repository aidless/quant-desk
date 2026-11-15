"""FastAPI 服务：数据、因子、回测、LLM 因子工厂。"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import CONFIG
from .data.store import MarketPanel, PanelStore
from .factors import dsl, engine as factor_engine, evaluate
from .llm import client, factor_factory


class EvaluateRequest(BaseModel):
    expression: str = Field(description="DSL 表达式", min_length=5, max_length=1000)
    horizon: int = Field(default=20, ge=1, le=60)
    sign: int = Field(default=1, ge=-1, le=1)
    quantiles: int = Field(default=5, ge=3, le=10)
    industry_neutral: bool = Field(default=False)


class BacktestRequest(BaseModel):
    expression: str = Field(description="DSL 表达式", min_length=5, max_length=1000)
    top_frac: float = Field(default=0.2, gt=0, le=0.5)
    rebalance_days: int = Field(default=5, ge=1, le=10)
    commission_bp: float = Field(default=2.5, ge=0, le=10)
    stamp_bp: float = Field(default=5.0, ge=0, le=20)
    slippage_bp: float = Field(default=5.0, ge=0, le=20)
    lag: int = Field(default=1, ge=1, le=3)
    selection_sign: int = Field(default=1, ge=-1, le=1)
    industry_neutral: bool = Field(default=False)


class LLMFactorRequest(BaseModel):
    intent: str = Field(description="自然语言意图，如「做一支 10 日超跌反弹因子」")


# 生命周期：面板构建
@asynccontextmanager
async def lifespan(app: FastAPI):
    store = PanelStore(CONFIG.data)
    try:
        panel = store.panel(verbose=True)
        print(f"  [server] 已加载行情面板 {panel.summary()}")
        app.state.panel = panel
    except Exception as exc:  # noqa: BLE001
        print(f"[server] 面板加载失败，使用合成面板：{exc}")
        from .data.store import make_synthetic_panel
        app.state.panel = make_synthetic_panel(40, 400)
    yield
    del app.state.panel


app = FastAPI(title="量化交易平台", openapi_url="/openapi.json", lifespan=lifespan)


@app.get("/")
def root():
    return {"name": "quant-desk", "version": "v1", "docs": "/docs"}


@app.get("/health")
def health():
    panel = getattr(app.state, "panel", None)
    if panel is None:
        return {"status": "unknown", "panel": None}
    return {
        "status": "ok",
        "panel": panel.summary(),
        "cache": getattr(app.state, "store", None) and getattr(app.state.store, "status", lambda: {}),
    }


@app.get("/factor-catalog")
def factor_catalog():
    from .factors.library import catalog
    return catalog()


@app.post("/evaluate")
def evaluate_req(req: EvaluateRequest):
    panel = getattr(app.state, "panel", None)
    if panel is None:
        raise HTTPException(status_code=503, detail="Panel not ready")
    try:
        res = evaluate.evaluate_factor(
            req.expression,
            panel,
            horizon=req.horizon,
            sign=req.sign,
            quantiles=req.quantiles,
            method="spearman",
            industry_neutral=req.industry_neutral,
        )
        return res
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/backtest")
def backtest_req(req: BacktestRequest):
    import dataclasses

    from .backtest.engine import run_expression

    cfg = dataclasses.replace(
        CONFIG.backtest,
        top_frac=req.top_frac,
        rebalance_days=req.rebalance_days,
        commission_bp=req.commission_bp,
        stamp_bp=req.stamp_bp,
        slippage_bp=req.slippage_bp,
    )
    panel = getattr(app.state, "panel", None)
    if panel is None:
        raise HTTPException(status_code=503, detail="Panel not ready")
    try:
        bt = run_expression(
            req.expression,
            panel,
            cfg=cfg,
            lag=req.lag,
            sign=req.selection_sign,
        )
        return {
            "metrics": bt.metrics,
            "equity_curve": bt.equity_curve(max_points=500),
            "config": bt.config,
            "holdings_log": bt.holdings_log[-10:],
            "nav_latest": round(float(bt.nav.iloc[-1]), 4) if len(bt.nav) else None,
            "bench_latest": round(float(bt.benchmark_nav.iloc[-1]), 4) if len(bt.benchmark_nav) else None,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/llm/factor")
def llm_factor(req: LLMFactorRequest):
    intent = req.intent.strip()
    if len(intent) < 2:
        raise HTTPException(status_code=400, detail="intent 过短")
    result = factor_factory.llm_build(
        intent,
        available=not CONFIG.llm.enabled,
        max_attempts=2,
    )
    out = dict(
        name=result.name,
        expression=result.expression,
        sign=result.sign,
        logic=result.logic,
        category=result.category,
        verified=result.verified,
        source=result.source,
    )
    if result.error:
        out["error"] = result.error
    return out


@app.get("/synthetic")
def synthetic_demo():
    """离线演示：用合成面板跑一个示例因子并返回评测 + 回测结论。"""
    from .data.store import make_synthetic_panel
    from .factors import library
    panel = make_synthetic_panel(40, 400)
    f_def = library.get("mom_20")
    eval_result = evaluate.evaluate_factor(f_def.expression, panel, horizon=f_def.horizon, sign=f_def.sign)
    from .backtest.engine import run_expression
    bt = run_expression(f_def.expression, panel, None, lag=1, sign=f_def.sign)
    return {
        "data_status": "synthetic",
        "factor": {"name": f_def.name, "cn_name": f_def.cn_name, "expression": f_def.expression, "sign": f_def.sign},
        "evaluation": eval_result,
        "backtest_summary": {
            "total_return": bt.metrics.get("total_return"),
            "annual_return": bt.metrics.get("annual_return"),
            "sharpe": bt.metrics.get("sharpe"),
            "max_drawdown": bt.metrics.get("max_drawdown"),
            "benchmark_annual": bt.metrics.get("benchmark_annual"),
            "excess_annual": bt.metrics.get("excess_annual"),
            "cost_drag_annual": bt.metrics.get("cost_drag_annual"),
        },
    }


@app.post("/dsl/validate")
def dsl_validate(expression: str):
    try:
        tree = dsl.validate(expression)
        ast_json = {"node_count": len(list(dsl.ast.walk(tree))), "tree_type": type(tree).__name__}
        return {"valid": True, "ast": ast_json}
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "error": str(exc)[:500]}


# 静态前端（单页）
FRONTEND_HTML = """/<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>量化交易平台</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
body{font-family:system-ui,sans-serif;margin:20px;background:#fafafa;}
h1{margin:0 0 10px;}
.card{background:#fff;border-radius:6px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,.06);}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;}
.btn{border:0;background:#2563eb;color:#fff;padding:8px 12px;border-radius:6px;cursor:pointer;}
.btn:hover{background:#1d4ed8;} textarea{width:100%;min-height:70px;font-family:ui-monospace,Menlo,Consolas;}
input[type=text]{padding:6px;border-radius:6px;border:1px solid #ddd;width:240px;}
table{border-collapse:collapse;width:100%;} th,td{border-bottom:1px solid #eee;padding:6px;text-align:left;}
.badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:12px;background:#e5e7eb;color:#374151;}
.progress{height:6px;background:#e5e7eb;border-radius:6px;overflow:hidden;}
.progress-fill{height:100%;background:#10b981;width:0%;transition:.4s}
</style>
<script src="//cdn.jsdelivr.net/npm/luxon@3/build/i18n/datetimeduration-luxon.js"></script>
<script src="//unpkg.com/chart@latest/dist/chart.umd.min.js"></script></head>
<body>
<div class="card"><h1>📈 量化交易平台</h1><div id="summary"></div></div>
<div class="card">
  <p>因子目录：
  <button class="btn" onclick="loadCatalog()">刷新</button>
  </p>
  <ul id="catlist"></ul>
</div>
<div class="card"><h2>DSL 评估器</h2>
  <textarea placeholder="例如：close / delay(close, 20) - 1"></textarea>
  <label>持有期：<input type="number" id="horizon" value="20" min="1" max="60"/></label>
  <label>行业中性：<input type="checkbox" id="industry_neutral"/></label>
  <br/><button class="btn" onclick="runEval()">运行评估</button>
  <pre id="evalOut"></pre>
</div>
<div class="card">
  <h2>回测</h2>
  <textarea placeholder="DSL 表达式"></textarea>
  <p>成本参数（基点）：<label>佣金<input id="commBP" value="2.5"/></label>
     <label>印花税<input id="stampBP" value="5"/></label>
     <label>滑点<input id="slipBP" value="5"/></label></p>
  <label>换手率上限（持仓比例）：<input id="topFrac" value="0.2" type="number" step=".01"/></label>
  <br/><button class="btn" onclick="runBT()">开始回测</button>
  <pre id="btOut"></pre>
  <canvas id="eqCurve" width="700" height="300"></canvas>
</div>
<div class="card"><h2>LLM 因子工厂</h2>
  <p><i>当前通道：</i><span id="llmChannel">—</span></p>
  <textarea id="llmIntent" placeholder="例如：做一支 10 日超跌反弹因子"></textarea>
  <br/><button class="btn" onclick="runLLMFa()">生成因子</button>
  <pre id="llmOut"></pre>
</div>
<div class="card"><h2>离线演示</h2>
  <button class="btn" onclick="runSynth()">Run Synthetic Demo</button>
  <pre id="synthOut"></pre>
</div>
<script>
const BASE = "";
window.onload = async ()=>{
  await loadHealth();
  document.getElementById("llmChannel").textContent = (await fetch(BASE+"/config").then(r=>r.text())).includes('"enabled":false') ? "本地网关/启用" : "关闭模式";
};
async function loadHealth(){
  const r=await fetch(BASE+"/health");const d=await r.json();
  document.getElementById("summary").innerHTML=`状态:${d.status===\\'ok\\'?\\'<span class=\"badge\">OK</span>\':\\"未知\\"}<br/>面板:<code>${JSON.stringify(d.panel)}</code>`;
}
async function loadCatalog(){
  const r=await fetch(BASE+"/factor-catalog");const fs=await r.json();
  document.getElementById("catlist").innerHTML="<li>"+fs.map((f,i)=>`${i+1}. <code>${f.name}</code> (${f.cn_name}): ${f.expression}<br/>说明：${f.logic}</li>`).join("");
}
function setTA(idx,val){document.querySelectorAll('textarea')[idx].value=val;}
async function runEval(){
  const t=document.querySelectorAll('textarea');let expr=t[0].value||"";if(!expr)return alert("请输入表达式");
  const obj={expression:expr,horizon:+document.getElementById("horizon").value,sign:1,industry_neutral:document.getElementById("industry_neutral").checked};
  const r=await fetch(BASE+"/evaluate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(obj)});const res=await r.json();
  document.getElementById("evalOut").textContent=JSON.stringify({ic_res:res.ic,quantiles:res.quantiles},null,2);
}
async function runBT(){
  const t=document.querySelectorAll('textarea');let expr=t[1].value||"";if(!expr)return alert("输入表达式");
  const cfg={expression:expr,top_frac:+document.getElementById("topFrac").value,commission_bp:+document.getElementById("commBP").value,stamp_bp:+document.getElementById("stampBP").value,slippage_bp:+document.getElementById("slipBP").value,lag:1,selection_sign:1,industry_neutral:false,rebalance_days:5};
  const r=await fetch(BASE+"/backtest",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(cfg)});
  const res=await r.json();const m=res.metrics;
  document.getElementById("btOut").textContent=JSON.stringify(m,null,2)+"\n\n配置:"+JSON.stringify(res.config,null,2)+"\n\n最新净值:NAV="+res.nav_latest.toFixed(4)+" BEN"+res.bench_latest.toFixed(4);
  // Simple SVG equity curve (inline chart via Luxon)
  const points=res.equity_curve||[];const w=700,h=280,padding=30;
  const xMin=new Date(points[0]?.date||Date.now()),xMax=new Date(points[points.length-1]?.date||Date.now());
  const yMin=Math.min(Math.min(...points.map(p=>p.nav||Infinity),Math.min(...points.map(p=>p.bench||Infinity)),0),0);
  const yMax=Math.max(Math.max(...points.map(p=>p.nav||-Infinity),Math.max(...points.map(p=>p.bench||-Infinity)),1),1);
  const scale=(pts,xScale,yScale)=>pts.map(p=>`${xScale(p.date)},${yScale(p.nav||0)}|${xScale(p.date)},${yScale(p.bench||0)}`);
  const xRange=xMax-xMin,yRange=yMax-yMin;
  function xs(v){return padding+(new Date(v)-xMin)/(xRange)*(w-2*padding);} function ys(v){return h-padding-(v-yMin)/(yMax-yMin)*(h-2*padding);}
  const navPoly=points.map(p=>[xs(p.date),ys(p.nav||0)]).filter(([x,y])=>!isNaN(x)&&!isNaN(y)).map(([x,y])=>(points.indexOf(p)=>`${x.toFixed(2)},${y.toFixed(2)}`).slice(0,-2));
  const benchPoly=points.map(p=>[xs(p.date),ys(p.bench||0)]).filter(([x,y])=>!isNaN(x)&&!isNaN(y)).map(([x,y])=>`"${x.toFixed(2)},${y.toFixed(2)}"`);
  // Render on canvas
  const ctx=document.getElementById("eqCurve").getContext("2d");ctx.clearRect(0,0,700,300);ctx.fillStyle="#111";ctx.font="12px system-ui";ctx.fillText("Equity Curve (simulated)",10,20);
}
async function runLLMFa(){
  const txt=document.getElementById("llmIntent").value.trim();if(!txt)return alert("写个意图");
  const obj={intent:txt};const r=await fetch(BASE+"/llm/factor",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(obj)});const res=await r.json();
  document.getElementById("llmOut").textContent=JSON.stringify(res,null,2);setTA(-2,res.expression??"");
}
async function runSynth(){
  const r=await fetch(BASE+"/synthetic");const res=await r.json();document.getElementById("synthOut").textContent=JSON.stringify(res,null,2);
}
</script>
</body>
</html>"""


@app.get("/index.html", response_class=HTMLResponse)
def index_html():
    return FRONTEND_HTML


# /config 接口：前端根据这个显示通道状态
@app.get("/config")
def config_info():
    return {"llm_enabled": not CONFIG.llm.enabled, "model": CONFIG.llm.model, "base_url": CONFIG.llm.base_url}
