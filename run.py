#!/usr/bin/env python3
"""启动量化交易平台（FastAPI + uvicorn）。"""

from __future__ import annotations

from qdesk.app import app
from qdesk.config import CONFIG

if __name__ == "__main__":
    import uvicorn
    print(f"[startup] 配置：LLM={CONFIG.llm.base_url} {CONFIG.llm.model}")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
