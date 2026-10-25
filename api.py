"""Account Intelligence - HTTP API + static web UI.

Run:   uvicorn api:app --reload
Open:  http://localhost:8000

Routes
  GET  /api/focus?segment=&region=&industry=   both lists + KPIs for the AE's book
  GET  /api/accounts/{account_id}              facts, reasons, brief, prompt
  POST /api/accounts/{account_id}/brief        live regeneration (optional; needs OLLAMA_MODEL)
  GET  /                                       web/index.html
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api_core import account_payload, focus_payload, scored
from src.insights import NotConfigured, generate_brief, segment_medians

WEB = Path(__file__).parent / "web"
app = FastAPI(title="Account Intelligence", version="0.1")


@app.get("/api/focus")
def focus(segment: str | None = None, region: str | None = None, industry: str | None = None):
    return focus_payload(segment or None, region or None, industry or None)


@app.get("/api/accounts/{account_id}")
def account(account_id: str):
    payload = account_payload(account_id)
    if payload is None:
        raise HTTPException(404, f"unknown account {account_id}")
    return payload


@app.post("/api/accounts/{account_id}/brief")
def regenerate(account_id: str):
    df = scored()
    if account_id not in set(df["account_id"]):
        raise HTTPException(404, f"unknown account {account_id}")
    acc = df.set_index("account_id").loc[account_id].to_dict() | {"account_id": account_id}
    try:
        return {"brief": generate_brief(acc, segment_medians(df, acc["segment"])), "source": os.getenv("OLLAMA_MODEL")}
    except NotConfigured as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:  # noqa: BLE001 - surface provider errors to the UI
        raise HTTPException(502, f"model call failed: {e}") from e


@app.get("/api/health")
def health():
    return {"ok": True, "live_model": os.getenv("OLLAMA_MODEL") or None}


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
