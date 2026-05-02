"""
FastAPI server exposing the OpenEnv environment over HTTP.
Sessions are persisted to PostgreSQL so state survives server restarts.

Required environment variable:
  DATABASE_URL  — standard postgres DSN, e.g.:
                  postgresql://user:password@host:5432/dbname
                  or with SSL:
                  postgresql://user:password@host:5432/dbname?sslmode=require

Uses psycopg (v3) with its native async connection pool. The pool is
initialised once at startup and closed cleanly at shutdown via FastAPI
lifespan. All DB calls are fully async — no thread-pool overhead.
"""
from __future__ import annotations

import asyncio
import os
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
import psycopg
from psycopg_pool import AsyncConnectionPool
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.environment import LicenseComplianceEnv
from env.models import Action, Observation, Reward, EnvironmentState, ScanFinding

DATABASE_URL: str = ""  # loaded at startup via lifespan
_pool: AsyncConnectionPool | None = None

DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT        PRIMARY KEY,
    task_id    TEXT        NOT NULL,
    seed       INTEGER     NOT NULL,
    step       INTEGER     NOT NULL DEFAULT 0,
    done       BOOLEAN     NOT NULL DEFAULT FALSE,
    findings   JSONB       NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions (updated_at);
"""

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global _pool, DATABASE_URL
    DATABASE_URL = os.environ["DATABASE_URL"]   # fail fast here, not at import
    _pool = AsyncConnectionPool(conninfo=DATABASE_URL, min_size=2, max_size=10, open=False)
    await _pool.open()
    async with _pool.connection() as conn:
        await conn.execute(DDL)
    yield
    await _pool.close()


app = FastAPI(
    title="License Compliance Scanner — OpenEnv",
    description="AI agent environment for OSS license compliance tasks.",
    version="1.0.0",
    lifespan=lifespan,
)


async def _create_session(session_id: str, task_id: str, seed: int) -> None:
    async with _pool.connection() as conn:
        await conn.execute(
            "INSERT INTO sessions (session_id, task_id, seed) VALUES (%s, %s, %s) ON CONFLICT (session_id) DO NOTHING",
            (session_id, task_id, seed),
        )


async def _load_env(session_id: str) -> LicenseComplianceEnv:
    async with _pool.connection() as conn:
        row = await conn.execute(
            "SELECT task_id, seed, step, done, findings FROM sessions WHERE session_id = %s",
            (session_id,),
        )
        record = await row.fetchone()

    if not record:
        raise HTTPException(status_code=404, detail="Session not found")

    task_id, seed, step, done, findings_raw = record
    env = LicenseComplianceEnv(task_id=task_id, seed=seed)
    env.reset()
    env._step_count = step
    env._done = bool(done)
    findings_list = findings_raw if isinstance(findings_raw, list) else json.loads(findings_raw)
    env._findings = [ScanFinding(**f) for f in findings_list]
    return env


async def _save_env(session_id: str, env: LicenseComplianceEnv) -> None:
    findings_json = json.dumps([f.model_dump() for f in env._findings])
    async with _pool.connection() as conn:
        await conn.execute(
            "UPDATE sessions SET step=%s, done=%s, findings=%s::jsonb, updated_at=NOW() WHERE session_id=%s",
            (env._step_count, env._done, findings_json, session_id),
        )


class CreateEnvRequest(BaseModel):
    task_id: str = "classify_licenses"
    seed: int = 42

class CreateEnvResponse(BaseModel):
    session_id: str
    observation: Observation

class StepRequest(BaseModel):
    session_id: str
    action: Action

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <html><head><title>License Compliance Scanner — OpenEnv</title></head>
    <body style="font-family:monospace;max-width:800px;margin:40px auto;padding:20px;color:#e8e8e8;background:#1a1a1a">
    <h1>License Compliance Scanner</h1>
    <p>OpenEnv environment for AI agents learning real-world license compliance.</p>
    <h2>Tasks</h2>
    <ul>
      <li><b>classify_licenses</b> (Easy) — Identify SPDX license from raw text</li>
      <li><b>detect_conflicts</b> (Medium) — Find incompatible dependencies</li>
      <li><b>generate_compliance_report</b> (Hard) — Full repo audit + report</li>
    </ul>
    <h2>API</h2>
    <pre style="background:#111;padding:12px;border-radius:6px">POST /env/create
    POST /env/step
    GET  /env/state/{session_id}
    GET  /env/score/{session_id}
    POST /env/reset/{session_id}
    GET  /health
    GET  /scan/github?url=&lt;github_url&gt;</pre>
    <p><a href="/docs" style="color:#60a0ff">Interactive API Docs</a></p>
    <p><a href="underwire_v2.html" style="color:#60a0ff">Live Scanner Dashboard</a></p>
    <p style="color:#888;font-size:12px">Sessions persisted to PostgreSQL.</p>
    </body></html>
    """

@app.post("/env/create", response_model=CreateEnvResponse)
async def create_env(req: CreateEnvRequest):
    session_id = str(uuid.uuid4())[:12]
    await _create_session(session_id, req.task_id, req.seed)
    env = LicenseComplianceEnv(task_id=req.task_id, seed=req.seed)
    obs = env.reset()
    await _save_env(session_id, env)
    return CreateEnvResponse(session_id=session_id, observation=obs)

@app.post("/env/step", response_model=StepResponse)
async def step(req: StepRequest):
    env = await _load_env(req.session_id)
    try:
        obs, reward, done, info = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await _save_env(req.session_id, env)
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/env/state/{session_id}", response_model=EnvironmentState)
async def get_state(session_id: str):
    env = await _load_env(session_id)
    return env.state()

@app.get("/env/score/{session_id}", response_model=Reward)
async def get_score(session_id: str):
    env = await _load_env(session_id)
    return env.final_score()

@app.post("/env/reset/{session_id}", response_model=Observation)
async def reset_env(session_id: str):
    env = await _load_env(session_id)
    obs = env.reset()
    await _save_env(session_id, env)
    return obs

@app.get("/health")
async def health():
    async with _pool.connection() as conn:
        row = await conn.execute("SELECT COUNT(*) FROM sessions")
        count = (await row.fetchone())[0]
    return {"status": "ok", "persisted_sessions": count, "db": "postgresql"}


@app.get("/scan/github")
async def scan_github(
    url: str = Query(..., description="GitHub repository URL, e.g. https://github.com/owner/repo"),
) -> dict[str, Any]:
    """
    Scan a GitHub repository for OSS license compliance issues.

    Fetches ``package.json`` (npm) or ``requirements.txt`` (Python) from the
    repository via raw.githubusercontent.com, parses each dependency, maps it
    to a known SPDX license, and runs the detect_conflicts grader.

    Returns structured findings JSON with:
    - ``dependencies``  — list of resolved DependencyEntry objects
    - ``findings``      — list of ScanFinding objects (conflicts + clear)
    - ``grader_result`` — Reward breakdown (precision, recall, F1, ...)
    - ``summary``       — human-readable scan summary
    """
    from scan_github import scan_github_repo
    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None, scan_github_repo, url
        )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Network error fetching repo: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return result


if __name__ == "__main__":
    import selectors
    import asyncio
    import uvicorn

    loop = asyncio.SelectorEventLoop(selectors.SelectSelector())
    asyncio.set_event_loop(loop)

    config = uvicorn.Config(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),
        loop="asyncio",
        reload=False,
    )
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())