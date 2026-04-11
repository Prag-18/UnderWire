from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import server
from server import app


def _unwrap_json(value):
    return getattr(value, "adapted", value)


class FakeCursor:
    def __init__(self, store: dict[str, dict[str, dict]]) -> None:
        self.store = store
        self._result = None

    def execute(self, query: str, params=None) -> None:
        normalized = " ".join(query.split()).upper()
        params = params or ()

        if normalized.startswith("CREATE TABLE IF NOT EXISTS SESSIONS"):
            self._result = None
            return

        if normalized.startswith("DELETE FROM SESSIONS WHERE CREATED_AT <"):
            cutoff = params[0]
            self.store["sessions"] = {
                session_id: row
                for session_id, row in self.store["sessions"].items()
                if row["created_at"] >= cutoff
            }
            self._result = None
            return

        if normalized.startswith("INSERT INTO SESSIONS"):
            session_id, task_id, seed, step, done, findings, created_at = params
            self.store["sessions"][session_id] = {
                "session_id": session_id,
                "task_id": task_id,
                "seed": seed,
                "step": step,
                "done": done,
                "findings": _unwrap_json(findings),
                "created_at": created_at,
            }
            self._result = None
            return

        if normalized.startswith("SELECT * FROM SESSIONS WHERE SESSION_ID ="):
            session_id = params[0]
            row = self.store["sessions"].get(session_id)
            self._result = dict(row) if row else None
            return

        if normalized.startswith("UPDATE SESSIONS SET STEP ="):
            step, done, findings, session_id = params
            row = self.store["sessions"][session_id]
            row["step"] = step
            row["done"] = done
            row["findings"] = _unwrap_json(findings)
            self._result = None
            return

        if normalized.startswith("SELECT SESSION_ID, TASK_ID, SEED, STEP, DONE, CREATED_AT FROM SESSIONS ORDER BY CREATED_AT DESC"):
            rows = sorted(
                self.store["sessions"].values(),
                key=lambda row: row["created_at"],
                reverse=True,
            )
            self._result = [
                {
                    "session_id": row["session_id"],
                    "task_id": row["task_id"],
                    "seed": row["seed"],
                    "step": row["step"],
                    "done": row["done"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
            return

        if normalized.startswith("DELETE FROM SESSIONS WHERE SESSION_ID ="):
            session_id = params[0]
            self.store["sessions"].pop(session_id, None)
            self._result = None
            return

        if normalized.startswith("SELECT COUNT(*) AS PERSISTED_SESSIONS FROM SESSIONS"):
            self._result = {"persisted_sessions": len(self.store["sessions"])}
            return

        msg = f"Unsupported query in fake DB: {query}"
        raise NotImplementedError(msg)

    def fetchone(self):
        return self._result

    def fetchall(self):
        return self._result or []

    def close(self) -> None:
        return None


class FakeConnection:
    def __init__(self, store: dict[str, dict[str, dict]]) -> None:
        self.store = store

    def cursor(self) -> FakeCursor:
        return FakeCursor(self.store)

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def fake_db(monkeypatch):
    stale_created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=25)
    store = {
        "sessions": {
            "stale-session": {
                "session_id": "stale-session",
                "task_id": "classify_licenses",
                "seed": 1,
                "step": 0,
                "done": False,
                "findings": [],
                "created_at": stale_created_at,
            }
        }
    }

    def fake_get_db():
        return FakeConnection(store)

    monkeypatch.setattr(server, "get_db", fake_get_db)
    return store


@pytest.fixture
def client(fake_db):
    with TestClient(app) as test_client:
        yield test_client


def test_startup_cleans_expired_sessions(client, fake_db):
    assert "stale-session" not in fake_db["sessions"]


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["persisted_sessions"] == 0


def test_create_session(client):
    payload = {"task_id": "classify_licenses", "seed": 42}
    response = client.post("/env/create", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "session_id" in data
    assert "observation" in data
    assert data["observation"]["task_id"] == "classify_licenses"


def test_step_flow(client):
    create = client.post("/env/create", json={"task_id": "classify_licenses", "seed": 42})
    session_id = create.json()["session_id"]
    target_id = create.json()["observation"]["files_to_scan"][0]["file_id"]

    action = {
        "action": {
            "action_type": "request_clarification",
            "target_id": target_id,
            "confidence": 0.5,
        },
        "session_id": session_id,
    }

    response = client.post("/env/step", json=action)

    assert response.status_code == 200
    data = response.json()

    assert "observation" in data
    assert "reward" in data
    assert "done" in data
    assert data["observation"]["step"] == 1


def test_invalid_session(client):
    response = client.post(
        "/env/step",
        json={
            "session_id": "invalid",
            "action": {
                "action_type": "request_clarification",
                "target_id": "file_1",
                "confidence": 0.5,
            },
        },
    )

    assert response.status_code == 404
