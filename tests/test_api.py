"""
Integration tests for the FastAPI deployment endpoints.
Run with: python -m pytest tests/test_api.py

Uses httpx.AsyncClient with ASGITransport to avoid TestClient version conflicts.
"""
import json
import sys
import os
import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
from deployment.app import app

TRANSPORT = httpx.ASGITransport(app=app)
BASE = "http://testserver"

PHASE1_GOLD = {
    "issue_type": "authentication",
    "root_cause": "expired_jwt",
    "priority": "high",
    "affected_component": "login_service",
}
PHASE2_GOLD = {
    "tool": "search_repo",
    "parameters": {"query": "authentication bugs"},
}


@pytest.mark.anyio
async def test_health():
    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE) as client:
        res = await client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


@pytest.mark.anyio
async def test_predict_phase1():
    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE) as client:
        res = await client.post("/predict", json={
            "text": "Login failures due to expired JWT tokens.",
            "phase": 1,
        })
    assert res.status_code == 200
    body = res.json()
    assert "raw_output" in body
    assert len(body["raw_output"]) > 0


@pytest.mark.anyio
async def test_predict_phase2():
    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE) as client:
        res = await client.post("/predict", json={
            "text": "Search authentication bugs in the repository.",
            "phase": 2,
        })
    assert res.status_code == 200
    body = res.json()
    assert "raw_output" in body


@pytest.mark.anyio
async def test_evaluate_phase1_simulate():
    payload = {
        "phase": 1,
        "simulate": True,
        "examples": [{"text": "JWT expired on login.", "label": PHASE1_GOLD}],
    }
    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE) as client:
        res = await client.post("/evaluate", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["metrics"]["json_validity"] == 100.0
    assert body["metrics"]["exact_match"] == 100.0


@pytest.mark.anyio
async def test_evaluate_phase2_simulate():
    payload = {
        "phase": 2,
        "simulate": True,
        "examples": [{"text": "Search auth bugs.", "label": PHASE2_GOLD}],
    }
    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE) as client:
        res = await client.post("/evaluate", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["metrics"]["tool_selection_accuracy"] == 100.0


@pytest.mark.anyio
async def test_metrics_endpoint():
    # Run evaluate first to populate metrics
    payload = {
        "phase": 1,
        "simulate": True,
        "examples": [{"text": "JWT expired.", "label": PHASE1_GOLD}],
    }
    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE) as client:
        await client.post("/evaluate", json=payload)
        res = await client.get("/metrics")
    assert res.status_code == 200
    assert "metrics" in res.json()


@pytest.mark.anyio
async def test_invalid_phase():
    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE) as client:
        res = await client.post("/predict", json={"text": "test", "phase": 99})
    assert res.status_code == 400


@pytest.mark.anyio
async def test_empty_examples():
    async with httpx.AsyncClient(transport=TRANSPORT, base_url=BASE) as client:
        res = await client.post("/evaluate", json={
            "phase": 1, "examples": [], "simulate": True
        })
    assert res.status_code == 400
