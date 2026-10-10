import base64
import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from hedge_fund.reporter.brief import CodexClient
from hedge_fund.reporter.config import ReporterConfig
from hedge_fund.reporter.evidence import Evidence, Quote
from hedge_fund.reporter.scheduler import Reporter
from hedge_fund.reporter.server import create_app
from hedge_fund.reporter.store import ReportStore

ANSWER = {"signal": "bearish", "confidence": 65, "headline": "down", "summary": "s", "sources": []}


class FakeClient(CodexClient):
    def __init__(self):
        super().__init__("http://unused")

    def complete(self, system, user, model):
        return json.dumps(ANSWER)

    def healthy(self):
        return True, "fake"


@contextmanager
def fake_open_client():
    yield object()


def fake_evidence(ticker, as_of, client):
    return Evidence(ticker=ticker, as_of=as_of, gathered_at="g", quote=Quote(price=2.0, as_of="q"))


def make(tmp_path, password=""):
    cfg = ReporterConfig(db_path=tmp_path / "r.db", seed_symbols=("NVDA",), llm_base_url="http://unused",
                         auth_password=password, auth_user="desk")
    rep = Reporter(cfg, ReportStore(cfg.db_path), FakeClient(), evidence_fn=fake_evidence, open_client=fake_open_client)
    return rep, TestClient(create_app(rep))


def test_pages_and_health(tmp_path):
    rep, c = make(tmp_path)
    assert c.get("/healthz").json() == {"ok": True, "scheduler": False}
    assert c.get("/").status_code == 200 and "aihf" in c.get("/").text
    assert c.get("/static/app.js").status_code == 200
    s = c.get("/api/status").json()
    assert s["symbols"] == ["NVDA"] and s["auth"] is False and s["llm"]["model"] == "gpt-6-luna"
    assert c.get("/api/status").headers["cache-control"] == "no-store"


def test_symbols_crud(tmp_path):
    rep, c = make(tmp_path)
    r = c.post("/api/symbols", json={"ticker": " brk-b "})
    assert r.status_code == 200 and r.json()["added"] is True and r.json()["symbols"] == ["NVDA", "BRK-B"]
    assert c.post("/api/symbols", json={"ticker": "BRK-B"}).json()["added"] is False
    assert c.post("/api/symbols", json={"ticker": "N V"}).status_code == 400
    assert c.delete("/api/symbols/brk-b").json()["symbols"] == ["NVDA"]
    assert c.delete("/api/symbols/BRK-B").status_code == 404


def test_latest_reports_and_detail(tmp_path):
    rep, c = make(tmp_path)
    latest = c.get("/api/latest").json()
    assert latest == [{"ticker": "NVDA", "report_id": None, "brief": None, "last_attempt": None}]

    run = rep.run_cycle("manual")
    latest = c.get("/api/latest").json()[0]
    assert latest["brief"]["signal"] == "bearish" and latest["brief"]["action"] == "avoid"
    assert "summary" not in latest["brief"]            # cards are light
    assert latest["last_attempt"]["status"] == "ok"

    rows = c.get("/api/reports", params={"ticker": "nvda"}).json()
    assert len(rows) == 1 and rows[0]["run_id"] == run["id"]
    detail = c.get(f"/api/reports/{rows[0]['id']}").json()
    assert detail["brief"]["summary"] == "s" and detail["brief"]["quote"]["price"] == 2.0
    assert c.get("/api/reports/999").status_code == 404
    assert c.get("/api/runs").json()[0]["n_ok"] == 1


def test_run_now_requires_the_loop(tmp_path):
    rep, c = make(tmp_path)
    r = c.post("/api/run", json={})
    assert r.status_code == 409 and "scheduler is not running" in r.json()["detail"]


def test_basic_auth(tmp_path):
    rep, c = make(tmp_path, password="s3cret")
    r = c.get("/api/status")
    assert r.status_code == 401 and r.headers["www-authenticate"].startswith("Basic")
    assert c.get("/").status_code == 401
    assert c.get("/healthz").status_code == 200       # probes need no password
    bad = {"Authorization": "Basic " + base64.b64encode(b"desk:nope").decode()}
    assert c.get("/api/status", headers=bad).status_code == 401
    good = {"Authorization": "Basic " + base64.b64encode(b"desk:s3cret").decode()}
    assert c.get("/api/status", headers=good).json()["auth"] is True
    assert c.get("/", headers=good).status_code == 200


@pytest.mark.parametrize("path", ["/api/symbols", "/api/latest", "/api/runs"])
def test_auth_covers_every_api_route(tmp_path, path):
    rep, c = make(tmp_path, password="x")
    assert c.get(path).status_code == 401
