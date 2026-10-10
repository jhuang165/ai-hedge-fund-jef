"""Web server tests — FastAPI TestClient, engine calls faked, no network."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hedge_fund import paths
from hedge_fund.models import Signal
from hedge_fund.pipeline.ledger import Carry
from hedge_fund.research.models import ResearchReport, SearchResult
from hedge_fund.web import server


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Ctx:
    """Stands in for FDClient / TavilyClient: constructible, closable, inert."""

    def __init__(self, *a, **k): ...
    def __enter__(self): return self
    def __exit__(self, *a): ...
    def close(self): ...


class _LLM:
    model = "fake-model"

    def complete(self, system, user):  # never called: diagnose is faked
        raise AssertionError("LLM should not be called")


def _report(ticker, signal="bullish", confidence=70, action="buy", position=None):
    return ResearchReport(
        ticker=ticker, as_of="2025-06-30", model="fake-model", signal=signal,
        confidence=confidence, action=action, action_rationale="because",
        thesis=f"{ticker} headline\nmore", prompt_key="k",
        sources=[SearchResult(query="q", title="t", url="https://x/1", content="c")],
        desk=[Signal(model_name="momentum", ticker=ticker, date="2025-06-30", value=0.4)],
        position=position,
    )


class _Dump:
    """Anything with model_dump(), for faked run_cycle / backtest_fund."""

    def __init__(self, **data):
        self._data = data

    def model_dump(self):
        return self._data


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MANDATES_DIR", tmp_path / "mandates")
    monkeypatch.setattr(paths, "RESEARCH_DIR", tmp_path / "research")
    monkeypatch.setattr("hedge_fund.tui.keys.ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("FINANCIAL_DATASETS_API_KEY", "fd-key")
    monkeypatch.setenv("TAVILY_API_KEY", "tv-key")
    monkeypatch.delenv("HEDGE_FUND_LLM_MODEL", raising=False)
    monkeypatch.setattr(server, "open_cached_client", _Ctx)
    monkeypatch.setattr(server, "TavilyClient", _Ctx)
    monkeypatch.setattr(server, "make_llm", lambda model=None: _LLM())

    def fake_diagnose(ticker, as_of, fd, search, llm, *, position=None):
        if ticker == "BAD":
            raise RuntimeError("no such company")
        signal, conf = {"AAPL": ("bullish", 80), "MSFT": ("neutral", 50)}.get(ticker, ("bearish", 90))
        action = "hold" if position else ("buy" if signal == "bullish" else "watch")
        from hedge_fund.research.models import PositionMark
        mark = PositionMark.mark(position, 100.0) if position else None
        return _report(ticker, signal, conf, action, mark)

    monkeypatch.setattr(server, "diagnose", fake_diagnose)
    monkeypatch.setattr(server, "run_carried", lambda fund, as_of, fd, universe: SimpleNamespace(
        record=_Dump(fund=fund.spec.name, as_of=as_of, universe=universe, nav=100_000.0,
                     model=server.os.environ.get("HEDGE_FUND_LLM_MODEL")),
        carry=Carry(cash=fund.spec.capital), path=tmp_path / "receipt.json"))

    def fake_backtest(fund, start, end, fd, universe, *, on_cycle=None):
        for i, (d, nav) in enumerate([("2025-01-03", 101_000.0), ("2025-01-10", 99_000.0)]):
            on_cycle(i, 2, type("Record", (), {"as_of": d, "nav": nav})())
        return _Dump(fund=fund.spec.name, start=start, end=end, nav=[101_000.0, 99_000.0])

    monkeypatch.setattr(server, "backtest_fund", fake_backtest)
    return TestClient(server.create_app())


def _wait(client, job_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.03)
    raise AssertionError("job did not finish")


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------

def test_index_and_status(client):
    assert "<title>aihf" in client.get("/").text
    s = client.get("/api/status").json()
    assert s["version"]
    assert any(k["env_var"] == "TAVILY_API_KEY" and k["set"] for k in s["keys"])
    assert any(m["model"] == s["default_model"] for m in s["llm_models"])


def test_models_and_strategies(client):
    models = {m["name"]: m["kind"] for m in client.get("/api/models").json()}
    assert models["buffett"] == "agent" and models["momentum"] == "quant"
    strategies = {s["name"]: s for s in client.get("/api/strategies").json()}
    assert strategies["multifactor"]["kind"] == "systematic"
    assert strategies["deep-value"]["kind"] == "discretionary"
    assert "momentum" in strategies["multifactor"]["description"].lower()


def test_keys_roundtrip(client, tmp_path):
    rows = client.post("/api/keys", json={"env_var": "ANTHROPIC_API_KEY", "value": "sk-ant-secret"}).json()
    row = next(r for r in rows if r["env_var"] == "ANTHROPIC_API_KEY")
    assert row["set"] and "secret" not in row["masked"]
    assert "ANTHROPIC_API_KEY=sk-ant-secret" in (tmp_path / ".env").read_text()
    assert client.post("/api/keys", json={"env_var": "EVIL", "value": "x"}).status_code == 400


# ---------------------------------------------------------------------------
# Mandates
# ---------------------------------------------------------------------------

def test_mandates_seeded_with_example(client):
    rows = client.get("/api/mandates").json()
    assert rows[0]["file"] == "example.yaml"
    assert rows[0]["spec"]["strategies"]


def test_new_mandate_lifecycle(client, tmp_path):
    body = {"name": "quant-fund", "strategies": [{"name": "multifactor", "weight": 2}, {"name": "trend"}],
            "capital": 50_000, "rebalance": "daily", "benchmark": "qqq"}
    created = client.post("/api/mandates", json=body).json()
    assert created["spec"]["benchmark"] == "QQQ"
    assert created["spec"]["strategies"][0]["weight"] == 2.0
    assert (tmp_path / "mandates" / "quant-fund.yaml").exists()
    assert "quant-fund.yaml" in {m["file"] for m in client.get("/api/mandates").json()}

    assert client.post("/api/mandates", json=body).status_code == 409
    assert client.post("/api/mandates", json={**body, "name": "x", "strategies": [{"name": "nope"}]}).status_code == 400
    assert client.delete("/api/mandates/quant-fund.yaml").status_code == 200
    assert client.delete("/api/mandates/quant-fund.yaml").status_code == 404
    assert client.delete("/api/mandates/../../etc").status_code in (400, 404)


# ---------------------------------------------------------------------------
# Research jobs
# ---------------------------------------------------------------------------

def test_research_job_ranks_saves_and_lists(client, tmp_path):
    job = client.post("/api/research", json={"targets": ["msft", "AAPL:10@90", "XYZ"], "as_of": "2025-06-30"}).json()
    assert job["status"] in ("queued", "running") and job["label"] == "MSFT, AAPL, XYZ"

    done = _wait(client, job["id"])
    assert done["status"] == "done", done
    tickers = [r["ticker"] for r in done["result"]["reports"]]
    assert tickers == ["AAPL", "MSFT", "XYZ"]  # bullish 80 > neutral > bearish
    aapl = done["result"]["reports"][0]
    assert aapl["action"] == "hold" and aapl["position"]["unrealized_pnl_pct"] == pytest.approx(10 / 90)
    assert done["progress"]["done"][0]["ticker"] == "AAPL"

    saved = client.get("/api/reports").json()
    assert {r["ticker"] for r in saved} == {"AAPL", "MSFT", "XYZ"}
    full = client.get(f"/api/reports/{saved[0]['id']}").json()
    assert full["ticker"] == saved[0]["ticker"]
    assert len(list((tmp_path / "research").glob("*.json"))) == 3
    assert client.get("/api/reports/../nope").status_code in (400, 404)


def test_research_batch_tolerates_one_failure(client):
    job = client.post("/api/research", json={"targets": ["AAPL", "BAD"]}).json()
    done = _wait(client, job["id"])
    assert done["status"] == "done"
    assert [r["ticker"] for r in done["result"]["reports"]] == ["AAPL"]
    assert done["result"]["failures"][0]["ticker"] == "BAD"


def test_research_all_failing_fails_job(client):
    done = _wait(client, client.post("/api/research", json={"targets": ["BAD"]}).json()["id"])
    assert done["status"] == "failed" and "no such company" in done["error"]


def test_research_validation(client, monkeypatch):
    assert client.post("/api/research", json={"targets": ["AAPL:abc"]}).status_code == 400
    assert client.post("/api/research", json={"targets": []}).status_code == 422
    monkeypatch.delenv("TAVILY_API_KEY")
    r = client.post("/api/research", json={"targets": ["AAPL"]})
    assert r.status_code == 400 and "TAVILY_API_KEY" in r.json()["detail"]


def test_unknown_job_404(client):
    assert client.get("/api/jobs/nope").status_code == 404


# ---------------------------------------------------------------------------
# Fund jobs
# ---------------------------------------------------------------------------

def test_cycle_job_routes_model_and_restores_env(client):
    job = client.post("/api/cycle", json={"mandate": "example.yaml", "tickers": ["aapl", "msft"],
                                          "as_of": "2025-06-30", "model": "claude-opus-5"}).json()
    done = _wait(client, job["id"])
    assert done["status"] == "done", done
    assert done["result"]["universe"] == ["AAPL", "MSFT"]
    assert done["result"]["model"] == "claude-opus-5"   # routed through the env seam...
    assert "HEDGE_FUND_LLM_MODEL" not in server.os.environ  # ...and restored after
    assert done["result"]["carried"].startswith("opened on")


def test_cycle_refuses_to_run_before_the_books_last_date(client, tmp_path):
    mandates = tmp_path / "mandates"
    (mandates / "example-fund-run-2025-07-01-000000-000000.json").write_text(json.dumps(
        {"fund": "example-fund", "as_of": "2025-07-01", "positions": {}, "cash": 1.0,
         "nav": 1.0, "equity_before": 1.0}))
    r = client.post("/api/cycle", json={"mandate": "example.yaml", "tickers": ["AAPL"],
                                        "as_of": "2025-06-30"})
    assert r.status_code == 409
    assert "2025-07-01" in r.json()["detail"]


def test_backtest_job_streams_nav(client):
    job = client.post("/api/backtest", json={"mandate": "example.yaml", "tickers": ["AAPL"], "as_of": "2025-06-30"}).json()
    assert "→" in job["label"]
    done = _wait(client, job["id"])
    assert done["status"] == "done", done
    assert done["progress"]["nav"] == [101_000.0, 99_000.0]
    assert done["progress"]["i"] == 2 and done["progress"]["n"] == 2
    assert done["result"]["nav"] == [101_000.0, 99_000.0]
    assert any(j["id"] == job["id"] for j in client.get("/api/jobs").json())


def test_fund_validation(client):
    assert client.post("/api/cycle", json={"mandate": "missing.yaml", "tickers": ["AAPL"]}).status_code == 404
    assert client.post("/api/cycle", json={"mandate": "example.yaml", "tickers": [" "]}).status_code == 400


def test_scorecard_grades_saved_reports(client, monkeypatch, tmp_path):
    from hedge_fund.research import save_report
    save_report(_report("AAPL"), tmp_path / "research")
    seen = {}

    def fake_grade(reports, fd, *, horizon_days, benchmark, as_of=None):
        seen["n"] = len(reports); seen["horizon"] = horizon_days; seen["benchmark"] = benchmark
        return _Dump(n_graded=0, n_pending=len(reports), horizon_days=horizon_days)

    monkeypatch.setattr(server, "grade", fake_grade)
    body = client.get("/api/scorecard?horizon=30&benchmark=qqq").json()
    assert seen == {"n": 1, "horizon": 30, "benchmark": "QQQ"}
    assert body["n_pending"] == 1
