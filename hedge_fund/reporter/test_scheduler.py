import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from hedge_fund.reporter.brief import CodexClient
from hedge_fund.reporter.config import ReporterConfig
from hedge_fund.reporter.evidence import Evidence, Quote
from hedge_fund.reporter.scheduler import Reporter, interval_minutes, is_active, next_run_at, next_window_open
from hedge_fund.reporter.store import ReportStore

NY = ZoneInfo("America/New_York")


def cfg(tmp_path, **overrides) -> ReporterConfig:
    c = ReporterConfig(db_path=tmp_path / "r.db", seed_symbols=("NVDA", "AAPL"), llm_base_url="http://unused",
                       interval_minutes=60, off_hours_interval_minutes=240, active_window=(7 * 60, 20 * 60),
                       timezone="America/New_York", parallel=2)
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


# ----------------------------------------------------------- schedule math

def test_is_active_weekdays_in_window():
    assert is_active(datetime(2026, 10, 9, 9, 30, tzinfo=NY), (420, 1200))      # Friday 09:30
    assert not is_active(datetime(2026, 10, 9, 20, 0, tzinfo=NY), (420, 1200))  # 20:00 is outside
    assert not is_active(datetime(2026, 10, 10, 12, 0, tzinfo=NY), (420, 1200))  # Saturday


def test_interval_minutes_by_window(tmp_path):
    c = cfg(tmp_path)
    assert interval_minutes(datetime(2026, 10, 9, 10, 0, tzinfo=NY), c) == 60
    assert interval_minutes(datetime(2026, 10, 9, 23, 0, tzinfo=NY), c) == 240


def test_next_window_open_skips_weekend():
    sat = datetime(2026, 10, 10, 12, 0, tzinfo=NY)
    assert next_window_open(sat, (420, 1200)) == datetime(2026, 10, 12, 7, 0, tzinfo=NY)
    fri_early = datetime(2026, 10, 9, 6, 0, tzinfo=NY)
    assert next_window_open(fri_early, (420, 1200)) == datetime(2026, 10, 9, 7, 0, tzinfo=NY)


def test_next_run_at_active_hours(tmp_path):
    finished = datetime(2026, 10, 9, 14, 5, tzinfo=timezone.utc)  # 10:05 NY, Friday
    assert next_run_at(finished, cfg(tmp_path)) == datetime(2026, 10, 9, 15, 5, tzinfo=timezone.utc)


def test_next_run_at_off_hours_but_not_past_the_open(tmp_path):
    finished = datetime(2026, 10, 9, 9, 30, tzinfo=timezone.utc)  # 05:30 NY: 240 min would be 09:30, open is 07:00
    assert next_run_at(finished, cfg(tmp_path)) == datetime(2026, 10, 9, 11, 0, tzinfo=timezone.utc)
    finished = datetime(2026, 10, 10, 2, 0, tzinfo=timezone.utc)  # Fri 22:00 NY -> +240 = Sat 02:00 NY
    assert next_run_at(finished, cfg(tmp_path)) == datetime(2026, 10, 10, 6, 0, tzinfo=timezone.utc)


def test_next_run_at_off_hours_disabled_waits_for_open(tmp_path):
    c = cfg(tmp_path, off_hours_interval_minutes=0)
    finished = datetime(2026, 10, 10, 15, 0, tzinfo=timezone.utc)  # Saturday
    assert next_run_at(finished, c) == datetime(2026, 10, 12, 11, 0, tzinfo=timezone.utc)  # Monday 07:00 NY


# ------------------------------------------------------------------ cycle

class FakeClient(CodexClient):
    def __init__(self, answers: dict[str, object]):
        super().__init__("http://unused")
        self.answers = answers
        self.seen: list[str] = []

    def complete(self, system, user, model):
        ticker = user.split("Ticker: ")[1].split(".")[0]
        self.seen.append(user)
        a = self.answers[ticker]
        if isinstance(a, Exception):
            raise a
        return json.dumps(a)

    def healthy(self):
        return True, "fake"


@contextmanager
def fake_open_client():
    yield object()


def fake_evidence(ticker, as_of, client):
    if ticker == "BAD":
        raise RuntimeError("no such name")
    return Evidence(ticker=ticker, as_of=as_of, gathered_at="g", quote=Quote(price=1.0, as_of="q"))


ANSWER = {"signal": "bullish", "confidence": 70, "headline": "up", "sources": []}


def make(tmp_path, answers, **overrides):
    c = cfg(tmp_path, **overrides)
    store = ReportStore(c.db_path)
    rep = Reporter(c, store, FakeClient(answers), evidence_fn=fake_evidence, open_client=fake_open_client)
    return rep, store


def test_run_cycle_saves_every_outcome(tmp_path):
    rep, store = make(tmp_path, {"NVDA": ANSWER, "AAPL": ValueError("bad json")})
    store.add_symbol("BAD")
    run = rep.run_cycle("manual")
    assert run["status"] == "done" and run["n_ok"] == 1 and run["n_failed"] == 2
    rows = {r["ticker"]: r for r in store.reports_in_run(run["id"])}
    assert rows["NVDA"]["status"] == "ok" and rows["NVDA"]["brief"]["action"] == "buy"
    assert rows["AAPL"]["error"] == "ValueError: bad json"
    assert rows["BAD"]["error"].startswith("evidence: no such name")
    assert rep.current is None


def test_run_cycle_feeds_previous_brief(tmp_path):
    rep, store = make(tmp_path, {"NVDA": ANSWER, "AAPL": ANSWER})
    rep.run_cycle("manual", ["NVDA"])
    rep.run_cycle("manual", ["NVDA"])
    second = rep.client.seen[1]
    assert "Previous brief (written" in second and "bullish 70, action buy" in second
    latest = store.latest_brief("NVDA")["brief"]
    assert latest["prev_signal"] == "bullish" and latest["prev_confidence"] == 70.0


def test_run_cycle_subset_and_status(tmp_path):
    rep, store = make(tmp_path, {"NVDA": ANSWER, "AAPL": ANSWER})
    run = rep.run_cycle("manual", ["aapl"])
    assert [r["ticker"] for r in store.reports_in_run(run["id"])] == ["AAPL"]
    s = rep.status()
    assert s["symbols"] == ["NVDA", "AAPL"] and s["llm"]["reachable"] is True
    assert s["scheduler"]["running"] is False and s["scheduler"]["last_run"]["id"] == run["id"]
    assert s["scheduler"]["active_hours"] == "07:00-20:00"


def test_loop_runs_on_boot_and_on_request(tmp_path):
    rep, store = make(tmp_path, {"NVDA": ANSWER, "AAPL": ANSWER}, run_on_boot=True)
    assert rep.request_run() is False  # not started yet
    rep.start()
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not store.runs():
            time.sleep(0.05)
        while time.time() < deadline and (store.last_run() or {}).get("status") == "running":
            time.sleep(0.05)
        assert store.last_run()["trigger"] == "schedule"
        assert rep.next_run is not None and rep.next_run > datetime.now(timezone.utc)
        assert rep.request_run(["NVDA"]) is True
        deadline = time.time() + 10
        while time.time() < deadline and len(store.runs()) < 2:
            time.sleep(0.05)
        while time.time() < deadline and (store.last_run() or {}).get("status") == "running":
            time.sleep(0.05)
        assert store.last_run()["trigger"] == "manual"
        assert [r["ticker"] for r in store.reports_in_run(store.last_run()["id"])] == ["NVDA"]
    finally:
        rep.stop()
