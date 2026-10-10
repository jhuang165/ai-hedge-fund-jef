from hedge_fund.reporter.store import ReportStore


def _store(tmp_path):
    return ReportStore(tmp_path / "r.db")


def test_seed_only_when_empty(tmp_path):
    s = _store(tmp_path)
    assert s.seed_symbols(["nvda", "AAPL"]) is True
    assert s.symbols() == ["NVDA", "AAPL"]
    assert s.seed_symbols(["MSFT"]) is False
    assert s.symbols() == ["NVDA", "AAPL"]


def test_add_remove_symbol(tmp_path):
    s = _store(tmp_path)
    assert s.add_symbol("msft") is True
    assert s.add_symbol("MSFT") is False
    assert s.symbols() == ["MSFT"]
    assert s.remove_symbol("msft") is True
    assert s.remove_symbol("MSFT") is False
    assert s.symbols() == []


def test_runs_and_reports_roundtrip(tmp_path):
    s = _store(tmp_path)
    s.seed_symbols(["NVDA", "AAPL"])
    run = s.create_run("schedule")
    rid = s.save_report(run_id=run, ticker="NVDA", model="m", brief={"signal": "bullish", "x": 1},
                        error=None, duration_s=1.5)
    s.save_report(run_id=run, ticker="AAPL", model="m", brief=None, error="boom", duration_s=0.1)
    s.finish_run(run, n_ok=1, n_failed=1)

    r = s.report(rid)
    assert r["brief"] == {"signal": "bullish", "x": 1} and r["status"] == "ok"
    assert s.report(999) is None
    assert s.latest_brief("AAPL") is None
    assert s.latest_brief("NVDA")["id"] == rid

    last = s.last_run()
    assert last["status"] == "done" and last["n_ok"] == 1 and last["n_failed"] == 1
    assert last["finished_at"] is not None

    latest = s.latest()
    assert latest["NVDA"]["brief"]["id"] == rid
    assert latest["AAPL"]["latest"]["status"] == "failed" and latest["AAPL"]["brief"] is None


def test_latest_brief_survives_a_failed_attempt(tmp_path):
    s = _store(tmp_path)
    s.seed_symbols(["NVDA"])
    r1 = s.create_run("schedule")
    good = s.save_report(run_id=r1, ticker="NVDA", model="m", brief={"signal": "neutral"}, error=None, duration_s=1)
    s.finish_run(r1, n_ok=1, n_failed=0)
    r2 = s.create_run("schedule")
    s.save_report(run_id=r2, ticker="NVDA", model="m", brief=None, error="timeout", duration_s=600)
    s.finish_run(r2, n_ok=0, n_failed=1)

    row = s.latest()["NVDA"]
    assert row["latest"]["status"] == "failed"
    assert row["brief"]["id"] == good
    assert [r["id"] for r in s.reports_for("NVDA", ok_only=True)] == [good]
    assert len(s.reports_for("NVDA")) == 2


def test_failed_run_records_error(tmp_path):
    s = _store(tmp_path)
    run = s.create_run("manual")
    s.finish_run(run, n_ok=0, n_failed=0, error="Traceback ...")
    assert s.last_run()["status"] == "failed"
    assert s.last_run()["error"].startswith("Traceback")
