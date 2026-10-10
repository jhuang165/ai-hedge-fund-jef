import json

import pytest

from hedge_fund.reporter.config import ReporterConfig
from hedge_fund.reporter.publish import export_site, publish
from hedge_fund.reporter.scheduler import Reporter
from hedge_fund.reporter.server import main
from hedge_fund.reporter.store import ReportStore
from hedge_fund.reporter.test_server import FakeClient, fake_evidence, fake_open_client


def make(tmp_path, **cfg_kw):
    cfg = ReporterConfig(db_path=tmp_path / "r.db", seed_symbols=("NVDA", "MU"), llm_base_url="http://unused", **cfg_kw)
    rep = Reporter(cfg, ReportStore(cfg.db_path), FakeClient(), evidence_fn=fake_evidence, open_client=fake_open_client)
    return rep


def read(path):
    return json.loads(path.read_text())


def test_export_mirrors_the_api(tmp_path):
    rep = make(tmp_path)
    rep.run_cycle("manual")
    rep.run_cycle("manual", ["NVDA"])
    out = tmp_path / "site"
    summary = export_site(rep, out)

    assert summary["symbols"] == 2 and summary["reports"] == 3
    html = (out / "index.html").read_text()
    assert '<html lang="en" data-mode="static">' in html
    assert (out / "static" / "app.js").exists() and (out / "static" / "style.css").exists()
    assert not (out / "static" / "index.html").exists()

    status = read(out / "data" / "status.json")
    assert status["mode"] == "static" and status["published_at"] and status["symbols"] == ["NVDA", "MU"]
    assert "base_url" not in status["llm"]
    latest = read(out / "data" / "latest.json")
    assert {c["ticker"] for c in latest} == {"NVDA", "MU"}
    nvda = next(c for c in latest if c["ticker"] == "NVDA")
    assert nvda["brief"]["signal"] == "bearish" and "evidence" not in nvda["brief"]
    history = read(out / "data" / "history" / "NVDA.json")
    assert [h["id"] for h in history] == [3, 1] and history[0]["brief"]["confidence"] == 65
    full = read(out / "data" / "reports" / "3.json")
    assert full["ticker"] == "NVDA" and full["brief"]["summary"] == "s"
    assert len(read(out / "data" / "runs.json")) == 2


def test_export_replaces_a_previous_site(tmp_path):
    rep = make(tmp_path)
    out = tmp_path / "site"
    out.mkdir()
    (out / "stale.json").write_text("{}")
    export_site(rep, out)
    assert not (out / "stale.json").exists()
    assert not (tmp_path / "site.tmp").exists()
    assert read(out / "data" / "latest.json")[0]["brief"] is None


def test_publish_runs_the_command_in_the_site(tmp_path):
    rep = make(tmp_path)
    out = tmp_path / "site"
    summary = publish(rep, out, "pwd > uploaded.txt && echo done")
    assert (out / "uploaded.txt").read_text().strip().endswith("site")
    assert summary["output"] == ["done"] and summary["took_s"] >= 0


def test_publish_failure_is_reported_not_raised_by_the_reporter(tmp_path):
    rep = make(tmp_path, publish_dir=tmp_path / "site", publish_cmd="echo boom >&2; exit 3")
    run = rep.run_cycle("manual", ["NVDA"])
    assert run["n_ok"] == 1
    last = rep.status()["publish"]["last"]
    assert last["ok"] is False and "exited 3" in last["detail"] and "boom" in last["detail"]
    assert (tmp_path / "site" / "data" / "latest.json").exists()  # the export itself still happened

    rep.cfg.publish_cmd = ""
    rep.run_cycle("manual", ["MU"])
    assert rep.status()["publish"]["last"]["ok"] is True
    assert rep.status()["publish"]["enabled"] is True
    with pytest.raises(RuntimeError):
        publish(rep, tmp_path / "site", "exit 1")


def test_cli_publish_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REPORTER_DB", str(tmp_path / "cli.db"))
    monkeypatch.setenv("REPORTER_SYMBOLS", "NVDA")
    monkeypatch.setenv("CODEX_WRAPPER_URL", "http://127.0.0.1:9")  # nothing listens: status says unreachable
    monkeypatch.setenv("REPORTER_PUBLISH_CMD", "")
    with pytest.raises(SystemExit) as exc:
        main(["--publish", "--publish-dir", str(tmp_path / "out")])
    assert exc.value.code == 0
    assert "published 1 symbols, 0 reports" in capsys.readouterr().err
    assert read(tmp_path / "out" / "data" / "status.json")["llm"]["reachable"] is False
