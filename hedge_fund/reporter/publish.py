"""Publish: a static copy of the dashboard for hosts that only serve files.

Firebase Hosting, GitHub Pages, S3, any CDN: none of them can run the
reporter (no Python, no scheduler, no Codex), but all of them can serve
the dashboard and its data as plain files. So after every cycle the
reporter can write the whole site to a directory -- index.html, the
assets, and the JSON the page would otherwise fetch from /api -- and run
a command that uploads it (`firebase deploy`). The published copy is
read-only: no Run now, no watchlist edits. Those stay on the box that
runs the reporter.

    REPORTER_PUBLISH_DIR=deploy/firebase/site      # export here after each cycle
    REPORTER_PUBLISH_CMD="firebase deploy --only hosting --non-interactive"
                                                   # run in that directory afterwards
    aihf reporter --publish                        # export (and upload) once, now

Layout of the exported directory, mirrored by app.js in static mode:

    index.html                  the dashboard, with data-mode="static"
    static/app.js, style.css
    data/status.json            /api/status  plus published_at
    data/latest.json            /api/latest
    data/runs.json              /api/runs?limit=20
    data/history/<TICKER>.json  /api/reports?ticker=<TICKER>&limit=48
    data/reports/<id>.json      /api/reports/<id>, for every id in a history
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hedge_fund.reporter.store import ReportStore, now_iso

if TYPE_CHECKING:
    from hedge_fund.reporter.scheduler import Reporter

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
HISTORY_LIMIT = 48
RUNS_LIMIT = 20

CARD_KEYS = ("ticker", "generated_at", "model", "signal", "confidence", "action", "headline",
             "nothing_new", "prev_signal", "prev_confidence", "prev_action", "prev_generated_at", "quote")


def card(brief: dict[str, Any]) -> dict[str, Any]:
    """The light version of a brief for list views: no evidence payload."""
    return {k: brief.get(k) for k in CARD_KEYS}


def latest_cards(store: ReportStore) -> list[dict[str, Any]]:
    """One card per symbol: the newest brief and the newest attempt
    (the shape of /api/latest)."""
    out = []
    for ticker, row in store.latest().items():
        brief = row["brief"]
        attempt = row["latest"]
        out.append({
            "ticker": ticker,
            "report_id": brief["id"] if brief else None,
            "brief": card(brief["brief"]) if brief else None,
            "last_attempt": {"status": attempt["status"], "generated_at": attempt["generated_at"],
                             "error": attempt["error"]} if attempt else None,
        })
    return out


def history_cards(store: ReportStore, ticker: str, limit: int = HISTORY_LIMIT) -> list[dict[str, Any]]:
    """The shape of /api/reports?ticker=."""
    rows = store.reports_for(ticker, limit=limit)
    return [{**r, "brief": card(r["brief"]) if r["brief"] else None} for r in rows]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def export_site(reporter: Reporter, out: Path, *, history: int = HISTORY_LIMIT, runs: int = RUNS_LIMIT) -> dict[str, Any]:
    """Write the static site to *out* (replacing it atomically: the new tree
    is built next to it and swapped in). Returns a small summary."""
    out = Path(out)
    store = reporter.store
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    # The page and its assets. The page is flagged so app.js reads files
    # instead of the API and hides the controls that need a server.
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    flagged = html.replace("<html lang=\"en\">", "<html lang=\"en\" data-mode=\"static\">", 1)
    if flagged == html:
        raise RuntimeError("index.html no longer has the <html lang=\"en\"> tag the exporter flags")
    (tmp / "index.html").write_text(flagged, encoding="utf-8")
    shutil.copytree(STATIC_DIR, tmp / "static", ignore=shutil.ignore_patterns("index.html"))

    status = reporter.status()
    status["published_at"] = now_iso()
    status["mode"] = "static"
    status["llm"] = {k: v for k, v in status["llm"].items() if k != "base_url"}
    _dump(tmp / "data" / "status.json", status)
    _dump(tmp / "data" / "latest.json", latest_cards(store))
    _dump(tmp / "data" / "runs.json", store.runs(limit=runs))

    n_reports = 0
    for ticker in store.symbols():
        rows = history_cards(store, ticker, limit=history)
        _dump(tmp / "data" / "history" / f"{ticker}.json", rows)
        for row in rows:
            if row["status"] != "ok":
                continue
            full = store.report(row["id"])
            if full is not None:
                _dump(tmp / "data" / "reports" / f"{row['id']}.json", full)
                n_reports += 1

    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return {"dir": str(out), "symbols": len(status["symbols"]), "reports": n_reports,
            "published_at": status["published_at"]}


def publish(reporter: Reporter, out: Path, command: str = "", *, timeout_s: int = 600) -> dict[str, Any]:
    """Export the site and, if *command* is set, run it (a shell line, in
    the exported directory) to upload it. Raises on failure; the caller
    decides whether that is fatal."""
    t0 = time.time()
    summary = export_site(reporter, out)
    if command:
        proc = subprocess.run(command, shell=True, cwd=str(out), capture_output=True, text=True, timeout=timeout_s)
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-8:]
        if proc.returncode != 0:
            raise RuntimeError(f"publish command exited {proc.returncode}: " + " | ".join(tail))
        summary["command"] = command
        summary["output"] = tail
    summary["took_s"] = round(time.time() - t0, 1)
    return summary
