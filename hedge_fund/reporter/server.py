"""The reporter's web app: a JSON API, one static page, HTTP basic auth.

    aihf reporter                          # 0.0.0.0:8788, scheduler on
    aihf reporter --port 9000 --host 127.0.0.1
    aihf reporter --once                   # one cycle, then exit (cron)
    aihf reporter --no-scheduler           # serve history, never run
    aihf reporter --publish                # export the static site once (publish.py), then exit

It is meant to sit on the public internet behind nothing but
REPORTER_PASSWORD, so: basic auth on every route except /healthz, no
secrets in responses, `no-store` on the API. Nothing here can place an
order or change a mandate; the only writes are the watchlist and the
briefs themselves. With REPORTER_PUBLISH_DIR set, every cycle also
exports a read-only copy of the dashboard for a static host.
"""

from __future__ import annotations

import argparse
import logging
import secrets
import sys
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hedge_fund.reporter.config import ReporterConfig
from hedge_fund.reporter.publish import history_cards, latest_cards, publish
from hedge_fund.reporter.scheduler import Reporter
from hedge_fund.reporter.store import ReportStore

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8788
log = logging.getLogger(__name__)


class SymbolBody(BaseModel):
    ticker: str = Field(min_length=1, max_length=12)
    note: str = ""


class RunBody(BaseModel):
    tickers: list[str] | None = None


def create_app(reporter: Reporter) -> FastAPI:
    cfg = reporter.cfg
    app = FastAPI(title="aihf reporter", docs_url=None, redoc_url=None)
    basic = HTTPBasic(auto_error=False)

    def guard(credentials: HTTPBasicCredentials | None = Depends(basic)) -> None:
        if not cfg.auth_enabled:
            return
        ok = (
            credentials is not None
            and secrets.compare_digest(credentials.username.encode(), cfg.auth_user.encode())
            and secrets.compare_digest(credentials.password.encode(), cfg.auth_password.encode())
        )
        if not ok:
            raise HTTPException(status_code=401, detail="sign in",
                                headers={"WWW-Authenticate": 'Basic realm="aihf reporter"'})

    auth = [Depends(guard)]

    @app.middleware("http")
    async def no_store(request: Request, call_next):
        response: Response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, Any]:
        return {"ok": True, "scheduler": reporter.running}

    @app.get("/api/status", dependencies=auth)
    def status() -> dict[str, Any]:
        return reporter.status()

    @app.get("/api/symbols", dependencies=auth)
    def symbols() -> list[str]:
        return reporter.store.symbols()

    @app.post("/api/symbols", dependencies=auth)
    def add_symbol(body: SymbolBody) -> dict[str, Any]:
        ticker = body.ticker.strip().upper()
        if not ticker.replace(".", "").replace("-", "").isalnum():
            raise HTTPException(400, f"{ticker!r} does not look like a ticker")
        added = reporter.store.add_symbol(ticker, body.note)
        return {"ticker": ticker, "added": added, "symbols": reporter.store.symbols()}

    @app.delete("/api/symbols/{ticker}", dependencies=auth)
    def remove_symbol(ticker: str) -> dict[str, Any]:
        removed = reporter.store.remove_symbol(ticker)
        if not removed:
            raise HTTPException(404, f"{ticker.upper()} is not on the watchlist")
        return {"ticker": ticker.upper(), "removed": True, "symbols": reporter.store.symbols()}

    @app.get("/api/latest", dependencies=auth)
    def latest() -> list[dict[str, Any]]:
        """One card per symbol: the newest brief and the newest attempt."""
        return latest_cards(reporter.store)

    @app.get("/api/reports", dependencies=auth)
    def reports(ticker: str, limit: int = 48) -> list[dict[str, Any]]:
        return history_cards(reporter.store, ticker, limit=max(1, min(limit, 500)))

    @app.get("/api/reports/{report_id}", dependencies=auth)
    def report(report_id: int) -> dict[str, Any]:
        row = reporter.store.report(report_id)
        if row is None:
            raise HTTPException(404, "no such report")
        return row

    @app.get("/api/runs", dependencies=auth)
    def runs(limit: int = 30) -> list[dict[str, Any]]:
        return reporter.store.runs(limit=max(1, min(limit, 200)))

    @app.post("/api/run", dependencies=auth)
    def run_now(body: RunBody | None = None) -> dict[str, Any]:
        tickers = [t.strip().upper() for t in (body.tickers if body and body.tickers else []) if t.strip()] or None
        if reporter.current is not None:
            raise HTTPException(409, "a cycle is already running")
        if reporter.request_run(tickers):
            return {"queued": True}
        if not reporter.running:
            raise HTTPException(409, "the scheduler is not running; start the server without --no-scheduler")
        raise HTTPException(409, "could not queue a run")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False, dependencies=auth)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aihf reporter",
        description="Continuously brief every name on a watchlist from live prices, "
        "live headlines, and a Codex-backed model that searches the web.",
        epilog="configuration is by environment variable; see hedge_fund/reporter/config.py",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--once", action="store_true", help="run one cycle and exit (for cron)")
    parser.add_argument("--no-scheduler", action="store_true", help="serve the history only")
    parser.add_argument("--tickers", help="with --once: only these, comma separated")
    parser.add_argument("--publish", action="store_true",
                        help="export the static site to REPORTER_PUBLISH_DIR (or --publish-dir), run "
                             "REPORTER_PUBLISH_CMD there, and exit")
    parser.add_argument("--publish-dir", help="with --publish: where to write the site")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = ReporterConfig()
    store = ReportStore(cfg.db_path)
    reporter = Reporter(cfg, store)

    if args.publish:
        out = Path(args.publish_dir) if args.publish_dir else cfg.publish_dir
        if out is None:
            parser.error("--publish needs --publish-dir or REPORTER_PUBLISH_DIR")
        summary = publish(reporter, out, cfg.publish_cmd, timeout_s=cfg.publish_timeout_s)
        print(f"published {summary['symbols']} symbols, {summary['reports']} reports to {summary['dir']} "
              f"in {summary['took_s']}s", file=sys.stderr)
        for line in summary.get("output", []):
            print(f"  {line}", file=sys.stderr)
        sys.exit(0)

    if args.once:
        tickers = [t.strip().upper() for t in args.tickers.replace(" ", ",").split(",") if t.strip()] if args.tickers else None
        run = reporter.run_cycle("manual", tickers)
        print(f"run {run['id']}: {run['n_ok']} ok, {run['n_failed']} failed", file=sys.stderr)
        sys.exit(0 if run["status"] == "done" and run["n_failed"] == 0 else 1)

    if not cfg.auth_enabled and args.host not in {"127.0.0.1", "localhost", "::1"}:
        log.warning("REPORTER_PASSWORD is not set and the server binds to %s: anyone who can reach it can read "
                    "the briefs and edit the watchlist", args.host)
    if not args.no_scheduler and cfg.autostart:
        reporter.start()

    import uvicorn

    uvicorn.run(create_app(reporter), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
