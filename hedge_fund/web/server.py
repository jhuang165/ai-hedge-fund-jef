"""The web app's server: a FastAPI JSON API plus one static page.

    aihf web                      # http://127.0.0.1:8787, opens a browser tab
    aihf web --port 9000 --no-browser

Design:
- Every endpoint is a thin wrapper over an engine call the CLI/TUI already
  make; nothing here decides anything about money.
- Slow work (research = LLM + search per name, backtests = many LLM calls)
  runs as a *job* on one worker thread; the page polls /api/jobs/{id} and
  draws progress. One worker, deliberately: the LLM providers rate-limit,
  and a per-job model choice is passed through the same HEDGE_FUND_LLM_MODEL
  seam the CLI uses, which is process-global.
- Research reports are saved under ~/.hedge-fund/research/ so the Reports
  page has a history. Nothing else is written: the web app, like `aihf
  research`, touches no mandate and no ledger.
- Binds to 127.0.0.1 by default. There is no auth — it is a local tool.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hedge_fund import paths
from hedge_fund.backtesting import backtest_fund
from hedge_fund.data import CachedDataClient, FDClient
from hedge_fund.fund import Fund, FundSpec, load_spec, load_strategy, normalize_universe
from hedge_fund.llm import DEFAULT_MODEL, is_supported, load_api_models, make_llm
from hedge_fund.llm.registry import PROVIDER_ENV_VARS
from hedge_fund.pipeline.ledger import LedgerError, carried_book, run_carried
from hedge_fund.research import (
    TavilyClient,
    diagnose,
    grade,
    load_reports,
    merge_targets,
    parse_target,
    save_report,
)
from hedge_fund.research.models import ResearchReport
from hedge_fund.signals import ALPHA_MODEL_REGISTRY, LLMAgent
from hedge_fund.tui.keys import apply_credentials, masked, save_credential
from hedge_fund.tui.shared import (
    _BACKTEST_WEEKS,
    DISPLAY_NAMES,
    STRATEGY_DIR,
    VERSION,
    _strategy_kind,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8787

# Keys the settings page knows about, in display order.
_KEY_LABELS: dict[str, str] = {
    "FINANCIAL_DATASETS_API_KEY": "Financial Datasets (prices, fundamentals, filings)",
    "TAVILY_API_KEY": "Tavily (web search for research)",
    **{env: f"{provider} (LLM)" for provider, env in PROVIDER_ENV_VARS.items()},
}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class Job:
    """One unit of slow work and everything the page needs to show it."""

    def __init__(self, kind: str, request: dict[str, Any], label: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.label = label
        self.request = request
        self.status: Literal["queued", "running", "done", "failed"] = "queued"
        self.created_at = _now()
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.progress: dict[str, Any] = {}
        self.result: Any = None
        self.error: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "label": self.label, "status": self.status,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at, "error": self.error, "request": self.request,
        }

    def detail(self) -> dict[str, Any]:
        return {**self.summary(), "progress": self.progress, "result": self.result}


class JobStore:
    """Jobs by id, run one at a time on a worker thread."""

    def __init__(self, workers: int = 1) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="aihf-job")

    def submit(self, job: Job, fn) -> Job:
        """Queue *fn(job, progress)*; its return value becomes job.result."""
        with self._lock:
            self._jobs[job.id] = job

        def progress(update: dict[str, Any]) -> None:
            job.progress = {**job.progress, **update}

        def run() -> None:
            job.status, job.started_at = "running", _now()
            try:
                job.result = fn(job, progress)
                job.status = "done"
            except Exception as exc:  # the page shows the error; the server survives
                job.error = f"{type(exc).__name__}: {exc}"
                job.progress = {**job.progress, "traceback": traceback.format_exc()}
                job.status = "failed"
            finally:
                job.finished_at = _now()

        self._executor.submit(run)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    targets: list[str] = Field(min_length=1, description="TICKER or TICKER:SHARES@COST, one per entry")
    as_of: str = Field(default_factory=lambda: _date.today().isoformat())
    model: str | None = None


class CycleRequest(BaseModel):
    mandate: str = Field(description="mandate file name under ~/.hedge-fund/mandates, e.g. example.yaml")
    tickers: list[str] = Field(min_length=1)
    as_of: str = Field(default_factory=lambda: _date.today().isoformat())
    model: str | None = None


class BacktestRequest(CycleRequest):
    start: str | None = None


class NewMandate(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    strategies: list[dict[str, Any]] = Field(min_length=1, description="[{name, weight}] from the library")
    risk: dict[str, float] = Field(default_factory=lambda: {"max_position_pct": 0.25, "max_gross_exposure": 1.0})
    capital: float = 100_000.0
    rebalance: Literal["daily", "weekly", "monthly"] = "weekly"
    benchmark: str = "SPY"


class KeyUpdate(BaseModel):
    env_var: str
    value: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    apply_credentials()
    paths.ensure_mandates_dir()
    app = FastAPI(title="aihf", version=VERSION)
    jobs = JobStore()
    app.state.jobs = jobs

    @app.middleware("http")
    async def no_store(request, call_next):
        # A local tool edited in place and polled for live state: nothing it
        # serves — page, scripts, or JSON — may be answered from a browser cache.
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    # -- meta -------------------------------------------------------------

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return {
            "version": VERSION,
            "today": _date.today().isoformat(),
            "default_model": os.environ.get("HEDGE_FUND_LLM_MODEL") or DEFAULT_MODEL,
            "llm_models": [
                {"display_name": d, "model": m, "provider": p, "supported": is_supported(p),
                 "key_set": bool(os.environ.get(PROVIDER_ENV_VARS.get(p, "")))}
                for d, m, p in load_api_models()
            ],
            "keys": _keys(),
            "mandates_dir": str(paths.MANDATES_DIR),
            "research_dir": str(paths.RESEARCH_DIR),
        }

    @app.get("/api/keys")
    def keys() -> list[dict[str, Any]]:
        return _keys()

    @app.post("/api/keys")
    def set_key(body: KeyUpdate) -> list[dict[str, Any]]:
        if body.env_var not in _KEY_LABELS:
            raise HTTPException(400, f"unknown key {body.env_var!r}")
        save_credential(body.env_var, body.value.strip())
        os.environ[body.env_var] = body.value.strip()
        return _keys()

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return [
            {"name": n, "display_name": DISPLAY_NAMES.get(n, n),
             "kind": "agent" if issubclass(cls, LLMAgent) else "quant"}
            for n, cls in ALPHA_MODEL_REGISTRY.items()
        ]

    @app.get("/api/strategies")
    def strategies() -> list[dict[str, Any]]:
        out = []
        for path in sorted(STRATEGY_DIR.glob("*.yaml")):
            s = load_strategy(path)
            out.append({**s.model_dump(), "title": s.title, "kind": _strategy_kind(s),
                        "description": _yaml_comment(path)})
        return out

    # -- mandates ---------------------------------------------------------

    @app.get("/api/mandates")
    def mandates() -> list[dict[str, Any]]:
        out = []
        for path in sorted(paths.MANDATES_DIR.glob("*.yaml")):
            try:
                spec = load_spec(path)
                out.append({"file": path.name, "name": spec.name, "spec": spec.model_dump(),
                            "kinds": sorted({_strategy_kind(s) for s in spec.strategies})})
            except Exception as exc:
                out.append({"file": path.name, "name": path.stem, "error": str(exc)})
        return out

    @app.post("/api/mandates")
    def new_mandate(body: NewMandate) -> dict[str, Any]:
        library = {load_strategy(p).name: p for p in STRATEGY_DIR.glob("*.yaml")}
        strategies = []
        for entry in body.strategies:
            name = entry.get("name")
            if name not in library:
                raise HTTPException(400, f"unknown strategy {name!r}; library: {sorted(library)}")
            s = load_strategy(library[name])
            strategies.append(s.model_copy(update={"weight": float(entry.get("weight", 1.0))}))
        try:
            spec = FundSpec(name=body.name, strategies=strategies, risk=body.risk,
                            capital=body.capital, rebalance=body.rebalance, benchmark=body.benchmark)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        path = paths.MANDATES_DIR / f"{body.name}.yaml"
        if path.exists():
            raise HTTPException(409, f"{path.name} already exists")
        path.write_text(yaml.safe_dump(spec.model_dump(exclude_none=True), sort_keys=False))
        return {"file": path.name, "name": spec.name, "spec": spec.model_dump()}

    @app.delete("/api/mandates/{file}")
    def delete_mandate(file: str) -> dict[str, str]:
        path = _mandate_path(file)
        path.unlink()
        return {"deleted": file}

    # -- jobs -------------------------------------------------------------

    @app.post("/api/research")
    def research(body: ResearchRequest) -> dict[str, Any]:
        try:
            targets = merge_targets([parse_target(t) for t in body.targets if t.strip()])
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        if not targets:
            raise HTTPException(400, "no tickers given")
        _require_keys("FINANCIAL_DATASETS_API_KEY", "TAVILY_API_KEY")
        label = ", ".join(t.ticker for t in targets)
        job = Job("research", body.model_dump(), label)

        def run(job: Job, progress) -> dict[str, Any]:
            with _llm_model(body.model):
                llm = make_llm(body.model)
                reports: list[ResearchReport] = []
                failures: list[dict[str, str]] = []
                with FDClient() as raw, TavilyClient() as search:
                    fd = CachedDataClient(raw)
                    for i, target in enumerate(targets):
                        progress({"i": i, "n": len(targets), "label": target.ticker})
                        try:
                            report = diagnose(target.ticker, body.as_of, fd, search, llm,
                                              position=target.position)
                        except Exception as exc:
                            failures.append({"ticker": target.ticker, "error": f"{type(exc).__name__}: {exc}"})
                            progress({"failures": failures})
                            continue
                        reports.append(report)
                        _save_report(report)
                        progress({"done": [_report_row(r) for r in _ranked(reports)]})
                if not reports and failures:
                    raise RuntimeError("; ".join(f"{f['ticker']}: {f['error']}" for f in failures))
                return {
                    "as_of": body.as_of,
                    "reports": [r.model_dump() for r in _ranked(reports)],
                    "failures": failures,
                }

        return jobs.submit(job, run).summary()

    @app.post("/api/cycle")
    def cycle(body: CycleRequest) -> dict[str, Any]:
        spec, universe = _fund_inputs(body)
        _require_keys("FINANCIAL_DATASETS_API_KEY")
        try:  # refuse up front, not as a failed job, a run that breaks the chain
            carried_book(spec, body.as_of)
        except LedgerError as exc:
            raise HTTPException(409, str(exc))
        job = Job("cycle", body.model_dump(), f"{spec.name} @ {body.as_of}")

        def run(job: Job, progress) -> dict[str, Any]:
            with _llm_model(body.model):
                fund = Fund(spec)
                progress({"label": f"{len(universe)} tickers x "
                          f"{sum(len(s) for _, s in fund.strategies)} models"})
                with FDClient() as raw:
                    ran = run_carried(fund, body.as_of, CachedDataClient(raw),
                                      universe)
                return {**ran.record.model_dump(),
                        "carried": ran.carry.describe(), "receipt": str(ran.path)}

        return jobs.submit(job, run).summary()

    @app.post("/api/backtest")
    def backtest(body: BacktestRequest) -> dict[str, Any]:
        spec, universe = _fund_inputs(body)
        _require_keys("FINANCIAL_DATASETS_API_KEY")
        start = body.start or (
            _date.fromisoformat(body.as_of) - timedelta(weeks=_BACKTEST_WEEKS)
        ).isoformat()
        job = Job("backtest", {**body.model_dump(), "start": start},
                  f"{spec.name} {start} → {body.as_of}")

        def run(job: Job, progress) -> dict[str, Any]:
            with _llm_model(body.model):
                fund = Fund(spec)
                dates: list[str] = []
                nav: list[float] = []

                def on_cycle(i: int, n: int, record) -> None:
                    dates.append(record.as_of)
                    nav.append(record.nav)
                    progress({"i": i + 1, "n": n, "label": record.as_of,
                              "dates": list(dates), "nav": list(nav)})

                with FDClient() as raw:
                    result = backtest_fund(fund, start, body.as_of, CachedDataClient(raw),
                                           universe, on_cycle=on_cycle)
                return result.model_dump()

        return jobs.submit(job, run).summary()

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [j.summary() for j in jobs.recent()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.detail()

    # -- saved reports ----------------------------------------------------

    @app.get("/api/reports")
    def reports() -> list[dict[str, Any]]:
        rows = []
        for path in sorted(paths.RESEARCH_DIR.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                rows.append({"id": path.stem, **_report_row(ResearchReport(**data))})
            except Exception:
                continue  # a corrupt file is skipped, not fatal
        return rows

    @app.get("/api/scorecard")
    def scorecard(horizon: int = 90, benchmark: str = "SPY") -> dict[str, Any]:
        """Grade every saved report old enough to judge against forward returns."""
        _require_keys("FINANCIAL_DATASETS_API_KEY")
        reports = load_reports(paths.RESEARCH_DIR)
        with FDClient() as raw:
            card = grade(reports, CachedDataClient(raw), horizon_days=horizon,
                         benchmark=benchmark.upper())
        return card.model_dump()

    @app.get("/api/reports/{report_id}")
    def report(report_id: str) -> dict[str, Any]:
        path = paths.RESEARCH_DIR / f"{_safe_name(report_id)}.json"
        if not path.exists():
            raise HTTPException(404, "no such report")
        return json.loads(path.read_text())

    # -- the page ---------------------------------------------------------

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _keys() -> list[dict[str, Any]]:
    return [
        {"env_var": env, "label": label, "set": bool(os.environ.get(env)),
         "masked": masked(os.environ[env]) if os.environ.get(env) else None}
        for env, label in _KEY_LABELS.items()
    ]


def _require_keys(*env_vars: str) -> None:
    missing = [k for k in env_vars if not os.environ.get(k)]
    if missing:
        raise HTTPException(
            400, f"missing API key(s): {', '.join(missing)} — set them on the Settings page "
            f"or in {paths.ENV_PATH}",
        )


class _llm_model:
    """Route every agent built inside the block to *model* via the same
    process-global seam the CLI uses (HEDGE_FUND_LLM_MODEL); restored after."""

    def __init__(self, model: str | None) -> None:
        self._model = model
        self._previous: str | None = None

    def __enter__(self) -> None:
        self._previous = os.environ.get("HEDGE_FUND_LLM_MODEL")
        if self._model:
            os.environ["HEDGE_FUND_LLM_MODEL"] = self._model

    def __exit__(self, *args) -> None:
        if self._model:
            if self._previous is None:
                os.environ.pop("HEDGE_FUND_LLM_MODEL", None)
            else:
                os.environ["HEDGE_FUND_LLM_MODEL"] = self._previous


def _safe_name(name: str) -> str:
    if "/" in name or "\\" in name or name.startswith(".") or not name:
        raise HTTPException(400, "bad name")
    return name


def _mandate_path(file: str) -> Path:
    path = paths.MANDATES_DIR / _safe_name(file)
    if path.suffix != ".yaml" or not path.exists():
        raise HTTPException(404, f"no mandate {file!r} in {paths.MANDATES_DIR}")
    return path


def _fund_inputs(body: CycleRequest) -> tuple[FundSpec, list[str]]:
    try:
        spec = load_spec(_mandate_path(body.mandate))
        universe = normalize_universe(body.tickers)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return spec, universe


def _ranked(reports: list[ResearchReport]) -> list[ResearchReport]:
    return sorted(reports, key=lambda r: r.score, reverse=True)


def _report_row(r: ResearchReport) -> dict[str, Any]:
    """The columns of the ranked table — enough to draw a row without the
    full report."""
    desk = {s.model_name: (None if s.metadata.get("abstained") else s.value) for s in r.desk}
    p = r.position
    return {
        "ticker": r.ticker, "as_of": r.as_of, "model": r.model,
        "signal": r.signal, "confidence": r.confidence, "score": r.score,
        "action": r.action, "action_rationale": r.action_rationale,
        "headline": r.thesis.splitlines()[0] if r.thesis else "",
        "position": p.model_dump() if p else None,
        "desk": desk,
        "last_close": r.technicals.last_close if r.technicals else None,
        "n_sources": len(r.sources), "warnings": r.warnings,
    }


def _save_report(report: ResearchReport) -> Path:
    return save_report(report, paths.RESEARCH_DIR)


def _yaml_comment(path: Path) -> str:
    """The leading comment block of a library strategy, as its description."""
    lines = []
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("# ").rstrip())
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aihf web",
        description="Run the local web app: the research desk and the fund runner "
        "in a browser, over the same engine as the CLI and TUI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args(argv)

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    print(f"aihf web · {url}  (ctrl-c to stop)")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
