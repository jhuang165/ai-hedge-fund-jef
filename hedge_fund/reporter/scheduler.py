"""The loop: a cycle over the watchlist, on a schedule, with status.

A cycle has two phases. Evidence is gathered one name at a time (the data
layer and its disk cache are not built for concurrent writers, and the
fetches are seconds each); the model calls then run in a small pool,
sized to the wrapper's own parallelism, because each one is minutes.
Every name's outcome — a brief or an error — is saved as soon as it is
known, so a cycle that dies halfway still leaves the first names fresh.

Cadence: during the active window on weekdays a cycle starts every
`interval_minutes` after the previous one finished; outside it (nights,
weekends) every `off_hours_interval_minutes`, or never if that is 0. A
manual run can be requested at any time and runs next.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

from hedge_fund.data import open_cached_client
from hedge_fund.reporter.brief import Brief, CodexClient, produce_brief
from hedge_fund.reporter.config import ReporterConfig
from hedge_fund.reporter.evidence import Evidence, fetch_headlines, gather_evidence
from hedge_fund.reporter.store import ReportStore, now_iso

log = logging.getLogger(__name__)

EvidenceFn = Callable[[str, str, Any], Evidence]


# ---------------------------------------------------------------------------
# Schedule arithmetic (pure, tested)
# ---------------------------------------------------------------------------

def is_active(local: datetime, window: tuple[int, int]) -> bool:
    """Inside the Monday-Friday active window?"""
    if local.weekday() >= 5:
        return False
    minutes = local.hour * 60 + local.minute
    start, end = window
    return start <= minutes < end


def interval_minutes(local: datetime, cfg: ReporterConfig) -> int:
    """How long after a cycle the next one is due, from the local time it
    finished. 0 means "not until the active window opens"."""
    return cfg.interval_minutes if is_active(local, cfg.active_window) else cfg.off_hours_interval_minutes


def next_window_open(local: datetime, window: tuple[int, int]) -> datetime:
    """The next weekday moment the active window opens, strictly after *local*."""
    start = window[0]
    candidate = local.replace(hour=start // 60, minute=start % 60, second=0, microsecond=0)
    while candidate <= local or candidate.weekday() >= 5:
        candidate = (candidate + timedelta(days=1)).replace(hour=start // 60, minute=start % 60)
    return candidate


def next_run_at(last_finished: datetime, cfg: ReporterConfig) -> datetime:
    """When the next scheduled cycle is due, given when the last one ended."""
    tz = ZoneInfo(cfg.timezone)
    local = last_finished.astimezone(tz)
    gap = interval_minutes(local, cfg)
    if gap <= 0:
        return next_window_open(local, cfg.active_window).astimezone(timezone.utc)
    due = local + timedelta(minutes=gap)
    # A night-time gap must not skip the morning: if the window opens before
    # the off-hours gap elapses, run at the open instead.
    if not is_active(local, cfg.active_window):
        opening = next_window_open(local, cfg.active_window)
        if opening < due:
            due = opening
    return due.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# The reporter
# ---------------------------------------------------------------------------

class Reporter:
    def __init__(
        self,
        cfg: ReporterConfig,
        store: ReportStore,
        client: CodexClient | None = None,
        *,
        evidence_fn: EvidenceFn | None = None,
        open_client: Callable[[], Any] = open_cached_client,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.client = client or CodexClient(
            cfg.llm_base_url, cfg.llm_api_key,
            timeout_s=cfg.llm_timeout_s, reasoning_effort=cfg.reasoning_effort,
        )
        self._evidence_fn = evidence_fn or self._default_evidence
        self._open_client = open_client
        self._lock = threading.Lock()          # one cycle at a time
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: list[str] | None = None  # tickers for a requested manual run
        self.current: dict[str, Any] | None = None
        self.next_run: datetime | None = None
        self.last_error: str | None = None
        self.last_publish: dict[str, Any] | None = None  # outcome of the last static publish
        store.seed_symbols(cfg.seed_symbols)

    # ------------------------------------------------------------ evidence

    def _default_evidence(self, ticker: str, as_of: str, data_client: Any) -> Evidence:
        return gather_evidence(
            ticker, as_of, data_client,
            headlines_fn=lambda t: fetch_headlines(
                t, lookback_hours=self.cfg.headline_lookback_hours, limit=self.cfg.max_headlines),
        )

    # --------------------------------------------------------------- cycle

    def run_cycle(self, trigger: str = "manual", tickers: list[str] | None = None) -> dict[str, Any]:
        """One pass over *tickers* (default: the watchlist). Returns the run
        row. Safe to call from any thread; a second caller waits."""
        with self._lock:
            tickers = [t.upper() for t in (tickers or self.store.symbols())]
            run_id = self.store.create_run(trigger)
            started = time.time()
            self.current = {"run_id": run_id, "trigger": trigger, "started_at": now_iso(),
                            "total": len(tickers), "done": 0, "phase": "evidence", "ticker": None}
            n_ok = n_failed = 0
            fatal: str | None = None
            try:
                as_of = datetime.now(timezone.utc).date().isoformat()
                evidence: dict[str, Evidence] = {}
                errors: dict[str, str] = {}
                with self._open_client() as data_client:
                    for t in tickers:
                        self.current["ticker"] = t
                        try:
                            evidence[t] = self._evidence_fn(t, as_of, data_client)
                        except Exception as exc:
                            log.exception("evidence for %s failed", t)
                            errors[t] = f"evidence: {exc}"
                self.current["phase"] = "briefs"
                self.current["ticker"] = None
                for t, err in errors.items():
                    self.store.save_report(run_id=run_id, ticker=t, model=self.cfg.model,
                                           brief=None, error=err, duration_s=0.0)
                    n_failed += 1
                    self.current["done"] += 1

                def one(t: str) -> tuple[str, Brief | None, str | None, float]:
                    t0 = time.time()
                    try:
                        prev = self.store.latest_brief(t)
                        b = produce_brief(evidence[t], prev["brief"] if prev else None, self.client, self.cfg.model)
                        return t, b, None, time.time() - t0
                    except Exception as exc:
                        log.warning("brief for %s failed: %s", t, exc)
                        return t, None, f"{type(exc).__name__}: {exc}", time.time() - t0

                with ThreadPoolExecutor(max_workers=max(1, self.cfg.parallel), thread_name_prefix="brief") as pool:
                    for t, brief, err, dur in pool.map(one, [t for t in tickers if t in evidence]):
                        self.store.save_report(
                            run_id=run_id, ticker=t, model=self.cfg.model,
                            brief=brief.model_dump() if brief else None, error=err, duration_s=dur,
                            generated_at=brief.generated_at if brief else None,
                        )
                        if brief:
                            n_ok += 1
                        else:
                            n_failed += 1
                        self.current["done"] += 1
            except Exception:
                fatal = traceback.format_exc()
                log.error("cycle %s failed:\n%s", run_id, fatal)
            finally:
                self.store.finish_run(run_id, n_ok=n_ok, n_failed=n_failed, error=fatal)
                self.last_error = fatal
                self.current = None
                log.info("cycle %s %s: %d ok, %d failed, %.0fs", run_id, trigger, n_ok, n_failed, time.time() - started)
            if self.cfg.publish_enabled:
                self.publish()
            return self.store.runs(limit=1)[0]

    # ------------------------------------------------------------ publish

    def publish(self) -> dict[str, Any]:
        """Export the static site (and upload it, if a command is set).
        Never raises: the outcome lands in `last_publish` and /api/status,
        because a broken upload must not stop the next cycle."""
        from hedge_fund.reporter.publish import publish  # local: keeps the import graph one-way

        at = now_iso()
        try:
            summary = publish(self, self.cfg.publish_dir, self.cfg.publish_cmd, timeout_s=self.cfg.publish_timeout_s)
            self.last_publish = {"at": at, "ok": True, "detail": f"{summary['reports']} reports in {summary['took_s']}s"}
            log.info("published %s: %s", summary["dir"], self.last_publish["detail"])
        except Exception as exc:
            log.exception("publish failed")
            self.last_publish = {"at": at, "ok": False, "detail": f"{type(exc).__name__}: {exc}"[:500]}
        return self.last_publish

    # ----------------------------------------------------------- the loop

    def request_run(self, tickers: list[str] | None = None) -> bool:
        """Ask the loop to run now. False if the loop is not running (then
        the caller may run_cycle() itself) or a run is already in progress."""
        if self._thread is None or not self._thread.is_alive():
            return False
        if self.current is not None:
            return False
        self._pending = tickers
        self._wake.set()
        return True

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="reporter-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self) -> None:
        now = datetime.now(timezone.utc)
        last = self.store.last_run()
        if self.cfg.run_on_boot:
            self.next_run = now
        elif last and last.get("finished_at"):
            self.next_run = next_run_at(datetime.fromisoformat(last["finished_at"]), self.cfg)
        else:
            self.next_run = now
        while not self._stop.is_set():
            wait = max(0.0, (self.next_run - datetime.now(timezone.utc)).total_seconds()) if self.next_run else 3600
            woke = self._wake.wait(timeout=min(wait, 60.0))
            if self._stop.is_set():
                break
            if woke:
                self._wake.clear()
                pending, self._pending = self._pending, None
                self.run_cycle("manual", pending)
                self.next_run = next_run_at(datetime.now(timezone.utc), self.cfg)
            elif self.next_run and datetime.now(timezone.utc) >= self.next_run:
                self.run_cycle("schedule")
                self.next_run = next_run_at(datetime.now(timezone.utc), self.cfg)

    # -------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        ok, detail = self.client.healthy()
        return {
            "now": now_iso(),
            "scheduler": {
                "running": self.running,
                "current": self.current,
                "next_run_at": self.next_run.isoformat(timespec="seconds") if self.next_run else None,
                "last_run": self.store.last_run(),
                "last_error": self.last_error,
                "interval_minutes": self.cfg.interval_minutes,
                "off_hours_interval_minutes": self.cfg.off_hours_interval_minutes,
                "active_hours": f"{self.cfg.active_window[0] // 60:02d}:{self.cfg.active_window[0] % 60:02d}-"
                                f"{self.cfg.active_window[1] // 60:02d}:{self.cfg.active_window[1] % 60:02d}",
                "timezone": self.cfg.timezone,
            },
            "llm": {"base_url": self.cfg.llm_base_url, "model": self.cfg.model,
                    "reasoning_effort": self.cfg.reasoning_effort or "default", "reachable": ok, "detail": detail},
            "symbols": self.store.symbols(),
            "auth": self.cfg.auth_enabled,
            "publish": {"enabled": self.cfg.publish_enabled, "last": self.last_publish},
        }
