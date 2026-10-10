"""Where briefs live: one SQLite file, three tables.

symbols  — the watchlist (managed in the UI; seeded once from config)
runs     — one row per cycle: when, why (schedule / manual), how it ended
reports  — one row per symbol per cycle: the validated brief as JSON, or
           the error that stopped it, so a failed name is visible rather
           than silently stale

SQLite because the reporter is one process on one small box: no server,
one file to back up, and `sqlite3` ships with Python. Every method takes
the connection lock, so the scheduler thread, the request handlers, and
the per-symbol worker threads can all write.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
    ticker     TEXT PRIMARY KEY,
    added_at   TEXT NOT NULL,
    note       TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    trigger     TEXT NOT NULL,
    status      TEXT NOT NULL,          -- running | done | failed
    n_ok        INTEGER NOT NULL DEFAULT 0,
    n_failed    INTEGER NOT NULL DEFAULT 0,
    error       TEXT
);
CREATE TABLE IF NOT EXISTS reports (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    ticker       TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    model        TEXT NOT NULL,
    status       TEXT NOT NULL,         -- ok | failed
    brief        TEXT,                  -- JSON (Brief.model_dump) when ok
    error        TEXT,
    duration_s   REAL
);
CREATE INDEX IF NOT EXISTS reports_ticker_time ON reports (ticker, generated_at DESC);
CREATE INDEX IF NOT EXISTS reports_run ON reports (run_id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ReportStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            if str(self.path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ symbols

    def symbols(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT ticker FROM symbols ORDER BY rowid").fetchall()
        return [r["ticker"] for r in rows]

    def seed_symbols(self, tickers: Iterable[str]) -> bool:
        """Insert *tickers* only if the watchlist is empty. True if seeded."""
        with self._lock:
            if self._conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]:
                return False
            stamp = now_iso()
            self._conn.executemany(
                "INSERT OR IGNORE INTO symbols (ticker, added_at) VALUES (?, ?)",
                [(t.strip().upper(), stamp) for t in tickers if t.strip()],
            )
            self._conn.commit()
            return True

    def add_symbol(self, ticker: str, note: str = "") -> bool:
        """True if added, False if it was already there."""
        ticker = ticker.strip().upper()
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO symbols (ticker, added_at, note) VALUES (?, ?, ?)",
                (ticker, now_iso(), note),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def remove_symbol(self, ticker: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM symbols WHERE ticker = ?", (ticker.strip().upper(),))
            self._conn.commit()
            return cur.rowcount == 1

    # --------------------------------------------------------------- runs

    def create_run(self, trigger: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO runs (started_at, trigger, status) VALUES (?, ?, 'running')",
                (now_iso(), trigger),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, *, n_ok: int, n_failed: int, error: str | None = None) -> None:
        status = "failed" if error else "done"
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, status = ?, n_ok = ?, n_failed = ?, error = ? WHERE id = ?",
                (now_iso(), status, n_ok, n_failed, error, run_id),
            )
            self._conn.commit()

    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def last_run(self) -> dict[str, Any] | None:
        runs = self.runs(limit=1)
        return runs[0] if runs else None

    # ------------------------------------------------------------ reports

    def save_report(
        self,
        *,
        run_id: int,
        ticker: str,
        model: str,
        brief: dict[str, Any] | None,
        error: str | None,
        duration_s: float,
        generated_at: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO reports (run_id, ticker, generated_at, model, status, brief, error, duration_s) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, ticker.upper(), generated_at or now_iso(), model,
                    "ok" if brief is not None else "failed",
                    json.dumps(brief) if brief is not None else None,
                    error, duration_s,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def report(self, report_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        return _row_to_report(row) if row else None

    def reports_for(self, ticker: str, limit: int = 48, *, ok_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM reports WHERE ticker = ?"
        if ok_only:
            sql += " AND status = 'ok'"
        sql += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(sql, (ticker.upper(), limit)).fetchall()
        return [_row_to_report(r) for r in rows]

    def latest_brief(self, ticker: str) -> dict[str, Any] | None:
        """The newest successful brief for *ticker*, or None."""
        rows = self.reports_for(ticker, limit=1, ok_only=True)
        return rows[0] if rows else None

    def latest(self) -> dict[str, dict[str, Any]]:
        """Per watchlist symbol: the newest report of any status, and the
        newest successful brief (which may be an older row when the latest
        attempt failed)."""
        out: dict[str, dict[str, Any]] = {}
        for ticker in self.symbols():
            rows = self.reports_for(ticker, limit=1)
            latest = rows[0] if rows else None
            ok = latest if latest and latest["status"] == "ok" else self.latest_brief(ticker)
            out[ticker] = {"latest": latest, "brief": ok}
        return out

    def reports_in_run(self, run_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM reports WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
        return [_row_to_report(r) for r in rows]


def _row_to_report(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["brief"] = json.loads(d["brief"]) if d.get("brief") else None
    return d
