"""Reporter settings: every knob is an environment variable so the same
code runs from a checkout, a container, or a cron entry."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from hedge_fund import paths

# The watchlist the store is seeded with when it has none. Symbols are
# otherwise managed in the UI (or REPORTER_SYMBOLS on first boot).
DEFAULT_SYMBOLS: tuple[str, ...] = (
    "SMCI", "NVDA", "QCOM", "GOOG", "AAPL", "AMZN", "MSFT", "META", "WFC", "MU",
)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {os.environ.get(name)!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "1" if default else "0").lower()
    return raw in {"1", "true", "yes", "on"}


def _parse_window(raw: str) -> tuple[int, int]:
    """'07:00-20:00' -> (420, 1200) minutes after local midnight."""
    try:
        start, end = raw.split("-")
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
    except ValueError as exc:
        raise ValueError(f"REPORTER_ACTIVE_HOURS must look like HH:MM-HH:MM, got {raw!r}") from exc
    return sh * 60 + sm, eh * 60 + em


@dataclass
class ReporterConfig:
    # Where briefs live. Under ~/.hedge-fund like every other user file.
    db_path: Path = field(default_factory=lambda: Path(_env("REPORTER_DB", str(paths.USER_DIR / "reporter.db"))))
    seed_symbols: tuple[str, ...] = field(default_factory=lambda: tuple(
        s.strip().upper() for s in _env("REPORTER_SYMBOLS", ",".join(DEFAULT_SYMBOLS)).replace(" ", ",").split(",") if s.strip()
    ))

    # The OpenAI-compatible endpoint (Codex-Wrapper) and the model behind it.
    llm_base_url: str = field(default_factory=lambda: _env("CODEX_WRAPPER_URL", "http://127.0.0.1:8020/v1").rstrip("/"))
    llm_api_key: str = field(default_factory=lambda: _env("CODEX_WRAPPER_API_KEY", ""))
    model: str = field(default_factory=lambda: _env("REPORTER_MODEL", "gpt-6-luna"))
    reasoning_effort: str = field(default_factory=lambda: _env("REPORTER_REASONING", ""))  # "", low, medium, high
    llm_timeout_s: int = field(default_factory=lambda: _env_int("REPORTER_LLM_TIMEOUT", 600))
    parallel: int = field(default_factory=lambda: _env_int("REPORTER_PARALLEL", 2))

    # Cadence. Active window applies Monday-Friday in `timezone`; outside it
    # (nights, weekends) the off-hours interval applies. 0 disables off-hours runs.
    interval_minutes: int = field(default_factory=lambda: _env_int("REPORTER_INTERVAL_MINUTES", 60))
    off_hours_interval_minutes: int = field(default_factory=lambda: _env_int("REPORTER_OFF_HOURS_INTERVAL_MINUTES", 240))
    active_window: tuple[int, int] = field(default_factory=lambda: _parse_window(_env("REPORTER_ACTIVE_HOURS", "07:00-20:00")))
    timezone: str = field(default_factory=lambda: _env("REPORTER_TIMEZONE", "America/New_York"))
    autostart: bool = field(default_factory=lambda: _env_bool("REPORTER_AUTOSTART", True))
    run_on_boot: bool = field(default_factory=lambda: _env_bool("REPORTER_RUN_ON_BOOT", True))

    # Who may read the dashboard. Empty password = no auth (local use only).
    auth_user: str = field(default_factory=lambda: _env("REPORTER_USER", "desk"))
    auth_password: str = field(default_factory=lambda: _env("REPORTER_PASSWORD", ""))

    headline_lookback_hours: int = field(default_factory=lambda: _env_int("REPORTER_HEADLINE_HOURS", 48))
    max_headlines: int = field(default_factory=lambda: _env_int("REPORTER_MAX_HEADLINES", 14))

    # Static publishing (see publish.py): after every cycle, export the site
    # to this directory and run this shell command there (e.g. `firebase
    # deploy --only hosting`). Empty = off.
    publish_dir: Path | None = field(default_factory=lambda: Path(_env("REPORTER_PUBLISH_DIR", "")) if _env("REPORTER_PUBLISH_DIR", "") else None)
    publish_cmd: str = field(default_factory=lambda: _env("REPORTER_PUBLISH_CMD", ""))
    publish_timeout_s: int = field(default_factory=lambda: _env_int("REPORTER_PUBLISH_TIMEOUT", 600))

    @property
    def auth_enabled(self) -> bool:
        return bool(self.auth_password)

    @property
    def publish_enabled(self) -> bool:
        return self.publish_dir is not None
