"""Where research reports live, and how they come back.

Every research run — CLI or web — is saved under ~/.hedge-fund/research/
as one JSON file per report, named `{as_of}_{TICKER}_{stamp}.json`. The
scorecard reads them back to grade the desk's calls against what the
market did next; the Reports page reads them back to show them.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from hedge_fund import paths
from hedge_fund.research.models import ResearchReport


def save_report(report: ResearchReport, directory: Path | None = None) -> Path:
    """Write one report; returns its path."""
    directory = directory or paths.RESEARCH_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%H%M%S%f")
    path = directory / f"{report.as_of}_{report.ticker}_{stamp}.json"
    path.write_text(report.model_dump_json(indent=2))
    return path


def load_reports(directory: Path | None = None) -> list[ResearchReport]:
    """Every readable report, oldest file first. A corrupt file is skipped,
    not fatal — one bad save must not hide the rest of the record."""
    directory = directory or paths.RESEARCH_DIR
    if not directory.exists():
        return []
    reports: list[ResearchReport] = []
    for path in sorted(directory.glob("*.json")):
        try:
            reports.append(ResearchReport(**json.loads(path.read_text())))
        except Exception:
            continue
    return reports
