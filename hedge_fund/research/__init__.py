"""Research — read-only stock diagnosis, separate from the
mandate/strategy/backtest pipeline. See hedge_fund/research/diagnose.py.
"""

from __future__ import annotations

from hedge_fund.research.diagnose import DEFAULT_QUERIES, default_desk, diagnose
from hedge_fund.research.models import (
    FLAT_ACTIONS,
    HELD_ACTIONS,
    Claim,
    Position,
    PositionMark,
    ResearchReport,
    SearchResult,
)
from hedge_fund.research.positions import (
    ResearchTarget,
    load_portfolio,
    merge_targets,
    parse_target,
)
from hedge_fund.research.scorecard import GradedCall, Scorecard, grade
from hedge_fund.research.search import SearchClient, SearchClientError, TavilyClient
from hedge_fund.research.store import load_reports, save_report

__all__ = [
    "Claim",
    "DEFAULT_QUERIES",
    "FLAT_ACTIONS",
    "GradedCall",
    "HELD_ACTIONS",
    "Position",
    "PositionMark",
    "ResearchReport",
    "ResearchTarget",
    "Scorecard",
    "SearchClient",
    "SearchClientError",
    "SearchResult",
    "TavilyClient",
    "default_desk",
    "diagnose",
    "grade",
    "load_portfolio",
    "load_reports",
    "merge_targets",
    "parse_target",
    "save_report",
]
