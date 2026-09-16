"""Research — read-only web-search stock diagnosis, separate from the
mandate/strategy/backtest pipeline. See hedge_fund/research/diagnose.py.
"""

from __future__ import annotations

from hedge_fund.research.diagnose import DEFAULT_QUERIES, diagnose
from hedge_fund.research.models import Claim, ResearchReport, SearchResult
from hedge_fund.research.search import SearchClient, SearchClientError, TavilyClient

__all__ = [
    "Claim",
    "DEFAULT_QUERIES",
    "ResearchReport",
    "SearchClient",
    "SearchClientError",
    "SearchResult",
    "TavilyClient",
    "diagnose",
]
