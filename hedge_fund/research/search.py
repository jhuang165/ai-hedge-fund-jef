"""Tavily search client — the web-search provider for `aihf research`.

Mirrors hedge_fund/data/client.py's (FDClient) house style: a requests.Session
with the API key on every request, a fail-loud contract (infrastructure
failures raise), and retry-with-backoff on 429. Tavily has no "404 means no
data" case — a search that finds nothing still succeeds with an empty list —
so every non-2xx response here raises.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Protocol, runtime_checkable

import requests

from hedge_fund.research.models import SearchResult

logger = logging.getLogger(__name__)


class SearchClientError(Exception):
    """A search request failed for infrastructure reasons (auth, rate limit,
    server error, network). Distinct from "zero results" — that's a normal,
    successful search outcome, not a failure.
    """

    def __init__(self, message: str, *, status_code: int | None = None, query: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.query = query


@runtime_checkable
class SearchClient(Protocol):
    """Structural interface every search provider satisfies — no inheritance
    required, matching hedge_fund/data/protocol.py's DataClient pattern.
    """

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class TavilyClient:
    """Tavily (https://tavily.com) search API client.

    Usage::

        with TavilyClient() as search:
            results = search.search("AAPL stock news")
    """

    BASE_URL = "https://api.tavily.com"
    _RETRY_DELAYS = (5, 15, 30)

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self._api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {self._api_key}"

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> TavilyClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run one search. `topic="finance"` biases results toward market and
        financial sources; `time_range="month"` keeps "up to date" honest.
        """
        resp = self._request("POST", "/search", json={
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "topic": "finance",
            "time_range": "month",
        })
        rows = resp.json().get("results") or []
        return [
            SearchResult(
                query=query,
                title=row.get("title", ""),
                url=row["url"],
                content=row.get("content", ""),
                score=row.get("score"),
                published_date=row.get("published_date"),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """HTTP request with retry on 429. Fail-loud: raises SearchClientError
        on network errors, HTTP errors, and exhausted rate-limit retries.
        """
        url = self.BASE_URL + path
        for attempt, delay in enumerate((*self._RETRY_DELAYS, None)):
            try:
                resp = self._session.request(method, url, timeout=self._timeout, **kwargs)
            except requests.RequestException as exc:
                raise SearchClientError(f"{method} {path} failed: {exc}") from exc

            if resp.status_code == 429 and delay is not None:
                logger.info(
                    "Rate limited (429), retrying in %ds (attempt %d/%d)",
                    delay, attempt + 1, len(self._RETRY_DELAYS),
                )
                time.sleep(delay)
                continue

            if resp.status_code >= 400:
                raise SearchClientError(
                    f"{method} {path} returned {resp.status_code}: {resp.text[:200]}",
                    status_code=resp.status_code,
                )

            return resp

        raise SearchClientError(
            f"{method} {path} rate limited (429) after {len(self._RETRY_DELAYS)} retries",
            status_code=429,
        )
