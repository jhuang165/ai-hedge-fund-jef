"""Which data source the fund reads from, and how to open it.

Two providers satisfy the DataClient protocol: FDClient (Financial Datasets,
needs FINANCIAL_DATASETS_API_KEY, point-in-time filings) and YFinanceClient
(Yahoo Finance through yfinance, no key, see its module docstring for what
it approximates). The choice is one function so every entry point agrees:

    HEDGE_FUND_DATA_SOURCE=fd | yfinance    forces a provider
    otherwise                                fd if the FD key is set, else yfinance

Each provider gets its own on-disk cache directory, so a row computed from
Yahoo's statements can never be served back as a Financial Datasets filing.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from hedge_fund.data.cached import CachedDataClient
from hedge_fund.data.protocol import DataClient
from hedge_fund.paths import CACHE_DIR

SOURCES = ("fd", "yfinance")


def data_source() -> str:
    """The provider name the current environment selects."""
    forced = os.environ.get("HEDGE_FUND_DATA_SOURCE", "").strip().lower()
    if forced:
        if forced not in SOURCES:
            raise ValueError(
                f"HEDGE_FUND_DATA_SOURCE={forced!r}; expected one of {', '.join(SOURCES)}"
            )
        return forced
    return "fd" if os.environ.get("FINANCIAL_DATASETS_API_KEY") else "yfinance"


def cache_dir_for(source: str | None = None):
    source = source or data_source()
    return CACHE_DIR / ("data" if source == "fd" else f"data-{source}")


@contextmanager
def open_data_client(source: str | None = None) -> Iterator[DataClient]:
    """The raw provider for *source* (default: whatever the environment
    selects), as a context manager."""
    source = source or data_source()
    if source == "fd":
        from hedge_fund.data.client import FDClient

        with FDClient() as client:
            yield client
    else:
        from hedge_fund.data.yfinance_client import YFinanceClient

        with YFinanceClient() as client:
            yield client


@contextmanager
def open_cached_client(source: str | None = None, refresh: bool = False) -> Iterator[CachedDataClient]:
    """The selected provider wrapped in its own disk cache — what every
    CLI and web entry point should use."""
    source = source or data_source()
    with open_data_client(source) as raw:
        yield CachedDataClient(raw, cache_dir=cache_dir_for(source), refresh=refresh)
