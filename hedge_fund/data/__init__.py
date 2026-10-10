"""v2 data pipeline — data provider protocol, FD client, and response models."""

from hedge_fund.data.cached import CachedDataClient
from hedge_fund.data.client import FDClient, FDClientError
from hedge_fund.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsData,
    EarningsRecord,
    Filing,
    FinancialMetrics,
    InsiderTrade,
    Price,
)
from hedge_fund.data.protocol import DataClient
from hedge_fund.data.source import cache_dir_for, data_source, open_cached_client, open_data_client
from hedge_fund.data.yfinance_client import YFinanceClient, YFinanceClientError

__all__ = [
    "YFinanceClient",
    "YFinanceClientError",
    "cache_dir_for",
    "data_source",
    "open_cached_client",
    "open_data_client",
    "CachedDataClient",
    "CompanyFacts",
    "CompanyNews",
    "DataClient",
    "Earnings",
    "EarningsData",
    "EarningsRecord",
    "FDClient",
    "FDClientError",
    "Filing",
    "FinancialMetrics",
    "InsiderTrade",
    "Price",
]
