"""YFinanceClient tests — a fake yfinance Ticker built from pandas frames,
no network. Pins the mapping from Yahoo's shapes to the fund's models: TTM
arithmetic over quarter ends, annual rows extending history, the assumed
filing lag doing point-in-time filtering, insider text normalized to Form 4
codes, and earnings surprises labelled BEAT/MISS/MEET."""

from datetime import date, timedelta

import pandas as pd
import pytest

from hedge_fund.data.yfinance_client import (
    FILING_LAG_DAYS,
    YFinanceClient,
    YFinanceClientError,
    normalize_transaction,
)

Q = ["2025-06-30", "2025-03-31", "2024-12-31", "2024-09-30", "2024-06-30"]   # newest first
A = ["2024-09-30", "2023-09-30", "2022-09-30"]


def _frame(rows: dict[str, list[float]], periods: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=[pd.Timestamp(p) for p in periods]).T


class FakeTicker:
    """Five quarters of statements (revenue 100/qtr, net income 10/qtr, EPS
    1/qtr), three fiscal years, a flat 50.00 price, two insider rows, and
    three earnings dates (two reported, one upcoming)."""

    def __init__(self, symbol):
        self.symbol = symbol
        n = len(Q)
        self.quarterly_income_stmt = _frame({
            "Total Revenue": [100.0] * n, "Gross Profit": [40.0] * n,
            "Operating Income": [20.0] * n, "Net Income": [10.0] * n,
            "EBITDA": [25.0] * n, "EBIT": [20.0] * n, "Diluted EPS": [1.0] * n,
            "Diluted Average Shares": [10.0] * n,
        }, Q)
        self.quarterly_balance_sheet = _frame({
            "Stockholders Equity": [200.0] * n, "Total Assets": [400.0] * n,
            "Total Debt": [100.0] * n, "Cash And Cash Equivalents": [50.0] * n,
            "Current Assets": [150.0] * n, "Current Liabilities": [100.0] * n,
            "Invested Capital": [300.0] * n, "Ordinary Shares Number": [10.0] * n,
        }, Q)
        self.quarterly_cashflow = _frame({
            "Free Cash Flow": [12.0] * n, "Operating Cash Flow": [15.0] * n,
            "Cash Dividends Paid": [-4.0] * n,
        }, Q)
        m = len(A)
        self.income_stmt = _frame({
            "Total Revenue": [400.0, 360.0, 300.0], "Gross Profit": [160.0] * m,
            "Operating Income": [80.0] * m, "Net Income": [40.0, 36.0, 30.0],
            "EBITDA": [100.0] * m, "EBIT": [80.0] * m, "Diluted EPS": [4.0, 3.6, 3.0],
        }, A)
        self.balance_sheet = _frame({
            "Stockholders Equity": [200.0, 180.0, 160.0], "Total Assets": [400.0] * m,
            "Total Debt": [100.0] * m, "Cash And Cash Equivalents": [50.0] * m,
            "Current Assets": [150.0] * m, "Current Liabilities": [100.0] * m,
            "Invested Capital": [300.0] * m, "Ordinary Shares Number": [10.0] * m,
        }, A)
        self.cashflow = _frame({
            "Free Cash Flow": [48.0] * m, "Operating Cash Flow": [60.0] * m,
            "Cash Dividends Paid": [-16.0] * m,
        }, A)
        self.insider_transactions = pd.DataFrame([
            {"Shares": 100, "Value": 5000.0, "Text": "Sale at price 50.00 per share.",
             "Insider": "DOE JANE", "Position": "CEO", "Start Date": pd.Timestamp("2025-06-10")},
            {"Shares": 200, "Value": 9000.0, "Text": "Purchase at price 45.00 per share.",
             "Insider": "ROE RICHARD", "Position": "Director", "Start Date": pd.Timestamp("2025-05-01")},
            {"Shares": 300, "Value": None, "Text": "Stock Gift",
             "Insider": "DOE JANE", "Position": "CEO", "Start Date": pd.Timestamp("2025-04-01")},
        ])
        self.earnings_history = pd.DataFrame(
            {"epsActual": [1.0, 1.0], "epsEstimate": [0.9, 1.1]},
            index=pd.Index([pd.Timestamp("2025-03-31"), pd.Timestamp("2025-06-30")], name="quarter"),
        )
        self._dates = pd.DataFrame(
            {"EPS Estimate": [1.2, 1.1, 0.9], "Reported EPS": [None, 1.0, 1.0],
             "Surprise(%)": [None, -9.09, 11.11]},
            index=pd.Index([pd.Timestamp("2025-10-30"), pd.Timestamp("2025-07-30"),
                            pd.Timestamp("2025-04-30")], name="Earnings Date"),
        )
        self.info = {"longName": "Fake Corp", "sector": "Tech", "industry": "Widgets",
                     "exchange": "NMS", "marketCap": 500.0}
        self.history_calls = []

    def history(self, start, end, **kw):
        self.history_calls.append((start, end))
        if self.symbol == "NOPE":
            raise Exception("NOPE: possibly delisted; no price data found")
        if self.symbol == "DOWN":
            raise Exception("connection reset")
        days = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
        return pd.DataFrame({"Open": 50.0, "High": 51.0, "Low": 49.0, "Close": 50.0,
                             "Volume": 1000}, index=days)

    def get_earnings_dates(self, limit):
        return self._dates


@pytest.fixture
def client():
    return YFinanceClient(ticker_factory=FakeTicker)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def test_prices_are_daily_bars_with_iso_dates(client):
    bars = client.get_prices("FAKE", "2025-06-02", "2025-06-06")
    assert [b.time for b in bars] == ["2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06"]
    assert bars[0].close == 50.0 and bars[0].volume == 1000


def test_unknown_ticker_is_empty_not_an_error(client):
    assert client.get_prices("NOPE", "2025-06-02", "2025-06-06") == []


def test_infrastructure_failure_raises(client):
    with pytest.raises(YFinanceClientError):
        client.get_prices("DOWN", "2025-06-02", "2025-06-06")


def test_only_daily_bars(client):
    with pytest.raises(YFinanceClientError):
        client.get_prices("FAKE", "2025-06-02", "2025-06-06", interval="minute")


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

def test_ttm_rows_sum_four_quarters_and_annual_rows_extend_history(client):
    rows = client.get_financial_metrics("FAKE", "2026-01-01", limit=10)
    periods = [m.report_period for m in rows]
    # Two quarter ends have four trailing quarters; every fiscal year end
    # older than the oldest of those extends the history.
    assert periods == ["2025-06-30", "2025-03-31", "2024-09-30", "2023-09-30", "2022-09-30"]
    latest = rows[0]
    assert latest.period == "ttm"
    assert latest.earnings_per_share == pytest.approx(4.0)
    assert latest.net_margin == pytest.approx(40 / 400)
    assert latest.gross_margin == pytest.approx(160 / 400)
    assert latest.return_on_equity == pytest.approx(40 / 200)
    assert latest.debt_to_equity == pytest.approx(0.5)
    assert latest.current_ratio == pytest.approx(1.5)
    assert latest.book_value_per_share == pytest.approx(20.0)
    assert latest.free_cash_flow_per_share == pytest.approx(4.8)
    assert latest.payout_ratio == pytest.approx(16 / 40)
    # Valuation struck at the period-end close of 50.
    assert latest.market_cap == pytest.approx(500.0)
    assert latest.price_to_earnings_ratio == pytest.approx(12.5)
    assert latest.price_to_book_ratio == pytest.approx(2.5)
    assert latest.free_cash_flow_yield == pytest.approx(4.8 / 50)
    assert latest.enterprise_value == pytest.approx(500 + 100 - 50)
    # Growth needs the prior year's window: absent for the quarterly rows
    # here (only five quarters), present for the annual ones.
    assert latest.revenue_growth is None
    fy2023 = rows[3]
    assert fy2023.revenue_growth == pytest.approx(360 / 300 - 1)
    assert fy2023.earnings_per_share_growth == pytest.approx(3.6 / 3.0 - 1)


def test_filing_lag_makes_rows_point_in_time(client):
    lag = timedelta(days=FILING_LAG_DAYS)
    rows = client.get_financial_metrics("FAKE", "2026-01-01")
    assert rows[0].filing_date == (date(2025, 6, 30) + lag).isoformat()
    # The day before the assumed filing, the newest quarter is not public yet.
    day_before = (date(2025, 6, 30) + lag - timedelta(days=1)).isoformat()
    assert [m.report_period for m in client.get_financial_metrics("FAKE", day_before)][0] == "2025-03-31"


def test_limit_and_period_contract(client):
    assert len(client.get_financial_metrics("FAKE", "2026-01-01", limit=1)) == 1
    with pytest.raises(YFinanceClientError):
        client.get_financial_metrics("FAKE", "2026-01-01", period="annual")


# ---------------------------------------------------------------------------
# Insiders, facts, earnings
# ---------------------------------------------------------------------------

def test_insider_rows_are_normalized_and_windowed(client):
    trades = client.get_insider_trades("FAKE", "2025-06-30", start_date="2025-04-15")
    assert [(t.name, t.transaction_type, t.transaction_shares, t.transaction_price_per_share)
            for t in trades] == [
        ("DOE JANE", "S-Sale", 100.0, 50.0),
        ("ROE RICHARD", "P-Purchase", 200.0, 45.0),
    ]
    assert trades[0].filing_date == "2025-06-10" and trades[0].title == "CEO"
    # The gift falls outside the window; inside it, it is a non-market code.
    everything = client.get_insider_trades("FAKE", "2025-06-30")
    assert everything[-1].transaction_type == "G-Gift"


@pytest.mark.parametrize("text,code", [
    ("Sale at price 10 per share.", "S-Sale"),
    ("Purchase at price 10 per share.", "P-Purchase"),
    ("Stock Gift", "G-Gift"),
    ("Conversion of Exercise of derivative security", "M-Exercise"),
    ("Stock Award(Grant)", "A-Award"),
    ("Payment of exercise price or tax liability", "M-Exercise"),
    ("", "J-Other"),
])
def test_normalize_transaction(text, code):
    assert normalize_transaction(text) == code


def test_company_facts_and_market_cap(client):
    facts = client.get_company_facts("FAKE")
    assert facts.name == "Fake Corp" and facts.sector == "Tech" and facts.industry == "Widgets"
    assert client.get_market_cap("FAKE", "2026-01-01") == 500.0


def test_earnings_history_labels_surprises_and_skips_upcoming(client):
    records = client.get_earnings_history("FAKE", limit=8)
    assert [(r.report_period, r.filing_date, r.quarterly.eps_surprise) for r in records] == [
        ("2025-06-30", "2025-07-30", "MISS"),
        ("2025-03-31", "2025-04-30", "BEAT"),
    ]
    assert all(r.source_type == "8-K" for r in records)
    latest = client.get_earnings("FAKE")
    assert latest.report_period == "2025-06-30"


def test_news_feed_is_not_served(client):
    assert client.get_news("FAKE", "2026-01-01") == []
