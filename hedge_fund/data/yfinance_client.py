"""YFinanceClient — a DataClient that needs no API key.

Backed by Yahoo Finance through the `yfinance` package: daily bars, the
quarterly and annual statements, Form 4 insider transactions, and the
earnings calendar with EPS surprises. It is the data source the fund falls
back to when FINANCIAL_DATASETS_API_KEY is not set (see
hedge_fund/data/source.py), and it exists so `aihf research` can run from a
bare checkout.

What it is good for: live research as of today, where "what was filed by
*end_date*" and "what is filed now" are the same thing.

What it is not: a point-in-time feed. Yahoo gives fiscal period ends, not
SEC filing dates, so every metrics row's filing_date is the period end plus
FILING_LAG_DAYS — a conservative stand-in for when a 10-Q would have become
public. A backtest over this client leaks less than one that ignores
filing dates entirely, but it is still an approximation; use the
Financial Datasets client for backtests you intend to believe.

Other deliberate differences from FDClient:

- get_financial_metrics returns trailing-twelve-month rows at the last few
  quarter ends (Yahoo keeps about five quarters) and, before those, at
  fiscal year ends from the annual statements. Every row is a genuine TTM
  figure as of its date, but the spacing is not uniform.
- Insider trades carry the trade date as filing_date (Form 4s are due within
  two business days) and a normalized transaction code so the insider-flow
  model can classify them.
- Earnings records are built from the earnings calendar: one 8-K-style
  record per reported quarter, with BEAT/MISS/MEET from the EPS surprise.
- get_news returns nothing: Yahoo's news feed is not a stable API. Search
  is the research desk's news source anyway.
"""

from __future__ import annotations

import math
from datetime import date as _date, timedelta
from typing import Any, Callable

from hedge_fund.data.models import (
    CompanyFacts,
    CompanyNews,
    Earnings,
    EarningsData,
    EarningsRecord,
    FinancialMetrics,
    InsiderTrade,
    Price,
)

FILING_LAG_DAYS = 40      # period end -> assumed public; 10-Qs are due in 40-45 days
_TAX_RATE = 0.21          # for ROIC when the statements carry no tax rate
_ANNUAL_LIMIT = 4         # fiscal years Yahoo serves for free


class YFinanceClientError(Exception):
    """An infrastructure failure talking to Yahoo (fail loud, never empty)."""


class YFinanceClient:
    """DataClient over yfinance. See the module docstring for the contract."""

    def __init__(
        self,
        *,
        filing_lag_days: int = FILING_LAG_DAYS,
        ticker_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._lag = timedelta(days=filing_lag_days)
        self._factory = ticker_factory or _default_factory
        self._tickers: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Context manager (nothing to close; mirrors FDClient for `with`)
    # ------------------------------------------------------------------

    def __enter__(self) -> YFinanceClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        self._tickers.clear()

    def _t(self, ticker: str) -> Any:
        ticker = ticker.upper()
        if ticker not in self._tickers:
            self._tickers[ticker] = self._factory(ticker)
        return self._tickers[ticker]

    # ------------------------------------------------------------------
    # Prices
    # ------------------------------------------------------------------

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        interval: str = "day",
        interval_multiplier: int = 1,
    ) -> list[Price]:
        """Unadjusted daily OHLCV bars in [start_date, end_date]."""
        if interval != "day" or interval_multiplier != 1:
            raise YFinanceClientError("YFinanceClient serves daily bars only")
        # yfinance treats `end` as exclusive.
        end_excl = (_date.fromisoformat(end_date) + timedelta(days=1)).isoformat()
        try:
            hist = self._t(ticker).history(
                start=start_date, end=end_excl, interval="1d",
                auto_adjust=False, actions=False, raise_errors=True,
            )
        except Exception as exc:  # yfinance raises a plain Exception
            if _is_unknown_ticker(exc):
                return []
            raise YFinanceClientError(f"{ticker}: price fetch failed: {exc}") from exc
        bars: list[Price] = []
        for ts, row in hist.iterrows():
            close = _f(row.get("Close"))
            if close is None:
                continue
            bars.append(Price(
                open=_f(row.get("Open")) or close,
                close=close,
                high=_f(row.get("High")) or close,
                low=_f(row.get("Low")) or close,
                volume=int(_f(row.get("Volume")) or 0),
                time=ts.date().isoformat(),
            ))
        return bars

    # ------------------------------------------------------------------
    # Financial metrics
    # ------------------------------------------------------------------

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        """TTM metrics rows assumed public by *end_date*, newest first."""
        if period != "ttm":
            raise YFinanceClientError("YFinanceClient serves period='ttm' only")
        t = self._t(ticker)
        try:
            q_inc, q_bal, q_cf = t.quarterly_income_stmt, t.quarterly_balance_sheet, t.quarterly_cashflow
            a_inc, a_bal, a_cf = t.income_stmt, t.balance_sheet, t.cashflow
        except Exception as exc:
            raise YFinanceClientError(f"{ticker}: statements fetch failed: {exc}") from exc

        rows: list[FinancialMetrics] = []
        quarterly = _periods(q_inc)
        for i, p in enumerate(quarterly):
            window = quarterly[i:i + 4]
            if len(window) < 4:
                break
            prior = quarterly[i + 4:i + 8]
            rows.append(self._ttm_row(
                ticker, p, inc=q_inc, bal=q_bal, cf=q_cf, window=window,
                prior=prior if len(prior) == 4 else None,
            ))
        # Annual rows only extend history backwards: a fiscal year end on or
        # after the oldest quarterly TTM row is already covered.
        oldest_ttm = rows[-1].report_period if rows else None
        annual = _periods(a_inc)[:_ANNUAL_LIMIT]
        for i, p in enumerate(annual):
            if oldest_ttm is not None and p >= oldest_ttm:
                continue
            prior = annual[i + 1:i + 2]
            rows.append(self._ttm_row(
                ticker, p, inc=a_inc, bal=a_bal, cf=a_cf, window=[p],
                prior=prior or None,
            ))

        rows.sort(key=lambda m: m.report_period, reverse=True)
        public = [m for m in rows if m.filing_date is not None and m.filing_date <= end_date]
        return public[:limit]

    def _ttm_row(
        self, ticker: str, p: str, *, inc, bal, cf, window: list[str], prior: list[str] | None,
    ) -> FinancialMetrics:
        """One TTM row at period end *p*: flows summed over *window*, stocks
        taken at *p*, growth against the same window one year earlier."""
        rev = _sum(inc, "Total Revenue", window)
        gross = _sum(inc, "Gross Profit", window)
        op = _sum(inc, "Operating Income", window)
        ni = _sum(inc, "Net Income", window)
        ebitda = _sum(inc, "EBITDA", window)
        ebit = _sum(inc, "EBIT", window) or op
        eps = _sum(inc, "Diluted EPS", window) or _sum(inc, "Basic EPS", window)
        fcf = _sum(cf, "Free Cash Flow", window)
        ocf = _sum(cf, "Operating Cash Flow", window)
        dividends = _sum(cf, "Cash Dividends Paid", window) or _sum(cf, "Common Stock Dividend Paid", window)

        equity = _at(bal, "Stockholders Equity", p) or _at(bal, "Common Stock Equity", p)
        assets = _at(bal, "Total Assets", p)
        debt = _at(bal, "Total Debt", p)
        cash = _at(bal, "Cash And Cash Equivalents", p)
        cur_assets = _at(bal, "Current Assets", p)
        cur_liab = _at(bal, "Current Liabilities", p)
        invested = _at(bal, "Invested Capital", p)
        shares = (_at(bal, "Ordinary Shares Number", p) or _at(bal, "Share Issued", p)
                  or _sum(inc, "Diluted Average Shares", window[:1]))

        close = self._close_near(ticker, p)
        market_cap = close * shares if close is not None and shares else None
        bvps = equity / shares if equity is not None and shares else None
        fcfps = fcf / shares if fcf is not None and shares else None
        ev = (market_cap + (debt or 0.0) - (cash or 0.0)) if market_cap is not None else None

        prior_rev = _sum(inc, "Total Revenue", prior) if prior else None
        prior_ni = _sum(inc, "Net Income", prior) if prior else None
        prior_eps = (_sum(inc, "Diluted EPS", prior) or _sum(inc, "Basic EPS", prior)) if prior else None
        prior_op = _sum(inc, "Operating Income", prior) if prior else None
        prior_fcf = _sum(cf, "Free Cash Flow", prior) if prior else None
        prior_ebitda = _sum(inc, "EBITDA", prior) if prior else None
        prior_equity = _at(bal, "Stockholders Equity", prior[0]) if prior else None
        prior_shares = _at(bal, "Ordinary Shares Number", prior[0]) if prior else None
        prior_bvps = prior_equity / prior_shares if prior_equity is not None and prior_shares else None

        filing = (_date.fromisoformat(p) + self._lag).isoformat()
        return FinancialMetrics(
            ticker=ticker,
            report_period=p,
            period="ttm",
            filing_date=filing,
            market_cap=market_cap,
            enterprise_value=ev,
            price_to_earnings_ratio=_div(close, eps) if eps and eps > 0 else None,
            price_to_book_ratio=_div(close, bvps) if bvps and bvps > 0 else None,
            price_to_sales_ratio=_div(market_cap, rev),
            enterprise_value_to_ebitda_ratio=_div(ev, ebitda) if ebitda and ebitda > 0 else None,
            enterprise_value_to_revenue_ratio=_div(ev, rev),
            free_cash_flow_yield=_div(fcfps, close),
            gross_margin=_div(gross, rev),
            operating_margin=_div(op, rev),
            net_margin=_div(ni, rev),
            return_on_equity=_div(ni, equity),
            return_on_assets=_div(ni, assets),
            return_on_invested_capital=(
                _div(ebit * (1 - _TAX_RATE), invested) if ebit is not None else None
            ),
            current_ratio=_div(cur_assets, cur_liab),
            operating_cash_flow_ratio=_div(ocf, cur_liab),
            debt_to_equity=_div(debt, equity),
            debt_to_assets=_div(debt, assets),
            revenue_growth=_growth(rev, prior_rev),
            earnings_growth=_growth(ni, prior_ni),
            earnings_per_share_growth=_growth(eps, prior_eps),
            operating_income_growth=_growth(op, prior_op),
            free_cash_flow_growth=_growth(fcf, prior_fcf),
            ebitda_growth=_growth(ebitda, prior_ebitda),
            book_value_growth=_growth(bvps, prior_bvps),
            payout_ratio=(
                _div(abs(dividends), ni) if dividends is not None and ni and ni > 0 else None
            ),
            earnings_per_share=eps,
            book_value_per_share=bvps,
            free_cash_flow_per_share=fcfps,
        )

    def _close_near(self, ticker: str, period_end: str) -> float | None:
        """The last close on or before the period end (what the filed
        valuation would have been struck at)."""
        start = (_date.fromisoformat(period_end) - timedelta(days=10)).isoformat()
        bars = self.get_prices(ticker, start, period_end)
        if not bars:
            return None
        return max(bars, key=lambda b: b.time).close

    # ------------------------------------------------------------------
    # News (none), insider trades, company facts, earnings
    # ------------------------------------------------------------------

    def get_news(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        """Yahoo's news feed is not served; the research desk searches the
        web instead. Empty here means "no feed", not "no news"."""
        return []

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        try:
            df = self._t(ticker).insider_transactions
        except Exception as exc:
            raise YFinanceClientError(f"{ticker}: insider fetch failed: {exc}") from exc
        if df is None or len(df) == 0:
            return []
        trades: list[InsiderTrade] = []
        for _, row in df.iterrows():
            when = row.get("Start Date")
            if when is None or (isinstance(when, float) and math.isnan(when)):
                continue
            day = _day(when)
            if day > end_date or (start_date is not None and day < start_date):
                continue
            text = str(row.get("Text") or "")
            trades.append(InsiderTrade(
                ticker=ticker.upper(),
                name=str(row.get("Insider") or "unknown"),
                filing_date=day,
                transaction_date=day,
                title=_none_if_nan(row.get("Position")),
                transaction_type=normalize_transaction(text),
                transaction_shares=_f(row.get("Shares")),
                transaction_value=_f(row.get("Value")),
                transaction_price_per_share=_price_in(text),
            ))
            if len(trades) >= limit:
                break
        return trades

    def get_company_facts(self, ticker: str) -> CompanyFacts | None:
        info = self._info(ticker)
        if not info:
            return None
        return CompanyFacts(
            ticker=ticker.upper(),
            name=info.get("longName") or info.get("shortName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            exchange=info.get("exchange"),
            location=", ".join(x for x in (info.get("city"), info.get("country")) if x) or None,
        )

    def get_earnings(self, ticker: str) -> Earnings | None:
        history = self.get_earnings_history(ticker, limit=1)
        if not history:
            return None
        r = history[0]
        return Earnings(ticker=r.ticker, report_period=r.report_period,
                        fiscal_period=r.fiscal_period, quarterly=r.quarterly)

    def get_earnings_history(
        self,
        ticker: str,
        limit: int = 12,
    ) -> list[EarningsRecord]:
        """One record per reported quarter from the earnings calendar."""
        t = self._t(ticker)
        try:
            dates = t.get_earnings_dates(limit=limit + 4)
            quarters = t.earnings_history
        except Exception as exc:
            raise YFinanceClientError(f"{ticker}: earnings fetch failed: {exc}") from exc
        if dates is None or len(dates) == 0:
            return []
        fiscal_ends = sorted(_day(q) for q in quarters.index) if quarters is not None and len(quarters) else []

        records: list[EarningsRecord] = []
        for when, row in dates.iterrows():
            reported = _f(row.get("Reported EPS"))
            if reported is None:
                continue  # a future date, or a quarter Yahoo has no actual for
            day = _day(when)
            estimate = _f(row.get("EPS Estimate"))
            surprise_pct = _f(row.get("Surprise(%)"))
            if surprise_pct is None and estimate is not None:
                surprise_pct = (reported - estimate) / abs(estimate) * 100 if estimate else None
            surprise = (
                None if surprise_pct is None
                else "BEAT" if surprise_pct > 0 else "MISS" if surprise_pct < 0 else "MEET"
            )
            records.append(EarningsRecord(
                ticker=ticker.upper(),
                report_period=_quarter_before(day, fiscal_ends),
                source_type="8-K",
                filing_date=day,
                quarterly=EarningsData(
                    earnings_per_share=reported,
                    estimated_earnings_per_share=estimate,
                    eps_surprise=surprise,
                ),
            ))
            if len(records) >= limit:
                break
        return records

    def get_market_cap(self, ticker: str, end_date: str) -> float | None:
        info = self._info(ticker)
        cap = _f(info.get("marketCap")) if info else None
        if cap is not None:
            return cap
        metrics = self.get_financial_metrics(ticker, end_date, limit=1)
        return metrics[0].market_cap if metrics else None

    def _info(self, ticker: str) -> dict:
        try:
            info = self._t(ticker).info
        except Exception as exc:
            raise YFinanceClientError(f"{ticker}: info fetch failed: {exc}") from exc
        return info or {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_transaction(text: str) -> str:
    """Yahoo's free-text transaction into a Form 4-style code the
    insider-flow model classifies: only open-market purchases (P) and sales
    (S) count; everything else is a non-market event."""
    low = text.strip().lower()
    if low.startswith("purchase") or low.startswith("buy"):
        return "P-Purchase"
    if low.startswith("sale") or low.startswith("sell"):
        return "S-Sale"
    if "gift" in low:
        return "G-Gift"
    if "conversion" in low or "exercise" in low:
        return "M-Exercise"
    if "award" in low or "grant" in low:
        return "A-Award"
    if "tax" in low:
        return "F-Tax"
    return "J-Other" if low else "J-Other"


def _default_factory(ticker: str) -> Any:
    import yfinance as yf  # imported lazily: the FD path never pays for it

    return yf.Ticker(ticker)


def _is_unknown_ticker(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "delisted" in msg or "no price data" in msg or "not found" in msg


def _periods(frame) -> list[str]:
    """Statement column period ends as ISO dates, newest first."""
    if frame is None or len(frame.columns) == 0:
        return []
    return sorted((_day(c) for c in frame.columns), reverse=True)


def _day(ts) -> str:
    if hasattr(ts, "date"):
        return ts.date().isoformat() if callable(ts.date) else str(ts)[:10]
    return str(ts)[:10]


def _cell(frame, row: str, period: str) -> float | None:
    if frame is None or row not in frame.index:
        return None
    for col in frame.columns:
        if _day(col) == period:
            return _f(frame.at[row, col])
    return None


def _at(frame, row: str, period: str) -> float | None:
    return _cell(frame, row, period)


def _sum(frame, row: str, periods: list[str] | None) -> float | None:
    """Sum of *row* over *periods*; None if any period is missing, so a
    three-quarter "TTM" never masquerades as four."""
    if not periods:
        return None
    total = 0.0
    for p in periods:
        v = _cell(frame, row, p)
        if v is None:
            return None
        total += v
    return total


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) or math.isinf(x) else x


def _none_if_nan(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return str(v)


def _div(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _growth(now: float | None, before: float | None) -> float | None:
    if now is None or before is None or before == 0:
        return None
    return (now - before) / abs(before)


def _price_in(text: str) -> float | None:
    """The per-share price Yahoo embeds in 'Sale at price 123.45 per share.'"""
    marker = "at price "
    if marker not in text:
        return None
    rest = text.split(marker, 1)[1].split(" ")[0].replace(",", "")
    try:
        return float(rest)
    except ValueError:
        return None


def _quarter_before(day: str, fiscal_ends: list[str]) -> str:
    """The fiscal quarter the earnings date on *day* reported: the latest
    known fiscal quarter end before it, else the calendar quarter end."""
    before = [q for q in fiscal_ends if q < day]
    if before:
        return before[-1]
    d = _date.fromisoformat(day)
    q_end_month = ((d.month - 1) // 3) * 3  # 0, 3, 6, 9 -> previous quarter end month
    if q_end_month == 0:
        return _date(d.year - 1, 12, 31).isoformat()
    last_day = {3: 31, 6: 30, 9: 30}[q_end_month]
    return _date(d.year, q_end_month, last_day).isoformat()
