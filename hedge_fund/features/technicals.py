"""Point-in-time price snapshot — what the tape says about a ticker.

A `PriceSnapshot` is the price-action counterpart to `FundamentalsSnapshot`:
everything an analyst may know about a stock's *trading* as of a date,
derived from daily bars on or before that date. Pure arithmetic over the
bars — trailing returns, momentum, realized volatility, drawdown, distance
from the 52-week range, RSI, moving averages, liquidity — so a model (quant
or LLM) reasons over facts rather than re-deriving them.

Point-in-time by construction: `compute_price_snapshot` drops every bar
after `as_of` before it computes anything, so the same function is honest
in a backtest and live.

Windows are in trading days (21 ≈ one month, 252 ≈ one year).
"""

from __future__ import annotations

import hashlib
import math
from datetime import date as _date
from datetime import timedelta

import numpy as np
from pydantic import BaseModel

from hedge_fund.data.models import Price
from hedge_fund.data.protocol import DataClient
from hedge_fund.features.errors import InsufficientData

# Trading-day windows.
MONTH = 21
QUARTER = 63
HALF_YEAR = 126
YEAR = 252

# Enough calendar days to cover a 252-bar lookback plus holidays and a
# little slack, without pulling years of history per call.
DEFAULT_LOOKBACK_DAYS = 400

# The minimum history to say anything at all (one quarter of bars).
MIN_BARS = QUARTER

# How far back to look for the most recent close: covers weekends, holiday
# clusters, and short trading halts without reaching into stale history.
MARK_LOOKBACK_DAYS = 7


class PriceSnapshot(BaseModel):
    """Price-action facts about *ticker* as of *as_of*. Every field derived
    from a window the bars did not fully cover is None, never a guess."""

    ticker: str
    as_of: str
    last_bar_date: str
    last_close: float
    n_bars: int

    # Trailing total returns (close-to-close, no dividends).
    ret_1m: float | None = None
    ret_3m: float | None = None
    ret_6m: float | None = None
    ret_12m: float | None = None

    # Classic 12-1 momentum: the year's return skipping the most recent month
    # (short-term reversal lives in that month, so it is excluded).
    momentum_12_1: float | None = None

    # Risk.
    vol_63d_ann: float | None = None       # annualized std of daily returns, last quarter
    max_drawdown_1y: float | None = None   # worst peak-to-trough over the last year, as a positive fraction

    # Range and trend.
    pct_from_52w_high: float | None = None  # <= 0: how far below the 52-week high
    pct_from_52w_low: float | None = None   # >= 0: how far above the 52-week low
    rsi_14: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None

    # Liquidity.
    avg_dollar_volume_20d: float | None = None

    @property
    def above_sma_200(self) -> bool | None:
        if self.sma_200 is None:
            return None
        return self.last_close > self.sma_200

    @property
    def content_hash(self) -> str:
        canonical = self.model_dump_json()
        return hashlib.sha256(canonical.encode()).hexdigest()[:24]

    def render(self) -> str:
        """Compact text block for an LLM prompt."""
        trend = "-"
        if self.sma_200 is not None:
            trend = "above" if self.above_sma_200 else "below"
        lines = [
            f"Price action for {self.ticker} (last close {self.last_close:.2f} on "
            f"{self.last_bar_date}; {self.n_bars} daily bars):",
            f"  Returns: 1m {_pct(self.ret_1m)}  |  3m {_pct(self.ret_3m)}  |  "
            f"6m {_pct(self.ret_6m)}  |  12m {_pct(self.ret_12m)}  |  "
            f"12-1 momentum {_pct(self.momentum_12_1)}",
            f"  Risk: realized vol (63d, ann.) {_upct(self.vol_63d_ann)}  |  "
            f"max drawdown (1y) {_upct(self.max_drawdown_1y)}",
            f"  Range: {_pct(self.pct_from_52w_high)} from 52w high  |  "
            f"{_pct(self.pct_from_52w_low)} above 52w low",
            f"  Trend: RSI(14) {_num(self.rsi_14)}  |  SMA50 {_num(self.sma_50)}  |  "
            f"SMA200 {_num(self.sma_200)} (price {trend} SMA200)",
            f"  Liquidity: avg dollar volume (20d) {_money(self.avg_dollar_volume_20d)}",
        ]
        return "\n".join(lines)


def last_close(
    ticker: str,
    as_of: str,
    data_client: DataClient,
) -> tuple[str, float] | None:
    """(bar date, close) of the last bar on or before *as_of* within the
    lookback, or None when nothing printed. The one mark every stage uses,
    so the pipeline, the personas, and the quant models never disagree on
    what a name was worth on a date."""
    start = (_date.fromisoformat(as_of) - timedelta(days=MARK_LOOKBACK_DAYS)).isoformat()
    bars = [p for p in data_client.get_prices(ticker, start, as_of) if p.time[:10] <= as_of]
    if not bars:
        return None
    last = max(bars, key=lambda p: p.time)
    return last.time[:10], last.close


def build_price_snapshot(
    ticker: str,
    as_of: str,
    data_client: DataClient,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> PriceSnapshot:
    """Fetch bars and compute the snapshot for (ticker, as_of).

    Raises InsufficientData with fewer than MIN_BARS bars on or before
    *as_of*. Data-layer failures propagate (fail loud).
    """
    start = (_date.fromisoformat(as_of) - timedelta(days=lookback_days)).isoformat()
    bars = data_client.get_prices(ticker, start, as_of)
    return compute_price_snapshot(ticker, as_of, bars)


def compute_price_snapshot(ticker: str, as_of: str, bars: list[Price]) -> PriceSnapshot:
    """Pure: the snapshot from a list of daily bars (any order, any dates).

    Bars after *as_of* are discarded first — no lookahead — then the rest
    are sorted by time. Raises InsufficientData below MIN_BARS.
    """
    pit = sorted((b for b in bars if b.time[:10] <= as_of), key=lambda b: b.time)
    if len(pit) < MIN_BARS:
        raise InsufficientData(
            f"{ticker} as of {as_of}: only {len(pit)} daily bars (need {MIN_BARS})"
        )

    closes = np.array([b.close for b in pit], dtype=float)
    volumes = np.array([b.volume for b in pit], dtype=float)
    n = len(closes)
    last = float(closes[-1])

    def ret(window: int) -> float | None:
        if n <= window or closes[-1 - window] <= 0:
            return None
        return float(closes[-1] / closes[-1 - window] - 1)

    momentum = None
    if n > YEAR and closes[-1 - YEAR] > 0:
        # Return from t-252 to t-21.
        momentum = float(closes[-1 - MONTH] / closes[-1 - YEAR] - 1)

    vol = None
    if n > QUARTER:
        daily = closes[-QUARTER - 1:][1:] / closes[-QUARTER - 1:][:-1] - 1
        vol = float(np.std(daily, ddof=1) * math.sqrt(YEAR))

    year_closes = closes[-YEAR:] if n >= YEAR else closes
    max_dd = _max_drawdown(year_closes) if n >= YEAR else None

    hi, lo = float(year_closes.max()), float(year_closes.min())
    pct_from_high = (last / hi - 1) if n >= YEAR and hi > 0 else None
    pct_from_low = (last / lo - 1) if n >= YEAR and lo > 0 else None

    return PriceSnapshot(
        ticker=ticker,
        as_of=as_of,
        last_bar_date=pit[-1].time[:10],
        last_close=last,
        n_bars=n,
        ret_1m=ret(MONTH),
        ret_3m=ret(QUARTER),
        ret_6m=ret(HALF_YEAR),
        ret_12m=ret(YEAR),
        momentum_12_1=momentum,
        vol_63d_ann=vol,
        max_drawdown_1y=max_dd,
        pct_from_52w_high=pct_from_high,
        pct_from_52w_low=pct_from_low,
        rsi_14=_rsi(closes, 14),
        sma_50=float(closes[-50:].mean()) if n >= 50 else None,
        sma_200=float(closes[-200:].mean()) if n >= 200 else None,
        avg_dollar_volume_20d=(
            float((closes[-20:] * volumes[-20:]).mean()) if n >= 20 else None
        ),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _max_drawdown(closes: np.ndarray) -> float:
    peak = closes[0]
    worst = 0.0
    for c in closes:
        if c > peak:
            peak = c
        dd = (peak - c) / peak if peak > 0 else 0.0
        if dd > worst:
            worst = dd
    return float(worst)


def _rsi(closes: np.ndarray, period: int) -> float | None:
    """Wilder's RSI on the last *period* changes. None with too few bars."""
    if len(closes) <= period:
        return None
    deltas = np.diff(closes[-period - 1:])
    gains = deltas[deltas > 0].sum()
    losses = -deltas[deltas < 0].sum()
    if losses == 0:
        return 100.0 if gains > 0 else 50.0
    rs = (gains / period) / (losses / period)
    return float(100.0 - 100.0 / (1.0 + rs))


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:+.1%}"


def _upct(v: float | None) -> str:
    """Unsigned: for quantities that are magnitudes (vol, drawdown)."""
    return "-" if v is None else f"{v:.1%}"


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def _money(v: float | None) -> str:
    if v is None:
        return "-"
    if abs(v) >= 1e9:
        return f"${v / 1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    return f"${v:,.0f}"
