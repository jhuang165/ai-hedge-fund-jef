"""PriceSnapshot tests — synthetic bars, no network."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from hedge_fund.data.models import Price
from hedge_fund.features.snapshot import InsufficientData
from hedge_fund.features.technicals import (
    MIN_BARS,
    YEAR,
    PriceSnapshot,
    build_price_snapshot,
    compute_price_snapshot,
)


def bars(closes: list[float], end: str = "2025-06-30", volume: int = 1_000) -> list[Price]:
    """Daily bars ending on *end*, one per weekday, oldest first."""
    out = []
    d = date.fromisoformat(end)
    for close in reversed(closes):
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        out.append(Price(open=close, close=close, high=close, low=close,
                         volume=volume, time=f"{d.isoformat()}T00:00:00Z"))
        d -= timedelta(days=1)
    return list(reversed(out))


def linear(n: int, start: float = 100.0, step: float = 0.1) -> list[float]:
    return [start + i * step for i in range(n)]


def test_too_few_bars_raises():
    with pytest.raises(InsufficientData):
        compute_price_snapshot("T", "2025-06-30", bars(linear(MIN_BARS - 1)))


def test_bars_after_as_of_are_dropped():
    # 300 bars ending 2025-06-30, but as_of is a month earlier: the later
    # closes must not leak into the snapshot.
    series = bars(linear(300), end="2025-06-30")
    as_of = "2025-05-30"
    snap = compute_price_snapshot("T", as_of, series)
    assert snap.last_bar_date <= as_of
    assert snap.n_bars == sum(1 for b in series if b.time[:10] <= as_of)


def test_returns_and_momentum_on_a_rising_series():
    n = YEAR + 30
    snap = compute_price_snapshot("T", "2025-06-30", bars(linear(n)))
    assert snap.ret_1m is not None and snap.ret_1m > 0
    assert snap.ret_12m is not None and snap.ret_12m > 0
    assert snap.momentum_12_1 is not None
    # 12-1 skips the last month, so it is smaller than the full 12m return.
    assert 0 < snap.momentum_12_1 < snap.ret_12m
    assert snap.above_sma_200 is True
    assert snap.pct_from_52w_high == pytest.approx(0.0)
    assert snap.pct_from_52w_low is not None and snap.pct_from_52w_low > 0
    assert snap.rsi_14 == 100.0  # monotonically rising: no losses
    assert snap.max_drawdown_1y == 0.0


def test_windows_short_of_data_are_none_not_guessed():
    snap = compute_price_snapshot("T", "2025-06-30", bars(linear(MIN_BARS + 5)))
    assert snap.ret_1m is not None
    assert snap.ret_3m is not None
    assert snap.ret_6m is None
    assert snap.ret_12m is None
    assert snap.momentum_12_1 is None
    assert snap.sma_200 is None
    assert snap.above_sma_200 is None
    assert snap.max_drawdown_1y is None


def test_drawdown_and_distance_from_high():
    # Up to 200 then straight down to 100 over a year.
    closes = linear(YEAR // 2, 100, 200 / YEAR) + linear(YEAR - YEAR // 2, 200, -200 / YEAR)
    snap = compute_price_snapshot("T", "2025-06-30", bars(closes))
    assert snap.max_drawdown_1y == pytest.approx(0.5, abs=0.02)
    assert snap.pct_from_52w_high == pytest.approx(-0.5, abs=0.02)
    assert snap.above_sma_200 is False


def test_avg_dollar_volume():
    snap = compute_price_snapshot("T", "2025-06-30", bars([50.0] * 80, volume=2_000))
    assert snap.avg_dollar_volume_20d == pytest.approx(100_000.0)
    assert snap.vol_63d_ann == 0.0
    assert snap.rsi_14 == 50.0  # flat: no gains, no losses


def test_render_and_hash_are_stable():
    a = compute_price_snapshot("T", "2025-06-30", bars(linear(300)))
    b = compute_price_snapshot("T", "2025-06-30", bars(linear(300)))
    assert a.content_hash == b.content_hash
    text = a.render()
    assert "Price action for T" in text
    assert "12-1 momentum" in text
    assert "SMA200" in text


def test_build_uses_data_client_with_pit_window():
    class Client:
        def __init__(self):
            self.calls = []

        def get_prices(self, ticker, start_date, end_date, **kw):
            self.calls.append((ticker, start_date, end_date))
            return bars(linear(300), end=end_date)

    client = Client()
    snap = build_price_snapshot("T", "2025-06-30", client)
    assert isinstance(snap, PriceSnapshot)
    ticker, start, end = client.calls[0]
    assert (ticker, end) == ("T", "2025-06-30")
    assert start < "2025-06-30"
