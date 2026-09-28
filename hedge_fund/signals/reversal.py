"""Mean-reversion alpha model — short-term reversal.

Over horizons of a month, single stocks tend to reverse (Jegadeesh 1990;
Lehmann 1990): last month's sharp losers bounce, last month's sharp winners
give some back. This is the mirror image of MomentumModel's skipped month.

The view is contrarian on the trailing one-month return, measured in units
of the stock's OWN typical monthly move: the 1m return is z-scored against
the distribution of rolling 21-day returns over the prior year, so a -8%
month on a sleepy utility is a big signal and on a volatile small-cap is
noise. RSI(14) confirms — an overbought/oversold reading pushes the same
way. Pure math over daily bars; point-in-time by construction.
"""

from __future__ import annotations

import math
from datetime import date as _date
from datetime import timedelta

import numpy as np

from hedge_fund.data.protocol import DataClient
from hedge_fund.features.snapshot import InsufficientData
from hedge_fund.features.technicals import (
    DEFAULT_LOOKBACK_DAYS,
    MONTH,
    QUARTER,
    compute_price_snapshot,
)
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

# Below this many rolling monthly returns the z-score is not meaningful.
_MIN_MONTHLY_SAMPLES = QUARTER


class MeanReversionModel(QuantModel):
    """Fade the last month's move, sized by how unusual it was for the stock."""

    def __init__(
        self,
        *,
        z_scale: float = 2.0,
        rsi_weight: float = 0.3,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    ) -> None:
        # z_scale: z-score at which the reversal view saturates toward ±1.
        # rsi_weight: share of the view coming from RSI vs the z-score.
        self._z_scale = z_scale
        self._rsi_weight = rsi_weight
        self._lookback_days = lookback_days

    @property
    def name(self) -> str:
        return "mean-reversion"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        start = (_date.fromisoformat(date) - timedelta(days=self._lookback_days)).isoformat()
        bars = data_client.get_prices(ticker, start, date)
        try:
            snap = compute_price_snapshot(ticker, date, bars)
        except InsufficientData as exc:
            return self._abstain(ticker, date, str(exc))

        pit = sorted((b for b in bars if b.time[:10] <= date), key=lambda b: b.time)
        closes = np.array([b.close for b in pit], dtype=float)
        needed = MONTH + _MIN_MONTHLY_SAMPLES + 1
        if len(closes) < needed or snap.ret_1m is None:
            return self._abstain(
                ticker, date,
                f"need {needed} bars for a monthly-return distribution (have {len(closes)})",
            )

        monthly = closes[MONTH:] / closes[:-MONTH] - 1
        history = monthly[:-1]  # the current month is not its own baseline
        sigma = float(np.std(history, ddof=1))
        if sigma <= 0:
            return self._abstain(ticker, date, "flat price history — no reversal baseline")
        z = (snap.ret_1m - float(history.mean())) / sigma

        # Contrarian: a big up-month is a negative view, a big down-month positive.
        z_view = -math.tanh(z / self._z_scale)
        # RSI 70 -> -0.66, RSI 30 -> +0.66, RSI 50 -> 0.
        rsi_view = 0.0 if snap.rsi_14 is None else -math.tanh((snap.rsi_14 - 50.0) / 25.0)
        value = (1 - self._rsi_weight) * z_view + self._rsi_weight * rsi_view

        stance = "fading the rally" if value < -0.05 else "buying the dip" if value > 0.05 else "nothing to fade"
        rsi_text = f"; RSI(14) {snap.rsi_14:.0f}" if snap.rsi_14 is not None else ""
        components = {
            "ret_1m": snap.ret_1m,
            "z_score_1m": z,
            "monthly_sigma": sigma,
            "z_view": z_view,
            "rsi_view": rsi_view,
        }
        if snap.rsi_14 is not None:
            components["rsi_14"] = snap.rsi_14

        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=value,
            reasoning=(
                f"1m return {snap.ret_1m:+.1%} is {z:+.2f} sigma vs the stock's own "
                f"monthly moves (sigma {sigma:.1%}){rsi_text} — {stance}"
            ),
            components=components,
            metadata={"last_close": snap.last_close, "last_bar_date": snap.last_bar_date},
        )
