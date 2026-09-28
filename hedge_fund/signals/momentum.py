"""Momentum alpha model — 12-1 price momentum, risk-adjusted.

The oldest documented equity anomaly (Jegadeesh & Titman, 1993): stocks
that went up over the past year keep going up for a while. The most recent
month is skipped because it reverses (see MeanReversionModel — the two are
deliberately complementary).

Conviction is the momentum divided by realized volatility — a Sharpe-like
ratio — squashed through tanh, so a 30% run on a 30%-vol name is a strong
view (+0.76) while the same 30% on a 90%-vol name is a modest one (+0.32).
Pure math over daily bars; point-in-time by construction.
"""

from __future__ import annotations

import math

from hedge_fund.data.protocol import DataClient
from hedge_fund.features.snapshot import InsufficientData
from hedge_fund.features.technicals import DEFAULT_LOOKBACK_DAYS, build_price_snapshot
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

# Floor on volatility in the denominator: a near-zero vol must not turn a
# tiny return into a maximal view.
_MIN_VOL = 0.05


class MomentumModel(QuantModel):
    """Long the year's winners, short its losers, scaled by their volatility."""

    def __init__(self, *, scale: float = 1.0, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> None:
        # `scale` multiplies the risk-adjusted momentum before tanh: >1 makes
        # the model more decisive, <1 more measured.
        self._scale = scale
        self._lookback_days = lookback_days

    @property
    def name(self) -> str:
        return "momentum"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        try:
            snap = build_price_snapshot(ticker, date, data_client, self._lookback_days)
        except InsufficientData as exc:
            return self._abstain(ticker, date, str(exc))

        if snap.momentum_12_1 is None or snap.vol_63d_ann is None:
            return self._abstain(
                ticker, date, f"need a year of bars for 12-1 momentum (have {snap.n_bars})"
            )

        vol = max(snap.vol_63d_ann, _MIN_VOL)
        score = snap.momentum_12_1 / vol
        value = math.tanh(score * self._scale)
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=value,
            reasoning=(
                f"12-1 momentum {snap.momentum_12_1:+.1%} on {snap.vol_63d_ann:.0%} realized vol "
                f"(risk-adjusted {score:+.2f}); 12m return {_pct(snap.ret_12m)}, "
                f"{_pct(snap.pct_from_52w_high)} from 52w high"
            ),
            components={
                "momentum_12_1": snap.momentum_12_1,
                "vol_63d_ann": snap.vol_63d_ann,
                "risk_adjusted": score,
            },
            metadata={"last_close": snap.last_close, "last_bar_date": snap.last_bar_date},
        )


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:+.1%}"
