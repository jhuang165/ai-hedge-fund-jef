"""Insider-flow alpha model — follow the people who know the company best.

Corporate insiders' open-market purchases predict returns (Lakonishok & Lee,
2001; Cohen, Malloy & Pomorski, 2012). Sales are far noisier — insiders sell
to diversify, pay taxes, buy houses — so they are down-weighted, and only
open-market transactions count: grants, option exercises, tax withholding
and gifts say nothing about a view on the stock.

The signal is the net purchase ratio over a trailing window of Form 4
filings public by the as-of date:

    NPR = (buy dollars - sell_weight * sell dollars) / (buy dollars + sell_weight * sell dollars)

(counts are used when dollar values are missing), lightly damped when only
one or two transactions exist. No filings in the window is a real neutral
(0.0), not an abstention: silence from insiders is information too.
Point-in-time via the data client's filing-date filter.
"""

from __future__ import annotations

import math
from datetime import date as _date
from datetime import timedelta

from hedge_fund.data.models import InsiderTrade
from hedge_fund.data.protocol import DataClient
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

# Form 4 transaction codes. P = open-market purchase, S = open-market sale.
_BUY_CODES = {"P"}
_SELL_CODES = {"S"}


class InsiderFlowModel(QuantModel):
    """Bullish on net open-market insider buying, bearish on net selling."""

    def __init__(self, *, window_days: int = 90, sell_weight: float = 0.5) -> None:
        self._window_days = window_days
        self._sell_weight = sell_weight

    @property
    def name(self) -> str:
        return "insider-flow"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        start = (_date.fromisoformat(date) - timedelta(days=self._window_days)).isoformat()
        trades = data_client.get_insider_trades(ticker, end_date=date, start_date=start)
        # Belt and braces on point-in-time: the client filters on filing date,
        # but a provider that ignores the parameter must not leak the future.
        trades = [t for t in trades if t.filing_date[:10] <= date]

        buys = [t for t in trades if _classify(t) == "buy"]
        sells = [t for t in trades if _classify(t) == "sell"]
        if not buys and not sells:
            return Signal(
                model_name=self.name, ticker=ticker, date=date, value=0.0,
                reasoning=f"no open-market insider trades filed in the last {self._window_days} days",
                components={"n_buys": 0.0, "n_sells": 0.0},
            )

        buy_value = sum(_value(t) for t in buys)
        sell_value = sum(_value(t) for t in sells)
        if buy_value > 0 or sell_value > 0:
            basis = "dollars"
            pos, neg = buy_value, self._sell_weight * sell_value
        else:
            basis = "counts"
            pos, neg = float(len(buys)), self._sell_weight * len(sells)
        npr = (pos - neg) / (pos + neg)

        # Damp thin evidence: one lone trade is a hint, five is a pattern.
        n = len(buys) + len(sells)
        damp = 1.0 - math.exp(-n / 2.0)
        value = npr * damp

        insiders_buying = {t.name for t in buys}
        insiders_selling = {t.name for t in sells}
        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=value,
            reasoning=(
                f"{len(buys)} open-market buy{'s' if len(buys) != 1 else ''} "
                f"(${buy_value:,.0f}, {len(insiders_buying)} insider{'s' if len(insiders_buying) != 1 else ''}) vs "
                f"{len(sells)} sale{'s' if len(sells) != 1 else ''} "
                f"(${sell_value:,.0f}, {len(insiders_selling)}) in {self._window_days} days; "
                f"net purchase ratio {npr:+.2f} by {basis}, sales weighted {self._sell_weight:.1f}x"
            ),
            components={
                "n_buys": float(len(buys)),
                "n_sells": float(len(sells)),
                "buy_value": buy_value,
                "sell_value": sell_value,
                "net_purchase_ratio": npr,
                "damping": damp,
            },
            metadata={"window_days": self._window_days, "basis": basis},
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _classify(trade: InsiderTrade) -> str | None:
    """'buy' / 'sell' for open-market transactions, None for everything else.

    Providers label transactions differently ("P-Purchase", "S", "Sale", …),
    so the leading Form 4 code is read first; a missing or unknown code falls
    back to the sign of the share count, and a change in ownership that is
    consistent with a purchase or sale.
    """
    code = (trade.transaction_type or "").strip().upper()
    if code:
        letter = code[0]
        if letter in _BUY_CODES or code.startswith("PURCHASE") or code.startswith("BUY"):
            return "buy"
        if letter in _SELL_CODES or code.startswith("SALE") or code.startswith("SELL"):
            return "sell"
        return None  # award, exercise, tax withholding, gift, conversion, ...

    shares = trade.transaction_shares
    if shares is None or shares == 0:
        before, after = trade.shares_owned_before_transaction, trade.shares_owned_after_transaction
        if before is None or after is None or before == after:
            return None
        shares = after - before
    return "buy" if shares > 0 else "sell"


def _value(trade: InsiderTrade) -> float:
    if trade.transaction_value is not None:
        return abs(float(trade.transaction_value))
    if trade.transaction_shares is not None and trade.transaction_price_per_share is not None:
        return abs(float(trade.transaction_shares) * float(trade.transaction_price_per_share))
    return 0.0
