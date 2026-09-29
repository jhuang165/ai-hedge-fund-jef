"""SimBroker — deterministic simulated broker for backtests.

Fills every order completely, at the order's reference price moved against
the fund by a modeled slippage, and charges a commission on the notional.
That determinism is the point: given the same orders and the same cost
model, a backtest replays to the same book.

The cost model is the honest floor of a backtest. The reference price is
the close the signal was formed on; a real fund cannot trade the closing
print it just observed, and `slippage_bps` stands in for the gap between
that decision price and the fill. `commission_bps` is the broker's fee on
the traded notional. Both default to zero here so the broker's arithmetic
is exact when tested alone; a mandate's ExecutionPolicy sets the realistic
values and every backtest and cycle runner passes them in.

Margin is not modeled: cash may go negative and stays visible. With an
unlevered mandate (gross_target <= 1), sells-before-buys ordering, and
floor-toward-zero sizing, a long book won't get there by more than the
costs — but nothing here pretends to enforce it.
"""

from __future__ import annotations

from hedge_fund.brokers.models import Fill, Order, Position

_BPS = 1e-4


class SimBroker:
    """In-memory broker: signed positions plus a cash balance."""

    def __init__(
        self,
        cash: float,
        *,
        commission_bps: float = 0.0,
        slippage_bps: float = 0.0,
        positions: dict[str, int] | None = None,
    ) -> None:
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("commission_bps and slippage_bps must be >= 0")
        self._cash = cash
        # A broker can open on a carried book — the ledger's last positions —
        # not only on cash.
        self._shares: dict[str, int] = {
            t: s for t, s in (positions or {}).items() if s != 0
        }
        self._commission = commission_bps * _BPS
        self._slippage = slippage_bps * _BPS

    def positions(self) -> dict[str, Position]:
        return {
            t: Position(ticker=t, shares=s)
            for t, s in self._shares.items()
            if s != 0
        }

    def cash(self) -> float:
        return self._cash

    def credit(self, amount: float, memo: str) -> None:
        self._cash += amount

    def place_order(self, order: Order) -> Fill:
        if order.price <= 0:
            raise ValueError(
                f"cannot fill {order.ticker} at price {order.price} — "
                "the caller must price every order"
            )

        # Slippage always moves the fill against the fund: buys pay up,
        # sells receive less.
        if order.side == "buy":
            fill_price = order.price * (1 + self._slippage)
        else:
            fill_price = order.price * (1 - self._slippage)
        notional = order.quantity * fill_price
        commission = notional * self._commission
        slippage = order.quantity * abs(fill_price - order.price)

        if order.side == "buy":
            self._shares[order.ticker] = self._shares.get(order.ticker, 0) + order.quantity
            self._cash -= notional + commission
        else:
            self._shares[order.ticker] = self._shares.get(order.ticker, 0) - order.quantity
            self._cash += notional - commission

        if self._shares[order.ticker] == 0:
            del self._shares[order.ticker]

        return Fill(
            ticker=order.ticker,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage,
        )
