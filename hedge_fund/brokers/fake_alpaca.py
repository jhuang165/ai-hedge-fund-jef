"""An in-memory stand-in for Alpaca's trading API, for tests.

Plugs into AlpacaBroker as its `session`. It keeps an account (cash and
signed positions), fills market orders at `prices` on the first status poll,
and behaves like the real venue where it matters to the broker: quantities
and prices are strings, a short's qty is negative with side "short", and an
order that would carry a position through zero is refused.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field


@dataclass
class _Response:
    status_code: int
    body: object = None

    @property
    def content(self) -> bytes:
        return b"" if self.body is None else b"x"

    @property
    def text(self) -> str:
        return str(self.body)

    def json(self):
        return self.body


@dataclass
class FakeAlpaca:
    cash: float = 100_000.0
    positions: dict[str, int] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)
    is_open: bool = True
    date: str = "2026-09-29"
    status: str = "ACTIVE"
    stuck: set[str] = field(default_factory=set)       # symbols whose orders never fill
    rejected: set[str] = field(default_factory=set)    # symbols the venue refuses
    headers: dict[str, str] = field(default_factory=dict)
    submitted: list[dict] = field(default_factory=list)
    canceled: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._orders: dict[str, dict] = {}
        self._ids = itertools.count(1)

    # requests.Session's one method the broker uses
    def request(self, method: str, url: str, json=None, timeout=None) -> _Response:
        path = "/" + url.split("/", 3)[3]
        if (method, path) == ("GET", "/v2/account"):
            return _Response(200, {"status": self.status, "cash": str(self.cash),
                                   "trading_blocked": False, "account_blocked": False})
        if (method, path) == ("GET", "/v2/clock"):
            return _Response(200, {"is_open": self.is_open,
                                   "timestamp": f"{self.date}T10:30:00.123456789-04:00",
                                   "next_open": f"{self.date}T09:30:00-04:00"})
        if (method, path) == ("GET", "/v2/positions"):
            return _Response(200, [self._row(t) for t in sorted(self.positions)])
        if method == "GET" and path.startswith("/v2/positions/"):
            ticker = path.rsplit("/", 1)[1]
            if ticker not in self.positions:
                return _Response(404, {"message": "position does not exist"})
            return _Response(200, self._row(ticker))
        if (method, path) == ("POST", "/v2/orders"):
            return self._submit(json)
        if method == "GET" and path.startswith("/v2/orders/"):
            return _Response(200, self._advance(path.rsplit("/", 1)[1]))
        if method == "DELETE" and path.startswith("/v2/orders/"):
            order_id = path.rsplit("/", 1)[1]
            self._orders[order_id]["status"] = "canceled"
            self.canceled.append(order_id)
            return _Response(204)
        return _Response(404, {"message": f"no route {method} {path}"})

    def _row(self, ticker: str) -> dict:
        shares = self.positions[ticker]
        return {"symbol": ticker, "qty": str(shares),
                "side": "short" if shares < 0 else "long"}

    def _submit(self, body: dict) -> _Response:
        self.submitted.append(body)
        ticker, qty = body["symbol"], int(body["qty"])
        if ticker in self.rejected:
            return _Response(422, {"message": f"asset {ticker} is not tradable"})
        current = self.positions.get(ticker, 0)
        after = current + (qty if body["side"] == "buy" else -qty)
        if current and after and (current > 0) != (after > 0):
            return _Response(403, {"message": "insufficient qty available for order"})
        order_id = f"o{next(self._ids)}"
        self._orders[order_id] = {"id": order_id, "status": "new", "filled_qty": "0",
                                  "filled_avg_price": None, **body}
        return _Response(200, dict(self._orders[order_id]))

    def _advance(self, order_id: str) -> dict:
        order = self._orders[order_id]
        if order["status"] == "new" and order["symbol"] not in self.stuck:
            qty, price = int(order["qty"]), self.prices[order["symbol"]]
            signed = qty if order["side"] == "buy" else -qty
            held = self.positions.get(order["symbol"], 0) + signed
            if held:
                self.positions[order["symbol"]] = held
            else:
                self.positions.pop(order["symbol"], None)
            self.cash -= signed * price
            order.update(status="filled", filled_qty=str(qty), filled_avg_price=str(price))
        return dict(order)
