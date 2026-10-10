"""AlpacaBroker — a paper-trading account at Alpaca, behind the Broker protocol.

Paper mode is the live clock plus a paper broker: the same run_cycle, but the
orders go to a real broker's paper venue and fill at real quotes. The account
is the source of truth for the book — positions and cash are read from it on
every call, never tracked locally — so a manual trade in the dashboard or a
run that died halfway still leaves the fund reading what it actually holds.

Only the paper endpoint is wired. The live endpoint takes the same requests;
trading real money stays an explicit, separate opt-in (see VISION.md).

Execution: every order is a market order, good for the day, submitted and
then polled until it reaches a terminal state. The Broker contract is
all-or-raise, so anything but a complete fill — rejected, canceled, expired,
or still working when the timeout lapses (the remainder is then canceled) —
raises AlpacaError with what did fill. Alpaca will not take one order that
flips a position through zero, so a flip goes out as two: close, then open.

Keys come from the environment, under the names Alpaca's own tools use:
APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys, from the paper
dashboard).
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Callable

import requests

from hedge_fund.brokers.models import Fill, Order, Position

PAPER_URL = "https://paper-api.alpaca.markets"
KEY_ID_VAR = "APCA_API_KEY_ID"
SECRET_VAR = "APCA_API_SECRET_KEY"

_TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced", "done_for_day"}


class AlpacaError(RuntimeError):
    """The broker refused, failed, or did not finish what was asked."""


class AlpacaBroker:
    """Alpaca paper account: signed positions and cash, read live."""

    # The account's cash already moves when a real dividend lands, so the
    # pipeline must not accrue an estimate on top of it.
    books_dividends = True

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        *,
        base_url: str = PAPER_URL,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        fill_timeout: float = 60.0,
        poll_interval: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not key_id or not secret_key:
            raise AlpacaError(
                f"Alpaca paper keys are not set — export {KEY_ID_VAR} and "
                f"{SECRET_VAR} (or save them in ~/.hedge-fund/.env)"
            )
        self._base = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
        })
        self._timeout = timeout
        self._fill_timeout = fill_timeout
        self._poll = poll_interval
        self._sleep = sleep

    @classmethod
    def from_env(cls, **kwargs) -> AlpacaBroker:
        return cls(os.environ.get(KEY_ID_VAR, ""), os.environ.get(SECRET_VAR, ""),
                   **kwargs)

    # ------------------------------------------------------------------
    # Readiness — checked before a cycle spends anything on analysts
    # ------------------------------------------------------------------

    def status(self) -> dict:
        """A read-only look at the account — safe any time, market open or
        not: whether it can trade, its cash, its book, and the clock."""
        account = self._request("GET", "/v2/account")
        clock = self._request("GET", "/v2/clock")
        return {
            "status": account.get("status"),
            "trading_blocked": bool(account.get("trading_blocked")),
            "cash": float(account["cash"]),
            "positions": {t: p.shares for t, p in self.positions().items()},
            "market_open": bool(clock.get("is_open")),
            "next_open": clock.get("next_open"),
        }

    def market_date(self) -> str:
        """Today's date on the exchange's clock (YYYY-MM-DD). Raises unless
        the account can trade and the market is open right now: a market
        order placed while closed would sit until the next open, and the
        cycle that sized it would record fills that have not happened."""
        account = self._request("GET", "/v2/account")
        if account.get("status") != "ACTIVE" or account.get("trading_blocked") \
                or account.get("account_blocked"):
            raise AlpacaError(
                f"the Alpaca account cannot trade (status {account.get('status')}, "
                f"trading_blocked={account.get('trading_blocked')})"
            )
        clock = self._request("GET", "/v2/clock")
        if not clock.get("is_open"):
            raise AlpacaError(
                f"the market is closed — next open {clock.get('next_open')}. "
                "Paper runs trade during market hours only"
            )
        # The exchange-local timestamp (e.g. 2026-09-29T10:30:00.123456789-04:00)
        # leads with the market's date; nanoseconds rule out fromisoformat.
        return clock["timestamp"][:10]

    # ------------------------------------------------------------------
    # Broker protocol
    # ------------------------------------------------------------------

    def positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for row in self._request("GET", "/v2/positions"):
            shares = _signed_shares(row)
            if shares:
                out[row["symbol"]] = Position(ticker=row["symbol"], shares=shares)
        return out

    def cash(self) -> float:
        return float(self._request("GET", "/v2/account")["cash"])

    def credit(self, amount: float, memo: str) -> None:
        """Real dividends land in the account on their own; nothing to book."""

    def place_order(self, order: Order) -> Fill:
        if order.price <= 0:
            raise ValueError(
                f"cannot size {order.ticker} at price {order.price} — "
                "the caller must price every order"
            )
        current = self._held(order.ticker)
        delta = order.quantity if order.side == "buy" else -order.quantity
        after = current + delta
        if current and after and (current > 0) != (after > 0):
            legs = [abs(current), abs(after)]  # through zero: close, then open
        else:
            legs = [order.quantity]

        filled: list[tuple[int, float]] = []
        for qty in legs:
            filled.append(self._execute(order.ticker, order.side, qty, done=filled))

        quantity = sum(q for q, _ in filled)
        price = sum(q * p for q, p in filled) / quantity
        # Positive: the fill was worse than the close the order was sized on.
        # A real fill can beat it, so unlike SimBroker this can go negative.
        sign = 1 if order.side == "buy" else -1
        return Fill(
            ticker=order.ticker,
            side=order.side,
            quantity=quantity,
            price=price,
            commission=0.0,
            slippage=sign * (price - order.price) * quantity,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _held(self, ticker: str) -> int:
        try:
            row = self._request("GET", f"/v2/positions/{ticker}")
        except _NotFound:
            return 0
        return _signed_shares(row)

    def _execute(self, ticker: str, side: str, qty: int,
                 done: list[tuple[int, float]]) -> tuple[int, float]:
        """Submit one market order and wait for it to fill completely."""
        placed = self._request("POST", "/v2/orders", json={
            "symbol": ticker,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "client_order_id": f"aihf-{uuid.uuid4().hex}",
        })
        order_id = placed["id"]
        state = placed
        waited = 0.0
        while state.get("status") not in _TERMINAL and waited < self._fill_timeout:
            self._sleep(self._poll)
            waited += self._poll
            state = self._request("GET", f"/v2/orders/{order_id}")

        status = state.get("status")
        filled_qty = int(float(state.get("filled_qty") or 0))
        if status == "filled" and filled_qty == qty:
            return qty, float(state["filled_avg_price"])

        if status not in _TERMINAL:
            try:
                self._request("DELETE", f"/v2/orders/{order_id}")
            except AlpacaError:
                pass  # it may have finished in the meantime; the error below stands
            status = f"{status}, canceled after {self._fill_timeout:.0f}s"
        earlier = sum(q for q, _ in done)
        raise AlpacaError(
            f"{side} {qty} {ticker} did not fill ({status}): {filled_qty} of "
            f"{qty} shares filled"
            + (f", after {earlier} shares of the same order already had" if earlier else "")
            + ". The account holds whatever did fill; the next run reads it from there"
        )

    def _request(self, method: str, path: str, *, json: dict | None = None):
        try:
            resp = self._session.request(method, self._base + path, json=json,
                                         timeout=self._timeout)
        except requests.RequestException as exc:
            raise AlpacaError(f"{method} {path}: {exc}") from exc
        if resp.status_code == 404:
            raise _NotFound(path)
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("message", resp.text)
            except ValueError:
                detail = resp.text
            raise AlpacaError(f"{method} {path} → {resp.status_code}: {detail}")
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()


class _NotFound(AlpacaError):
    """404 — for a position, it means flat."""


def _signed_shares(row: dict) -> int:
    """Alpaca reports qty as a string and marks shorts with side="short";
    accept either sign on qty so the book is right whichever it sends."""
    qty = abs(int(float(row["qty"])))
    return -qty if row.get("side") == "short" else qty
