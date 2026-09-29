"""Broker data models — positions, orders, fills.

Two order verbs, not four: positions are signed share counts, so "short" is
just selling past zero and "cover" is buying back toward it. Long/short
labeling is a display concern, not an execution one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Position(BaseModel):
    """Signed share count in one ticker. Negative = short."""

    ticker: str
    shares: int


class Order(BaseModel):
    """An instruction to trade. `price` is the reference price the caller
    computed (the as-of close): SimBroker fills there plus its modeled
    slippage, a live broker fills at its own quote — the Fill always carries
    the truth."""

    ticker: str
    side: Literal["buy", "sell"]
    quantity: int = Field(gt=0)
    price: float


class Fill(BaseModel):
    """An executed order at its actual fill price, plus what it cost to get
    there. `slippage` is the dollars lost to filling away from the order's
    reference price (quantity x |fill - reference|); `commission` is the
    broker's fee. Both are already reflected in the broker's cash. A real
    broker's fill can beat the reference, so there `slippage` may be
    negative."""

    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    price: float
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def cost(self) -> float:
        """Total friction on this fill, in dollars."""
        return self.commission + self.slippage
