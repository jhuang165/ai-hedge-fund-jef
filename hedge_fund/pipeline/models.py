"""Pipeline records — the serialized truth of every cycle.

A CycleRecord captures one tick of the fund end to end: what the analysts
saw, what they said, how views became weights, what risk clamped, what was
ordered and filled, and what the book looks like after. The ledger persists
these; `fund why AAPL` will answer from them alone.
"""

from __future__ import annotations

from pydantic import BaseModel

from hedge_fund.brokers.models import Fill, Order
from hedge_fund.fund.spec import FundSpec
from hedge_fund.models import Signal
from hedge_fund.risk.limits import ClampEvent


class TickerSkip(BaseModel):
    """A requested name that could not be traded this cycle, and why."""

    ticker: str
    reason: str


class StrategyRecord(BaseModel):
    """One strategy's slice of a cycle: its analysts' views and its sleeve."""

    name: str
    slice: float                        # normalized capital slice of the fund
    signals: list[Signal]               # this strategy's analysts x tradeable tickers
    convictions: dict[str, float]       # blended views, pre-scaling
    weights: dict[str, float]           # the sleeve, before netting across strategies


class CycleRecord(BaseModel):
    """One tick of the fund, fully serialized — every stage's inputs and
    outputs. `model_dump_json()` round-trips; nothing about a decision
    lives anywhere else."""

    fund: str
    as_of: str
    spec: FundSpec                      # self-contained audit copy
    universe: list[str]                 # the tickers this cycle was asked to trade
    marks: dict[str, float]             # ticker -> close used for sizing and NAV
    vols: dict[str, float] = {}         # ticker -> realized vol used for risk-scaled sizing
    skipped: list[TickerSkip]
    strategies: list[StrategyRecord]    # every sleeve, incl. each thesis
    target_weights: dict[str, float]    # the NETTED book, pre-risk
    clamps: list[ClampEvent]
    final_weights: dict[str, float]     # post-risk
    equity_before: float
    cash_before: float
    dividends: dict[str, float] = {}    # ticker -> dividend cash accrued since the prior cycle (negative: owed on a short)
    prev_as_of: str | None = None       # the prior cycle this one carried its book from (None: opened on cash)
    peak_nav: float | None = None       # high-water mark the caller supplied (None: no history)
    drawdown: float | None = None       # equity_before's fall from peak_nav, as a fraction
    orders: list[Order]
    fills: list[Fill]
    positions: dict[str, int]           # signed shares after fills
    cash: float
    nav: float                          # cash + sum(shares * mark)
    costs: float = 0.0                  # commission + slippage paid this cycle, already out of cash
