"""Pydantic models for the research command's output.

A `ResearchReport` is the read-only counterpart to `Signal` (see
hedge_fund/models.py): both speak the same `signal`/`confidence` vocabulary,
but a report is a one-off, user-facing artifact with cited sources, not a
portfolio input.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hedge_fund.features.technicals import PriceSnapshot
from hedge_fund.models import Signal

# What the report recommends. Which subset applies depends on whether the
# user declared a position: held names get add/hold/trim/exit, names not
# held get buy/watch/avoid. One field, one vocabulary, so a ranked table
# across a mixed list reads in one column.
HELD_ACTIONS: tuple[str, ...] = ("add", "hold", "trim", "exit")
FLAT_ACTIONS: tuple[str, ...] = ("buy", "watch", "avoid")
Action = Literal["add", "hold", "trim", "exit", "buy", "watch", "avoid"]


class SearchResult(BaseModel):
    """One search hit, kept verbatim so a human can verify a citation."""

    query: str  # which templated query produced it
    title: str
    url: str
    content: str
    score: float | None = None
    published_date: str | None = None


class Claim(BaseModel):
    """One catalyst or risk statement, tied back to the evidence for it."""

    text: str
    source_indices: list[int] = Field(default_factory=list)


class Position(BaseModel):
    """A position the user declares they hold. Negative shares = short."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    shares: float
    cost_basis: float = Field(gt=0, description="average cost per share")

    @field_validator("ticker")
    @classmethod
    def _upper(cls, ticker: str) -> str:
        return ticker.strip().upper()

    @field_validator("shares")
    @classmethod
    def _nonzero(cls, shares: float) -> float:
        if shares == 0:
            raise ValueError("a position needs a non-zero share count")
        return shares

    @property
    def side(self) -> str:
        return "long" if self.shares > 0 else "short"


class PositionMark(BaseModel):
    """The declared position marked at the latest close the report saw."""

    ticker: str
    shares: float
    cost_basis: float
    side: Literal["long", "short"]
    last_close: float | None = None
    market_value: float | None = None       # signed: negative for a short
    unrealized_pnl: float | None = None
    unrealized_pnl_pct: float | None = None  # on cost, signed by side

    @classmethod
    def mark(cls, position: Position, last_close: float | None) -> PositionMark:
        if last_close is None:
            return cls(ticker=position.ticker, shares=position.shares,
                       cost_basis=position.cost_basis, side=position.side)
        pnl = position.shares * (last_close - position.cost_basis)
        return cls(
            ticker=position.ticker,
            shares=position.shares,
            cost_basis=position.cost_basis,
            side=position.side,
            last_close=last_close,
            market_value=position.shares * last_close,
            unrealized_pnl=pnl,
            unrealized_pnl_pct=pnl / abs(position.shares * position.cost_basis),
        )


class ResearchReport(BaseModel):
    """The complete output of `diagnose()` — a cited stock diagnosis."""

    ticker: str
    as_of: str
    model: str
    signal: Literal["bullish", "neutral", "bearish"]
    confidence: float  # 0-100, same scale as Signal.metadata["confidence"]
    action: Action
    action_rationale: str
    thesis: str
    catalysts: list[Claim] = Field(default_factory=list)
    risks: list[Claim] = Field(default_factory=list)
    sources: list[SearchResult] = Field(default_factory=list)  # index-addressable
    queries: list[str] = Field(default_factory=list)  # queries that actually succeeded
    desk: list[Signal] = Field(default_factory=list)  # every quant model's point-in-time view
    technicals: PriceSnapshot | None = None  # None when price history was unavailable
    position: PositionMark | None = None     # None when the user declared no position
    snapshot_hash: str | None = None  # None when fundamentals were unavailable
    prompt_key: str
    warnings: list[str] = Field(default_factory=list)

    @property
    def score(self) -> float:
        """Signed conviction in [-100, 100] for ranking a list of reports."""
        sign = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}[self.signal]
        return sign * self.confidence
