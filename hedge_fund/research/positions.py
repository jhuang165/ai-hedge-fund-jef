"""Positions as research input — what the user already holds.

`aihf research` can be told about an existing position so the diagnosis
answers the question actually being asked ("should I add, hold, trim or
exit?") instead of the generic one ("is this a buy?"). Two ways in:

    aihf research AAPL:100@150.25        # 100 shares, average cost $150.25
    aihf research --portfolio holdings.yaml

where the YAML is

    positions:
      - ticker: AAPL
        shares: 100
        cost_basis: 150.25
      - ticker: MSFT
        shares: -20          # negative = short
        cost_basis: 410

Nothing here talks to a broker or a ledger: positions are declared by the
user, used for one report, and never persisted by this command.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from hedge_fund.research.models import Position

# TICKER[:SHARES@COST] — shares may be fractional or negative (short).
_ARG = re.compile(
    r"^(?P<ticker>[A-Za-z][A-Za-z0-9.\-]*)"
    r"(?::(?P<shares>-?\d+(?:\.\d+)?)@(?P<cost>\d+(?:\.\d+)?))?$"
)


class ResearchTarget(BaseModel):
    """One ticker to research, with or without a declared position."""

    model_config = ConfigDict(extra="forbid")

    ticker: str
    position: Position | None = None

    @field_validator("ticker")
    @classmethod
    def _upper(cls, ticker: str) -> str:
        return ticker.strip().upper()


class Portfolio(BaseModel):
    """The `--portfolio` file shape."""

    model_config = ConfigDict(extra="forbid")

    positions: list[Position] = Field(default_factory=list)


def parse_target(arg: str) -> ResearchTarget:
    """Parse one CLI argument: `AAPL` or `AAPL:100@150.25`.

    Raises ValueError with the grammar in the message — the CLI turns that
    into a usage error.
    """
    m = _ARG.match(arg.strip())
    if m is None:
        raise ValueError(
            f"cannot parse {arg!r}: expected TICKER or TICKER:SHARES@COST, e.g. AAPL:100@150.25"
        )
    ticker = m["ticker"].upper()
    if m["shares"] is None:
        return ResearchTarget(ticker=ticker)
    return ResearchTarget(
        ticker=ticker,
        position=Position(ticker=ticker, shares=float(m["shares"]), cost_basis=float(m["cost"])),
    )


def load_portfolio(path: str | Path) -> list[ResearchTarget]:
    """Load positions from a YAML file (see module docstring for the shape).
    A bare list at the top level is accepted too."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, list):
        data = {"positions": data}
    portfolio = Portfolio(**data)
    return [ResearchTarget(ticker=p.ticker, position=p) for p in portfolio.positions]


def merge_targets(*groups: list[ResearchTarget]) -> list[ResearchTarget]:
    """Combine targets from several sources. One entry per ticker, first
    appearance keeps its place in the order; a later entry with a position
    overrides an earlier one without (so `--portfolio` can enrich a bare
    ticker, and a positional `AAPL:100@150` can override the file)."""
    by_ticker: dict[str, ResearchTarget] = {}
    for group in groups:
        for target in group:
            existing = by_ticker.get(target.ticker)
            if existing is None or target.position is not None:
                by_ticker[target.ticker] = target
    return list(by_ticker.values())
