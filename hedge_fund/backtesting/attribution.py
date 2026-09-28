"""Attribution — which strategy earned the fund's return.

The fund trades one netted book, so no broker fill belongs to a single
pod. What each pod does own is its sleeve: the target weights it handed
the fund each cycle. Its *paper* return over the next period is the return
of that sleeve at the marks the fund itself recorded, and its paper NAV is
its capital slice compounded at those returns — the track record the pod
would show if it ran alone at its slice. An allocator that feeds winners
and cuts losers reads exactly this.

The netting identity makes the sleeves add up: the fund's pre-risk target
book is the slice-weighted sum of the sleeves, so the slice-weighted sum
of sleeve returns IS the return that book would have earned. The realized
fund return differs from it by everything that happens after netting —
master-risk clamps, share flooring, cash drag, and trading costs — and that
gap is reported as the residual, never hidden in a pod's number.

Pure function over the cycle records: it reads the receipts and nothing
else, so it will run unchanged over the persistent ledger.
"""

from __future__ import annotations

from pydantic import BaseModel

from hedge_fund.backtesting.metrics import (
    annualized_return,
    max_drawdown,
    period_returns,
    sharpe_ratio,
)
from hedge_fund.pipeline.models import CycleRecord


class StrategyAttribution(BaseModel):
    """One strategy's paper track record inside the fund."""

    name: str
    slice: float                       # normalized capital slice
    nav: list[float]                   # paper sleeve NAV per cycle date; starts at capital * slice
    returns: list[float]               # sleeve return per period (first period is 0.0: no prior sleeve)
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    contribution_pct: float            # sum over periods of slice * sleeve return


class Attribution(BaseModel):
    """Every strategy's attribution, and what the sleeves do not explain."""

    strategies: list[StrategyAttribution]
    fund_return_pct: float             # arithmetic sum of the fund's period returns
    residual_pct: float                # fund_return_pct minus every contribution: risk, sizing, cash, costs


def attribute_strategies(
    records: list[CycleRecord],
    capital: float,
    cadence: str,
) -> Attribution:
    """Attribute the fund's return to its strategies from its cycle records.

    Period k runs from cycle k-1 to cycle k. A sleeve's return over it is
    sum_t weight_{s,t,k-1} * (mark_{t,k} / mark_{t,k-1} - 1) over the
    tickers the sleeve weighted and the fund marked at both ends; a name
    with no mark at one end contributes nothing (it could not have been
    held either — run_cycle refuses to carry an unpriced position). The
    first period, capital to cycle 0, has no prior sleeve: every sleeve
    returns 0.0 and the fund's first-trade costs land in the residual.
    """
    if not records:
        raise ValueError("attribution needs at least one cycle record")

    names = [sr.name for sr in records[0].strategies]
    slices = {sr.name: sr.slice for sr in records[0].strategies}

    sleeve_returns: dict[str, list[float]] = {n: [0.0] for n in names}
    for prev, cur in zip(records, records[1:]):
        prior = {sr.name: sr.weights for sr in prev.strategies}
        for name in names:
            r = 0.0
            for ticker, weight in prior.get(name, {}).items():
                before = prev.marks.get(ticker)
                after = cur.marks.get(ticker)
                if weight and before and after:
                    r += weight * (after / before - 1)
            sleeve_returns[name].append(r)

    fund_curve = [capital] + [r.nav for r in records]
    fund_period = period_returns(fund_curve)
    fund_total_arith = float(fund_period.sum())

    strategies: list[StrategyAttribution] = []
    contributed = 0.0
    for name in names:
        returns = sleeve_returns[name]
        nav = [capital * slices[name]]
        for r in returns:
            nav.append(nav[-1] * (1 + r))
        curve = nav[1:]                      # one point per cycle date, like the fund
        total = curve[-1] / nav[0] - 1
        contribution = slices[name] * sum(returns)
        contributed += contribution
        strategies.append(StrategyAttribution(
            name=name,
            slice=slices[name],
            nav=[round(v, 2) for v in curve],
            returns=[round(r, 8) for r in returns],
            total_return_pct=round(total, 6),
            annualized_return_pct=round(
                annualized_return(total, records[0].as_of, records[-1].as_of), 6),
            sharpe_ratio=round(sharpe_ratio(period_returns(nav), cadence), 4),
            max_drawdown_pct=round(max_drawdown(nav), 6),
            contribution_pct=round(contribution, 6),
        ))

    return Attribution(
        strategies=strategies,
        fund_return_pct=round(fund_total_arith, 6),
        residual_pct=round(fund_total_arith - contributed, 6),
    )
