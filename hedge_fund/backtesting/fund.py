"""Backtest a fund — run_cycle in a loop over history.

`run_cycle`'s docstring makes the promise: "a backtest is run_cycle in a
loop over history with a SimBroker; paper trading is the same loop on a
live clock." This module is that loop. Nothing here re-implements pipeline
mechanics — every tick is the real run_cycle against a persistent broker,
so anything true of one cycle (point-in-time data, fail-loud pricing,
master risk on the netted book) is true of every backtested tick by
construction.

Nothing here assumes what the fund trades on. The rebalance cadence comes
from the mandate (FundSpec.rebalance): a fundamentals fund says weekly, a
news-driven fund can say daily. The trading-day grid derives from the
mandate's benchmark's actual bars — holidays and half-weeks fall out
naturally, no exchange calendar math.

This is the fund-level counterpart to the per-model harness in engine.py
(BacktestEngine simulates one alpha model's views with fixed mechanics;
backtest_fund runs the whole shop).
"""

from __future__ import annotations

from datetime import date as _date
from typing import Callable, Literal

from pydantic import BaseModel

from hedge_fund.backtesting.attribution import Attribution, attribute_strategies
from hedge_fund.backtesting.metrics import (
    annualized_return,
    max_drawdown,
    period_returns,
    sharpe_ratio,
)
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.data.protocol import DataClient
from hedge_fund.fund.spec import Fund, normalize_universe
from hedge_fund.fund.universe import DatedUniverse, Universe, survivorship_warning
from hedge_fund.pipeline.models import CycleRecord
from hedge_fund.pipeline.run_cycle import run_cycle

class FundBacktestMetrics(BaseModel):
    """The numbers that say whether the fund worked, and against what."""

    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    benchmark_return_pct: float
    excess_return_pct: float          # fund total minus benchmark total
    n_cycles: int
    n_orders: int
    total_costs: float                # commission + slippage over the run, in dollars
    costs_pct: float                  # total_costs / starting capital — the friction drag
    total_dividends: float = 0.0      # dividend cash accrued on the book over the run
    kill_switch_date: str | None = None  # first cycle the drawdown kill-switch flattened the book


class FundBacktestResult(BaseModel):
    """A full backtest, serialized: the curve, the stats, and — because every
    tick is a CycleRecord — every thesis, clamp, order, and fill behind it.
    `model_dump_json()` round-trips; this is the receipts file."""

    fund: str
    start: str                        # first grid date actually traded
    end: str                          # last grid date actually traded
    rebalance: str
    benchmark: str
    universe: list[str]               # every ticker this backtest traded (union, for a dated universe)
    universe_kind: Literal["static", "dated"]
    warnings: list[str]               # what the reader must know before trusting the curve
    capital: float
    dates: list[str]
    nav: list[float]                  # NAV after each cycle, one per date
    benchmark_nav: list[float]        # benchmark scaled to the same capital
    metrics: FundBacktestMetrics
    attribution: Attribution          # each strategy's paper sleeve, and the residual
    records: list[CycleRecord]


def backtest_fund(
    fund: Fund,
    start: str,
    end: str,
    data_client: DataClient,
    universe: Universe,
    *,
    on_cycle: Callable[[int, int, CycleRecord], None] | None = None,
) -> FundBacktestResult:
    """Run *fund* over *universe* through history from *start* to *end*.

    One run_cycle per grid date against a persistent SimBroker — positions
    and cash carry across ticks, so the fund rebalances rather than
    restarts. `on_cycle(i, n, record)` fires after each tick (progress UIs).
    The universe is the study's input, not the mandate's: the same fund can
    be backtested over different names.

    A plain list is traded on every date and the result carries the
    survivorship warning. A DatedUniverse is asked for its members as of
    each grid date, so names join and leave the book as they did in
    history.

    The fund's high-water mark is tracked here and handed to each cycle, so
    the mandate's drawdown kill-switch can fire; the first date it did is
    on the metrics.

    Fail loud: no benchmark bars in the window raises — a backtest with no
    trading grid is an infrastructure problem, not an empty result.
    """
    spec = fund.spec
    if isinstance(universe, DatedUniverse):
        kind: Literal["static", "dated"] = "dated"
        members = universe.as_of
        traded = universe.tickers
    else:
        kind = "static"
        static = normalize_universe(universe)
        members = lambda _date: static  # noqa: E731
        traded = static
    bars = data_client.get_prices(spec.benchmark, start, end)
    closes = {b.time[:10]: b.close for b in bars if start <= b.time[:10] <= end}
    if not closes:
        raise ValueError(
            f"{spec.name}: no {spec.benchmark} bars in [{start}, {end}] — "
            "cannot build the trading grid"
        )
    grid = rebalance_grid(sorted(closes), spec.rebalance)

    broker = SimBroker(
        cash=spec.capital,
        commission_bps=spec.execution.commission_bps,
        slippage_bps=spec.execution.slippage_bps,
    )
    records: list[CycleRecord] = []
    nav: list[float] = []
    benchmark_nav: list[float] = []
    base_close = closes[grid[0]]
    peak = spec.capital
    prev: str | None = None
    for i, as_of in enumerate(grid):
        record = run_cycle(fund, as_of, broker, data_client, members(as_of),
                           peak_nav=peak, prev_as_of=prev)
        records.append(record)
        nav.append(record.nav)
        peak = max(peak, record.nav)
        prev = as_of
        # Price return, plus the mandate's assumed yield accrued continuously.
        years = (_date.fromisoformat(as_of) - _date.fromisoformat(grid[0])).days / 365
        income = (1 + spec.dividends.benchmark_yield) ** years
        benchmark_nav.append(spec.capital * closes[as_of] / base_close * income)
        if on_cycle is not None:
            on_cycle(i, len(grid), record)

    warnings = []
    if (w := survivorship_warning(universe, grid[0])) is not None:
        warnings.append(w)

    return FundBacktestResult(
        fund=spec.name,
        start=grid[0],
        end=grid[-1],
        rebalance=spec.rebalance,
        benchmark=spec.benchmark,
        universe=traded,
        universe_kind=kind,
        warnings=warnings,
        capital=spec.capital,
        dates=grid,
        nav=nav,
        benchmark_nav=benchmark_nav,
        metrics=_metrics(spec.capital, grid, nav, benchmark_nav,
                         spec.rebalance, records),
        attribution=attribute_strategies(records, spec.capital, spec.rebalance),
        records=records,
    )


def rebalance_grid(days: list[str], cadence: str) -> list[str]:
    """Pick the rebalance dates out of sorted trading *days* (YYYY-MM-DD).

    daily: every day. weekly: the last trading day of each ISO week.
    monthly: the last trading day of each calendar month.
    """
    if cadence == "daily":
        return list(days)
    if cadence not in ("weekly", "monthly"):
        raise ValueError(f"unknown rebalance cadence {cadence!r}")

    last_of_period: dict[tuple[int, int], str] = {}
    for day in days:
        d = _date.fromisoformat(day)
        if cadence == "weekly":
            iso = d.isocalendar()
            key = (iso[0], iso[1])
        else:
            key = (d.year, d.month)
        last_of_period[key] = day  # days are sorted — the last write wins
    return sorted(last_of_period.values())


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _metrics(
    capital: float,
    grid: list[str],
    nav: list[float],
    benchmark_nav: list[float],
    cadence: str,
    records: list[CycleRecord],
) -> FundBacktestMetrics:
    total = nav[-1] / capital - 1
    annualized = annualized_return(total, grid[0], grid[-1])

    # Per-period returns over the curve including the starting capital, so
    # the first tick's move counts too.
    curve = [capital] + nav
    sharpe = sharpe_ratio(period_returns(curve), cadence)
    max_dd = max_drawdown(curve)

    benchmark_return = benchmark_nav[-1] / capital - 1
    total_costs = sum(r.costs for r in records)

    return FundBacktestMetrics(
        total_return_pct=round(total, 6),
        annualized_return_pct=round(annualized, 6),
        sharpe_ratio=round(float(sharpe), 4),
        max_drawdown_pct=round(float(max_dd), 6),
        benchmark_return_pct=round(benchmark_return, 6),
        excess_return_pct=round(total - benchmark_return, 6),
        n_cycles=len(nav),
        n_orders=sum(len(r.orders) for r in records),
        total_costs=round(total_costs, 2),
        costs_pct=round(total_costs / capital, 6),
        total_dividends=round(sum(sum(r.dividends.values()) for r in records), 2),
        kill_switch_date=next(
            (r.as_of for r in records
             if any(c.limit == "max_drawdown_pct" for c in r.clamps)),
            None,
        ),
    )
