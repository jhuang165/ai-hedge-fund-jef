"""Scorecard — did the research desk's calls beat the market?

Every saved report is a call: a signal (bullish / neutral / bearish), a
confidence, and an action, made on a date. Nothing in the app checks
whether "buy" beat "avoid". This does. Each call that is old enough is
joined to the stock's return over the horizon after it, less the
benchmark's return over the same days, and graded on whether that excess
return had the sign the call implied.

Three numbers say whether the desk has an edge:

- hit rate: the share of directional calls whose excess return went the
  way they said;
- the bull/bear spread: average excess return of bullish calls minus that
  of bearish ones — a desk that separates winners from losers shows a
  positive spread even when the market drags every name one way;
- the rank IC: the Spearman correlation between each call's signed
  conviction and its excess return — do stronger views pay more?

Calls whose horizon has not elapsed are pending, not wrong. A ticker with
no price at either end is skipped and says why. Pure over its inputs:
give it reports and a data client and it never touches the disk.
"""

from __future__ import annotations

from datetime import date as _date
from datetime import timedelta

import pandas as pd
from pydantic import BaseModel

from hedge_fund.data.protocol import DataClient
from hedge_fund.features.technicals import last_close
from hedge_fund.research.models import ResearchReport

DIRECTION = {"bullish": 1, "neutral": 0, "bearish": -1}


class GradedCall(BaseModel):
    """One matured call and what the market did after it."""

    ticker: str
    as_of: str
    matured_on: str                  # as_of + horizon
    action: str
    signal: str
    confidence: float
    score: float                     # signed conviction, for ranking
    direction: int                   # +1 bullish, 0 neutral, -1 bearish
    start_close: float
    end_close: float
    return_pct: float
    benchmark_return_pct: float
    excess_pct: float
    hit: bool | None                 # excess went the called way; None for a neutral call


class PendingCall(BaseModel):
    ticker: str
    as_of: str
    action: str
    signal: str
    matures_on: str


class SkippedCall(BaseModel):
    ticker: str
    as_of: str
    reason: str


class GroupStats(BaseModel):
    """Hit rate and excess return for one action or one signal."""

    key: str
    n: int
    hit_rate: float | None           # None when nothing in the group is directional
    avg_excess_pct: float
    median_excess_pct: float


class Scorecard(BaseModel):
    graded_as_of: str
    horizon_days: int
    benchmark: str
    n_reports: int
    n_graded: int
    n_pending: int
    n_skipped: int
    hit_rate: float | None           # over directional graded calls
    bull_bear_spread_pct: float | None
    rank_ic: float | None            # Spearman(score, excess) over graded calls
    by_action: list[GroupStats]
    by_signal: list[GroupStats]
    calls: list[GradedCall]
    pending: list[PendingCall]
    skipped: list[SkippedCall]


def grade(
    reports: list[ResearchReport],
    data_client: DataClient,
    *,
    horizon_days: int = 90,
    benchmark: str = "SPY",
    as_of: str | None = None,
) -> Scorecard:
    """Grade every report old enough to judge, as of *as_of* (default today).

    A ticker researched twice on one day is one call: the last report
    wins. Returns are close-to-close over [as_of, as_of + horizon_days],
    using the last close on or before each end (so a weekend maturity
    lands on the Friday), and excess is against *benchmark* over the same
    two dates.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    graded_as_of = as_of or _date.today().isoformat()

    latest: dict[tuple[str, str], ResearchReport] = {}
    for report in reports:
        latest[(report.ticker, report.as_of)] = report
    calls_in = [latest[k] for k in sorted(latest)]

    graded: list[GradedCall] = []
    pending: list[PendingCall] = []
    skipped: list[SkippedCall] = []
    bench_cache: dict[str, float | None] = {}

    def bench_close(day: str) -> float | None:
        if day not in bench_cache:
            mark = last_close(benchmark, day, data_client)
            bench_cache[day] = mark[1] if mark else None
        return bench_cache[day]

    for report in calls_in:
        matures_on = (_date.fromisoformat(report.as_of) + timedelta(days=horizon_days)).isoformat()
        if matures_on > graded_as_of:
            pending.append(PendingCall(ticker=report.ticker, as_of=report.as_of,
                                       action=report.action, signal=report.signal,
                                       matures_on=matures_on))
            continue
        start = last_close(report.ticker, report.as_of, data_client)
        end = last_close(report.ticker, matures_on, data_client)
        b0, b1 = bench_close(report.as_of), bench_close(matures_on)
        if start is None or end is None:
            skipped.append(SkippedCall(ticker=report.ticker, as_of=report.as_of,
                                       reason=f"no close for {report.ticker} on "
                                              f"{report.as_of if start is None else matures_on}"))
            continue
        if b0 is None or b1 is None:
            skipped.append(SkippedCall(ticker=report.ticker, as_of=report.as_of,
                                       reason=f"no {benchmark} close to measure against"))
            continue
        ret = end[1] / start[1] - 1
        bench_ret = b1 / b0 - 1
        excess = ret - bench_ret
        direction = DIRECTION[report.signal]
        graded.append(GradedCall(
            ticker=report.ticker, as_of=report.as_of, matured_on=matures_on,
            action=report.action, signal=report.signal, confidence=report.confidence,
            score=report.score, direction=direction,
            start_close=start[1], end_close=end[1],
            return_pct=round(ret, 6), benchmark_return_pct=round(bench_ret, 6),
            excess_pct=round(excess, 6),
            hit=(None if direction == 0 else (excess * direction) > 0),
        ))

    directional = [c for c in graded if c.hit is not None]
    bulls = [c.excess_pct for c in graded if c.direction > 0]
    bears = [c.excess_pct for c in graded if c.direction < 0]

    return Scorecard(
        graded_as_of=graded_as_of,
        horizon_days=horizon_days,
        benchmark=benchmark,
        n_reports=len(calls_in),
        n_graded=len(graded),
        n_pending=len(pending),
        n_skipped=len(skipped),
        hit_rate=_hit_rate(directional),
        bull_bear_spread_pct=(round(_mean(bulls) - _mean(bears), 6)
                              if bulls and bears else None),
        rank_ic=_rank_ic(graded),
        by_action=_groups(graded, "action"),
        by_signal=_groups(graded, "signal"),
        calls=graded,
        pending=pending,
        skipped=skipped,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _hit_rate(calls: list[GradedCall]) -> float | None:
    if not calls:
        return None
    return round(sum(1 for c in calls if c.hit) / len(calls), 4)


def _rank_ic(calls: list[GradedCall]) -> float | None:
    """Spearman correlation between conviction and excess return; None
    below three calls or when one side has no variation."""
    if len(calls) < 3:
        return None
    scores = pd.Series([c.score for c in calls])
    excess = pd.Series([c.excess_pct for c in calls])
    if scores.nunique() < 2 or excess.nunique() < 2:
        return None
    ic = scores.rank().corr(excess.rank())
    return None if pd.isna(ic) else round(float(ic), 4)


def _groups(calls: list[GradedCall], field: str) -> list[GroupStats]:
    keys = sorted({getattr(c, field) for c in calls})
    out: list[GroupStats] = []
    for key in keys:
        group = [c for c in calls if getattr(c, field) == key]
        excess = sorted(c.excess_pct for c in group)
        out.append(GroupStats(
            key=key,
            n=len(group),
            hit_rate=_hit_rate([c for c in group if c.hit is not None]),
            avg_excess_pct=round(_mean(excess), 6),
            median_excess_pct=round(float(pd.Series(excess).median()), 6),
        ))
    return out
