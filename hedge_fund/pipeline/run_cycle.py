"""run_cycle — one tick of the fund, the same code path in every mode.

    point-in-time data -> analysts -> blend -> risk -> execution -> record

This is the fund's heartbeat. A backtest is run_cycle in a loop over history
with a SimBroker; paper trading is the same loop on a live clock with a
PaperBroker; live is the same loop with a real broker. Only the clock and the
broker change — the pipeline never does.

run_cycle is the pipeline's only impure piece: it talks to the data client
and the broker. Every stage it delegates to (blend_signals, apply_limits,
build_orders) is a pure function. Determinism: given the same spec, date,
broker state, and data responses, the returned record is byte-identical —
with the one caveat that a cold LLM cache makes an agent's first live call
nondeterministic; the prompt cache makes every replay exact.

Notable behavior, chosen deliberately:
- Targets are the complete statement of the desired book. If every analyst
  abstains or goes neutral, the targets are all zero and the fund closes to
  flat. The record shows `abstained` on each signal, so an outer loop (the
  Day-5 daemon) can decide to skip a tick instead — that guard belongs
  outside the pipeline.
- A universe ticker with no price and no position is skipped (recorded, its
  analysts never called): unlisted/delisted names are normal in history.
  A HELD ticker with no price raises — a fund that cannot price its own
  book has an infrastructure problem, and its NAV would be a lie.
"""

from __future__ import annotations

from datetime import date as _date
from datetime import timedelta

from hedge_fund.brokers.models import Fill
from hedge_fund.brokers.protocol import Broker
from hedge_fund.data.protocol import DataClient
from hedge_fund.features.snapshot import InsufficientData
from hedge_fund.features.technicals import MARK_LOOKBACK_DAYS, build_price_snapshot, last_close
from hedge_fund.features.valuation import dividend_per_share
from hedge_fund.fund.spec import Fund, normalize_universe
from hedge_fund.models import Signal
from hedge_fund.pipeline.execution import build_orders
from hedge_fund.pipeline.models import CycleRecord, StrategyRecord, TickerSkip
from hedge_fund.portfolio.construction import blend_signals
from hedge_fund.risk.limits import apply_limits


def run_cycle(
    fund: Fund,
    as_of: str,
    broker: Broker,
    data_client: DataClient,
    universe: list[str],
    *,
    peak_nav: float | None = None,
    prev_as_of: str | None = None,
) -> CycleRecord:
    """Run one tick of *fund* over *universe* as of *as_of* (YYYY-MM-DD).

    The universe is an argument, not a mandate field: a fund is its desk —
    strategies, staff, risk, capital — and can be pointed at any names. What
    it was asked to trade this tick is recorded on the returned CycleRecord.

    `peak_nav` is the fund's high-water mark before this tick, from whoever
    holds the history (the backtest loop, later the ledger). With it, the
    drawdown kill-switch can judge; without it, that limit cannot fire.
    `prev_as_of` is the prior tick's date, from the same source: dividends
    accrue on the held book for the days in between. None: nothing to
    accrue.
    """
    spec = fund.spec
    universe = normalize_universe(universe)
    held = broker.positions()

    marks, skipped = _mark_prices(
        sorted(set(universe) | set(held)), as_of, held, data_client,
    )

    dividends: dict[str, float] = {}
    if spec.dividends.accrue and prev_as_of is not None and held:
        dividends = _accrue_dividends(held, prev_as_of, as_of, data_client)
        for ticker, amount in dividends.items():
            broker.credit(amount, f"dividend accrual {ticker} {prev_as_of}..{as_of}")

    cash_before = broker.cash()
    equity_before = cash_before + sum(
        p.shares * marks[t] for t, p in held.items()
    )
    if equity_before <= 0:
        raise ValueError(
            f"{spec.name}: equity is {equity_before:.2f} as of {as_of} — "
            "cannot size positions against a non-positive book"
        )

    tradeable = [t for t in universe if t in marks]

    # Realized vol per name, for risk-scaled sizing — fetched once per cycle
    # and shared by every sleeve that asks for it. The same bars the momentum
    # and reversal models read, so with a disk cache this is free.
    vols: dict[str, float] = {}
    if any(strategy.blend.vol_scaled for strategy, _ in fund.strategies):
        vols = _realized_vols(tradeable, as_of, data_client)

    # Each strategy runs its own analysts and blends its own sleeve; the fund
    # nets the sleeves by capital slice. A persona staffed into two strategies
    # is asked twice, but the second ask is a prompt-cache hit, not spend.
    total_slice = sum(s.weight for s, _ in fund.strategies)
    strategy_records: list[StrategyRecord] = []
    netted: dict[str, float] = {t: 0.0 for t in tradeable}
    for strategy, staff in fund.strategies:
        signals: list[Signal] = []
        for ticker in tradeable:
            for model in staff:
                signals.append(model.predict(ticker, as_of, data_client))
        blend = blend_signals(
            signals, strategy.model_weights, strategy.blend.gross_target,
            market_neutral=strategy.blend.market_neutral,
            vols=vols if strategy.blend.vol_scaled else None,
        )
        slice_ = strategy.weight / total_slice
        for ticker, weight in blend.weights.items():
            netted[ticker] += slice_ * weight
        strategy_records.append(StrategyRecord(
            name=strategy.name,
            slice=slice_,
            signals=signals,
            convictions=blend.convictions,
            weights=blend.weights,
        ))

    drawdown = None
    if peak_nav is not None:
        drawdown = max(0.0, 1 - equity_before / max(peak_nav, equity_before))
    risk = apply_limits(netted, spec.risk, drawdown=drawdown)

    orders = build_orders(risk.weights, held, marks, equity_before,
                          min_trade_pct=spec.execution.min_trade_pct)
    fills: list[Fill] = [broker.place_order(o) for o in orders]

    positions_after = {t: p.shares for t, p in broker.positions().items()}
    cash_after = broker.cash()
    nav = cash_after + sum(s * marks[t] for t, s in positions_after.items())

    return CycleRecord(
        fund=spec.name,
        as_of=as_of,
        spec=spec,
        universe=universe,
        marks=marks,
        vols=vols,
        skipped=skipped,
        dividends=dividends,
        strategies=strategy_records,
        target_weights=netted,
        clamps=risk.clamps,
        final_weights=risk.weights,
        equity_before=equity_before,
        cash_before=cash_before,
        peak_nav=peak_nav,
        drawdown=drawdown,
        orders=orders,
        fills=fills,
        positions=positions_after,
        cash=cash_after,
        nav=nav,
        costs=sum(f.cost for f in fills),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _mark_prices(
    tickers: list[str],
    as_of: str,
    held: dict,
    data_client: DataClient,
) -> tuple[dict[str, float], list[TickerSkip]]:
    """Last close on or before *as_of* for each ticker, within the lookback.

    No bar and not held -> TickerSkip (the caller then never runs analysts
    on it). No bar but HELD -> raise: the book cannot be honestly valued.
    """
    marks: dict[str, float] = {}
    skipped: list[TickerSkip] = []

    for ticker in tickers:
        mark = last_close(ticker, as_of, data_client)
        if mark is not None:
            marks[ticker] = mark[1]
        elif ticker in held:
            raise ValueError(
                f"held position {ticker} has no price within "
                f"{MARK_LOOKBACK_DAYS} days of {as_of} — cannot value the book"
            )
        else:
            skipped.append(TickerSkip(
                ticker=ticker,
                reason=f"no close within {MARK_LOOKBACK_DAYS} days of {as_of}",
            ))

    return marks, skipped


def _accrue_dividends(
    held: dict,
    prev_as_of: str,
    as_of: str,
    data_client: DataClient,
) -> dict[str, float]:
    """Dividend cash per held ticker for (prev_as_of, as_of]: shares x
    trailing dividend per share x days / 365. The rate comes from the
    latest filing public on *as_of*, so it is point-in-time; a name with
    no payout on file, or a loss-maker, accrues nothing. Signed by the
    position: a short owes the dividend."""
    days = (_date.fromisoformat(as_of) - _date.fromisoformat(prev_as_of)).days
    if days <= 0:
        return {}
    accrued: dict[str, float] = {}
    for ticker, position in held.items():
        metrics = data_client.get_financial_metrics(ticker, as_of, period="ttm", limit=1)
        if not metrics:
            continue
        dps = dividend_per_share(metrics[0])
        if not dps:
            continue
        accrued[ticker] = position.shares * dps * days / 365
    return accrued


def _realized_vols(
    tickers: list[str],
    as_of: str,
    data_client: DataClient,
) -> dict[str, float]:
    """Annualized 63-day realized vol per ticker, from the point-in-time
    price snapshot. A name without a quarter of bars is left out — the
    blend then sizes it as a typical-vol name. Data-layer errors propagate.
    """
    vols: dict[str, float] = {}
    for ticker in tickers:
        try:
            snap = build_price_snapshot(ticker, as_of, data_client)
        except InsufficientData:
            continue
        if snap.vol_63d_ann is not None:
            vols[ticker] = snap.vol_63d_ann
    return vols
