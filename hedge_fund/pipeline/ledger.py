"""The ledger — a fund's book carried from one run to the next.

Every run writes a receipt: the full CycleRecord, positions and cash and NAV
included. The ledger is the other half: it reads the newest receipt back and
opens the next run's broker on that book, so NAV moves between runs and
becomes a track record instead of resetting to the mandate's capital.

    receipts on disk -> Carry (book, prior date, high-water mark)
                     -> SimBroker opened on it -> run_cycle -> new receipt

The chain is ordered by as-of date, not by when a run happened to execute.
A run may repeat the latest date (a second tick that day) or move forward;
it may not go back in time — a book that already knows the future cannot
honestly trade the past. History is what `--backtest` is for.

A backtest never touches the ledger: it opens on cash and loops its own
broker. Only live-clock runs (`aihf MANDATE`, the app's run, the web runner)
carry and record.

Where the book lives depends on the mandate's `execution.broker`:

- sim: the receipts ARE the book. The broker is a SimBroker opened on the
  newest receipt's positions and cash.
- alpaca-paper: the paper account is the book. Positions and cash are read
  from it; the receipts still supply the chain (prior date, high-water mark)
  and are checked against the account, and any drift is flagged on the new
  receipt. A paper run trades today, while the market is open, and a fund's
  first paper run needs a flat account — otherwise it would liquidate
  positions it never opened.

Each broker keeps its own chain: switching a fund to paper starts a fresh
track record rather than splicing real fills onto simulated history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from hedge_fund import paths
from hedge_fund.brokers.alpaca import AlpacaBroker
from hedge_fund.brokers.protocol import Broker
from hedge_fund.brokers.sim import SimBroker
from hedge_fund.data.protocol import DataClient
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.pipeline.models import CycleRecord
from hedge_fund.pipeline.run_cycle import run_cycle


class LedgerError(ValueError):
    """The requested run would break the fund's chain of receipts."""


@dataclass(frozen=True)
class Carry:
    """What a fund brings into its next cycle. `as_of` None: no prior run
    on this broker — a simulated fund opens on its mandate's capital, a
    paper fund on whatever its account holds."""

    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    as_of: str | None = None          # the prior run's date — dividends accrue from here
    peak_nav: float | None = None     # high-water mark over the whole chain
    source: Path | None = None        # the receipt the chain was read from
    broker: str = "sim"               # the mandate's execution.broker

    def describe(self) -> str:
        """One line for a run's footer: where this book came from."""
        n = len(self.positions)
        book = (f"{n} {'position' if n == 1 else 'positions'}, "
                f"${self.cash:,.0f} cash")
        if self.broker == "alpaca-paper":
            last = "first run" if self.as_of is None else f"last run {self.as_of}"
            return f"read the book from the Alpaca paper account ({book}; {last})"
        if self.as_of is None:
            return f"opened on ${self.cash:,.0f} of capital (first run)"
        return f"carried the book from the {self.as_of} run ({book})"


@dataclass(frozen=True)
class LedgerRun:
    record: CycleRecord
    path: Path                        # where this run's receipt was saved
    carry: Carry                      # what it opened on


def run_receipts(name: str, root: Path | None = None,
                 broker: str | None = None) -> list[tuple[Path, dict]]:
    """A fund's run receipts, oldest first by (as_of, run stamp), as raw
    dicts — only *broker*'s chain when given. Unreadable files and other
    funds' receipts (a name that is a prefix of another's) are skipped;
    backtests never match."""
    root = root or paths.MANDATES_DIR
    out: list[tuple[Path, dict]] = []
    for path in root.glob(f"{name}-run-*.json"):
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("fund") != name or "as_of" not in d:
            continue
        if broker is not None and _broker_of(d) != broker:
            continue
        out.append((path, d))
    out.sort(key=lambda pd: (pd[1]["as_of"], pd[0].name))
    return out


def carried_book(spec: FundSpec, as_of: str, root: Path | None = None,
                 *, today: str | None = None) -> Carry:
    """The book *spec*'s fund carries into a run as of *as_of*: the newest
    receipt's positions and cash, its date, and the chain's high-water mark,
    from the chain of the mandate's broker. No receipts: the mandate's
    capital, flat. Raises LedgerError if *as_of* is before the newest
    receipt, or — on a paper broker — is not *today* (New York)."""
    broker = spec.execution.broker
    if broker != "sim":
        today = today or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
        if as_of != today:
            raise LedgerError(
                f"{spec.name} trades on the {broker} broker, which fills now: "
                f"run it as of today ({today}), not {as_of}. --backtest is for "
                "any other date."
            )
    receipts = run_receipts(spec.name, root, broker)
    if not receipts:
        # A paper account's opening equity is whatever it holds, not the
        # mandate's capital, so there is no high-water mark until it trades.
        return Carry(cash=spec.capital, broker=broker,
                     peak_nav=spec.capital if broker == "sim" else None)

    path, last = receipts[-1]
    if as_of < last["as_of"]:
        raise LedgerError(
            f"{spec.name}'s book was last run as of {last['as_of']}; a run as "
            f"of {as_of} would trade the past with a book that has seen the "
            "future. Run as of that date or later, or --backtest for history."
        )
    # The first receipt's opening equity is the capital the fund started
    # with, so a fund that has only lost money still has a peak to fall from.
    peak = max(
        max(d.get("equity_before", d["nav"]), d["nav"]) for _, d in receipts
    )
    return Carry(
        cash=last["cash"],
        positions={t: int(s) for t, s in last.get("positions", {}).items()},
        as_of=last["as_of"],
        peak_nav=peak,
        source=path,
        broker=broker,
    )


def open_broker(spec: FundSpec, carry: Carry, as_of: str, *,
                adopt_account: bool = False) -> tuple[Broker, Carry, list[str]]:
    """The broker a run trades through, the carry as that broker sees it,
    and anything worth flagging. For a paper account this is where the
    run checks, before any analyst is paid for: that the market is open
    today, that a first run starts flat (unless *adopt_account* hands the
    account's positions to the fund), and whether the account drifted from
    the last receipt."""
    ex = spec.execution
    if ex.broker == "sim":
        broker = SimBroker(
            cash=carry.cash,
            positions=carry.positions,
            commission_bps=ex.commission_bps,
            slippage_bps=ex.slippage_bps,
        )
        return broker, carry, []

    alpaca = AlpacaBroker.from_env()
    market_day = alpaca.market_date()
    if market_day != as_of:
        raise LedgerError(
            f"the market's date is {market_day}; a paper run as of {as_of} "
            "cannot fill at today's prices"
        )
    held = {t: p.shares for t, p in alpaca.positions().items()}
    if carry.as_of is None and held and not adopt_account:
        # Also where a first run that failed partway lands: whatever did fill
        # is in the account, with no receipt yet to say it was the fund's.
        raise LedgerError(
            f"{spec.name}'s first paper run needs a flat account, and this one "
            f"holds {', '.join(sorted(held))} — the fund would sell what it never "
            "bought. Use a paper account of its own, reset this one in the "
            "Alpaca dashboard, or pass --adopt-account to hand these positions "
            "to the fund."
        )
    if carry.as_of is None:
        warnings = [
            f"adopted the account's {t} position ({held[t]} shares) into the fund's book"
            for t in sorted(held)
        ]
    else:
        warnings = [
            f"{t}: the {carry.as_of} receipt holds {carry.positions.get(t, 0)} shares, "
            f"the Alpaca account {held.get(t, 0)} — trading from the account's book"
            for t in sorted(set(held) | set(carry.positions))
            if held.get(t, 0) != carry.positions.get(t, 0)
        ]
    return alpaca, replace(carry, cash=alpaca.cash(), positions=held), warnings


def save_receipt(record: CycleRecord, root: Path | None = None) -> Path:
    """Write *record* where the ledger and the app's history pane read it."""
    root = root or paths.MANDATES_DIR
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S-%f")
    path = root / f"{record.fund}-run-{stamp}.json"
    path.write_text(record.model_dump_json(indent=2))
    return path


def run_carried(
    fund: Fund,
    as_of: str,
    data_client: DataClient,
    universe: list[str],
    *,
    root: Path | None = None,
    adopt_account: bool = False,
) -> LedgerRun:
    """One ledgered tick: open the mandate's broker on the carried book, run
    the cycle, save the receipt. The only way a live-clock run should trade."""
    spec = fund.spec
    carry = carried_book(spec, as_of, root)
    broker, carry, warnings = open_broker(spec, carry, as_of,
                                          adopt_account=adopt_account)
    record = run_cycle(fund, as_of, broker, data_client, universe,
                       peak_nav=carry.peak_nav, prev_as_of=carry.as_of)
    record.warnings = warnings
    return LedgerRun(record=record, path=save_receipt(record, root), carry=carry)


def _broker_of(receipt: dict) -> str:
    """The broker a receipt traded through; receipts from before the field
    existed were simulated."""
    return ((receipt.get("spec") or {}).get("execution") or {}).get("broker", "sim")
