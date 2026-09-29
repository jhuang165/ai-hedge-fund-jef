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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from hedge_fund.brokers.sim import SimBroker
from hedge_fund.data.protocol import DataClient
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund import paths
from hedge_fund.pipeline.models import CycleRecord
from hedge_fund.pipeline.run_cycle import run_cycle


class LedgerError(ValueError):
    """The requested run would break the fund's chain of receipts."""


@dataclass(frozen=True)
class Carry:
    """What a fund brings into its next cycle. `as_of` None: no prior run,
    the fund opens on its mandate's capital."""

    cash: float
    positions: dict[str, int] = field(default_factory=dict)
    as_of: str | None = None          # the prior run's date — dividends accrue from here
    peak_nav: float | None = None     # high-water mark over the whole chain
    source: Path | None = None        # the receipt the book was read from

    def describe(self) -> str:
        """One line for a run's footer: where this book came from."""
        if self.as_of is None:
            return f"opened on ${self.cash:,.0f} of capital (first run)"
        n = len(self.positions)
        return (f"carried the book from the {self.as_of} run "
                f"({n} {'position' if n == 1 else 'positions'}, "
                f"${self.cash:,.0f} cash)")


@dataclass(frozen=True)
class LedgerRun:
    record: CycleRecord
    path: Path                        # where this run's receipt was saved
    carry: Carry                      # what it opened on


def run_receipts(name: str, root: Path | None = None) -> list[tuple[Path, dict]]:
    """A fund's run receipts, oldest first by (as_of, run stamp), as raw
    dicts. Unreadable files and other funds' receipts (a name that is a
    prefix of another's) are skipped; backtests never match."""
    root = root or paths.MANDATES_DIR
    out: list[tuple[Path, dict]] = []
    for path in root.glob(f"{name}-run-*.json"):
        try:
            d = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("fund") != name or "as_of" not in d:
            continue
        out.append((path, d))
    out.sort(key=lambda pd: (pd[1]["as_of"], pd[0].name))
    return out


def carried_book(spec: FundSpec, as_of: str, root: Path | None = None) -> Carry:
    """The book *spec*'s fund carries into a run as of *as_of*: the newest
    receipt's positions and cash, its date, and the chain's high-water mark.
    No receipts: the mandate's capital, flat. Raises LedgerError if *as_of*
    is before the newest receipt."""
    receipts = run_receipts(spec.name, root)
    if not receipts:
        return Carry(cash=spec.capital, peak_nav=spec.capital)

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
    )


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
) -> LedgerRun:
    """One ledgered tick: open a broker on the carried book, run the cycle,
    save the receipt. The only way a live-clock run should trade."""
    spec = fund.spec
    carry = carried_book(spec, as_of, root)
    broker = SimBroker(
        cash=carry.cash,
        positions=carry.positions,
        commission_bps=spec.execution.commission_bps,
        slippage_bps=spec.execution.slippage_bps,
    )
    record = run_cycle(fund, as_of, broker, data_client, universe,
                       peak_nav=carry.peak_nav, prev_as_of=carry.as_of)
    return LedgerRun(record=record, path=save_receipt(record, root), carry=carry)
