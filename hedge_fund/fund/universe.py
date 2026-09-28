"""Universe — which names the fund may trade, and since when.

A backtest over a list of tickers typed in today is survivorship-biased by
construction: every name on it is one you already know made it. The
bankruptcies, the delistings, and the acquisitions that would have been in
a real fund's book are missing, and the curve leans upward for it.

The honest input is point-in-time membership: which names were in the
investable set on each date. A `DatedUniverse` is that — a list of
snapshots, each in force from its `from` date until the next one — and
the backtest asks it for the members as of every cycle. A name that drops
out of the universe stops receiving signals, so the fund closes it at the
next cycle's mark; a name that joins starts being traded from that date.

A plain list is still accepted everywhere and is the right thing for a
live cycle (today's list, as of today). A backtest over one carries the
survivorship warning on its result, in the receipt and on every screen —
the bias is not fixed by naming it, but it is no longer hidden.
"""

from __future__ import annotations

from datetime import date as _date
from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class UniverseEntry(BaseModel):
    """The investable names from `from_date` until the next entry."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_date: str = Field(alias="from", description="YYYY-MM-DD this membership takes effect")
    tickers: list[str] = Field(min_length=1)

    @field_validator("from_date", mode="before")
    @classmethod
    def _iso_date(cls, value: object) -> str:
        """YAML parses a bare 2024-01-05 into a date; accept both spellings,
        and insist on YYYY-MM-DD so string comparison orders correctly."""
        if isinstance(value, _date):
            return value.isoformat()
        if not isinstance(value, str):
            raise ValueError(f"`from` must be a YYYY-MM-DD date, got {value!r}")
        try:
            _date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"`from` must be a YYYY-MM-DD date, got {value!r}") from exc
        return value

    @field_validator("tickers")
    @classmethod
    def _normalize(cls, tickers: list[str]) -> list[str]:
        return normalize_tickers(tickers)


class DatedUniverse(BaseModel):
    """Point-in-time membership: sorted snapshots, each in force until the next."""

    model_config = ConfigDict(extra="forbid")

    entries: list[UniverseEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _sorted_and_unique(self) -> DatedUniverse:
        dates = [e.from_date for e in self.entries]
        if len(set(dates)) != len(dates):
            raise ValueError(f"duplicate `from` dates in universe: {sorted(dates)}")
        self.entries.sort(key=lambda e: e.from_date)
        return self

    @property
    def start(self) -> str:
        """The first date this universe says anything about."""
        return self.entries[0].from_date

    @property
    def tickers(self) -> list[str]:
        """Every name that was ever a member, in order of first appearance."""
        seen: list[str] = []
        for entry in self.entries:
            for t in entry.tickers:
                if t not in seen:
                    seen.append(t)
        return seen

    def as_of(self, date: str) -> list[str]:
        """Members on *date* (YYYY-MM-DD). Raises before the first entry:
        a backtest cannot guess who was investable before the file starts."""
        current = None
        for entry in self.entries:
            if entry.from_date <= date:
                current = entry
            else:
                break
        if current is None:
            raise ValueError(
                f"universe has no membership as of {date} — its first entry "
                f"starts {self.start}"
            )
        return list(current.tickers)


Universe = Union[list[str], DatedUniverse]


def normalize_tickers(tickers: list[str]) -> list[str]:
    """Upper-cased, de-duped, order preserved. Empty raises."""
    out: list[str] = []
    for ticker in tickers:
        upper = ticker.strip().upper()
        if upper and upper not in out:
            out.append(upper)
    if not out:
        raise ValueError("universe is empty — a run needs at least one ticker")
    return out


def load_universe(path: str | Path) -> DatedUniverse:
    """Load a dated universe from YAML: a list of `{from, tickers}` entries,
    or a mapping with an `entries` key holding that list."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if isinstance(data, list):
        data = {"entries": data}
    return DatedUniverse(**data)


def survivorship_warning(universe: Universe, start: str) -> str | None:
    """The warning a backtest carries when its universe was hand-picked."""
    if isinstance(universe, DatedUniverse):
        return None
    names = ", ".join(universe[:6]) + (", …" if len(universe) > 6 else "")
    return (
        f"Survivorship bias: the universe ({names}) was chosen today and "
        f"backtested from {start}. It holds only names known to have survived, "
        "so the curve is biased upward. Backtest over a dated universe file "
        "(--universe FILE) for point-in-time membership."
    )
