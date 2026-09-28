"""Risk limits — hard caps the analysts cannot override.

"Conviction requests, risk disposes": portfolio construction proposes target
weights, and this stage clamps them against the fund's limits. Everything
here is deterministic arithmetic — the LLM's influence over the book ends at
the Signal, and no clamp is ever negotiable.

Exposure removed by a clamp is NOT redistributed to other names; it stays in
cash. Redistributing would let the risk stage *increase* positions, which
inverts its job. The net cap obeys the same rule: it shrinks the side that
is too big rather than adding to the other side.

The drawdown limit is a kill-switch, not a dial. When the book has fallen
that far from its peak, the fund closes to flat — and because a flat book
never recovers, every later cycle finds the same drawdown and stays flat.
Restarting a killed fund is a human decision, made by editing the mandate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RiskLimits(BaseModel):
    """The fund's hard limits, set in its FundSpec."""

    model_config = ConfigDict(extra="forbid")

    max_position_pct: float = Field(
        gt=0, le=1.0, description="max |weight| per ticker, as a fraction of equity"
    )
    max_gross_exposure: float = Field(
        gt=0, description="max sum of |weights| across the book (1.0 = unlevered)"
    )
    max_net_exposure: float | None = Field(
        default=None, ge=0,
        description="max |sum of weights| — how far the book may lean long or "
        "short (0 = dollar-neutral, 1.0 = fully long allowed). None: uncapped",
    )
    max_drawdown_pct: float | None = Field(
        default=None, gt=0, lt=1.0,
        description="kill-switch: when equity has fallen this fraction from "
        "its peak NAV, close to flat and stay flat. None: no kill-switch",
    )


ClampKind = Literal[
    "max_position_pct", "max_net_exposure", "max_gross_exposure", "max_drawdown_pct",
]


class ClampEvent(BaseModel):
    """One limit firing — recorded so every clamp is explainable."""

    limit: ClampKind
    ticker: str | None = None  # None for the portfolio-level clamps
    before: float
    after: float


class RiskResult(BaseModel):
    """Clamped weights plus the audit trail of every limit that fired."""

    weights: dict[str, float]
    clamps: list[ClampEvent]

    @property
    def killed(self) -> bool:
        """True when the drawdown kill-switch flattened the book."""
        return any(c.limit == "max_drawdown_pct" for c in self.clamps)


def apply_limits(
    weights: dict[str, float],
    limits: RiskLimits,
    *,
    drawdown: float | None = None,
) -> RiskResult:
    """Clamp target weights against the fund's hard limits.

    `drawdown` is the fund's current fall from its peak NAV as a fraction
    (0.12 = 12% below the high-water mark); None when the caller has no
    history to measure it from, which disables the kill-switch for this
    call.

    Order matters and makes the sequence idempotent:
    0. Kill-switch: if drawdown >= max_drawdown_pct, every weight becomes
       zero. One ClampEvent (before = requested gross, after = 0) and
       nothing else runs — there is nothing left to cap.
    1. Per-ticker cap: any |weight| above max_position_pct is clamped to the
       cap, preserving sign. One ClampEvent per clamped ticker.
    2. Net cap: if |sum of weights| exceeds max_net_exposure, the dominant
       side is scaled down until the net sits exactly at the cap. Only
       shrinks, so it cannot re-violate the per-ticker cap.
    3. Gross cap: if the summed |weights| still exceed max_gross_exposure,
       every weight is scaled down proportionally. Scaling both sides by
       the same factor shrinks the net too, so it cannot re-violate the
       net cap.
    """
    clamps: list[ClampEvent] = []

    if (
        limits.max_drawdown_pct is not None
        and drawdown is not None
        and drawdown >= limits.max_drawdown_pct
    ):
        gross = sum(abs(w) for w in weights.values())
        clamps.append(ClampEvent(limit="max_drawdown_pct", before=gross, after=0.0))
        return RiskResult(weights={t: 0.0 for t in weights}, clamps=clamps)

    clamped: dict[str, float] = {}
    for ticker in sorted(weights):
        w = weights[ticker]
        cap = limits.max_position_pct
        if abs(w) > cap:
            new_w = cap if w > 0 else -cap
            clamps.append(ClampEvent(
                limit="max_position_pct", ticker=ticker, before=w, after=new_w,
            ))
            clamped[ticker] = new_w
        else:
            clamped[ticker] = w

    if limits.max_net_exposure is not None:
        net = sum(clamped.values())
        if abs(net) > limits.max_net_exposure + 1e-12:
            longs = sum(w for w in clamped.values() if w > 0)
            shorts = -sum(w for w in clamped.values() if w < 0)
            cap = limits.max_net_exposure
            if net > 0:
                # Shrink the longs until longs - shorts == cap.
                factor = (cap + shorts) / longs
                clamped = {t: (w * factor if w > 0 else w) for t, w in clamped.items()}
            else:
                factor = (cap + longs) / shorts
                clamped = {t: (w * factor if w < 0 else w) for t, w in clamped.items()}
            clamps.append(ClampEvent(
                limit="max_net_exposure", before=net, after=cap if net > 0 else -cap,
            ))

    gross = sum(abs(w) for w in clamped.values())
    if gross > limits.max_gross_exposure:
        scale = limits.max_gross_exposure / gross
        clamped = {t: w * scale for t, w in clamped.items()}
        clamps.append(ClampEvent(
            limit="max_gross_exposure", before=gross, after=limits.max_gross_exposure,
        ))

    return RiskResult(weights=clamped, clamps=clamps)
