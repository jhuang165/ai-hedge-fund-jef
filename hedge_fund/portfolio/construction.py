"""Portfolio construction — blend analyst views into target weights.

This is the fund's portfolio manager: it takes every analyst's Signal and
produces one target weight per ticker. Pure arithmetic, no I/O — given the
same signals (and the same vols) it always produces the same book.

v0 policy: conviction-weighted, risk-scaled. Capital flows to tickers in
proportion to their blended conviction divided by their realized
volatility, scaled so the whole book deploys `gross_target`. Two names the
desk likes equally get equal RISK, not equal dollars — a 90%-vol name gets
a sixth of the dollars a 15%-vol name does. Without vols the policy
degrades to plain conviction weighting.

Known wart, accepted deliberately: the cross-sectional normalization
ignores *absolute* conviction — a lone weak view would receive the full
gross target, which the risk stage then clamps ("conviction requests, risk
disposes"). A min-conviction floor is the obvious knob once `evaluate()`
can measure it.
"""

from __future__ import annotations

from statistics import median

from pydantic import BaseModel

from hedge_fund.models import Signal

# Realized vol below this is treated as this: a sleepy or thinly traded
# name must not absorb the whole book because its vol printed near zero.
VOL_FLOOR = 0.05


class BlendResult(BaseModel):
    """Per-ticker blended convictions and the target weights they imply."""

    convictions: dict[str, float]  # blended view per ticker, pre-scaling
    weights: dict[str, float]      # target weight per ticker; sum(|w|) <= gross_target


def blend_signals(
    signals: list[Signal],
    model_weights: dict[str, float],
    gross_target: float,
    market_neutral: bool = False,
    vols: dict[str, float] | None = None,
) -> BlendResult:
    """Blend model signals into target weights.

    Per ticker, the conviction is a weighted mean over *voting* models:

        conviction_t = sum(w_m * value_mt) / sum(w_m)

    An abstained signal (metadata.abstained is True — LLM failure or
    insufficient data) is excluded from numerator AND denominator: "no
    opinion" must not masquerade as "opinion: neutral". A non-abstained 0.0
    (e.g. PEAD outside its window) is a real neutral vote and dilutes.

    With vols (ticker -> annualized realized vol), each conviction is
    divided by its name's vol, floored at VOL_FLOOR, and expressed relative
    to the median vol so a typical name is unchanged. A ticker with no vol
    is treated as median-vol. None or empty means no risk scaling.

    With market_neutral, the (risk-scaled) convictions are demeaned
    cross-sectionally before scaling — what a pod does with analyst
    rankings: long the names the desk likes most *relative to the others*,
    short the least liked, sleeve sums to zero dollars. Uniform convictions
    demean to a flat book.

    Cross-sectionally, weights are the result normalized to the gross
    target: weight_t = scaled_t / sum(|scaled|) * gross_target. All-zero
    convictions produce an all-zero (flat) book.

    Args:
        signals:        Every model's Signal for every ticker this cycle.
        model_weights:  model_name -> blend weight from the StrategySpec.
        gross_target:   Desired sum of |weights| when views exist.
        market_neutral: Demean convictions before scaling (dollar-neutral).
        vols:           ticker -> annualized realized vol, for risk scaling.
    """
    weighted_sum: dict[str, float] = {}
    weight_total: dict[str, float] = {}
    for signal in signals:
        if signal.metadata.get("abstained") is True:
            continue
        w = model_weights[signal.model_name]
        weighted_sum[signal.ticker] = weighted_sum.get(signal.ticker, 0.0) + w * signal.value
        weight_total[signal.ticker] = weight_total.get(signal.ticker, 0.0) + w

    tickers = sorted({s.ticker for s in signals})
    convictions = {
        t: (weighted_sum[t] / weight_total[t]) if weight_total.get(t) else 0.0
        for t in tickers
    }

    scaled = convictions
    if vols:
        scaled = {t: c * factor for (t, c), factor in
                  zip(scaled.items(), _risk_factors(tickers, vols).values())}

    if market_neutral and tickers:
        mean = sum(scaled.values()) / len(scaled)
        scaled = {t: c - mean for t, c in scaled.items()}

    # Threshold, not == 0: demeaning identical convictions leaves ~1e-16
    # residue, and dividing by it would normalize noise into a full book.
    gross = sum(abs(c) for c in scaled.values())
    if gross < 1e-9:
        weights = {t: 0.0 for t in tickers}
    else:
        weights = {t: c / gross * gross_target for t, c in scaled.items()}

    return BlendResult(convictions=convictions, weights=weights)


def _risk_factors(tickers: list[str], vols: dict[str, float]) -> dict[str, float]:
    """median_vol / vol_t per ticker (floored); 1.0 where the vol is unknown."""
    floored = {t: max(vols[t], VOL_FLOOR) for t in tickers if vols.get(t) is not None}
    if not floored:
        return {t: 1.0 for t in tickers}
    typical = median(floored.values())
    return {t: (typical / floored[t]) if t in floored else 1.0 for t in tickers}
