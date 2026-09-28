"""Quality-value alpha model — cheap versus its own history, and good.

A systematic take on what the value personas do by hand, using the same
point-in-time financial metrics they see. Two halves, averaged:

- **Value** — where today's multiples sit against the stock's OWN filed
  history: the percentile of the current P/E, P/B, EV/EBITDA (lower is
  cheaper) and FCF yield (higher is cheaper) among the trailing periods.
  A stock at the cheap end of its own range scores +1, the expensive end
  -1. Own-history percentiles need no cross-sectional universe and stay
  point-in-time by construction.
- **Quality** — the level of the latest filed ROE, ROIC, operating margin
  and leverage, each mapped through tanh around a sensible anchor (ROE of
  10% is "fine", 25% is "great"; debt/equity above 1 costs points).

Cheap-and-good is the strongest long; expensive-and-deteriorating the
strongest short. Abstains below four filed periods, like the snapshot the
LLM personas use.
"""

from __future__ import annotations

import math

from hedge_fund.data.models import FinancialMetrics
from hedge_fund.data.protocol import DataClient
from hedge_fund.features.snapshot import MIN_PERIODS
from hedge_fund.features.technicals import last_close
from hedge_fund.features.valuation import reprice
from hedge_fund.models import Signal
from hedge_fund.signals.base import QuantModel

# (attribute, lower-is-cheaper)
_VALUE_FIELDS: tuple[tuple[str, bool], ...] = (
    ("price_to_earnings_ratio", True),
    ("price_to_book_ratio", True),
    ("enterprise_value_to_ebitda_ratio", True),
    ("free_cash_flow_yield", False),
)

# (attribute, anchor, scale, sign): score = sign * tanh((x - anchor) / scale)
_QUALITY_FIELDS: tuple[tuple[str, float, float, float], ...] = (
    ("return_on_equity", 0.10, 0.15, 1.0),
    ("return_on_invested_capital", 0.08, 0.12, 1.0),
    ("operating_margin", 0.10, 0.15, 1.0),
    ("debt_to_equity", 1.0, 1.5, -1.0),
)


class QualityValueModel(QuantModel):
    """Own-history value percentiles blended with absolute quality levels."""

    def __init__(self, *, periods: int = 20, value_weight: float = 0.5) -> None:
        self._periods = periods
        self._value_weight = value_weight

    @property
    def name(self) -> str:
        return "quality-value"

    def predict(self, ticker: str, date: str, data_client: DataClient) -> Signal:
        metrics = data_client.get_financial_metrics(ticker, date, period="ttm", limit=self._periods)
        if len(metrics) < MIN_PERIODS:
            return self._abstain(
                ticker, date, f"only {len(metrics)} filed periods (need {MIN_PERIODS})"
            )
        # Today's multiples at today's price: the filed row's valuation was
        # struck at filing time. The history it is ranked against stays as
        # filed — that is what the stock's own range looked like.
        latest = metrics[0]
        mark = last_close(ticker, date, data_client)
        if mark is not None:
            latest = reprice(latest, mark[1])

        value_scores = _value_scores(latest, metrics)
        quality_scores = _quality_scores(latest)
        if not value_scores and not quality_scores:
            return self._abstain(ticker, date, "no valuation or quality metrics filed")

        value = _mean(value_scores)
        quality = _mean(quality_scores)
        if value is None:
            composite = quality
        elif quality is None:
            composite = value
        else:
            composite = self._value_weight * value + (1 - self._value_weight) * quality

        return Signal(
            model_name=self.name,
            ticker=ticker,
            date=date,
            value=self._normalize_to_signal(composite),
            reasoning=_reasoning(latest, value, quality, value_scores, quality_scores),
            components={
                **({"value": value} if value is not None else {}),
                **({"quality": quality} if quality is not None else {}),
                **{f"value.{k}": v for k, v in value_scores.items()},
                **{f"quality.{k}": v for k, v in quality_scores.items()},
            },
            metadata={
                "report_period": latest.report_period,
                "filing_date": latest.filing_date,
                "n_periods": len(metrics),
                "price": mark[1] if mark is not None else None,
            },
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _value_scores(latest: FinancialMetrics, history: list[FinancialMetrics]) -> dict[str, float]:
    """Per-multiple score in [-1, +1]: +1 at the cheapest end of the stock's
    own PRIOR history, -1 at the most expensive. Negative P/E and P/B
    (losses, negative equity) are not "cheap" — those multiples are skipped."""
    scores: dict[str, float] = {}
    for field, lower_is_cheaper in _VALUE_FIELDS:
        current = getattr(latest, field)
        if current is None or (lower_is_cheaper and current <= 0):
            continue
        prior = [
            v for m in history[1:]
            if (v := getattr(m, field)) is not None and not (lower_is_cheaper and v <= 0)
        ]
        if len(prior) < MIN_PERIODS - 1:
            continue
        pct = QuantModel._percentile_rank(current, prior) / 100.0  # share of prior periods below current
        # Lower-is-cheaper: at the bottom of its range pct≈0 -> +1.
        scores[field] = (1 - 2 * pct) if lower_is_cheaper else (2 * pct - 1)
    return scores


def _quality_scores(latest: FinancialMetrics) -> dict[str, float]:
    scores: dict[str, float] = {}
    for field, anchor, scale, sign in _QUALITY_FIELDS:
        x = getattr(latest, field)
        if x is None:
            continue
        scores[field] = sign * math.tanh((x - anchor) / scale)
    return scores


def _mean(scores: dict[str, float]) -> float | None:
    return sum(scores.values()) / len(scores) if scores else None


def _reasoning(
    latest: FinancialMetrics,
    value: float | None,
    quality: float | None,
    value_scores: dict[str, float],
    quality_scores: dict[str, float],
) -> str:
    parts = []
    if value is not None:
        cheapest_multiple = max(value_scores, key=value_scores.get)
        richest_multiple = min(value_scores, key=value_scores.get)
        spread = (
            f"on {_short(cheapest_multiple)} only" if len(value_scores) == 1
            else f"cheapest on {_short(cheapest_multiple)}, richest on {_short(richest_multiple)}"
        )
        parts.append(
            f"value {value:+.2f} vs own history "
            f"(P/E {_num(latest.price_to_earnings_ratio)}, P/B {_num(latest.price_to_book_ratio)}, "
            f"FCF yield {_pct(latest.free_cash_flow_yield)}; {spread})"
        )
    if quality is not None:
        parts.append(
            f"quality {quality:+.2f} "
            f"(ROE {_pct(latest.return_on_equity)}, ROIC {_pct(latest.return_on_invested_capital)}, "
            f"op margin {_pct(latest.operating_margin)}, D/E {_num(latest.debt_to_equity)})"
        )
    return f"filed {latest.filing_date or latest.report_period}: " + "; ".join(parts)


_SHORT = {
    "price_to_earnings_ratio": "P/E",
    "price_to_book_ratio": "P/B",
    "enterprise_value_to_ebitda_ratio": "EV/EBITDA",
    "free_cash_flow_yield": "FCF yield",
}


def _short(field: str) -> str:
    return _SHORT.get(field, field)


def _num(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1%}"
