"""Performance arithmetic shared by the fund backtest and its attribution.

Pure functions over NAV curves. One definition of Sharpe and drawdown, so a
sleeve's numbers are computed exactly the way the fund's are and the two
can be read side by side.
"""

from __future__ import annotations

from datetime import date as _date

import numpy as np

PERIODS_PER_YEAR = {"daily": 252, "weekly": 52, "monthly": 12}


def period_returns(curve: list[float]) -> np.ndarray:
    """Simple returns between consecutive points of a NAV curve."""
    arr = np.asarray(curve, dtype=float)
    return arr[1:] / arr[:-1] - 1


def sharpe_ratio(returns: np.ndarray, cadence: str) -> float:
    """Annualized mean/std of per-period returns; 0.0 when undefined."""
    if len(returns) > 1 and float(returns.std(ddof=1)) > 0:
        return float(returns.mean() / returns.std(ddof=1)) * float(
            np.sqrt(PERIODS_PER_YEAR[cadence])
        )
    return 0.0


def max_drawdown(curve: list[float]) -> float:
    """Largest peak-to-trough fall as a fraction of the peak."""
    peak = curve[0]
    worst = 0.0
    for value in curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > worst:
            worst = drawdown
    return worst


def annualized_return(total_return: float, start: str, end: str) -> float:
    """Compound *total_return* over [start, end] to a per-year rate."""
    calendar_days = (_date.fromisoformat(end) - _date.fromisoformat(start)).days
    years = max(calendar_days / 365.25, 0.01)
    return (1 + total_return) ** (1 / years) - 1
