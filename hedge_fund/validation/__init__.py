"""v2 validation — is the backtest telling the truth?

Ships: a hold-out split (`holdout`) and a parameter sweep (`sweep`), the
cheap checks that catch most fitted results. Planned: Combinatorial Purged
Cross-Validation (CPCV) and the Probability of Backtest Overfitting (PBO).
"""

from hedge_fund.validation.holdout import HoldoutResult, holdout
from hedge_fund.validation.sensitivity import (
    SweepPoint,
    SweepResult,
    parse_sweep,
    set_path,
    sweep,
)

__all__ = [
    "HoldoutResult",
    "SweepPoint",
    "SweepResult",
    "holdout",
    "parse_sweep",
    "set_path",
    "sweep",
]
