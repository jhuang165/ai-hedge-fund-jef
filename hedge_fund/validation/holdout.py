"""Hold-out split — does the mandate survive a window it was not tuned on?

Every parameter in a mandate — a model's scale, a strategy's weight, the
band, the caps — was chosen while looking at some backtest. The cheapest
honest check is to backtest the halves separately: the half you looked at
(train) and the half you did not (test). A fund that only works on the
first half was fitted, not found.

The two halves are independent backtests, each starting flat at the
mandate's capital, so the test half is what a fresh launch on the split
date would have seen. Nothing here decides for you; it reports both
halves side by side and how much the Sharpe decayed between them.
"""

from __future__ import annotations

from pydantic import BaseModel

from hedge_fund.backtesting.fund import FundBacktestMetrics, backtest_fund
from hedge_fund.data.protocol import DataClient
from hedge_fund.fund.spec import Fund
from hedge_fund.fund.universe import Universe

# A test Sharpe below this fraction of train's is called a degradation.
_DECAY_FRACTION = 0.5


class HoldoutResult(BaseModel):
    fund: str
    split: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train: FundBacktestMetrics
    test: FundBacktestMetrics
    sharpe_decay: float              # train sharpe minus test sharpe
    holds: bool
    verdict: str


def holdout(
    fund: Fund,
    start: str,
    split: str,
    end: str,
    data_client: DataClient,
    universe: Universe,
) -> HoldoutResult:
    """Backtest [start, split) and [split, end] separately and compare."""
    if not start < split < end:
        raise ValueError(f"need start < split < end, got {start} < {split} < {end}")
    train = backtest_fund(fund, start, _day_before(split), data_client, universe)
    test = backtest_fund(fund, split, end, data_client, universe)
    decay = train.metrics.sharpe_ratio - test.metrics.sharpe_ratio
    holds, verdict = _judge(train.metrics, test.metrics)
    return HoldoutResult(
        fund=fund.spec.name,
        split=split,
        train_start=train.start, train_end=train.end,
        test_start=test.start, test_end=test.end,
        train=train.metrics, test=test.metrics,
        sharpe_decay=round(decay, 4),
        holds=holds,
        verdict=verdict,
    )


def _judge(train: FundBacktestMetrics, test: FundBacktestMetrics) -> tuple[bool, str]:
    if train.sharpe_ratio <= 0:
        return (test.sharpe_ratio > 0,
                "train half did not work; nothing to validate"
                + (" — but the test half did" if test.sharpe_ratio > 0 else ""))
    if test.sharpe_ratio <= 0:
        return False, "does not survive out of sample: positive in train, not in test"
    if test.sharpe_ratio < _DECAY_FRACTION * train.sharpe_ratio:
        return False, (f"degrades out of sample: test sharpe is "
                       f"{test.sharpe_ratio / train.sharpe_ratio:.0%} of train's")
    return True, "holds out of sample"


def _day_before(day: str) -> str:
    from datetime import date as _date
    from datetime import timedelta
    return (_date.fromisoformat(day) - timedelta(days=1)).isoformat()
