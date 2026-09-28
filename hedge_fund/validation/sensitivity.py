"""Parameter sweep — how much does the result depend on the knob?

A backtest is one point in parameter space. If moving a model's scale from
1.0 to 1.5 turns a Sharpe of 1.2 into 0.1, the 1.2 was the knob, not the
edge. The sweep re-runs the backtest over a grid of values for any field
in the mandate — reached by dotted path, `strategies.0.models.0.params.scale`
— and reports every point, so the shape of the surface is visible, not
just its peak.

With a split, each point is also backtested on the test half alone, so a
setting that only wins in train shows up as such. The verdict is
deliberately blunt: a surface whose Sharpe changes sign across the grid,
or swings by more than a full point, is fragile.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any, Callable

import yaml
from pydantic import BaseModel

from hedge_fund.backtesting.fund import FundBacktestMetrics, backtest_fund
from hedge_fund.data.protocol import DataClient
from hedge_fund.fund.spec import Fund, FundSpec
from hedge_fund.fund.universe import Universe

# A Sharpe range wider than this across the grid is called fragile.
_FRAGILE_RANGE = 1.0


class SweepPoint(BaseModel):
    overrides: dict[str, Any]
    metrics: FundBacktestMetrics
    test: FundBacktestMetrics | None = None   # the test half alone, when a split was given


class SweepResult(BaseModel):
    fund: str
    paths: list[str]
    split: str | None
    points: list[SweepPoint]
    sharpe_min: float
    sharpe_max: float
    sharpe_median: float
    share_positive: float            # fraction of points with a positive Sharpe
    fragile: bool
    verdict: str


def sweep(
    spec: FundSpec,
    overrides: dict[str, list[Any]],
    start: str,
    end: str,
    data_client: DataClient,
    universe: Universe,
    *,
    split: str | None = None,
    build_fund: Callable[[FundSpec], Fund] = Fund,
) -> SweepResult:
    """Backtest every combination of *overrides* (path -> values).

    `build_fund` turns each modified spec into a Fund; tests inject fakes
    through it, production lets the registry staff the desk.
    """
    if not overrides:
        raise ValueError("sweep needs at least one path with values")
    paths = list(overrides)
    points: list[SweepPoint] = []
    for combo in itertools.product(*(overrides[p] for p in paths)):
        chosen = dict(zip(paths, combo))
        data = spec.model_dump()
        for path, value in chosen.items():
            set_path(data, path, value)
        fund = build_fund(FundSpec(**data))
        full = backtest_fund(fund, start, end, data_client, universe)
        test = None
        if split is not None:
            test = backtest_fund(build_fund(FundSpec(**data)), split, end,
                                 data_client, universe).metrics
        points.append(SweepPoint(overrides=chosen, metrics=full.metrics, test=test))

    sharpes = sorted(p.metrics.sharpe_ratio for p in points)
    lo, hi = sharpes[0], sharpes[-1]
    median = sharpes[len(sharpes) // 2] if len(sharpes) % 2 else (
        (sharpes[len(sharpes) // 2 - 1] + sharpes[len(sharpes) // 2]) / 2)
    positive = sum(1 for s_ in sharpes if s_ > 0) / len(sharpes)
    fragile, verdict = _judge(lo, hi, positive)
    return SweepResult(
        fund=spec.name, paths=paths, split=split, points=points,
        sharpe_min=round(lo, 4), sharpe_max=round(hi, 4), sharpe_median=round(median, 4),
        share_positive=round(positive, 4), fragile=fragile, verdict=verdict,
    )


def set_path(data: dict, path: str, value: Any) -> None:
    """Assign *value* at a dotted *path* into nested dicts and lists, in place.

    `strategies.0.models.1.weight` walks dict keys and list indexes. A
    missing key is created (so `params.scale` can be set on a model with
    no params); a bad index or a key on a list raises.
    """
    parts = path.split(".")
    node: Any = data
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        if isinstance(node, list):
            try:
                index = int(part)
            except ValueError:
                raise ValueError(f"{path}: {part!r} is not a list index") from None
            if not 0 <= index < len(node):
                raise ValueError(f"{path}: index {index} out of range")
            if last:
                node[index] = value
            else:
                node = node[index]
        elif isinstance(node, dict):
            if last:
                node[part] = value
            else:
                node = node.setdefault(part, {})
        else:
            raise ValueError(f"{path}: cannot descend into {type(node).__name__} at {part!r}")


def parse_sweep(specs: list[str]) -> dict[str, list[Any]]:
    """`PATH=v1,v2,...` strings (the CLI form) -> {path: [values]}.

    Values are read as YAML scalars, so 0.5 is a float, 2 an int, true a
    bool, and anything else a string.
    """
    overrides: dict[str, list[Any]] = {}
    for item in specs:
        if "=" not in item:
            raise ValueError(f"sweep {item!r}: expected PATH=V1,V2,...")
        path, _, values = item.partition("=")
        parsed = [yaml.safe_load(v.strip()) for v in values.split(",") if v.strip()]
        if not parsed:
            raise ValueError(f"sweep {item!r}: no values")
        overrides[path.strip()] = deepcopy(parsed)
    return overrides


def _judge(lo: float, hi: float, positive: float) -> tuple[bool, str]:
    if lo < 0 < hi:
        return True, "fragile: the Sharpe changes sign across the grid"
    if hi - lo > _FRAGILE_RANGE:
        return True, f"fragile: the Sharpe swings by {hi - lo:.1f} across the grid"
    if positive == 0:
        return False, "robustly negative: no point on the grid works"
    return False, "robust: every point on the grid tells the same story"
