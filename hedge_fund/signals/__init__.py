"""Alpha models — view-forming components of the quant stack.

See hedge_fund/signals/base.py for the AlphaModel / QuantModel interface.
Concrete models register here as they are implemented. Two flavors, one
interface: LLM investor agents (persona system prompts on LLMAgent) and
quant models (pure math).
"""

from __future__ import annotations

from hedge_fund.signals.base import AlphaModel, QuantModel
from hedge_fund.signals.buffett import BuffettAgent
from hedge_fund.signals.druckenmiller import DruckenmillerAgent
from hedge_fund.signals.graham import GrahamAgent
from hedge_fund.signals.insider import InsiderFlowModel
from hedge_fund.signals.llm_agent import LLMAgent
from hedge_fund.signals.lynch import LynchAgent
from hedge_fund.signals.momentum import MomentumModel
from hedge_fund.signals.munger import MungerAgent
from hedge_fund.signals.pead import PEADModel
from hedge_fund.signals.quality_value import QualityValueModel
from hedge_fund.signals.reversal import MeanReversionModel

ALPHA_MODEL_REGISTRY: dict[str, type[AlphaModel]] = {
    # Quant models
    "pead": PEADModel,
    "momentum": MomentumModel,
    "mean-reversion": MeanReversionModel,
    "insider-flow": InsiderFlowModel,
    "quality-value": QualityValueModel,
    # LLM investor agents
    "buffett": BuffettAgent,
    "munger": MungerAgent,
    "graham": GrahamAgent,
    "lynch": LynchAgent,
    "druckenmiller": DruckenmillerAgent,
}

# Every quant model, in the order a research desk readout lists them.
QUANT_MODEL_NAMES: tuple[str, ...] = tuple(
    name for name, cls in ALPHA_MODEL_REGISTRY.items() if not issubclass(cls, LLMAgent)
)

__all__ = [
    "AlphaModel",
    "QuantModel",
    "LLMAgent",
    "BuffettAgent",
    "MungerAgent",
    "GrahamAgent",
    "LynchAgent",
    "DruckenmillerAgent",
    "PEADModel",
    "MomentumModel",
    "MeanReversionModel",
    "InsiderFlowModel",
    "QualityValueModel",
    "QUANT_MODEL_NAMES",
    "ALPHA_MODEL_REGISTRY",
]
