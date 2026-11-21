"""mergersim: calibrated merger simulation (logit demand + Nash-Bertrand)."""

from .demand import LinearDemand, LogitDemand
from .market import Market, equilibrium
from .simulation import MergerResult, report, simulate_merger, upward_pricing_pressure

__all__ = [
    "LinearDemand",
    "LogitDemand",
    "Market",
    "equilibrium",
    "MergerResult",
    "simulate_merger",
    "upward_pricing_pressure",
    "report",
]

__version__ = "0.1.0"
