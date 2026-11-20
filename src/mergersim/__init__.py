"""Calibrated merger simulation (logit demand, Bertrand pricing)."""

from .merger import diversion_ratio, hhi, simulate_merger, upward_pricing_pressure
from .pricing import bertrand_equilibrium

__all__ = [
    "bertrand_equilibrium",
    "diversion_ratio",
    "hhi",
    "simulate_merger",
    "upward_pricing_pressure",
]
