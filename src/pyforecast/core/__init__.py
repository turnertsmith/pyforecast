"""Core decline curve analysis models and fitting."""

from .models import HyperbolicModel, ForecastResult
from .fitting import DeclineFitter, FittingConfig
from .selection import evaluate_fit_quality
from .regime_detection import RegimeDetectionConfig, detect_regime_change_improved

__all__ = [
    "HyperbolicModel",
    "ForecastResult",
    "DeclineFitter",
    "FittingConfig",
    "evaluate_fit_quality",
    "RegimeDetectionConfig",
    "detect_regime_change_improved",
]
