"""Refinement module for measuring, logging, and improving decline curve fit quality.

This module provides capabilities to:
- Measure forecast accuracy through hindcast validation
- Log fit parameters and metrics to persistent storage
- Analyze residuals for systematic fit errors
- Learn optimal fitting parameters from accumulated results

All features are disabled by default and observation-only - they don't modify
the core fitting behavior, only observe and report on it.

Usage:
    from pyforecast.refinement import (
        # Data classes
        FitLogRecord,
        HindcastResult,
        ResidualDiagnostics,
        # Validators and analyzers
        HindcastValidator,
        ResidualAnalyzer,
        FitLogger,
        ParameterLearner,
        # Storage
        FitLogStorage,
    )

    # Run hindcast validation
    validator = HindcastValidator(holdout_months=6)
    result = validator.validate(well, product, fitter)

    # Log fit results
    logger = FitLogger(storage_path="~/.pyforecast/fit_logs.db")
    logger.log(fit_result, well, product)

    # Analyze residuals
    analyzer = ResidualAnalyzer()
    diagnostics = analyzer.analyze(residuals)
"""

from .schemas import (
    FitLogRecord,
    HindcastResult,
    ResidualDiagnostics,
)
from .storage import FitLogStorage
from .hindcast import HindcastValidator
from .residual_analysis import ResidualAnalyzer
from .fit_logger import FitLogger
from .parameter_learning import ParameterLearner

__all__ = [
    # Data classes
    "FitLogRecord",
    "HindcastResult",
    "ResidualDiagnostics",
    # Storage
    "FitLogStorage",
    # Validators and analyzers
    "HindcastValidator",
    "ResidualAnalyzer",
    "FitLogger",
    "ParameterLearner",
]
