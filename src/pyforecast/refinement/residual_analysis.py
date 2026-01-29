"""Residual analysis for detecting systematic fit errors.

Analyzes residuals (actual - predicted) to identify patterns that indicate
fitting issues like autocorrelation, early/late bias, and underfitting.
"""

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import numpy as np

from .schemas import ResidualDiagnostics
from ..validation.result import (
    ValidationIssue,
    ValidationResult,
    IssueCategory,
    IssueSeverity,
)

if TYPE_CHECKING:
    from ..core.models import ForecastResult

logger = logging.getLogger(__name__)


# Validation issue codes for residual analysis
RESIDUAL_ISSUE_CODES = {
    "RD001": "Significant autocorrelation in residuals",
    "RD002": "Systematic early/late bias pattern",
    "RD003": "Non-zero mean residuals",
    "RD004": "High residual variance",
}


@dataclass
class ResidualAnalysisConfig:
    """Configuration for residual analysis.

    Attributes:
        autocorr_threshold: Threshold for flagging autocorrelation (default 0.5)
        dw_low_threshold: Durbin-Watson lower threshold (default 1.5)
        dw_high_threshold: Durbin-Watson upper threshold (default 2.5)
        bias_threshold: Threshold for flagging early/late bias (default 0.15)
        mean_threshold: Threshold for flagging non-zero mean (default 0.1)
    """

    autocorr_threshold: float = 0.5
    dw_low_threshold: float = 1.5
    dw_high_threshold: float = 2.5
    bias_threshold: float = 0.15
    mean_threshold: float = 0.1


class ResidualAnalyzer:
    """Analyzes fit residuals to detect systematic patterns.

    Computes residual diagnostics and generates validation issues
    when concerning patterns are detected.

    Example:
        analyzer = ResidualAnalyzer()
        diagnostics = analyzer.analyze(actual, predicted)

        if diagnostics.has_systematic_pattern:
            print("Warning: Fit shows systematic errors")
            print(f"  Durbin-Watson: {diagnostics.durbin_watson:.2f}")
            print(f"  Early bias: {diagnostics.early_bias:.1%}")

        # Get validation issues
        issues = analyzer.get_validation_issues(diagnostics, well_id, product)
    """

    def __init__(self, config: ResidualAnalysisConfig | None = None):
        """Initialize analyzer.

        Args:
            config: Analysis configuration (uses defaults if None)
        """
        self.config = config or ResidualAnalysisConfig()

    def analyze(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> ResidualDiagnostics:
        """Analyze residuals from a fit.

        Args:
            actual: Actual observed values
            predicted: Model predicted values

        Returns:
            ResidualDiagnostics with computed metrics
        """
        return ResidualDiagnostics.compute(actual, predicted)

    def analyze_from_result(
        self,
        t: np.ndarray,
        q: np.ndarray,
        result: "ForecastResult",
    ) -> ResidualDiagnostics:
        """Analyze residuals from a ForecastResult.

        Args:
            t: Time array used for fitting
            q: Production data used for fitting
            result: ForecastResult from fitting

        Returns:
            ResidualDiagnostics with computed metrics
        """
        # Get the portion of data used for fitting
        regime_start = result.regime_start_idx
        t_fit = t[regime_start:] - t[regime_start]
        q_actual = q[regime_start:]

        # Get predicted values
        q_predicted = result.model.rate(t_fit)

        return self.analyze(q_actual, q_predicted)

    def get_validation_issues(
        self,
        diagnostics: ResidualDiagnostics,
        well_id: str,
        product: str,
    ) -> ValidationResult:
        """Generate validation issues from residual diagnostics.

        Args:
            diagnostics: ResidualDiagnostics to check
            well_id: Well identifier
            product: Product type

        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult(well_id=well_id, product=product)

        # Check for autocorrelation (Durbin-Watson)
        if diagnostics.durbin_watson < self.config.dw_low_threshold:
            result.add_issue(ValidationIssue(
                code="RD001",
                category=IssueCategory.FITTING_RESULT,
                severity=IssueSeverity.WARNING,
                message=f"Positive autocorrelation in residuals (DW={diagnostics.durbin_watson:.2f})",
                guidance=(
                    "Residuals show systematic patterns suggesting the model may be "
                    "underfitting. Consider adjusting b-factor bounds or checking for "
                    "regime changes."
                ),
                details={
                    "durbin_watson": diagnostics.durbin_watson,
                    "autocorr_lag1": diagnostics.autocorr_lag1,
                    "threshold": self.config.dw_low_threshold,
                },
            ))
        elif diagnostics.durbin_watson > self.config.dw_high_threshold:
            result.add_issue(ValidationIssue(
                code="RD001",
                category=IssueCategory.FITTING_RESULT,
                severity=IssueSeverity.WARNING,
                message=f"Negative autocorrelation in residuals (DW={diagnostics.durbin_watson:.2f})",
                guidance=(
                    "Residuals show oscillating patterns. This may indicate overfitting "
                    "or issues with the decline model specification."
                ),
                details={
                    "durbin_watson": diagnostics.durbin_watson,
                    "autocorr_lag1": diagnostics.autocorr_lag1,
                    "threshold": self.config.dw_high_threshold,
                },
            ))

        # Check for early/late bias
        if abs(diagnostics.early_bias) > self.config.bias_threshold:
            direction = "over" if diagnostics.early_bias < 0 else "under"
            result.add_issue(ValidationIssue(
                code="RD002",
                category=IssueCategory.FITTING_RESULT,
                severity=IssueSeverity.WARNING,
                message=f"Early period {direction}-prediction bias ({diagnostics.early_bias:.1%})",
                guidance=(
                    f"Model systematically {direction}-predicts in the early time period. "
                    "This may indicate incorrect initial rate (qi) or early decline behavior."
                ),
                details={
                    "early_bias": diagnostics.early_bias,
                    "late_bias": diagnostics.late_bias,
                    "threshold": self.config.bias_threshold,
                },
            ))

        if abs(diagnostics.late_bias) > self.config.bias_threshold:
            direction = "over" if diagnostics.late_bias < 0 else "under"
            result.add_issue(ValidationIssue(
                code="RD002",
                category=IssueCategory.FITTING_RESULT,
                severity=IssueSeverity.WARNING,
                message=f"Late period {direction}-prediction bias ({diagnostics.late_bias:.1%})",
                guidance=(
                    f"Model systematically {direction}-predicts in the late time period. "
                    "This may indicate incorrect terminal decline (Dmin) or b-factor."
                ),
                details={
                    "early_bias": diagnostics.early_bias,
                    "late_bias": diagnostics.late_bias,
                    "threshold": self.config.bias_threshold,
                },
            ))

        # Check for non-zero mean residuals
        mean_actual_approx = diagnostics.std * 10  # Rough approximation
        if mean_actual_approx > 0:
            relative_mean = abs(diagnostics.mean) / mean_actual_approx
            if relative_mean > self.config.mean_threshold:
                result.add_issue(ValidationIssue(
                    code="RD003",
                    category=IssueCategory.FITTING_RESULT,
                    severity=IssueSeverity.INFO,
                    message=f"Non-zero mean residuals (mean={diagnostics.mean:.2f})",
                    guidance=(
                        "Residuals have a non-zero mean, indicating slight systematic "
                        "over or under-prediction across the fit period."
                    ),
                    details={
                        "residual_mean": diagnostics.mean,
                        "residual_std": diagnostics.std,
                    },
                ))

        return result

    def analyze_and_validate(
        self,
        t: np.ndarray,
        q: np.ndarray,
        result: "ForecastResult",
        well_id: str,
        product: str,
    ) -> tuple[ResidualDiagnostics, ValidationResult]:
        """Analyze residuals and generate validation issues in one call.

        Args:
            t: Time array used for fitting
            q: Production data used for fitting
            result: ForecastResult from fitting
            well_id: Well identifier
            product: Product type

        Returns:
            Tuple of (ResidualDiagnostics, ValidationResult)
        """
        diagnostics = self.analyze_from_result(t, q, result)
        validation = self.get_validation_issues(diagnostics, well_id, product)
        return diagnostics, validation


def compute_residuals(
    t: np.ndarray,
    q: np.ndarray,
    result: "ForecastResult",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute residuals from a ForecastResult.

    Args:
        t: Time array used for fitting
        q: Production data used for fitting
        result: ForecastResult from fitting

    Returns:
        Tuple of (t_fit, residuals) arrays
    """
    # Get the portion of data used for fitting
    regime_start = result.regime_start_idx
    t_fit = t[regime_start:] - t[regime_start]
    q_actual = q[regime_start:]

    # Get predicted values
    q_predicted = result.model.rate(t_fit)

    # Compute residuals
    residuals = q_actual - q_predicted

    return t_fit, residuals


def summarize_residual_results(
    diagnostics_list: list[ResidualDiagnostics],
) -> dict:
    """Summarize residual diagnostics across multiple fits.

    Args:
        diagnostics_list: List of ResidualDiagnostics objects

    Returns:
        Dictionary with summary statistics
    """
    if not diagnostics_list:
        return {"count": 0}

    dw_values = [d.durbin_watson for d in diagnostics_list]
    autocorr_values = [d.autocorr_lag1 for d in diagnostics_list]
    early_biases = [d.early_bias for d in diagnostics_list]
    late_biases = [d.late_bias for d in diagnostics_list]
    systematic_count = sum(1 for d in diagnostics_list if d.has_systematic_pattern)

    return {
        "count": len(diagnostics_list),
        "durbin_watson": {
            "mean": float(np.mean(dw_values)),
            "std": float(np.std(dw_values)),
            "min": float(np.min(dw_values)),
            "max": float(np.max(dw_values)),
        },
        "autocorr_lag1": {
            "mean": float(np.mean(autocorr_values)),
            "std": float(np.std(autocorr_values)),
        },
        "early_bias": {
            "mean": float(np.mean(early_biases)),
            "std": float(np.std(early_biases)),
        },
        "late_bias": {
            "mean": float(np.mean(late_biases)),
            "std": float(np.std(late_biases)),
        },
        "systematic_pattern_pct": systematic_count / len(diagnostics_list) * 100,
    }
