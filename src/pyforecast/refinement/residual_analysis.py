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


@dataclass
class FittingAdjustment:
    """A suggested adjustment to improve fitting.

    Attributes:
        parameter: Parameter to adjust (qi, di, b, dmin, model_type)
        direction: Direction of adjustment (increase, decrease, or specific value)
        confidence: Confidence in the suggestion (low, medium, high)
        reason: Explanation of why this adjustment is suggested
    """
    parameter: str
    direction: str
    confidence: str
    reason: str


def suggest_fitting_adjustments(
    diagnostics: ResidualDiagnostics,
    result: "ForecastResult" = None,
    bias_threshold: float = 0.15,
    dw_low: float = 1.5,
    dw_high: float = 2.5,
) -> list[FittingAdjustment]:
    """Suggest fitting adjustments based on residual diagnostics.

    Analyzes residual patterns to provide actionable suggestions for
    improving fit quality.

    Args:
        diagnostics: ResidualDiagnostics from a fit
        result: Optional ForecastResult for additional context
        bias_threshold: Threshold for flagging early/late bias
        dw_low: Durbin-Watson lower threshold for positive autocorrelation
        dw_high: Durbin-Watson upper threshold for negative autocorrelation

    Returns:
        List of FittingAdjustment suggestions
    """
    suggestions = []

    # Check for early bias (model systematically over/under-predicts early time)
    if diagnostics.early_bias > bias_threshold:
        # Model under-predicts early (actual > predicted)
        # Suggests qi is too low
        suggestions.append(FittingAdjustment(
            parameter="qi",
            direction="increase",
            confidence="medium",
            reason=f"Early under-prediction ({diagnostics.early_bias:.1%}): "
                   "Initial rate (qi) may be too low"
        ))
    elif diagnostics.early_bias < -bias_threshold:
        # Model over-predicts early (actual < predicted)
        # Suggests qi is too high or Di is too low
        suggestions.append(FittingAdjustment(
            parameter="qi",
            direction="decrease",
            confidence="medium",
            reason=f"Early over-prediction ({diagnostics.early_bias:.1%}): "
                   "Initial rate (qi) may be too high"
        ))

    # Check for late bias
    if diagnostics.late_bias > bias_threshold:
        # Model under-predicts late (actual > predicted)
        # Suggests Dmin is too high or b is too low
        suggestions.append(FittingAdjustment(
            parameter="dmin",
            direction="decrease",
            confidence="medium",
            reason=f"Late under-prediction ({diagnostics.late_bias:.1%}): "
                   "Terminal decline (Dmin) may be too aggressive"
        ))
        suggestions.append(FittingAdjustment(
            parameter="b",
            direction="increase",
            confidence="low",
            reason=f"Late under-prediction ({diagnostics.late_bias:.1%}): "
                   "b-factor may be too low, causing too-fast decline"
        ))
    elif diagnostics.late_bias < -bias_threshold:
        # Model over-predicts late (actual < predicted)
        # Suggests Dmin is too low or b is too high
        suggestions.append(FittingAdjustment(
            parameter="dmin",
            direction="increase",
            confidence="medium",
            reason=f"Late over-prediction ({diagnostics.late_bias:.1%}): "
                   "Terminal decline (Dmin) may be too conservative"
        ))

    # Check for positive autocorrelation (DW < lower threshold)
    if diagnostics.durbin_watson < dw_low:
        suggestions.append(FittingAdjustment(
            parameter="model_type",
            direction="try_alternative",
            confidence="medium",
            reason=f"Positive autocorrelation (DW={diagnostics.durbin_watson:.2f}): "
                   "Model may be underfitting. Consider regime detection or different model type."
        ))
        # If we have model info, suggest specific adjustments
        if result is not None and result.model.b < 0.3:
            suggestions.append(FittingAdjustment(
                parameter="b",
                direction="increase",
                confidence="medium",
                reason="Near-exponential model with autocorrelation suggests "
                       "hyperbolic behavior may be present"
            ))

    # Check for negative autocorrelation (DW > upper threshold)
    if diagnostics.durbin_watson > dw_high:
        suggestions.append(FittingAdjustment(
            parameter="model_type",
            direction="simplify",
            confidence="low",
            reason=f"Negative autocorrelation (DW={diagnostics.durbin_watson:.2f}): "
                   "Model may be overfitting or data is noisy. Consider exponential model."
        ))

    # Check for combined early+late bias pattern
    if (abs(diagnostics.early_bias) > bias_threshold and
        abs(diagnostics.late_bias) > bias_threshold and
        np.sign(diagnostics.early_bias) != np.sign(diagnostics.late_bias)):
        # Opposite biases early vs late suggests b-factor issue
        suggestions.append(FittingAdjustment(
            parameter="b",
            direction="adjust",
            confidence="high",
            reason="Opposite early/late bias pattern strongly suggests "
                   "b-factor mismatch with actual decline curvature"
        ))

    return suggestions


def format_adjustment_suggestions(
    suggestions: list[FittingAdjustment],
) -> str:
    """Format adjustment suggestions as readable text.

    Args:
        suggestions: List of FittingAdjustment objects

    Returns:
        Formatted string with suggestions
    """
    if not suggestions:
        return "No specific adjustments suggested. Fit appears reasonable."

    lines = ["Suggested fitting adjustments:"]
    for i, adj in enumerate(suggestions, 1):
        lines.append(f"  {i}. [{adj.confidence.upper()}] {adj.parameter}: {adj.direction}")
        lines.append(f"     Reason: {adj.reason}")

    return "\n".join(lines)
