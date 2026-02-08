"""Fitting validation for decline curve analysis.

Pre-fit checks verify data is suitable for fitting.
Post-fit checks verify the fit quality and parameter reasonableness.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import linregress

from .result import (
    ValidationResult,
    ValidationIssue,
    IssueSeverity,
    IssueCategory,
)

if TYPE_CHECKING:
    from ..data.well import Well
    from ..core.models import ForecastResult


class FittingValidator:
    """Validates pre-fit requirements and post-fit quality.

    Pre-fit error codes:
        FP001: Insufficient data points
        FP002: Increasing trend (not declining)
        FP003: Flat trend (no decline)

    Post-fit error codes:
        FR001: Poor fit (R² < threshold)
        FR003: B-factor at lower bound
        FR004: B-factor at upper bound
        FR005: Very high decline rate (>100%/yr)
    """

    def __init__(
        self,
        min_points: int = 6,
        min_r_squared: float = 0.5,
        b_min: float = 0.01,
        b_max: float = 1.5,
        max_annual_decline: float = 1.0,
    ):
        """Initialize fitting validator.

        Args:
            min_points: Minimum data points required for fitting
            min_r_squared: Minimum acceptable R² value
            b_min: Lower bound for b-factor
            b_max: Upper bound for b-factor
            max_annual_decline: Maximum acceptable annual decline rate
        """
        self.min_points = min_points
        self.min_r_squared = min_r_squared
        self.b_min = b_min
        self.b_max = b_max
        self.max_annual_decline = max_annual_decline

    def validate_pre_fit(
        self,
        well: "Well",
        product: str,
    ) -> ValidationResult:
        """Validate data before fitting.

        Checks:
            - Sufficient data points
            - Declining trend (not increasing)
            - Not flat (some decline present)

        Args:
            well: Well object to validate
            product: Product to validate

        Returns:
            ValidationResult with any pre-fit issues
        """
        result = ValidationResult(well_id=well.well_id, product=product)

        try:
            values = well.production.get_product(product)
            times = well.production.time_months
        except (AttributeError, KeyError):
            return result  # Product not present

        if values is None or len(values) == 0:
            return result

        values = np.asarray(values, dtype=float)
        times = np.asarray(times, dtype=float)

        # Filter to non-zero values for analysis
        nonzero_mask = values > 0
        if np.sum(nonzero_mask) < 2:
            result.add_issue(ValidationIssue(
                code="FP001",
                category=IssueCategory.FITTING_PREREQ,
                severity=IssueSeverity.ERROR,
                message=f"Insufficient non-zero {product} data points for fitting",
                guidance="Need at least 2 non-zero production values to analyze trend",
                details={
                    "nonzero_count": int(np.sum(nonzero_mask)),
                    "total_count": len(values),
                    "min_required": 2,
                },
            ))
            return result

        # Check data point count
        n_points = len(values)
        if n_points < self.min_points:
            result.add_issue(ValidationIssue(
                code="FP001",
                category=IssueCategory.FITTING_PREREQ,
                severity=IssueSeverity.ERROR,
                message=f"Insufficient {product} data points: {n_points} < {self.min_points}",
                guidance=f"Need at least {self.min_points} months of data for reliable fitting",
                details={
                    "point_count": n_points,
                    "min_required": self.min_points,
                },
            ))

        # Check trend direction using non-zero values
        t_nz = times[nonzero_mask]
        q_nz = values[nonzero_mask]

        if len(q_nz) >= 3:
            # Use log-linear regression to detect trend
            log_q = np.log(q_nz)
            slope, _, r_value, _, _ = linregress(t_nz, log_q)

            # Positive slope = increasing production
            if slope > 0.01:  # >1% monthly increase
                result.add_issue(ValidationIssue(
                    code="FP002",
                    category=IssueCategory.FITTING_PREREQ,
                    severity=IssueSeverity.WARNING,
                    message=f"Increasing {product} trend detected (not declining)",
                    guidance="Decline curve fitting requires declining production; data may be early-time or ramping",
                    details={
                        "monthly_slope": float(slope),
                        "annual_rate": float(slope * 12),
                        "r_squared": float(r_value ** 2),
                    },
                ))
            elif abs(slope) < 0.001 and r_value ** 2 > 0.5:  # Very flat with good fit
                result.add_issue(ValidationIssue(
                    code="FP003",
                    category=IssueCategory.FITTING_PREREQ,
                    severity=IssueSeverity.WARNING,
                    message=f"Flat {product} trend detected (minimal decline)",
                    guidance="Very stable production may not fit hyperbolic model well",
                    details={
                        "monthly_slope": float(slope),
                        "r_squared": float(r_value ** 2),
                    },
                ))

        return result

    def validate_post_fit(
        self,
        well: "Well",
        product: str,
    ) -> ValidationResult:
        """Validate fitting results.

        Checks:
            - R² quality
            - B-factor not at bounds
            - Reasonable decline rate

        Args:
            well: Well object with forecast
            product: Product that was fitted

        Returns:
            ValidationResult with any post-fit issues
        """
        result = ValidationResult(well_id=well.well_id, product=product)

        forecast = well.get_forecast(product)
        if forecast is None:
            return result

        self._check_fit_quality(result, forecast, product)
        return result

    def _check_fit_quality(
        self,
        result: ValidationResult,
        forecast: "ForecastResult",
        product: str = "",
    ) -> None:
        """Check fit quality metrics and add issues to result.

        Shared logic between validate_post_fit() and validate_fit_result().

        Args:
            result: ValidationResult to add issues to
            forecast: ForecastResult to check
            product: Product label for messages
        """
        # Check R² quality
        if forecast.r_squared < self.min_r_squared:
            prefix = f"{product} " if product else ""
            result.add_issue(ValidationIssue.poor_fit(
                product=prefix.strip() or "overall",
                r_squared=float(forecast.r_squared),
                threshold=self.min_r_squared,
                rmse=float(forecast.rmse),
            ))

        # Check b-factor bounds
        model = forecast.model
        b_tolerance = 0.001
        prefix = f"{product} " if product else ""

        if abs(model.b - self.b_min) < b_tolerance:
            result.add_issue(ValidationIssue(
                code="FR003",
                category=IssueCategory.FITTING_RESULT,
                severity=IssueSeverity.INFO,
                message=f"{prefix}b-factor at lower bound ({model.b:.3f})",
                guidance="B at lower bound suggests near-exponential decline; may be constrained by bound",
                details={"b": float(model.b), "b_min": self.b_min, "b_max": self.b_max},
            ))

        if abs(model.b - self.b_max) < b_tolerance:
            result.add_issue(ValidationIssue(
                code="FR004",
                category=IssueCategory.FITTING_RESULT,
                severity=IssueSeverity.WARNING,
                message=f"{prefix}b-factor at upper bound ({model.b:.3f})",
                guidance="B at upper bound may indicate transient flow or data issues; review fit",
                details={"b": float(model.b), "b_min": self.b_min, "b_max": self.b_max},
            ))

        # Check decline rate
        annual_decline = model.di * 12
        if annual_decline > self.max_annual_decline:
            result.add_issue(ValidationIssue(
                code="FR005",
                category=IssueCategory.FITTING_RESULT,
                severity=IssueSeverity.WARNING,
                message=f"Very high {prefix}decline rate: {annual_decline:.0%}/year",
                guidance="Decline >100%/year is unusual; verify data quality and fit",
                details={"annual_decline": float(annual_decline), "monthly_decline": float(model.di), "threshold": self.max_annual_decline},
            ))

    def validate_fit_result(
        self,
        forecast: "ForecastResult",
        well_id: str | None = None,
        product: str | None = None,
    ) -> ValidationResult:
        """Validate a ForecastResult directly.

        Alternative to validate_post_fit when you have the result
        but not the well object.

        Args:
            forecast: ForecastResult to validate
            well_id: Well identifier for result
            product: Product name for result

        Returns:
            ValidationResult with any issues
        """
        result = ValidationResult(well_id=well_id, product=product)
        self._check_fit_quality(result, forecast, product or "")
        return result
