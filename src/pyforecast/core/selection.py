"""Model selection and fit quality evaluation."""

from .models import ForecastResult


# Default grading thresholds (can be overridden by config)
DEFAULT_GRADE_THRESHOLDS = {
    "A": 0.95,
    "B": 0.85,
    "C": 0.70,
    "D": 0.50,
}


def evaluate_fit_quality(
    result: ForecastResult,
    grade_thresholds: dict | None = None,
) -> dict:
    """Evaluate the quality of a decline curve fit.

    Args:
        result: ForecastResult from curve fitting
        grade_thresholds: Optional dict with grade thresholds {"A": 0.95, "B": 0.85, ...}

    Returns:
        Dictionary with quality assessment and recommendations
    """
    thresholds = grade_thresholds or DEFAULT_GRADE_THRESHOLDS
    acceptable_threshold = result.acceptable_r_squared

    assessment = {
        "acceptable": result.is_acceptable,
        "r_squared": result.r_squared,
        "rmse": result.rmse,
        "aic": result.aic,
        "bic": result.bic,
        "quality_grade": _grade_fit(result.r_squared, thresholds),
        "warnings": [],
        "decline_type": result.model.decline_type,
    }

    # Check for potential issues using configurable thresholds
    marginal_threshold = thresholds.get("D", 0.50)
    if result.r_squared < marginal_threshold:
        assessment["warnings"].append(
            f"Poor fit (R² < {marginal_threshold}): Consider manual review of data quality"
        )
    elif result.r_squared < acceptable_threshold:
        assessment["warnings"].append(
            f"Marginal fit (R² < {acceptable_threshold}): Forecast may have significant uncertainty"
        )

    if result.model.b < 0.1:
        assessment["warnings"].append(
            "Near-exponential decline (b < 0.1): Well may be in boundary-dominated flow"
        )

    if result.model.b > 1.2:
        assessment["warnings"].append(
            "High b-factor (> 1.2): Unusual decline behavior, verify data quality"
        )

    if result.data_points_used < 12:
        assessment["warnings"].append(
            f"Limited data ({result.data_points_used} months): Forecast uncertainty is high"
        )

    if result.regime_start_idx > 0:
        assessment["warnings"].append(
            f"Regime change detected at index {result.regime_start_idx}: "
            f"Only using {result.data_points_used} months of recent data"
        )

    return assessment


def _grade_fit(r_squared: float, thresholds: dict | None = None) -> str:
    """Assign letter grade based on R² value.

    Args:
        r_squared: Coefficient of determination
        thresholds: Optional dict with grade thresholds

    Returns:
        Letter grade (A, B, C, D, F)
    """
    thresholds = thresholds or DEFAULT_GRADE_THRESHOLDS
    if r_squared >= thresholds.get("A", 0.95):
        return "A"
    elif r_squared >= thresholds.get("B", 0.85):
        return "B"
    elif r_squared >= thresholds.get("C", 0.70):
        return "C"
    elif r_squared >= thresholds.get("D", 0.50):
        return "D"
    else:
        return "F"


def compare_fits(
    results: list[ForecastResult],
    acceptable_r_squared: float | None = None,
) -> ForecastResult:
    """Compare multiple fit results and select the best one.

    Selection criteria (in order of priority):
    1. Must meet minimum R² threshold (uses result's acceptable_r_squared)
    2. Lowest BIC (parsimony-adjusted fit quality)

    Args:
        results: List of ForecastResult objects to compare
        acceptable_r_squared: Override threshold (uses result's threshold if None)

    Returns:
        Best ForecastResult based on selection criteria

    Raises:
        ValueError: If no acceptable fits found
    """
    # Filter acceptable fits
    if acceptable_r_squared is not None:
        acceptable = [r for r in results if r.r_squared >= acceptable_r_squared]
    else:
        acceptable = [r for r in results if r.is_acceptable]

    if not acceptable:
        # Return best available even if below threshold
        if results:
            return min(results, key=lambda r: -r.r_squared)
        raise ValueError("No fit results to compare")

    # Select by lowest BIC among acceptable fits
    return min(acceptable, key=lambda r: r.bic)
