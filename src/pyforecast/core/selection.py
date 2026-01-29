"""Model selection and fit quality evaluation."""

from .models import ForecastResult


def evaluate_fit_quality(result: ForecastResult) -> dict:
    """Evaluate the quality of a decline curve fit.

    Args:
        result: ForecastResult from curve fitting

    Returns:
        Dictionary with quality assessment and recommendations
    """
    assessment = {
        "acceptable": result.is_acceptable,
        "r_squared": result.r_squared,
        "rmse": result.rmse,
        "aic": result.aic,
        "bic": result.bic,
        "quality_grade": _grade_fit(result.r_squared),
        "warnings": [],
        "decline_type": result.model.decline_type,
    }

    # Check for potential issues
    if result.r_squared < 0.5:
        assessment["warnings"].append(
            "Poor fit (R² < 0.5): Consider manual review of data quality"
        )
    elif result.r_squared < 0.7:
        assessment["warnings"].append(
            "Marginal fit (R² < 0.7): Forecast may have significant uncertainty"
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


def _grade_fit(r_squared: float) -> str:
    """Assign letter grade based on R² value.

    Args:
        r_squared: Coefficient of determination

    Returns:
        Letter grade (A, B, C, D, F)
    """
    if r_squared >= 0.95:
        return "A"
    elif r_squared >= 0.85:
        return "B"
    elif r_squared >= 0.70:
        return "C"
    elif r_squared >= 0.50:
        return "D"
    else:
        return "F"


def compare_fits(results: list[ForecastResult]) -> ForecastResult:
    """Compare multiple fit results and select the best one.

    Selection criteria (in order of priority):
    1. Must meet minimum R² threshold (0.7)
    2. Lowest BIC (parsimony-adjusted fit quality)

    Args:
        results: List of ForecastResult objects to compare

    Returns:
        Best ForecastResult based on selection criteria

    Raises:
        ValueError: If no acceptable fits found
    """
    # Filter acceptable fits
    acceptable = [r for r in results if r.is_acceptable]

    if not acceptable:
        # Return best available even if below threshold
        if results:
            return min(results, key=lambda r: -r.r_squared)
        raise ValueError("No fit results to compare")

    # Select by lowest BIC among acceptable fits
    return min(acceptable, key=lambda r: r.bic)
