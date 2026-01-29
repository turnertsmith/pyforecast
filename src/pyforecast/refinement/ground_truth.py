"""Ground truth comparison against ARIES projections.

Compares pyforecast's auto-fitted decline curves against expert/approved
ARIES forecasts to measure fitting accuracy.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .schemas import GroundTruthResult

if TYPE_CHECKING:
    from ..data.well import Well
    from ..import_.aries_forecast import AriesForecastImporter


@dataclass
class GroundTruthConfig:
    """Configuration for ground truth comparison.

    Attributes:
        comparison_months: Number of months to compare forecasts (default 60)
    """
    comparison_months: int = 60


class GroundTruthValidator:
    """Compare pyforecast fits against ARIES ground truth.

    Takes an ARIES forecast importer and compares fitted wells against
    the expert/approved projections.

    Example:
        importer = AriesForecastImporter()
        importer.load(Path("aries_forecasts.csv"))

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        if result and result.is_good_match:
            print(f"Good match: MAPE={result.mape:.1f}%")
    """

    def __init__(
        self,
        aries_importer: "AriesForecastImporter",
        config: GroundTruthConfig | None = None,
    ) -> None:
        """Initialize the validator.

        Args:
            aries_importer: Loaded ARIES forecast importer
            config: Comparison configuration (optional)
        """
        self.importer = aries_importer
        self.config = config or GroundTruthConfig()

    def validate(
        self,
        well: "Well",
        product: str,
    ) -> GroundTruthResult | None:
        """Compare pyforecast fit against ARIES ground truth.

        Args:
            well: Well with fitted forecast
            product: Product to compare (oil, gas, water)

        Returns:
            GroundTruthResult if both pyforecast and ARIES data exist,
            None if either is missing
        """
        # Check pyforecast has a forecast
        forecast = well.get_forecast(product)
        if forecast is None:
            return None

        # Check ARIES has data for this well
        # Try multiple identifier types
        propnum = well.identifier.propnum or well.identifier.api or well.identifier.entity_id
        if not propnum:
            return None

        aries_params = self.importer.get(propnum, product)
        if aries_params is None:
            # Try alternate identifiers
            for alt_id in [well.identifier.api, well.identifier.entity_id, well.identifier.propnum]:
                if alt_id:
                    aries_params = self.importer.get(alt_id, product)
                    if aries_params:
                        break

        if aries_params is None:
            return None

        # Get pyforecast model parameters
        pyf_model = forecast.model
        pyf_qi = pyf_model.qi
        pyf_di = pyf_model.di
        pyf_b = pyf_model.b

        # Get ARIES parameters
        aries_qi = aries_params.qi
        aries_di = aries_params.di
        aries_b = aries_params.b

        # Calculate parameter differences
        qi_pct_diff = (pyf_qi - aries_qi) / aries_qi * 100 if aries_qi != 0 else 0
        di_pct_diff = (pyf_di - aries_di) / aries_di * 100 if aries_di != 0 else 0
        b_abs_diff = pyf_b - aries_b

        # Generate forecast arrays
        months = self.config.comparison_months
        t = np.arange(months, dtype=float)

        # Get pyforecast rates
        pyf_rates = pyf_model.rate(t)

        # Get ARIES rates using HyperbolicModel
        aries_model = self.importer.to_model(aries_params)
        aries_rates = aries_model.rate(t)

        # Calculate comparison metrics
        mape = self._calculate_mape(aries_rates, pyf_rates)
        correlation = self._calculate_correlation(aries_rates, pyf_rates)
        bias = self._calculate_bias(aries_rates, pyf_rates)

        # Calculate cumulative difference
        pyf_cumulative = pyf_model.cumulative(months - 1)
        aries_cumulative = aries_model.cumulative(months - 1)
        if aries_cumulative.size > 0 and aries_cumulative[0] != 0:
            cumulative_diff_pct = (pyf_cumulative[0] - aries_cumulative[0]) / aries_cumulative[0] * 100
        else:
            cumulative_diff_pct = 0.0

        return GroundTruthResult(
            well_id=well.well_id,
            product=product,
            aries_qi=aries_qi,
            aries_di=aries_di,
            aries_b=aries_b,
            aries_decline_type=aries_params.decline_type,
            pyf_qi=pyf_qi,
            pyf_di=pyf_di,
            pyf_b=pyf_b,
            qi_pct_diff=qi_pct_diff,
            di_pct_diff=di_pct_diff,
            b_abs_diff=b_abs_diff,
            comparison_months=months,
            mape=mape,
            correlation=correlation,
            bias=bias,
            cumulative_diff_pct=cumulative_diff_pct,
            forecast_months=t,
            aries_rates=aries_rates,
            pyf_rates=pyf_rates,
        )

    def _calculate_mape(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """Calculate Mean Absolute Percentage Error.

        Args:
            actual: Actual (ARIES) values
            predicted: Predicted (pyforecast) values

        Returns:
            MAPE as percentage
        """
        # Avoid division by zero
        mask = actual > 0.1  # Minimum rate threshold
        if not np.any(mask):
            return 0.0

        abs_pct_error = np.abs((predicted[mask] - actual[mask]) / actual[mask])
        return float(np.mean(abs_pct_error) * 100)

    def _calculate_correlation(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """Calculate Pearson correlation coefficient.

        Args:
            actual: Actual (ARIES) values
            predicted: Predicted (pyforecast) values

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(actual) < 2:
            return 0.0

        # Check for constant arrays
        if np.std(actual) == 0 or np.std(predicted) == 0:
            return 1.0 if np.allclose(actual, predicted) else 0.0

        corr_matrix = np.corrcoef(actual, predicted)
        return float(corr_matrix[0, 1])

    def _calculate_bias(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float:
        """Calculate systematic bias.

        Args:
            actual: Actual (ARIES) values
            predicted: Predicted (pyforecast) values

        Returns:
            Bias as fraction (positive = pyf over-predicts)
        """
        mean_actual = np.mean(actual)
        if mean_actual == 0:
            return 0.0

        mean_diff = np.mean(predicted - actual)
        return float(mean_diff / mean_actual)


def summarize_ground_truth_results(
    results: list[GroundTruthResult],
) -> dict:
    """Summarize ground truth comparison results.

    Args:
        results: List of GroundTruthResult objects

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {
            "count": 0,
            "avg_mape": None,
            "median_mape": None,
            "avg_correlation": None,
            "avg_cumulative_diff_pct": None,
            "good_match_count": 0,
            "good_match_pct": 0.0,
            "grade_distribution": {},
        }

    mapes = [r.mape for r in results]
    correlations = [r.correlation for r in results]
    cum_diffs = [r.cumulative_diff_pct for r in results]
    good_matches = [r for r in results if r.is_good_match]

    # Grade distribution
    grades = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in results:
        grades[r.match_grade] += 1

    return {
        "count": len(results),
        "avg_mape": float(np.mean(mapes)),
        "median_mape": float(np.median(mapes)),
        "avg_correlation": float(np.mean(correlations)),
        "avg_cumulative_diff_pct": float(np.mean(cum_diffs)),
        "good_match_count": len(good_matches),
        "good_match_pct": len(good_matches) / len(results) * 100,
        "grade_distribution": grades,
        "by_product": _summarize_by_product(results),
    }


def _summarize_by_product(results: list[GroundTruthResult]) -> dict:
    """Summarize results by product type."""
    by_product = {}

    products = set(r.product for r in results)
    for product in products:
        product_results = [r for r in results if r.product == product]
        if not product_results:
            continue

        mapes = [r.mape for r in product_results]
        good_matches = [r for r in product_results if r.is_good_match]

        by_product[product] = {
            "count": len(product_results),
            "avg_mape": float(np.mean(mapes)),
            "good_match_pct": len(good_matches) / len(product_results) * 100,
        }

    return by_product
