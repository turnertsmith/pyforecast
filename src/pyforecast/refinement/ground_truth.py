"""Ground truth comparison against ARIES projections.

Compares pyforecast's auto-fitted decline curves against expert/approved
ARIES forecasts to measure fitting accuracy.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from .schemas import GroundTruthResult

if TYPE_CHECKING:
    from ..data.well import Well
    from ..import_.aries_forecast import AriesForecastImporter

from ..import_.aries_forecast import normalize_well_id

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthConfig:
    """Configuration for ground truth comparison.

    Attributes:
        comparison_months: Number of months to compare forecasts (default 60)
    """
    comparison_months: int = 60


@dataclass
class GroundTruthSummary:
    """Summary of ground truth comparison batch results.

    Attributes:
        results: List of individual comparison results
        wells_matched: Number of wells successfully matched
        wells_in_pyf_only: Wells with pyforecast data but no ARIES data
        wells_in_aries_only: Wells with ARIES data but not in pyforecast batch
    """
    results: list[GroundTruthResult]
    wells_matched: int
    wells_in_pyf_only: list[str]
    wells_in_aries_only: list[str]


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

        # Validate rates for NaN, infinite, or negative values
        pyf_rates = self._validate_rates(pyf_rates, f"{well.well_id}/pyforecast")
        aries_rates = self._validate_rates(aries_rates, f"{well.well_id}/ARIES")

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

        # Get start dates for alignment check
        aries_start = aries_params.start_date
        pyf_start: date | None = None

        # Calculate pyforecast start date (month after last production)
        if well.production.last_date:
            last = well.production.last_date
            if last.month == 12:
                pyf_start = date(last.year + 1, 1, 1)
            else:
                pyf_start = date(last.year, last.month + 1, 1)

        # Check alignment and generate warning if dates differ
        alignment_warning: str | None = None
        if aries_start and pyf_start:
            diff_months = abs(
                (aries_start.year - pyf_start.year) * 12 +
                (aries_start.month - pyf_start.month)
            )
            if diff_months > 0:
                alignment_warning = (
                    f"Start dates differ by {diff_months} month(s): "
                    f"ARIES={aries_start:%Y-%m}, pyf={pyf_start:%Y-%m}"
                )

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
            aries_start_date=aries_start,
            pyf_start_date=pyf_start,
            alignment_warning=alignment_warning,
        )

    def _validate_rates(
        self,
        rates: np.ndarray,
        source: str,
    ) -> np.ndarray:
        """Validate and clean rate array.

        Checks for NaN, infinite, and negative values, logging warnings
        for any issues found and applying fixes.

        Args:
            rates: Array of forecast rates to validate
            source: Description of data source for log messages

        Returns:
            Cleaned rate array with invalid values replaced
        """
        rates = rates.copy()

        if np.any(np.isnan(rates)):
            nan_count = np.sum(np.isnan(rates))
            logger.warning(
                f"{source}: {nan_count} NaN values in forecast rates, replacing with 0"
            )
            rates = np.nan_to_num(rates, nan=0.0)

        if np.any(np.isinf(rates)):
            inf_count = np.sum(np.isinf(rates))
            logger.warning(
                f"{source}: {inf_count} infinite values in forecast rates, clipping"
            )
            rates = np.clip(rates, 0, 1e9)

        if np.any(rates < 0):
            neg_count = np.sum(rates < 0)
            logger.warning(
                f"{source}: {neg_count} negative rates detected, clipping to 0"
            )
            rates = np.maximum(rates, 0)

        return rates

    def _calculate_mape(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> float | None:
        """Calculate Mean Absolute Percentage Error.

        Args:
            actual: Actual (ARIES) values
            predicted: Predicted (pyforecast) values

        Returns:
            MAPE as percentage, or None if insufficient valid data points
            (fewer than 3 points above minimum rate threshold)
        """
        # Avoid division by zero
        mask = actual > 0.1  # Minimum rate threshold
        valid_count = np.sum(mask)

        if valid_count < 3:
            logger.warning(
                f"Insufficient valid data points for MAPE calculation: "
                f"{valid_count} points above threshold (need 3)"
            )
            return None

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

    def validate_batch(
        self,
        wells: list["Well"],
        products: list[str],
    ) -> GroundTruthSummary:
        """Validate all wells and return summary with mismatch info.

        Args:
            wells: List of wells with fitted forecasts
            products: Products to validate (e.g., ["oil", "gas"])

        Returns:
            GroundTruthSummary with results and ID matching diagnostics
        """
        results = []

        # Collect pyforecast well IDs (normalized)
        pyf_ids: set[str] = set()
        for well in wells:
            propnum = (
                well.identifier.propnum
                or well.identifier.api
                or well.identifier.entity_id
            )
            if propnum:
                pyf_ids.add(normalize_well_id(propnum))

        # Get ARIES well IDs
        aries_ids = set(self.importer.list_wells())

        # Find mismatches
        wells_in_pyf_only = sorted(pyf_ids - aries_ids)
        wells_in_aries_only = sorted(aries_ids - pyf_ids)

        # Log mismatches if any
        if wells_in_pyf_only:
            logger.info(
                f"Wells in pyforecast but not ARIES ({len(wells_in_pyf_only)}): "
                f"{wells_in_pyf_only[:5]}{'...' if len(wells_in_pyf_only) > 5 else ''}"
            )
        if wells_in_aries_only:
            logger.info(
                f"Wells in ARIES but not pyforecast ({len(wells_in_aries_only)}): "
                f"{wells_in_aries_only[:5]}{'...' if len(wells_in_aries_only) > 5 else ''}"
            )

        # Validate each well/product combination
        for well in wells:
            for product in products:
                result = self.validate(well, product)
                if result is not None:
                    results.append(result)

        return GroundTruthSummary(
            results=results,
            wells_matched=len(results),
            wells_in_pyf_only=wells_in_pyf_only,
            wells_in_aries_only=wells_in_aries_only,
        )

    def validate_batch_parallel(
        self,
        wells: list["Well"],
        products: list[str],
        max_workers: int = 4,
    ) -> GroundTruthSummary:
        """Validate all wells in parallel for large batches.

        Uses ThreadPoolExecutor for concurrent validation. Provides
        speedup for large datasets at the cost of slightly higher
        memory usage.

        Args:
            wells: List of wells with fitted forecasts
            products: Products to validate (e.g., ["oil", "gas"])
            max_workers: Maximum number of parallel workers

        Returns:
            GroundTruthSummary with results and ID matching diagnostics
        """
        results: list[GroundTruthResult] = []

        # Collect pyforecast well IDs (normalized)
        pyf_ids: set[str] = set()
        for well in wells:
            propnum = (
                well.identifier.propnum
                or well.identifier.api
                or well.identifier.entity_id
            )
            if propnum:
                pyf_ids.add(normalize_well_id(propnum))

        # Get ARIES well IDs
        aries_ids = set(self.importer.list_wells())

        # Find mismatches
        wells_in_pyf_only = sorted(pyf_ids - aries_ids)
        wells_in_aries_only = sorted(aries_ids - pyf_ids)

        # Log mismatches if any
        if wells_in_pyf_only:
            logger.info(
                f"Wells in pyforecast but not ARIES ({len(wells_in_pyf_only)}): "
                f"{wells_in_pyf_only[:5]}{'...' if len(wells_in_pyf_only) > 5 else ''}"
            )
        if wells_in_aries_only:
            logger.info(
                f"Wells in ARIES but not pyforecast ({len(wells_in_aries_only)}): "
                f"{wells_in_aries_only[:5]}{'...' if len(wells_in_aries_only) > 5 else ''}"
            )

        # Build list of tasks
        tasks = [(well, product) for well in wells for product in products]

        # Validate in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.validate, well, product): (well.well_id, product)
                for well, product in tasks
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    well_id, product = futures[future]
                    logger.warning(f"Validation failed for {well_id}/{product}: {e}")

        return GroundTruthSummary(
            results=results,
            wells_matched=len(results),
            wells_in_pyf_only=wells_in_pyf_only,
            wells_in_aries_only=wells_in_aries_only,
        )


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
            "mape_unavailable_count": 0,
        }

    # Filter out None MAPE values for statistics
    valid_mapes = [r.mape for r in results if r.mape is not None]
    mape_unavailable_count = len(results) - len(valid_mapes)

    correlations = [r.correlation for r in results]
    cum_diffs = [r.cumulative_diff_pct for r in results]
    good_matches = [r for r in results if r.is_good_match]

    # Grade distribution (includes "X" for insufficient data)
    grades = {"A": 0, "B": 0, "C": 0, "D": 0, "X": 0}
    for r in results:
        grade = r.match_grade
        if grade in grades:
            grades[grade] += 1

    return {
        "count": len(results),
        "avg_mape": float(np.mean(valid_mapes)) if valid_mapes else None,
        "median_mape": float(np.median(valid_mapes)) if valid_mapes else None,
        "avg_correlation": float(np.mean(correlations)),
        "avg_cumulative_diff_pct": float(np.mean(cum_diffs)),
        "good_match_count": len(good_matches),
        "good_match_pct": len(good_matches) / len(results) * 100,
        "grade_distribution": grades,
        "mape_unavailable_count": mape_unavailable_count,
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

        valid_mapes = [r.mape for r in product_results if r.mape is not None]
        good_matches = [r for r in product_results if r.is_good_match]

        by_product[product] = {
            "count": len(product_results),
            "avg_mape": float(np.mean(valid_mapes)) if valid_mapes else None,
            "good_match_pct": len(good_matches) / len(product_results) * 100,
        }

    return by_product
