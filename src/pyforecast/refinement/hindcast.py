"""Hindcast validation framework for measuring forecast accuracy.

Hindcast (backtesting) splits historical data into training and holdout
periods, fits on training data, and measures prediction accuracy on holdout.
This provides a realistic measure of how well forecasts would have performed.
"""

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from .schemas import HindcastResult

if TYPE_CHECKING:
    from ..core.fitting import DeclineFitter
    from ..data.well import Well

logger = logging.getLogger(__name__)


@dataclass
class HindcastConfig:
    """Configuration for hindcast validation.

    Attributes:
        holdout_months: Number of months to hold out for validation (default 6)
        min_training_months: Minimum months required for training (default 12)
        min_holdout_rate: Minimum rate in holdout to avoid div-by-zero issues
    """

    holdout_months: int = 6
    min_training_months: int = 12
    min_holdout_rate: float = 0.1  # Minimum rate to include in MAPE calculation


class HindcastValidator:
    """Validates forecast accuracy through hindcast (backtesting).

    Hindcast splits historical production data into:
    - Training period: Used to fit the decline curve
    - Holdout period: Used to measure prediction accuracy

    This provides a realistic measure of forecast accuracy by simulating
    how a forecast made in the past would have performed.

    Example:
        validator = HindcastValidator(holdout_months=6)
        result = validator.validate(well, "oil", fitter)

        if result.is_good_hindcast:
            print(f"Good forecast: MAPE={result.mape:.1f}%")
        else:
            print(f"Poor forecast: MAPE={result.mape:.1f}%")
    """

    def __init__(
        self,
        holdout_months: int = 6,
        min_training_months: int = 12,
    ):
        """Initialize hindcast validator.

        Args:
            holdout_months: Number of months to hold out for validation
            min_training_months: Minimum months required for training
        """
        self.config = HindcastConfig(
            holdout_months=holdout_months,
            min_training_months=min_training_months,
        )

    def can_validate(self, n_months: int) -> bool:
        """Check if there's enough data for hindcast validation.

        Args:
            n_months: Total number of months of data

        Returns:
            True if sufficient data for training + holdout
        """
        min_required = self.config.min_training_months + self.config.holdout_months
        return n_months >= min_required

    def validate(
        self,
        well: "Well",
        product: str,
        fitter: "DeclineFitter",
    ) -> HindcastResult | None:
        """Run hindcast validation for a well.

        Splits data, fits on training period, validates on holdout.

        Args:
            well: Well object with production data
            product: Product to validate (oil, gas, water)
            fitter: DeclineFitter to use for fitting

        Returns:
            HindcastResult with validation metrics, or None if insufficient data
        """
        t = well.production.time_months
        q = well.production.get_product_daily(product)
        n_months = len(t)

        if not self.can_validate(n_months):
            logger.info(
                f"Skipping hindcast for {well.well_id}/{product}: "
                f"{n_months} months < {self.config.min_training_months + self.config.holdout_months} required"
            )
            return None

        result = self.validate_with_data(t, q, fitter, well.well_id, product)

        if result is None and np.max(q[n_months - self.config.holdout_months:]) >= self.config.min_holdout_rate:
            logger.warning(f"Hindcast fit failed for {well.well_id}/{product}")

        return result

    def validate_with_data(
        self,
        t: np.ndarray,
        q: np.ndarray,
        fitter: "DeclineFitter",
        well_id: str = "",
        product: str = "",
    ) -> HindcastResult | None:
        """Run hindcast validation on raw data arrays.

        Useful for validation without a Well object.

        Args:
            t: Time array (months)
            q: Production rate array
            fitter: DeclineFitter to use
            well_id: Well identifier for result
            product: Product name for result

        Returns:
            HindcastResult with validation metrics, or None if insufficient data
        """
        n_months = len(t)

        if not self.can_validate(n_months):
            return None

        training_end = n_months - self.config.holdout_months
        t_train = t[:training_end]
        q_train = q[:training_end]
        t_holdout = t[training_end:]
        q_holdout = q[training_end:]

        if np.max(q_holdout) < self.config.min_holdout_rate:
            return None

        try:
            fit_result = fitter.fit(t_train, q_train)
        except ValueError:
            return None

        t_pred = t_holdout - t_train[fit_result.regime_start_idx]
        q_pred = fit_result.model.rate(t_pred)
        metrics = self._calculate_metrics(q_holdout, q_pred)

        return HindcastResult(
            well_id=well_id,
            product=product,
            training_months=training_end,
            holdout_months=self.config.holdout_months,
            training_r_squared=fit_result.r_squared,
            training_qi=fit_result.model.qi,
            training_di=fit_result.model.di,
            training_b=fit_result.model.b,
            mape=metrics["mape"],
            correlation=metrics["correlation"],
            bias=metrics["bias"],
            holdout_actual=q_holdout,
            holdout_predicted=q_pred,
            holdout_months_array=t_holdout,
        )

    def _calculate_metrics(
        self,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> dict:
        """Calculate hindcast accuracy metrics.

        Args:
            actual: Actual production values
            predicted: Predicted production values

        Returns:
            Dictionary with mape, correlation, bias
        """
        # Filter out very low rates to avoid div-by-zero in MAPE
        mask = actual >= self.config.min_holdout_rate
        actual_valid = actual[mask]
        predicted_valid = predicted[mask]

        if len(actual_valid) == 0:
            return {"mape": 100.0, "correlation": 0.0, "bias": 0.0}

        # MAPE: Mean Absolute Percentage Error
        ape = np.abs(actual_valid - predicted_valid) / actual_valid * 100
        mape = float(np.mean(ape))

        # Correlation
        if len(actual_valid) > 1 and np.std(actual_valid) > 0 and np.std(predicted_valid) > 0:
            correlation, _ = stats.pearsonr(actual_valid, predicted_valid)
            correlation = float(correlation)
        else:
            correlation = 0.0

        # Bias: systematic over/under prediction
        # Positive = over-prediction, Negative = under-prediction
        mean_actual = np.mean(actual_valid)
        if mean_actual > 0:
            bias = float(np.mean(predicted_valid - actual_valid) / mean_actual)
        else:
            bias = 0.0

        return {
            "mape": mape,
            "correlation": correlation,
            "bias": bias,
        }


def run_hindcast_batch(
    wells: list["Well"],
    product: str,
    fitter: "DeclineFitter",
    holdout_months: int = 6,
    min_training_months: int = 12,
) -> list[HindcastResult]:
    """Run hindcast validation on multiple wells.

    Args:
        wells: List of wells to validate
        product: Product to validate
        fitter: DeclineFitter to use
        holdout_months: Months to hold out
        min_training_months: Minimum training months

    Returns:
        List of HindcastResults (only for wells with sufficient data)
    """
    validator = HindcastValidator(
        holdout_months=holdout_months,
        min_training_months=min_training_months,
    )

    results = []
    for well in wells:
        result = validator.validate(well, product, fitter)
        if result is not None:
            results.append(result)

    return results


def summarize_hindcast_results(results: list[HindcastResult]) -> dict:
    """Summarize hindcast results across multiple wells.

    Args:
        results: List of HindcastResults

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {
            "count": 0,
            "avg_mape": None,
            "avg_correlation": None,
            "avg_bias": None,
            "good_hindcast_pct": None,
        }

    mapes = [r.mape for r in results]
    correlations = [r.correlation for r in results]
    biases = [r.bias for r in results]
    good_count = sum(1 for r in results if r.is_good_hindcast)

    return {
        "count": len(results),
        "avg_mape": float(np.mean(mapes)),
        "median_mape": float(np.median(mapes)),
        "std_mape": float(np.std(mapes)),
        "avg_correlation": float(np.mean(correlations)),
        "avg_bias": float(np.mean(biases)),
        "good_hindcast_pct": good_count / len(results) * 100,
        "good_hindcast_count": good_count,
    }
