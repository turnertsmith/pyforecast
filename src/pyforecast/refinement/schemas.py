"""Data classes for refinement module.

Defines structured data types for fit logging, hindcast validation,
residual analysis, and ground truth comparison.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import uuid

import numpy as np


@dataclass
class FitLogRecord:
    """Complete record of a decline curve fit for logging and analysis.

    Captures all parameters, metrics, and optional diagnostics from a fit
    operation for persistent storage and subsequent analysis.

    Attributes:
        fit_id: Unique identifier for this fit (UUID)
        timestamp: When the fit was performed
        well_id: Well identifier
        product: Product type (oil, gas, water)

        # Well context (optional metadata)
        basin: Basin/play name if available
        formation: Formation name if available

        # Input characteristics
        data_points_total: Total data points available
        data_points_used: Data points used after regime detection
        regime_start_idx: Index where current regime starts

        # Parameters used for fitting
        b_min: Minimum b-factor constraint
        b_max: Maximum b-factor constraint
        dmin_annual: Terminal decline rate (annual fraction)
        recency_half_life: Half-life for recency weighting (months)
        regime_threshold: Threshold for regime change detection

        # Fit results
        qi: Initial rate from fit
        di: Initial decline rate (monthly)
        b: Hyperbolic exponent
        r_squared: Coefficient of determination
        rmse: Root mean squared error
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion

        # Residual diagnostics (optional)
        residual_mean: Mean of residuals
        residual_std: Standard deviation of residuals
        durbin_watson: Durbin-Watson statistic (2.0 = no autocorrelation)
        early_bias: Systematic error in early time period
        late_bias: Systematic error in late time period

        # Hindcast results (optional)
        hindcast_mape: Mean Absolute Percentage Error on holdout
        hindcast_correlation: Pearson correlation on holdout
        hindcast_bias: Systematic over/under prediction
    """

    # Required fields
    fit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    well_id: str = ""
    product: str = ""

    # Well context
    basin: str | None = None
    formation: str | None = None

    # Input characteristics
    data_points_total: int = 0
    data_points_used: int = 0
    regime_start_idx: int = 0

    # Parameters used
    b_min: float = 0.01
    b_max: float = 1.5
    dmin_annual: float = 0.06
    recency_half_life: float = 12.0
    regime_threshold: float = 1.0

    # Fit results
    qi: float = 0.0
    di: float = 0.0
    b: float = 0.0
    r_squared: float = 0.0
    rmse: float = 0.0
    aic: float = 0.0
    bic: float = 0.0

    # Residual diagnostics
    residual_mean: float | None = None
    residual_std: float | None = None
    durbin_watson: float | None = None
    early_bias: float | None = None
    late_bias: float | None = None

    # Hindcast results
    hindcast_mape: float | None = None
    hindcast_correlation: float | None = None
    hindcast_bias: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "fit_id": self.fit_id,
            "timestamp": self.timestamp.isoformat(),
            "well_id": self.well_id,
            "product": self.product,
            "basin": self.basin,
            "formation": self.formation,
            "data_points_total": self.data_points_total,
            "data_points_used": self.data_points_used,
            "regime_start_idx": self.regime_start_idx,
            "b_min": self.b_min,
            "b_max": self.b_max,
            "dmin_annual": self.dmin_annual,
            "recency_half_life": self.recency_half_life,
            "regime_threshold": self.regime_threshold,
            "qi": self.qi,
            "di": self.di,
            "b": self.b,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "aic": self.aic,
            "bic": self.bic,
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "durbin_watson": self.durbin_watson,
            "early_bias": self.early_bias,
            "late_bias": self.late_bias,
            "hindcast_mape": self.hindcast_mape,
            "hindcast_correlation": self.hindcast_correlation,
            "hindcast_bias": self.hindcast_bias,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FitLogRecord":
        """Create from dictionary."""
        # Handle timestamp conversion
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            fit_id=data.get("fit_id", str(uuid.uuid4())),
            timestamp=timestamp,
            well_id=data.get("well_id", ""),
            product=data.get("product", ""),
            basin=data.get("basin"),
            formation=data.get("formation"),
            data_points_total=data.get("data_points_total", 0),
            data_points_used=data.get("data_points_used", 0),
            regime_start_idx=data.get("regime_start_idx", 0),
            b_min=data.get("b_min", 0.01),
            b_max=data.get("b_max", 1.5),
            dmin_annual=data.get("dmin_annual", 0.06),
            recency_half_life=data.get("recency_half_life", 12.0),
            regime_threshold=data.get("regime_threshold", 1.0),
            qi=data.get("qi", 0.0),
            di=data.get("di", 0.0),
            b=data.get("b", 0.0),
            r_squared=data.get("r_squared", 0.0),
            rmse=data.get("rmse", 0.0),
            aic=data.get("aic", 0.0),
            bic=data.get("bic", 0.0),
            residual_mean=data.get("residual_mean"),
            residual_std=data.get("residual_std"),
            durbin_watson=data.get("durbin_watson"),
            early_bias=data.get("early_bias"),
            late_bias=data.get("late_bias"),
            hindcast_mape=data.get("hindcast_mape"),
            hindcast_correlation=data.get("hindcast_correlation"),
            hindcast_bias=data.get("hindcast_bias"),
        )


@dataclass
class HindcastResult:
    """Result of hindcast (backtesting) validation.

    Hindcast splits historical data into training and holdout periods,
    fits on training data, and measures prediction accuracy on holdout.

    Attributes:
        well_id: Well identifier
        product: Product type
        training_months: Number of months used for training
        holdout_months: Number of months held out for validation

        # Training fit quality
        training_r_squared: R-squared on training data
        training_qi: Initial rate from training fit
        training_di: Decline rate from training fit
        training_b: B-factor from training fit

        # Holdout prediction accuracy
        mape: Mean Absolute Percentage Error (%)
        correlation: Pearson correlation coefficient
        bias: Mean (predicted - actual) / mean(actual), positive = over-prediction

        # Arrays for plotting (stored as lists for serialization)
        holdout_actual: Actual production in holdout period
        holdout_predicted: Predicted production in holdout period
        holdout_months_array: Month indices for holdout period
    """

    well_id: str
    product: str
    training_months: int
    holdout_months: int

    # Training fit quality
    training_r_squared: float
    training_qi: float
    training_di: float
    training_b: float

    # Holdout accuracy metrics
    mape: float  # Mean Absolute Percentage Error (%)
    correlation: float  # Pearson correlation
    bias: float  # Systematic over/under prediction

    # Arrays for plotting
    holdout_actual: np.ndarray = field(default_factory=lambda: np.array([]))
    holdout_predicted: np.ndarray = field(default_factory=lambda: np.array([]))
    holdout_months_array: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def is_good_hindcast(self) -> bool:
        """Check if hindcast meets quality threshold.

        A good hindcast has:
        - MAPE < 30%
        - Correlation > 0.7
        - Absolute bias < 0.3 (30%)
        """
        return (
            self.mape < 30.0
            and self.correlation > 0.7
            and abs(self.bias) < 0.3
        )

    def summary(self) -> dict[str, Any]:
        """Return summary dictionary."""
        return {
            "well_id": self.well_id,
            "product": self.product,
            "training_months": self.training_months,
            "holdout_months": self.holdout_months,
            "training_r_squared": self.training_r_squared,
            "mape": self.mape,
            "correlation": self.correlation,
            "bias": self.bias,
            "is_good_hindcast": self.is_good_hindcast,
        }


@dataclass
class ResidualDiagnostics:
    """Diagnostics from residual analysis.

    Analyzes residuals (actual - predicted) to detect systematic
    fit errors like autocorrelation and time-varying bias.

    Attributes:
        residuals: Array of residuals (actual - predicted)
        mean: Mean of residuals (should be near 0)
        std: Standard deviation of residuals
        autocorr_lag1: Lag-1 autocorrelation coefficient
        durbin_watson: Durbin-Watson statistic
            - ~2.0 = no autocorrelation
            - <1.5 = positive autocorrelation (underfit pattern)
            - >2.5 = negative autocorrelation (overfit oscillation)
        early_bias: Mean error in first half of data (fraction of mean rate)
        late_bias: Mean error in last half of data (fraction of mean rate)
        has_systematic_pattern: Whether residuals show concerning patterns
    """

    residuals: np.ndarray
    mean: float
    std: float
    autocorr_lag1: float
    durbin_watson: float
    early_bias: float
    late_bias: float
    has_systematic_pattern: bool

    @classmethod
    def compute(
        cls,
        actual: np.ndarray,
        predicted: np.ndarray,
    ) -> "ResidualDiagnostics":
        """Compute residual diagnostics from actual vs predicted values.

        Args:
            actual: Actual observed values
            predicted: Model predicted values

        Returns:
            ResidualDiagnostics with computed metrics
        """
        residuals = actual - predicted
        n = len(residuals)

        # Basic statistics
        mean = float(np.mean(residuals))
        std = float(np.std(residuals))

        # Autocorrelation at lag 1
        if n > 1:
            r = residuals - mean
            autocorr_lag1 = float(
                np.sum(r[:-1] * r[1:]) / np.sum(r ** 2)
            ) if np.sum(r ** 2) > 0 else 0.0
        else:
            autocorr_lag1 = 0.0

        # Durbin-Watson statistic
        if n > 1:
            diff_sq = np.sum(np.diff(residuals) ** 2)
            res_sq = np.sum(residuals ** 2)
            durbin_watson = float(diff_sq / res_sq) if res_sq > 0 else 2.0
        else:
            durbin_watson = 2.0

        # Early/late bias (as fraction of mean actual value)
        mean_actual = np.mean(actual) if len(actual) > 0 else 1.0
        if mean_actual == 0:
            mean_actual = 1.0  # Avoid division by zero

        half = n // 2
        if half > 0:
            early_bias = float(np.mean(residuals[:half]) / mean_actual)
            late_bias = float(np.mean(residuals[half:]) / mean_actual)
        else:
            early_bias = 0.0
            late_bias = 0.0

        # Detect systematic patterns
        has_systematic_pattern = (
            durbin_watson < 1.5  # Positive autocorrelation
            or durbin_watson > 2.5  # Negative autocorrelation
            or abs(early_bias) > 0.15  # >15% early bias
            or abs(late_bias) > 0.15  # >15% late bias
            or abs(autocorr_lag1) > 0.5  # Strong autocorrelation
        )

        return cls(
            residuals=residuals,
            mean=mean,
            std=std,
            autocorr_lag1=autocorr_lag1,
            durbin_watson=durbin_watson,
            early_bias=early_bias,
            late_bias=late_bias,
            has_systematic_pattern=has_systematic_pattern,
        )

    def summary(self) -> dict[str, Any]:
        """Return summary dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "autocorr_lag1": self.autocorr_lag1,
            "durbin_watson": self.durbin_watson,
            "early_bias": self.early_bias,
            "late_bias": self.late_bias,
            "has_systematic_pattern": self.has_systematic_pattern,
        }


@dataclass
class ParameterSuggestion:
    """Suggested fitting parameters learned from historical fits.

    Contains recommended parameter values based on analysis of
    accumulated fit logs for a specific grouping (global, basin, formation).

    Attributes:
        grouping: Description of grouping (e.g., "global", "Permian/Wolfcamp")
        sample_count: Number of fits used to derive suggestion
        product: Product type

        # Suggested parameter values
        suggested_recency_half_life: Recommended half-life for recency weighting
        suggested_regime_threshold: Recommended threshold for regime detection
        suggested_regime_window: Recommended window for regime detection
        suggested_regime_sustained_months: Recommended sustained months

        # Performance metrics from historical fits
        avg_r_squared: Average R-squared of fits with these parameters
        avg_hindcast_mape: Average hindcast MAPE (if available)
        confidence: Confidence level (based on sample count)
    """

    grouping: str
    sample_count: int
    product: str

    # Suggested values
    suggested_recency_half_life: float
    suggested_regime_threshold: float
    suggested_regime_window: int
    suggested_regime_sustained_months: int

    # Performance metrics
    avg_r_squared: float
    avg_hindcast_mape: float | None = None
    confidence: str = "low"  # "low", "medium", "high"

    def __post_init__(self):
        """Set confidence level based on sample count."""
        if self.sample_count >= 100:
            self.confidence = "high"
        elif self.sample_count >= 20:
            self.confidence = "medium"
        else:
            self.confidence = "low"

    def summary(self) -> dict[str, Any]:
        """Return summary dictionary."""
        return {
            "grouping": self.grouping,
            "sample_count": self.sample_count,
            "product": self.product,
            "suggested_recency_half_life": self.suggested_recency_half_life,
            "suggested_regime_threshold": self.suggested_regime_threshold,
            "suggested_regime_window": self.suggested_regime_window,
            "suggested_regime_sustained_months": self.suggested_regime_sustained_months,
            "avg_r_squared": self.avg_r_squared,
            "avg_hindcast_mape": self.avg_hindcast_mape,
            "confidence": self.confidence,
        }


@dataclass
class GroundTruthResult:
    """Result of comparing pyforecast fit against ARIES ground truth.

    Compares fitted decline curve parameters and forecasted production
    against expert/approved ARIES projections.

    Attributes:
        well_id: Well identifier
        product: Product type (oil, gas, water)

        # ARIES parameters (stored in daily rate / monthly decline)
        aries_qi: ARIES initial rate (daily)
        aries_di: ARIES initial decline (monthly fraction)
        aries_b: ARIES hyperbolic exponent
        aries_decline_type: ARIES decline type (EXP, HYP, HRM)

        # pyforecast parameters
        pyf_qi: pyforecast initial rate (daily)
        pyf_di: pyforecast initial decline (monthly fraction)
        pyf_b: pyforecast hyperbolic exponent

        # Parameter differences
        qi_pct_diff: (pyf - aries) / aries * 100
        di_pct_diff: (pyf - aries) / aries * 100
        b_abs_diff: pyf_b - aries_b

        # Forecast comparison metrics
        comparison_months: Number of months compared
        mape: Mean Absolute Percentage Error (%)
        correlation: Pearson correlation coefficient
        bias: Systematic over/under prediction (positive = pyf over-predicts)
        cumulative_diff_pct: (pyf_cum - aries_cum) / aries_cum * 100

        # Arrays for plotting (optional, can be large)
        forecast_months: Time array for forecast comparison
        aries_rates: ARIES forecasted rates (daily)
        pyf_rates: pyforecast forecasted rates (daily)
    """

    well_id: str
    product: Literal["oil", "gas", "water"]

    # ARIES parameters
    aries_qi: float
    aries_di: float
    aries_b: float
    aries_decline_type: str

    # pyforecast parameters
    pyf_qi: float
    pyf_di: float
    pyf_b: float

    # Parameter differences
    qi_pct_diff: float
    di_pct_diff: float
    b_abs_diff: float

    # Forecast comparison metrics
    comparison_months: int
    mape: float | None  # None when insufficient valid data points
    correlation: float
    bias: float
    cumulative_diff_pct: float

    # Arrays for plotting
    forecast_months: np.ndarray = field(default_factory=lambda: np.array([]))
    aries_rates: np.ndarray = field(default_factory=lambda: np.array([]))
    pyf_rates: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def mape_valid(self) -> bool:
        """Check if MAPE was successfully calculated.

        MAPE may be None when there are insufficient valid data points
        (fewer than 3 points above the minimum rate threshold).

        Returns:
            True if MAPE is available
        """
        return self.mape is not None

    @property
    def is_good_match(self) -> bool:
        """Check if pyforecast fit is a good match to ARIES ground truth.

        A good match meets all criteria:
        - MAPE < 20% (forecast curves are similar)
        - Correlation > 0.95 (curves track well together)
        - Cumulative diff < 15% (total volumes are similar)
        - b-factor diff < 0.3 (decline shape is similar)

        Returns None MAPE as a failed match (insufficient data).

        Returns:
            True if all criteria are met
        """
        if self.mape is None:
            return False
        return (
            self.mape < 20.0
            and self.correlation > 0.95
            and abs(self.cumulative_diff_pct) < 15.0
            and abs(self.b_abs_diff) < 0.3
        )

    @property
    def match_grade(self) -> str:
        """Return a grade for the match quality.

        Returns:
            "A" (excellent), "B" (good), "C" (fair), "D" (poor),
            or "X" (insufficient data - MAPE unavailable)
        """
        # Return X grade if MAPE couldn't be calculated
        if self.mape is None:
            return "X"

        score = 0

        # MAPE scoring
        if self.mape < 10:
            score += 3
        elif self.mape < 20:
            score += 2
        elif self.mape < 30:
            score += 1

        # Correlation scoring
        if self.correlation > 0.98:
            score += 3
        elif self.correlation > 0.95:
            score += 2
        elif self.correlation > 0.90:
            score += 1

        # Cumulative diff scoring
        abs_cum = abs(self.cumulative_diff_pct)
        if abs_cum < 5:
            score += 3
        elif abs_cum < 15:
            score += 2
        elif abs_cum < 25:
            score += 1

        # b-factor diff scoring
        abs_b = abs(self.b_abs_diff)
        if abs_b < 0.1:
            score += 3
        elif abs_b < 0.3:
            score += 2
        elif abs_b < 0.5:
            score += 1

        # Grade based on total score (max 12)
        if score >= 10:
            return "A"
        elif score >= 7:
            return "B"
        elif score >= 4:
            return "C"
        else:
            return "D"

    def summary(self) -> dict[str, Any]:
        """Return summary dictionary."""
        return {
            "well_id": self.well_id,
            "product": self.product,
            "aries_qi": self.aries_qi,
            "aries_di": self.aries_di,
            "aries_b": self.aries_b,
            "aries_decline_type": self.aries_decline_type,
            "pyf_qi": self.pyf_qi,
            "pyf_di": self.pyf_di,
            "pyf_b": self.pyf_b,
            "qi_pct_diff": self.qi_pct_diff,
            "di_pct_diff": self.di_pct_diff,
            "b_abs_diff": self.b_abs_diff,
            "comparison_months": self.comparison_months,
            "mape": self.mape,
            "correlation": self.correlation,
            "bias": self.bias,
            "cumulative_diff_pct": self.cumulative_diff_pct,
            "is_good_match": self.is_good_match,
            "match_grade": self.match_grade,
        }
