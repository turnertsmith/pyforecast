"""Data classes for refinement module.

Defines structured data types for fit logging, hindcast validation,
residual analysis, and ground truth comparison.
"""

from dataclasses import asdict, dataclass, field, fields as dataclass_fields
from datetime import date, datetime
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
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FitLogRecord":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        # Build kwargs from dataclass fields, using defaults for missing keys
        defaults = cls()
        kwargs = {"timestamp": timestamp}
        for f in dataclass_fields(cls):
            if f.name == "timestamp":
                continue
            if f.name in data and data[f.name] is not None:
                kwargs[f.name] = data[f.name]
            elif f.name == "fit_id" and f.name not in data:
                kwargs[f.name] = str(uuid.uuid4())
            else:
                kwargs[f.name] = getattr(defaults, f.name)
        return cls(**kwargs)


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

    _SUMMARY_EXCLUDE = {"holdout_actual", "holdout_predicted", "holdout_months_array"}

    def summary(self) -> dict[str, Any]:
        """Return summary dictionary."""
        result = {k: v for k, v in asdict(self).items() if k not in self._SUMMARY_EXCLUDE}
        result["is_good_hindcast"] = self.is_good_hindcast
        return result


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
        return {k: v for k, v in asdict(self).items() if k != "residuals"}


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
        return asdict(self)


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

    # Date alignment fields
    aries_start_date: date | None = None
    pyf_start_date: date | None = None
    alignment_warning: str | None = None

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

    @staticmethod
    def _tier_score(value: float, thresholds: tuple, reverse: bool = False) -> int:
        """Score a value against 3 thresholds (3/2/1/0 points)."""
        for i, thresh in enumerate(thresholds):
            if (value > thresh) if reverse else (value < thresh):
                return 3 - i
        return 0

    @property
    def match_grade(self) -> str:
        """Return a grade for the match quality.

        Returns:
            "A" (excellent), "B" (good), "C" (fair), "D" (poor),
            or "X" (insufficient data - MAPE unavailable)
        """
        if self.mape is None:
            return "X"

        score = (
            self._tier_score(self.mape, (10, 20, 30))
            + self._tier_score(self.correlation, (0.98, 0.95, 0.90), reverse=True)
            + self._tier_score(abs(self.cumulative_diff_pct), (5, 15, 25))
            + self._tier_score(abs(self.b_abs_diff), (0.1, 0.3, 0.5))
        )

        if score >= 10:
            return "A"
        elif score >= 7:
            return "B"
        elif score >= 4:
            return "C"
        return "D"

    _SUMMARY_EXCLUDE = {"forecast_months", "aries_rates", "pyf_rates"}

    def summary(self) -> dict[str, Any]:
        """Return summary dictionary."""
        result = {k: v for k, v in asdict(self).items() if k not in self._SUMMARY_EXCLUDE}
        result["is_good_match"] = self.is_good_match
        result["match_grade"] = self.match_grade
        # Convert dates to ISO strings
        for key in ("aries_start_date", "pyf_start_date"):
            val = result.get(key)
            if isinstance(val, date):
                result[key] = val.isoformat()
        return result
