"""Decline curve fitting with regime detection and recency weighting.

Features:
- Regime change detection (RTP, refrac) to identify relevant data window
- Exponential decay weighting to favor recent production data
- scipy curve_fit with fallback to differential_evolution
"""

from dataclasses import dataclass
import numpy as np
from scipy import optimize
from scipy.stats import linregress

from .models import HyperbolicModel, ForecastResult
from .regime_detection import detect_regime_change_improved, RegimeDetectionConfig


@dataclass
class FittingConfig:
    """Configuration for decline curve fitting.

    Attributes:
        b_min: Minimum b factor (default 0.01, exponential-like)
        b_max: Maximum b factor (default 1.5, super-harmonic)
        dmin_annual: Terminal decline rate as annual fraction (default 0.06 = 6%)
        regime_threshold: Fractional increase to detect regime change (default 1.0 = 100%)
        regime_window: Window size for trend fitting in regime detection (default 6)
        regime_sustained_months: Months elevation must be sustained to confirm regime change (default 2)
        recency_half_life: Half-life in months for exponential decay weighting (default 12)
            Lower values = more aggressive weighting toward recent data
        min_points: Minimum data points required for fitting (default 6)
    """
    b_min: float = 0.01
    b_max: float = 1.5
    dmin_annual: float = 0.06
    regime_threshold: float = 1.0
    regime_window: int = 6
    regime_sustained_months: int = 2
    recency_half_life: float = 12.0
    min_points: int = 6

    @property
    def dmin_monthly(self) -> float:
        """Convert annual Dmin to monthly."""
        return self.dmin_annual / 12.0

    @classmethod
    def from_pyforecast_config(
        cls,
        config: "PyForecastConfig",  # noqa: F821
        product: str,
    ) -> "FittingConfig":
        """Create FittingConfig from PyForecastConfig for a specific product.

        Args:
            config: PyForecastConfig instance
            product: Product name (oil, gas, water)

        Returns:
            FittingConfig for the specified product
        """
        product_config = config.get_product_config(product)
        return cls(
            b_min=product_config.b_min,
            b_max=product_config.b_max,
            dmin_annual=product_config.dmin,
            regime_threshold=config.regime.threshold,
            regime_window=config.regime.window,
            regime_sustained_months=config.regime.sustained_months,
            recency_half_life=config.fitting.recency_half_life,
            min_points=config.fitting.min_points,
        )


class DeclineFitter:
    """Fits hyperbolic decline curves to production data."""

    def __init__(self, config: FittingConfig | None = None):
        """Initialize fitter with configuration.

        Args:
            config: Fitting configuration, uses defaults if None
        """
        self.config = config or FittingConfig()

    def detect_regime_change(
        self,
        rates: np.ndarray,
        window: int | None = None
    ) -> int:
        """Detect the most recent regime change in production data.

        Uses trend extrapolation to detect regime changes (RTP, refrac).
        A regime change is confirmed when:
        1. Production exceeds projected trend by threshold percentage
        2. The elevation is sustained for configured number of months

        Args:
            rates: Production rates array (chronological order)
            window: Window size for trend fitting (uses config default if None)

        Returns:
            Index of the start of the current regime (0 if no change detected)
        """
        regime_config = RegimeDetectionConfig(
            window_size=window or self.config.regime_window,
            n_sigma=2.5,
            min_pct_increase=self.config.regime_threshold,
            sustained_months=self.config.regime_sustained_months,
            min_data_points=self.config.min_points,
        )
        return detect_regime_change_improved(rates, regime_config)

    def compute_weights(
        self,
        n_points: int,
        half_life: float | None = None
    ) -> np.ndarray:
        """Compute exponential decay weights favoring recent data.

        Args:
            n_points: Number of data points
            half_life: Decay half-life in months (uses config default if None)

        Returns:
            Array of weights (most recent = highest weight)
        """
        if half_life is None:
            half_life = self.config.recency_half_life

        # Time from most recent point (backwards)
        t_from_end = np.arange(n_points - 1, -1, -1, dtype=float)

        # Exponential decay: w = exp(-ln(2) * t / half_life)
        decay_rate = np.log(2) / half_life
        weights = np.exp(-decay_rate * t_from_end)

        # Normalize so sum = n_points (preserves effective sample size interpretation)
        weights = weights * n_points / np.sum(weights)

        return weights

    def initial_guess(
        self,
        t: np.ndarray,
        q: np.ndarray,
        weights: np.ndarray | None = None
    ) -> tuple[float, float, float]:
        """Estimate initial parameters from data.

        Uses log-linear regression for initial Di estimate and data range for qi.

        Args:
            t: Time array (months)
            q: Production rate array
            weights: Optional weights array

        Returns:
            Tuple of (qi_guess, di_guess, b_guess)
        """
        # Filter out zero/negative rates for log transform
        mask = q > 0
        if np.sum(mask) < 2:
            # Fallback to simple estimates
            return float(np.max(q)), 0.1, 0.5

        t_valid = t[mask]
        q_valid = q[mask]

        # qi estimate: extrapolate to t=0 or use first point
        qi_guess = float(q_valid[0]) if len(q_valid) > 0 else 100.0

        # Di estimate from log-linear regression (exponential assumption)
        log_q = np.log(q_valid)
        slope, intercept, _, _, _ = linregress(t_valid, log_q)

        # -slope is the exponential decline rate
        di_guess = max(0.001, min(1.0, -slope))

        # Refine qi from intercept
        qi_guess = max(qi_guess, np.exp(intercept))

        # Start with moderate b
        b_guess = 0.5

        return qi_guess, di_guess, b_guess

    def fit(
        self,
        t: np.ndarray,
        q: np.ndarray,
        apply_regime_detection: bool = True,
        apply_weights: bool = True,
        capture_residuals: bool = False,
    ) -> ForecastResult:
        """Fit hyperbolic decline model to production data.

        Args:
            t: Time array in months (relative, starting from 0)
            q: Production rate array
            apply_regime_detection: Whether to detect and use only current regime
            apply_weights: Whether to apply exponential decay weighting
            capture_residuals: Whether to capture residuals in the result
                (for residual analysis). Adds t_fit and residuals to result.

        Returns:
            ForecastResult with fitted model and quality metrics

        Raises:
            ValueError: If insufficient data points for fitting
        """
        t = np.asarray(t, dtype=float)
        q = np.asarray(q, dtype=float)

        # Detect regime change
        if apply_regime_detection:
            regime_start = self.detect_regime_change(q)
        else:
            regime_start = 0

        # Use data from current regime only
        t_fit = t[regime_start:] - t[regime_start]  # Reset time to 0 at regime start
        q_fit = q[regime_start:]

        if len(q_fit) < self.config.min_points:
            raise ValueError(
                f"Insufficient data: {len(q_fit)} points, need {self.config.min_points}"
            )

        # Compute weights
        if apply_weights:
            weights = self.compute_weights(len(q_fit))
        else:
            weights = np.ones(len(q_fit))

        # Get initial parameter guesses
        qi0, di0, b0 = self.initial_guess(t_fit, q_fit, weights)

        # Parameter bounds
        bounds_lower = [0.1, 0.001, self.config.b_min]
        bounds_upper = [qi0 * 3, 2.0, self.config.b_max]

        # Ensure initial guess is within bounds
        qi0 = np.clip(qi0, bounds_lower[0], bounds_upper[0])
        di0 = np.clip(di0, bounds_lower[1], bounds_upper[1])
        b0 = np.clip(b0, bounds_lower[2], bounds_upper[2])

        # Define objective function (hyperbolic without Dmin for fitting)
        def hyperbolic(t, qi, di, b):
            # Avoid numerical issues with very small b
            if b < 0.02:
                return qi * np.exp(-di * t)
            return qi / np.power(1 + b * di * t, 1 / b)

        # Try curve_fit first
        try:
            popt, pcov = optimize.curve_fit(
                hyperbolic,
                t_fit,
                q_fit,
                p0=[qi0, di0, b0],
                bounds=(bounds_lower, bounds_upper),
                sigma=1.0 / np.sqrt(weights),  # Higher weight = lower sigma
                absolute_sigma=False,
                method='trf',
                maxfev=5000
            )
            qi_fit, di_fit, b_fit = popt

        except (RuntimeError, optimize.OptimizeWarning):
            # Fallback to differential evolution
            def objective(params):
                qi, di, b = params
                pred = hyperbolic(t_fit, qi, di, b)
                residuals = (q_fit - pred) ** 2 * weights
                return np.sum(residuals)

            result = optimize.differential_evolution(
                objective,
                bounds=list(zip(bounds_lower, bounds_upper)),
                seed=42,
                maxiter=1000,
                tol=1e-7
            )
            qi_fit, di_fit, b_fit = result.x

        # Create model with Dmin
        model = HyperbolicModel(
            qi=qi_fit,
            di=di_fit,
            b=b_fit,
            dmin=self.config.dmin_monthly
        )

        # Calculate fit quality metrics
        q_pred = model.rate(t_fit)
        metrics = self._calculate_metrics(q_fit, q_pred, n_params=3)

        # Capture residuals if requested
        t_fit_out = None
        residuals_out = None
        if capture_residuals:
            t_fit_out = t_fit.copy()
            residuals_out = q_fit - q_pred

        return ForecastResult(
            model=model,
            r_squared=metrics['r_squared'],
            rmse=metrics['rmse'],
            aic=metrics['aic'],
            bic=metrics['bic'],
            regime_start_idx=regime_start,
            data_points_used=len(q_fit),
            t_fit=t_fit_out,
            residuals=residuals_out,
        )

    def _calculate_metrics(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        n_params: int = 3
    ) -> dict:
        """Calculate fit quality metrics.

        Args:
            observed: Observed values
            predicted: Predicted values
            n_params: Number of model parameters

        Returns:
            Dictionary with r_squared, rmse, aic, bic
        """
        n = len(observed)
        residuals = observed - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)

        # R-squared
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # RMSE
        rmse = np.sqrt(ss_res / n)

        # Log-likelihood (assuming normal errors)
        if ss_res > 0:
            sigma2 = ss_res / n
            log_likelihood = -n / 2 * (np.log(2 * np.pi * sigma2) + 1)
        else:
            log_likelihood = 0

        # AIC and BIC
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(n) - 2 * log_likelihood

        return {
            'r_squared': r_squared,
            'rmse': rmse,
            'aic': aic,
            'bic': bic
        }
