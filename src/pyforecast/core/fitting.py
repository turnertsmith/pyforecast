"""Decline curve fitting with regime detection and recency weighting.

Features:
- Regime change detection (RTP, refrac) to identify relevant data window
- Exponential decay weighting to favor recent production data
- scipy curve_fit with fallback to differential_evolution
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import optimize
from scipy.stats import linregress

from .models import HyperbolicModel, ForecastResult
from .regime_detection import detect_regime_change_improved, RegimeDetectionConfig


class ModelSelection(str, Enum):
    """Model selection strategy for decline curve fitting."""
    HYPERBOLIC = "hyperbolic"
    EXPONENTIAL = "exponential"
    HARMONIC = "harmonic"
    AUTO = "auto"


def _hyperbolic_model(t, qi, di, b):
    """Hyperbolic decline model function for curve fitting.

    Handles the exponential special case (b < 0.02) to avoid numerical issues.

    Args:
        t: Time array
        qi: Initial rate
        di: Initial decline rate
        b: Hyperbolic exponent

    Returns:
        Production rate array
    """
    if b < 0.02:
        return qi * np.exp(-di * t)
    return qi / np.power(1 + b * di * t, 1 / b)


@dataclass
class _PreparedFitData:
    """Internal container for prepared fit data after regime detection."""
    t_fit: np.ndarray
    q_fit: np.ndarray
    weights: np.ndarray
    regime_start: int
    effective_dmin: float


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
        acceptable_r_squared: Threshold for ForecastResult.is_acceptable (default 0.7)
        model_selection: Model selection strategy (default ModelSelection.HYPERBOLIC)
        estimate_dmin: Whether to estimate terminal decline from late-time data (default False)
        dmin_min_annual: Minimum allowable Dmin when estimating (default 0.024 = 2.4%)
        dmin_max_annual: Maximum allowable Dmin when estimating (default 0.24 = 24%)
        adaptive_regime_detection: Use CV-adaptive thresholds for regime detection (default False)
    """
    b_min: float = 0.01
    b_max: float = 1.5
    dmin_annual: float = 0.06
    regime_threshold: float = 1.0
    regime_window: int = 6
    regime_sustained_months: int = 2
    recency_half_life: float = 12.0
    min_points: int = 6
    acceptable_r_squared: float = 0.7
    model_selection: str | ModelSelection = ModelSelection.HYPERBOLIC
    estimate_dmin: bool = False
    dmin_min_annual: float = 0.024
    dmin_max_annual: float = 0.24
    adaptive_regime_detection: bool = False

    @property
    def dmin_monthly(self) -> float:
        """Convert annual Dmin to monthly."""
        return self.dmin_annual / 12.0

    @property
    def model_selection_enum(self) -> ModelSelection:
        """Get model_selection as a ModelSelection enum."""
        if isinstance(self.model_selection, ModelSelection):
            return self.model_selection
        try:
            return ModelSelection(self.model_selection.lower())
        except ValueError:
            return ModelSelection.HYPERBOLIC

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

        # Get product-specific recency half-life if available, fallback to default
        product_half_life = getattr(product_config, 'recency_half_life', None)
        recency_half_life = product_half_life if product_half_life is not None else config.fitting.recency_half_life

        # Get fitting-level settings with defaults
        fitting = config.fitting
        model_selection = getattr(fitting, 'model_selection', 'hyperbolic')
        estimate_dmin = getattr(fitting, 'estimate_dmin', False)
        dmin_min_annual = getattr(fitting, 'dmin_min_annual', 0.024)
        dmin_max_annual = getattr(fitting, 'dmin_max_annual', 0.24)
        adaptive_regime_detection = getattr(fitting, 'adaptive_regime_detection', False)

        return cls(
            b_min=product_config.b_min,
            b_max=product_config.b_max,
            dmin_annual=product_config.dmin,
            regime_threshold=config.regime.threshold,
            regime_window=config.regime.window,
            regime_sustained_months=config.regime.sustained_months,
            recency_half_life=recency_half_life,
            min_points=config.fitting.min_points,
            acceptable_r_squared=config.validation.acceptable_r_squared,
            model_selection=model_selection,
            estimate_dmin=estimate_dmin,
            dmin_min_annual=dmin_min_annual,
            dmin_max_annual=dmin_max_annual,
            adaptive_regime_detection=adaptive_regime_detection,
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
            adaptive=self.config.adaptive_regime_detection,
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

    def estimate_terminal_decline(
        self,
        t: np.ndarray,
        q: np.ndarray,
        min_months: int = 36,
        late_fraction: float = 0.25,
        min_r_squared: float = 0.8,
    ) -> float | None:
        """Estimate terminal decline rate from late-time production data.

        Fits an exponential decline to the last portion of production history
        to estimate the terminal decline rate (Dmin).

        Requirements:
        - At least 36 months of history (default)
        - Good fit (R² > 0.8) to late-time data
        - Result within configured bounds

        Args:
            t: Time array (months)
            q: Production rate array
            min_months: Minimum months required for estimation (default 36)
            late_fraction: Fraction of data to use for late-time fit (default 0.25)
            min_r_squared: Minimum R² required to trust estimate (default 0.8)

        Returns:
            Estimated Dmin (monthly) or None if estimation not possible/reliable
        """
        if len(t) < min_months:
            return None

        # Use last 25% of data
        n_late = max(6, int(len(t) * late_fraction))
        t_late = t[-n_late:]
        q_late = q[-n_late:]

        # Filter positive values
        mask = q_late > 0
        if np.sum(mask) < 4:
            return None

        t_valid = t_late[mask]
        q_valid = q_late[mask]

        try:
            # Fit exponential: log(q) = log(q0) - D*t
            log_q = np.log(q_valid)
            slope, intercept, r_value, _, _ = linregress(
                t_valid - t_valid[0], log_q
            )

            r_squared = r_value ** 2

            if r_squared < min_r_squared:
                return None

            # Decline rate is negative of slope
            dmin_monthly = -slope

            # Clip to configured bounds (convert annual to monthly)
            dmin_min = self.config.dmin_min_annual / 12.0
            dmin_max = self.config.dmin_max_annual / 12.0

            if dmin_monthly < dmin_min or dmin_monthly > dmin_max:
                # Outside reasonable bounds, don't use
                return None

            return float(dmin_monthly)

        except (ValueError, RuntimeError):
            return None

    def get_effective_dmin(
        self,
        t: np.ndarray,
        q: np.ndarray,
    ) -> float:
        """Get effective Dmin, estimating from data if configured.

        Args:
            t: Time array
            q: Production rates

        Returns:
            Effective Dmin (monthly)
        """
        if self.config.estimate_dmin:
            estimated = self.estimate_terminal_decline(t, q)
            if estimated is not None:
                return estimated

        return self.config.dmin_monthly

    def initial_guess(
        self,
        t: np.ndarray,
        q: np.ndarray,
        weights: np.ndarray | None = None
    ) -> tuple[float, float, float]:
        """Estimate initial parameters from data.

        Uses log-linear regression for initial Di estimate, data range for qi,
        and curvature analysis for b-factor estimation.

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

        # Curvature-based b-factor estimation
        b_guess = self._estimate_b_from_curvature(t_valid, log_q)

        return qi_guess, di_guess, b_guess

    def _estimate_b_from_curvature(
        self,
        t: np.ndarray,
        log_q: np.ndarray
    ) -> float:
        """Estimate b-factor from curvature of log(q) vs t.

        For hyperbolic decline: log(q) = log(qi) - (1/b)*log(1 + b*Di*t)
        The curvature of log(q) vs t relates to b: higher b = more curvature.

        Uses quadratic fit to log(q) vs t:
            log(q) = a*t^2 + b_coef*t + c

        The b-factor can be estimated from the ratio of quadratic to linear terms.
        For exponential (b~0): a ~ 0
        For harmonic (b~1): significant positive curvature

        Args:
            t: Time array (normalized)
            log_q: Log of production rates

        Returns:
            Estimated b-factor clipped to [b_min, b_max]
        """
        if len(t) < 4:
            # Not enough points for curvature estimation
            return 0.5

        try:
            # Normalize time to [0, 1] to improve numerical stability
            t_norm = (t - t[0]) / (t[-1] - t[0]) if t[-1] > t[0] else t

            # Fit quadratic: log(q) = a*t^2 + b_coef*t + c
            coeffs = np.polyfit(t_norm, log_q, 2)
            a, b_coef, c = coeffs

            if abs(b_coef) < 0.01:
                # Very slow decline, default to moderate b
                b_guess = 0.5
            else:
                # Estimate b from curvature ratio
                b_guess = -2.0 * a / b_coef

                # Validate and adjust based on physical interpretation
                if a >= 0 and abs(a) < abs(b_coef) * 0.05:
                    b_guess = 0.1  # Near-exponential

        except (np.linalg.LinAlgError, ValueError):
            b_guess = 0.5

        # Clip to configured bounds
        return float(np.clip(b_guess, self.config.b_min, self.config.b_max))

    def _prepare_fit_data(
        self,
        t: np.ndarray,
        q: np.ndarray,
        apply_regime_detection: bool = True,
        apply_weights: bool = True,
    ) -> _PreparedFitData:
        """Prepare data for fitting: regime detection, slicing, weighting.

        This is the shared preprocessing step used by all fitting methods.

        Args:
            t: Time array in months
            q: Production rate array
            apply_regime_detection: Whether to detect and use only current regime
            apply_weights: Whether to apply exponential decay weighting

        Returns:
            _PreparedFitData with t_fit, q_fit, weights, regime_start, effective_dmin

        Raises:
            ValueError: If insufficient data points for fitting
        """
        t = np.asarray(t, dtype=float)
        q = np.asarray(q, dtype=float)

        # Detect regime change
        regime_start = self.detect_regime_change(q) if apply_regime_detection else 0

        # Use data from current regime only
        t_fit = t[regime_start:] - t[regime_start]
        q_fit = q[regime_start:]

        if len(q_fit) < self.config.min_points:
            raise ValueError(
                f"Insufficient data: {len(q_fit)} points, need {self.config.min_points}"
            )

        # Compute weights
        weights = self.compute_weights(len(q_fit)) if apply_weights else np.ones(len(q_fit))

        # Get effective Dmin (may be estimated from data)
        effective_dmin = self.get_effective_dmin(t, q)

        return _PreparedFitData(
            t_fit=t_fit,
            q_fit=q_fit,
            weights=weights,
            regime_start=regime_start,
            effective_dmin=effective_dmin,
        )

    def _build_bounds(
        self,
        qi0: float,
        dmin: float,
        include_b: bool = True,
    ) -> tuple[list[float], list[float]]:
        """Build parameter bounds for optimization.

        Args:
            qi0: Initial qi guess (used for upper bound)
            dmin: Effective Dmin (monthly)
            include_b: Whether to include b-factor bounds

        Returns:
            Tuple of (bounds_lower, bounds_upper) lists
        """
        di_min = max(0.001, dmin * 1.1)
        if include_b:
            return [0.1, di_min, self.config.b_min], [qi0 * 3, 2.0, self.config.b_max]
        return [0.1, di_min], [qi0 * 3, 2.0]

    def _build_result(
        self,
        model: HyperbolicModel,
        t_fit: np.ndarray,
        q_fit: np.ndarray,
        regime_start: int,
        n_params: int = 3,
        capture_residuals: bool = False,
    ) -> ForecastResult:
        """Build a ForecastResult from a fitted model.

        Args:
            model: Fitted HyperbolicModel
            t_fit: Time array used for fitting
            q_fit: Production data used for fitting
            regime_start: Regime start index
            n_params: Number of model parameters
            capture_residuals: Whether to capture residuals

        Returns:
            ForecastResult with quality metrics
        """
        q_pred = model.rate(t_fit)
        metrics = self._calculate_metrics(q_fit, q_pred, n_params=n_params)

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
            acceptable_r_squared=self.config.acceptable_r_squared,
        )

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
        data = self._prepare_fit_data(t, q, apply_regime_detection, apply_weights)

        # Get initial parameter guesses
        qi0, di0, b0 = self.initial_guess(data.t_fit, data.q_fit, data.weights)

        # Parameter bounds
        bounds_lower, bounds_upper = self._build_bounds(qi0, data.effective_dmin)

        # Ensure initial guess is within bounds
        qi0 = np.clip(qi0, bounds_lower[0], bounds_upper[0])
        di0 = np.clip(di0, bounds_lower[1], bounds_upper[1])
        b0 = np.clip(b0, bounds_lower[2], bounds_upper[2])

        def objective(params):
            qi, di, b = params
            pred = _hyperbolic_model(data.t_fit, qi, di, b)
            resid = (data.q_fit - pred) ** 2 * data.weights
            return np.sum(resid)

        # Adaptive optimizer selection based on data characteristics
        n_points = len(data.q_fit)
        cv = np.std(data.q_fit) / np.mean(data.q_fit) if np.mean(data.q_fit) > 0 else 0.5

        # Choose optimization strategy
        if n_points < 12:
            maxfev = 3000
            use_de = False
        elif cv > 0.5:
            maxfev = 5000
            use_de = True
        else:
            maxfev = 5000
            use_de = False

        fit_success = False
        qi_fit, di_fit, b_fit = qi0, di0, b0

        # Primary optimization
        if not use_de:
            try:
                popt, pcov = optimize.curve_fit(
                    _hyperbolic_model,
                    data.t_fit,
                    data.q_fit,
                    p0=[qi0, di0, b0],
                    bounds=(bounds_lower, bounds_upper),
                    sigma=1.0 / np.sqrt(data.weights),
                    absolute_sigma=False,
                    method='trf',
                    maxfev=maxfev
                )
                qi_fit, di_fit, b_fit = popt
                fit_success = True
            except (RuntimeError, optimize.OptimizeWarning):
                pass

        # Fallback or primary DE
        if not fit_success or use_de:
            try:
                de_maxiter = 500 if cv > 0.5 else 1000

                result = optimize.differential_evolution(
                    objective,
                    bounds=list(zip(bounds_lower, bounds_upper)),
                    seed=42,
                    maxiter=de_maxiter,
                    tol=1e-7
                )
                qi_fit, di_fit, b_fit = result.x
                fit_success = True
            except Exception:
                pass

        # Check if fit is acceptable; if not, try basin hopping
        if fit_success:
            test_model = HyperbolicModel(qi=qi_fit, di=di_fit, b=b_fit, dmin=data.effective_dmin)
            q_test = test_model.rate(data.t_fit)
            test_metrics = self._calculate_metrics(data.q_fit, q_test, n_params=3)

            if test_metrics['r_squared'] < 0.8 and n_points >= 12:
                try:
                    bh_result = self._basin_hopping_fit(
                        data.t_fit, data.q_fit, data.weights,
                        [qi0, di0, b0],
                        bounds_lower, bounds_upper,
                        objective,
                    )
                    if bh_result is not None:
                        qi_bh, di_bh, b_bh = bh_result
                        bh_model = HyperbolicModel(qi=qi_bh, di=di_bh, b=b_bh, dmin=data.effective_dmin)
                        q_bh = bh_model.rate(data.t_fit)
                        bh_metrics = self._calculate_metrics(data.q_fit, q_bh, n_params=3)

                        if bh_metrics['r_squared'] > test_metrics['r_squared']:
                            qi_fit, di_fit, b_fit = qi_bh, di_bh, b_bh
                except Exception:
                    pass

        model = HyperbolicModel(
            qi=qi_fit,
            di=di_fit,
            b=b_fit,
            dmin=data.effective_dmin
        )

        return self._build_result(
            model, data.t_fit, data.q_fit, data.regime_start,
            n_params=3, capture_residuals=capture_residuals,
        )

    def _basin_hopping_fit(
        self,
        t_fit: np.ndarray,
        q_fit: np.ndarray,
        weights: np.ndarray,
        x0: list,
        bounds_lower: list,
        bounds_upper: list,
        objective_func,
        n_iterations: int = 50,
    ) -> tuple | None:
        """Perform basin hopping optimization for difficult fits.

        Basin hopping is a global optimization method that combines random
        perturbations with local minimization to escape local minima.

        Args:
            t_fit: Time array
            q_fit: Production rates
            weights: Fitting weights
            x0: Initial parameter guess [qi, di, b]
            bounds_lower: Lower bounds
            bounds_upper: Upper bounds
            objective_func: Objective function to minimize
            n_iterations: Number of basin hopping iterations

        Returns:
            Tuple of (qi, di, b) or None if failed
        """

        class BoundsConstraint:
            """Bounds constraint for basin hopping."""
            def __init__(self, lower, upper):
                self.lower = np.array(lower)
                self.upper = np.array(upper)

            def __call__(self, **kwargs):
                x = kwargs["x_new"]
                return np.all(x >= self.lower) and np.all(x <= self.upper)

        try:
            minimizer_kwargs = {
                "method": "L-BFGS-B",
                "bounds": list(zip(bounds_lower, bounds_upper)),
            }

            result = optimize.basinhopping(
                objective_func,
                x0,
                minimizer_kwargs=minimizer_kwargs,
                niter=n_iterations,
                accept_test=BoundsConstraint(bounds_lower, bounds_upper),
                seed=42,
            )

            if result.success or result.fun < objective_func(x0):
                return tuple(result.x)

        except Exception:
            pass

        return None

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

    def compute_prediction_intervals(
        self,
        t: np.ndarray,
        q: np.ndarray,
        forecast_months: int = 120,
        n_bootstrap: int = 100,
        percentiles: tuple = (5, 50, 95),
        apply_regime_detection: bool = True,
        apply_weights: bool = True,
        seed: int = 42,
    ) -> dict:
        """Compute bootstrap prediction intervals for the forecast.

        Uses residual bootstrap to estimate uncertainty in the forecast.
        Resamples residuals, adds them to fitted values, refits, and
        generates forecast from each bootstrap sample.

        Args:
            t: Time array in months
            q: Production rate array
            forecast_months: Months to forecast beyond data
            n_bootstrap: Number of bootstrap resamples (default 100)
            percentiles: Percentiles to compute (default P5, P50, P95)
            apply_regime_detection: Whether to detect regime changes
            apply_weights: Whether to apply recency weighting
            seed: Random seed for reproducibility

        Returns:
            Dict with keys:
            - 't_forecast': Time array for forecast period
            - 'P5', 'P50', 'P95': Rate arrays for each percentile
            - 'mean': Mean forecast across bootstrap samples
        """
        rng = np.random.default_rng(seed)

        t = np.asarray(t, dtype=float)
        q = np.asarray(q, dtype=float)

        # Get base fit
        base_result = self.fit(
            t, q,
            apply_regime_detection=apply_regime_detection,
            apply_weights=apply_weights,
            capture_residuals=True,
        )

        if base_result.residuals is None or base_result.t_fit is None:
            raise ValueError("Base fit did not capture residuals")

        t_fit = base_result.t_fit
        residuals = base_result.residuals
        q_pred = base_result.model.rate(t_fit)

        # Forecast time array
        t_last = t_fit[-1]
        t_forecast = np.linspace(t_last, t_last + forecast_months, forecast_months + 1)

        # Store bootstrap forecasts
        bootstrap_forecasts = []

        for _ in range(n_bootstrap):
            # Resample residuals with replacement
            resampled_residuals = rng.choice(residuals, size=len(residuals), replace=True)

            # Create bootstrap sample: fitted + resampled residuals
            q_bootstrap = q_pred + resampled_residuals

            # Ensure positive rates
            q_bootstrap = np.maximum(q_bootstrap, 0.1)

            # Refit to bootstrap sample
            try:
                bootstrap_result = self._fit_bootstrap(
                    t_fit, q_bootstrap,
                    base_result.regime_start_idx,
                    apply_weights,
                    effective_dmin=base_result.model.dmin,
                )

                # Generate forecast from bootstrap fit
                q_forecast = bootstrap_result.model.rate(t_forecast)
                bootstrap_forecasts.append(q_forecast)

            except (ValueError, RuntimeError):
                continue

        if len(bootstrap_forecasts) < 10:
            raise ValueError(f"Too few successful bootstrap fits: {len(bootstrap_forecasts)}")

        # Stack forecasts and compute percentiles
        forecast_array = np.array(bootstrap_forecasts)

        result = {
            't_forecast': t_forecast,
            'mean': np.mean(forecast_array, axis=0),
        }

        for p in percentiles:
            result[f'P{p}'] = np.percentile(forecast_array, p, axis=0)

        return result

    def _fit_bootstrap(
        self,
        t_fit: np.ndarray,
        q_fit: np.ndarray,
        regime_start: int,
        apply_weights: bool,
        effective_dmin: float | None = None,
    ) -> ForecastResult:
        """Internal method to fit a bootstrap sample.

        Args:
            t_fit: Time array (already regime-adjusted)
            q_fit: Bootstrap production rates
            regime_start: Original regime start index
            apply_weights: Whether to apply weights
            effective_dmin: Effective Dmin to use (defaults to config dmin_monthly)

        Returns:
            ForecastResult from bootstrap fit
        """
        dmin = effective_dmin if effective_dmin is not None else self.config.dmin_monthly
        weights = self.compute_weights(len(q_fit)) if apply_weights else np.ones(len(q_fit))

        qi0, di0, b0 = self.initial_guess(t_fit, q_fit, weights)

        bounds_lower, bounds_upper = self._build_bounds(qi0, dmin)

        qi0 = np.clip(qi0, bounds_lower[0], bounds_upper[0])
        di0 = np.clip(di0, bounds_lower[1], bounds_upper[1])
        b0 = np.clip(b0, bounds_lower[2], bounds_upper[2])

        try:
            popt, _ = optimize.curve_fit(
                _hyperbolic_model,
                t_fit,
                q_fit,
                p0=[qi0, di0, b0],
                bounds=(bounds_lower, bounds_upper),
                sigma=1.0 / np.sqrt(weights),
                absolute_sigma=False,
                method='trf',
                maxfev=2000
            )
            qi_fit, di_fit, b_fit = popt
        except (RuntimeError, optimize.OptimizeWarning):
            qi_fit, di_fit, b_fit = qi0, di0, b0

        model = HyperbolicModel(
            qi=qi_fit,
            di=di_fit,
            b=b_fit,
            dmin=dmin
        )

        return self._build_result(
            model, t_fit, q_fit, regime_start, n_params=3,
        )

    def fit_with_intervals(
        self,
        t: np.ndarray,
        q: np.ndarray,
        forecast_months: int = 120,
        n_bootstrap: int = 100,
        apply_regime_detection: bool = True,
        apply_weights: bool = True,
        capture_residuals: bool = False,
    ) -> ForecastResult:
        """Fit model and compute prediction intervals.

        Combines fitting with bootstrap interval computation.

        Args:
            t: Time array in months
            q: Production rate array
            forecast_months: Months to forecast for intervals
            n_bootstrap: Number of bootstrap samples
            apply_regime_detection: Whether to detect regime changes
            apply_weights: Whether to apply recency weighting
            capture_residuals: Whether to capture residuals

        Returns:
            ForecastResult with prediction_intervals populated
        """
        result = self.fit(
            t, q,
            apply_regime_detection=apply_regime_detection,
            apply_weights=apply_weights,
            capture_residuals=capture_residuals,
        )

        try:
            intervals = self.compute_prediction_intervals(
                t, q,
                forecast_months=forecast_months,
                n_bootstrap=n_bootstrap,
                apply_regime_detection=apply_regime_detection,
                apply_weights=apply_weights,
            )
            result.prediction_intervals = intervals
        except ValueError:
            pass

        return result

    def _fit_with_fixed_b(
        self,
        t_fit: np.ndarray,
        q_fit: np.ndarray,
        weights: np.ndarray,
        b_fixed: float,
        regime_start: int,
        capture_residuals: bool = False,
    ) -> ForecastResult:
        """Fit decline model with a fixed b-factor.

        Used for exponential (b~0) and harmonic (b=1) model types.

        Args:
            t_fit: Time array (already adjusted for regime)
            q_fit: Production rate array
            weights: Weights array
            b_fixed: Fixed b-factor value
            regime_start: Regime start index
            capture_residuals: Whether to capture residuals

        Returns:
            ForecastResult with fitted model
        """
        qi0, di0, _ = self.initial_guess(t_fit, q_fit, weights)

        bounds_lower, bounds_upper = self._build_bounds(qi0, self.config.dmin_monthly, include_b=False)

        qi0 = np.clip(qi0, bounds_lower[0], bounds_upper[0])
        di0 = np.clip(di0, bounds_lower[1], bounds_upper[1])

        def model_func(t, qi, di):
            return _hyperbolic_model(t, qi, di, b_fixed)

        try:
            popt, _ = optimize.curve_fit(
                model_func,
                t_fit,
                q_fit,
                p0=[qi0, di0],
                bounds=(bounds_lower, bounds_upper),
                sigma=1.0 / np.sqrt(weights),
                absolute_sigma=False,
                method='trf',
                maxfev=5000
            )
            qi_fit, di_fit = popt

        except (RuntimeError, optimize.OptimizeWarning):
            def objective(params):
                qi, di = params
                pred = model_func(t_fit, qi, di)
                residuals = (q_fit - pred) ** 2 * weights
                return np.sum(residuals)

            try:
                result = optimize.differential_evolution(
                    objective,
                    bounds=list(zip(bounds_lower, bounds_upper)),
                    seed=42,
                    maxiter=1000,
                    tol=1e-7
                )
                qi_fit, di_fit = result.x
            except (RuntimeError, ValueError):
                qi_fit, di_fit = qi0, di0

        model = HyperbolicModel(
            qi=qi_fit,
            di=di_fit,
            b=b_fixed,
            dmin=self.config.dmin_monthly
        )

        return self._build_result(
            model, t_fit, q_fit, regime_start,
            n_params=2, capture_residuals=capture_residuals,
        )

    def _fit_hyperbolic_free_b(
        self,
        data: _PreparedFitData,
        capture_residuals: bool = False,
    ) -> ForecastResult:
        """Fit hyperbolic decline with free b-factor.

        Args:
            data: Prepared fit data
            capture_residuals: Whether to capture residuals

        Returns:
            ForecastResult with fitted hyperbolic model
        """
        qi0, di0, b0 = self.initial_guess(data.t_fit, data.q_fit, data.weights)
        bounds_lower, bounds_upper = self._build_bounds(qi0, data.effective_dmin)

        qi0 = np.clip(qi0, bounds_lower[0], bounds_upper[0])
        di0 = np.clip(di0, bounds_lower[1], bounds_upper[1])
        b0 = np.clip(b0, bounds_lower[2], bounds_upper[2])

        try:
            popt, _ = optimize.curve_fit(
                _hyperbolic_model,
                data.t_fit, data.q_fit,
                p0=[qi0, di0, b0],
                bounds=(bounds_lower, bounds_upper),
                sigma=1.0 / np.sqrt(data.weights),
                absolute_sigma=False,
                method='trf', maxfev=5000,
            )
            qi_fit, di_fit, b_fit = popt
        except (RuntimeError, optimize.OptimizeWarning):
            def objective(params):
                qi, di, b = params
                pred = _hyperbolic_model(data.t_fit, qi, di, b)
                return np.sum((data.q_fit - pred) ** 2 * data.weights)

            opt_result = optimize.differential_evolution(
                objective,
                bounds=list(zip(bounds_lower, bounds_upper)),
                seed=42, maxiter=1000, tol=1e-7,
            )
            qi_fit, di_fit, b_fit = opt_result.x

        model = HyperbolicModel(qi=qi_fit, di=di_fit, b=b_fit, dmin=data.effective_dmin)
        return self._build_result(
            model, data.t_fit, data.q_fit, data.regime_start,
            n_params=3, capture_residuals=capture_residuals,
        )

    def fit_multimodel(
        self,
        t: np.ndarray,
        q: np.ndarray,
        apply_regime_detection: bool = True,
        apply_weights: bool = True,
        capture_residuals: bool = False,
    ) -> ForecastResult:
        """Fit multiple decline models and select the best by BIC.

        Fits exponential (b~0), harmonic (b=1), and hyperbolic (b free)
        models, then selects the best based on BIC (penalizes complexity).

        Args:
            t: Time array in months
            q: Production rate array
            apply_regime_detection: Whether to detect regime changes
            apply_weights: Whether to apply recency weighting
            capture_residuals: Whether to capture residuals

        Returns:
            Best ForecastResult based on BIC selection
        """
        from .selection import compare_fits

        data = self._prepare_fit_data(t, q, apply_regime_detection, apply_weights)

        results = []

        # Fit exponential (b ~ 0.01)
        try:
            exp_result = self._fit_with_fixed_b(
                data.t_fit, data.q_fit, data.weights, 0.01, data.regime_start, capture_residuals
            )
            results.append(exp_result)
        except Exception:
            pass

        # Fit harmonic (b = 1.0)
        try:
            harm_result = self._fit_with_fixed_b(
                data.t_fit, data.q_fit, data.weights, 1.0, data.regime_start, capture_residuals
            )
            results.append(harm_result)
        except Exception:
            pass

        # Fit hyperbolic (b free)
        try:
            hyp_result = self._fit_hyperbolic_free_b(data, capture_residuals)
            results.append(hyp_result)
        except Exception:
            pass

        if not results:
            raise ValueError("All model fits failed")

        return compare_fits(results, self.config.acceptable_r_squared)

    def fit_auto(
        self,
        t: np.ndarray,
        q: np.ndarray,
        apply_regime_detection: bool = True,
        apply_weights: bool = True,
        capture_residuals: bool = False,
    ) -> ForecastResult:
        """Fit using the configured model selection strategy.

        Dispatches to appropriate fitting method based on config.model_selection:
        - 'auto': Try all models, select best by BIC
        - 'hyperbolic': Fit hyperbolic with free b
        - 'exponential': Fit exponential (b ~ 0)
        - 'harmonic': Fit harmonic (b = 1)

        Args:
            t: Time array in months
            q: Production rate array
            apply_regime_detection: Whether to detect regime changes
            apply_weights: Whether to apply recency weighting
            capture_residuals: Whether to capture residuals

        Returns:
            ForecastResult from selected fitting strategy
        """
        selection = self.config.model_selection_enum

        if selection == ModelSelection.AUTO:
            return self.fit_multimodel(
                t, q,
                apply_regime_detection=apply_regime_detection,
                apply_weights=apply_weights,
                capture_residuals=capture_residuals,
            )
        elif selection in (ModelSelection.EXPONENTIAL, ModelSelection.HARMONIC):
            b_fixed = 0.01 if selection == ModelSelection.EXPONENTIAL else 1.0
            data = self._prepare_fit_data(t, q, apply_regime_detection, apply_weights)
            return self._fit_with_fixed_b(
                data.t_fit, data.q_fit, data.weights, b_fixed,
                data.regime_start, capture_residuals,
            )
        else:  # HYPERBOLIC or default
            return self.fit(
                t, q,
                apply_regime_detection=apply_regime_detection,
                apply_weights=apply_weights,
                capture_residuals=capture_residuals,
            )
