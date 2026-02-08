"""Arps decline curve models for oil and gas production forecasting.

This module implements the hyperbolic decline model with Dmin terminal decline
switch. Exponential (b~0) and Harmonic (b=1) are special cases of hyperbolic.

Mathematical Background
-----------------------

The Arps hyperbolic decline equation is the industry standard for production
forecasting:

    q(t) = qi / (1 + b * Di * t)^(1/b)

Where:
    q(t) = Production rate at time t
    qi   = Initial production rate at t=0
    Di   = Initial nominal decline rate (fraction/time)
    b    = Hyperbolic exponent (dimensionless, typically 0 to 1.5)
    t    = Time from start

Special Cases:
    - Exponential (b → 0): q(t) = qi * exp(-Di * t)
    - Harmonic (b = 1):    q(t) = qi / (1 + Di * t)

Terminal Decline Switch:
    The hyperbolic equation with b > 0 implies decline rate approaches zero
    as time increases, leading to unrealistically long well lives. The terminal
    decline switch addresses this by switching to exponential decline when the
    instantaneous decline rate falls to Dmin:

    t_switch = (Di/Dmin - 1) / (b * Di)

    For t > t_switch:
        q(t) = q_switch * exp(-Dmin * (t - t_switch))

Cumulative Production:
    General hyperbolic (b ≠ 0, 1):
        Np(t) = qi / ((1-b) * Di) * [1 - (1 + b*Di*t)^((b-1)/b)]

    Exponential (b → 0):
        Np(t) = qi / Di * (1 - exp(-Di * t))

    Harmonic (b = 1):
        Np(t) = qi / Di * ln(1 + Di * t)

References:
    Arps, J.J. (1945). "Analysis of Decline Curves". Trans. AIME, 160, 228-247.
"""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass
class HyperbolicModel:
    """Hyperbolic decline model with terminal decline switch.

    Implements the Arps hyperbolic decline equation:

        q(t) = qi / (1 + b * Di * t)^(1/b)

    With automatic switch to exponential terminal decline when the
    instantaneous decline rate D(t) = Di / (1 + b*Di*t) falls below Dmin.

    The switch time is calculated as:

        t_switch = (Di/Dmin - 1) / (b * Di)

    After t_switch, the model uses exponential decline at Dmin rate:

        q(t) = q_switch * exp(-Dmin * (t - t_switch))

    Attributes:
        qi: Initial rate at t=0 (units: bbl/day or mcf/day for fitting,
            output is same units as input)
        di: Initial nominal decline rate (fraction/month). This is the
            instantaneous decline rate at t=0.
        b: Hyperbolic exponent (dimensionless). Typical ranges:
            - 0.01-0.1: Near-exponential (conventional wells)
            - 0.3-0.8: Typical unconventional (tight oil/gas)
            - 0.8-1.0: Harmonic-like (transient flow)
            - 1.0-1.5: Super-harmonic (rare, early-time transient)
        dmin: Terminal decline rate (fraction/month). When instantaneous
            decline D(t) reaches this value, switches to exponential.
            Typical: 0.005 (6%/year) to 0.0083 (10%/year)
        t_switch: Time (months) when decline switches to terminal exponential.
            Calculated automatically from di, dmin, and b.

    Example:
        >>> model = HyperbolicModel(qi=100, di=0.05, b=0.5, dmin=0.005)
        >>> rates = model.rate([0, 12, 24, 36])  # Monthly rates
        >>> cumulative = model.cumulative(60)    # 5-year EUR
    """
    qi: float
    di: float
    b: float
    dmin: float = 0.005  # 6% annual ~ 0.5% monthly
    t_switch: float = field(init=False)

    def __post_init__(self) -> None:
        """Calculate switch time from hyperbolic to terminal exponential."""
        if self.b <= 0.01:
            # Essentially exponential, no switch needed
            self.t_switch = float('inf')
        else:
            # Instantaneous decline D(t) = Di / (1 + b*Di*t)
            # Solve for t when D(t) = Dmin: t_switch = (Di/Dmin - 1) / (b*Di)
            if self.di > self.dmin:
                self.t_switch = (self.di / self.dmin - 1) / (self.b * self.di)
            else:
                self.t_switch = 0.0

    def rate(self, t: np.ndarray | float) -> np.ndarray:
        """Calculate production rate at time t.

        Args:
            t: Time in months from start (scalar or array)

        Returns:
            Production rate (same units as qi)
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        q = np.zeros_like(t)

        # Before switch: hyperbolic decline
        mask_hyp = t <= self.t_switch
        if np.any(mask_hyp):
            t_hyp = t[mask_hyp]
            if self.b <= 0.01:
                # Use exponential form to avoid numerical issues
                q[mask_hyp] = self.qi * np.exp(-self.di * t_hyp)
            else:
                q[mask_hyp] = self.qi / np.power(1 + self.b * self.di * t_hyp, 1 / self.b)

        # After switch: exponential terminal decline
        mask_exp = t > self.t_switch
        if np.any(mask_exp):
            # Rate at switch time
            if self.b <= 0.01:
                q_switch = self.qi * np.exp(-self.di * self.t_switch)
            else:
                q_switch = self.qi / np.power(1 + self.b * self.di * self.t_switch, 1 / self.b)

            # Exponential decline from switch point
            t_after = t[mask_exp] - self.t_switch
            q[mask_exp] = q_switch * np.exp(-self.dmin * t_after)

        return q

    def cumulative(self, t: np.ndarray | float) -> np.ndarray:
        """Calculate cumulative production from 0 to t.

        Args:
            t: Time in months from start (scalar or array)

        Returns:
            Cumulative production (same volume units as qi * time)
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        cum = np.zeros_like(t)

        # Before switch: hyperbolic cumulative
        mask_hyp = t <= self.t_switch
        if np.any(mask_hyp):
            cum[mask_hyp] = self._cumulative_hyperbolic_vec(t[mask_hyp])

        # After switch: hyperbolic cumulative through switch + exponential after
        mask_exp = t > self.t_switch
        if np.any(mask_exp):
            cum_switch = self._cumulative_hyperbolic_vec(np.array([self.t_switch]))[0]
            q_switch = self.rate(self.t_switch)[0]
            t_after = t[mask_exp] - self.t_switch
            cum_exp = (q_switch / self.dmin) * (1 - np.exp(-self.dmin * t_after))
            cum[mask_exp] = cum_switch + cum_exp

        return cum

    def _cumulative_hyperbolic_vec(self, t: np.ndarray) -> np.ndarray:
        """Vectorized cumulative production during hyperbolic phase."""
        if self.b <= 0.01:
            # Exponential: Np = qi/Di * (1 - exp(-Di*t))
            return (self.qi / self.di) * (1 - np.exp(-self.di * t))
        elif abs(self.b - 1.0) < 0.01:
            # Harmonic: Np = qi/Di * ln(1 + Di*t)
            return (self.qi / self.di) * np.log(1 + self.di * t)
        else:
            # General hyperbolic: Np = qi / ((1-b)*Di) * (1 - (1+b*Di*t)^((b-1)/b))
            term = np.power(1 + self.b * self.di * t, (self.b - 1) / self.b)
            return (self.qi / ((1 - self.b) * self.di)) * (1 - term)

    def instantaneous_decline(self, t: np.ndarray | float) -> np.ndarray:
        """Calculate instantaneous decline rate at time t.

        Args:
            t: Time in months from start

        Returns:
            Instantaneous decline rate (fraction/month)
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        d = np.zeros_like(t)

        mask_hyp = t <= self.t_switch
        if np.any(mask_hyp):
            if self.b <= 0.01:
                d[mask_hyp] = self.di
            else:
                d[mask_hyp] = self.di / (1 + self.b * self.di * t[mask_hyp])

        mask_exp = t > self.t_switch
        if np.any(mask_exp):
            d[mask_exp] = self.dmin

        return d

    @property
    def decline_type(self) -> Literal["EXP", "HYP", "HRM"]:
        """Return ARIES-compatible decline type string."""
        if self.b <= 0.1:
            return "EXP"
        elif self.b >= 0.95:
            return "HRM"
        else:
            return "HYP"

    def forecast(self, months: int, start_month: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Generate forecast time series.

        Args:
            months: Number of months to forecast
            start_month: Starting month offset (for continuing from historical data)

        Returns:
            Tuple of (time_array, rate_array)
        """
        t = np.arange(start_month, start_month + months, dtype=float)
        return t, self.rate(t)


@dataclass
class ForecastResult:
    """Result of decline curve fitting and forecasting.

    Attributes:
        model: Fitted HyperbolicModel
        r_squared: Coefficient of determination
        rmse: Root mean squared error
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        regime_start_idx: Index where current regime starts (after detected changes)
        data_points_used: Number of production points used in fit
        t_fit: Optional time array used for fitting (for residual analysis)
        residuals: Optional residuals array (actual - predicted)
        acceptable_r_squared: Threshold for is_acceptable check (default 0.7)
        prediction_intervals: Optional dict with P5, P50, P95 forecast arrays
    """
    model: HyperbolicModel
    r_squared: float
    rmse: float
    aic: float
    bic: float
    regime_start_idx: int
    data_points_used: int
    t_fit: np.ndarray | None = None
    residuals: np.ndarray | None = None
    acceptable_r_squared: float = 0.7
    prediction_intervals: dict | None = None

    @property
    def is_acceptable(self) -> bool:
        """Check if fit meets minimum quality threshold."""
        return self.r_squared >= self.acceptable_r_squared

    def summary(self) -> dict:
        """Return summary dictionary of fit results."""
        return {
            "qi": self.model.qi,
            "di": self.model.di,
            "di_annual": self.model.di * 12,  # Convert to annual
            "b": self.model.b,
            "dmin": self.model.dmin,
            "dmin_annual": self.model.dmin * 12,
            "decline_type": self.model.decline_type,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
            "aic": self.aic,
            "bic": self.bic,
            "regime_start_idx": self.regime_start_idx,
            "data_points_used": self.data_points_used,
        }
