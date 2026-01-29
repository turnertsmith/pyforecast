"""Arps decline curve models for oil and gas production forecasting.

Implements hyperbolic decline model with Dmin terminal decline switch.
Exponential (b~0) and Harmonic (b=1) are special cases of hyperbolic.
"""

from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass
class HyperbolicModel:
    """Hyperbolic decline model: q(t) = qi / (1 + b*Di*t)^(1/b)

    With Dmin terminal decline: switches to exponential when instantaneous
    decline rate falls below Dmin.

    Attributes:
        qi: Initial rate at t=0 (bbl/month or mcf/month)
        di: Initial nominal decline rate (fraction/month)
        b: Hyperbolic exponent (0.01-1.5 typical range)
        dmin: Terminal decline rate (fraction/month), switches to exponential
        t_switch: Time (months) when decline switches to terminal exponential
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

        for i, ti in enumerate(t):
            if ti <= self.t_switch:
                cum[i] = self._cumulative_hyperbolic(ti)
            else:
                # Cumulative through switch + exponential after
                cum_switch = self._cumulative_hyperbolic(self.t_switch)
                q_switch = self.rate(self.t_switch)[0]
                t_after = ti - self.t_switch
                cum_exp = (q_switch / self.dmin) * (1 - np.exp(-self.dmin * t_after))
                cum[i] = cum_switch + cum_exp

        return cum

    def _cumulative_hyperbolic(self, t: float) -> float:
        """Cumulative production during hyperbolic phase."""
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

    @property
    def is_acceptable(self) -> bool:
        """Check if fit meets minimum quality threshold."""
        return self.r_squared >= 0.7

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
