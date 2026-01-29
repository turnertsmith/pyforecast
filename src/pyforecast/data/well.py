"""Data models for wells and production data."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal
import numpy as np
import pandas as pd

from ..core.models import ForecastResult


@dataclass
class WellIdentifier:
    """Unique identifier for a well across different data sources.

    Attributes:
        api: API number (e.g., "42-001-00001")
        propnum: ARIES property number
        entity_id: Enverus entity ID
        well_name: Human-readable well name
    """
    api: str | None = None
    propnum: str | None = None
    entity_id: str | None = None
    well_name: str | None = None

    @property
    def primary_id(self) -> str:
        """Return the best available identifier."""
        return self.api or self.propnum or self.entity_id or self.well_name or "UNKNOWN"

    def __str__(self) -> str:
        return self.primary_id


@dataclass
class ProductionData:
    """Monthly production data for a well.

    Attributes:
        dates: Array of production dates (first of month)
        oil: Oil production volumes (bbl/month total)
        gas: Gas production volumes (mcf/month total)
        water: Water production volumes (bbl/month total), optional
        time_months: Time in months from first production (computed)
        days_in_month: Days in each month (computed from dates)
    """
    dates: np.ndarray  # datetime64
    oil: np.ndarray
    gas: np.ndarray
    water: np.ndarray | None = None
    time_months: np.ndarray = field(init=False)
    days_in_month: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Compute time_months and days_in_month from dates."""
        if len(self.dates) > 0:
            # Convert to months from first date
            dates_dt = pd.to_datetime(self.dates)
            first_date = dates_dt.min()
            months_diff = (
                (dates_dt.year - first_date.year) * 12 +
                (dates_dt.month - first_date.month)
            )
            self.time_months = months_diff.values.astype(float)

            # Compute days in each month
            self.days_in_month = dates_dt.to_series().dt.daysinmonth.values.astype(float)
        else:
            self.time_months = np.array([], dtype=float)
            self.days_in_month = np.array([], dtype=float)

    def get_product(self, product: Literal["oil", "gas", "water"]) -> np.ndarray:
        """Get monthly production volume array for specified product.

        Args:
            product: Product type ("oil", "gas", or "water")

        Returns:
            Monthly production volume array

        Raises:
            ValueError: If product not available
        """
        if product == "oil":
            return self.oil
        elif product == "gas":
            return self.gas
        elif product == "water":
            if self.water is None:
                raise ValueError("Water production data not available")
            return self.water
        else:
            raise ValueError(f"Unknown product: {product}")

    def get_product_daily(self, product: Literal["oil", "gas", "water"]) -> np.ndarray:
        """Get daily production rate array for specified product.

        Converts monthly volumes to daily rates by dividing by days in month.
        This normalizes for varying month lengths (28-31 days).

        Args:
            product: Product type ("oil", "gas", or "water")

        Returns:
            Daily production rate array

        Raises:
            ValueError: If product not available
        """
        monthly = self.get_product(product)
        if len(self.days_in_month) == 0:
            return monthly
        return monthly / self.days_in_month

    @property
    def oil_daily(self) -> np.ndarray:
        """Oil production as daily rate (bbl/day)."""
        if len(self.days_in_month) == 0:
            return self.oil
        return self.oil / self.days_in_month

    @property
    def gas_daily(self) -> np.ndarray:
        """Gas production as daily rate (mcf/day)."""
        if len(self.days_in_month) == 0:
            return self.gas
        return self.gas / self.days_in_month

    @property
    def water_daily(self) -> np.ndarray | None:
        """Water production as daily rate (bbl/day)."""
        if self.water is None:
            return None
        if len(self.days_in_month) == 0:
            return self.water
        return self.water / self.days_in_month

    @property
    def n_months(self) -> int:
        """Number of months of production data."""
        return len(self.dates)

    @property
    def first_date(self) -> date | None:
        """First production date."""
        if len(self.dates) > 0:
            return pd.Timestamp(self.dates.min()).date()
        return None

    @property
    def last_date(self) -> date | None:
        """Last production date."""
        if len(self.dates) > 0:
            return pd.Timestamp(self.dates.max()).date()
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        data = {
            "date": self.dates,
            "time_months": self.time_months,
            "oil": self.oil,
            "gas": self.gas,
        }
        if self.water is not None:
            data["water"] = self.water
        return pd.DataFrame(data)


@dataclass
class Well:
    """Complete well data including identifier, production, and forecast.

    Attributes:
        identifier: Well identification info
        production: Historical production data
        forecast_oil: Forecast result for oil (if fitted)
        forecast_gas: Forecast result for gas (if fitted)
        forecast_water: Forecast result for water (if fitted)
        metadata: Additional well metadata
    """
    identifier: WellIdentifier
    production: ProductionData
    forecast_oil: ForecastResult | None = None
    forecast_gas: ForecastResult | None = None
    forecast_water: ForecastResult | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def well_id(self) -> str:
        """Shorthand for primary identifier."""
        return self.identifier.primary_id

    def has_sufficient_data(self, min_months: int = 6) -> bool:
        """Check if well has minimum data for forecasting.

        Args:
            min_months: Minimum months of production required

        Returns:
            True if sufficient data available
        """
        return self.production.n_months >= min_months

    def get_forecast(self, product: Literal["oil", "gas", "water"]) -> ForecastResult | None:
        """Get forecast result for specified product.

        Args:
            product: Product type

        Returns:
            ForecastResult or None if not fitted
        """
        if product == "oil":
            return self.forecast_oil
        elif product == "gas":
            return self.forecast_gas
        elif product == "water":
            return self.forecast_water
        return None

    def set_forecast(self, product: Literal["oil", "gas", "water"], result: ForecastResult) -> None:
        """Set forecast result for specified product.

        Args:
            product: Product type
            result: ForecastResult from fitting
        """
        if product == "oil":
            self.forecast_oil = result
        elif product == "gas":
            self.forecast_gas = result
        elif product == "water":
            self.forecast_water = result
