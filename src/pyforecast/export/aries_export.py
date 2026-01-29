"""Export forecasts in ARIES AC_FORECAST format."""

from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from ..data.well import Well
from ..core.models import ForecastResult


class AriesExporter:
    """Export decline forecasts in ARIES-compatible format.

    ARIES AC_FORECAST format includes:
    - PROPNUM: Property identifier
    - SEQUENCE: Forecast sequence number
    - KEYWORD: Product (OIL, GAS)
    - START_DATE: Forecast start (YYYYMM)
    - END_DATE: Forecast end (YYYYMM)
    - START_VALUE: Initial rate (qi)
    - END_VALUE: Economic limit rate
    - DECLINE_TYPE: EXP, HYP, or HRM
    - DECLINE_RATE: Nominal decline rate
    - B_FACTOR: Hyperbolic b factor
    """

    def __init__(
        self,
        economic_limit_oil: float = 5.0,
        economic_limit_gas: float = 10.0,
        forecast_years: int = 50,
    ):
        """Initialize exporter.

        Args:
            economic_limit_oil: Minimum oil rate (bbl/month) to forecast
            economic_limit_gas: Minimum gas rate (mcf/month) to forecast
            forecast_years: Maximum forecast duration in years
        """
        self.economic_limit_oil = economic_limit_oil
        self.economic_limit_gas = economic_limit_gas
        self.forecast_years = forecast_years

    def _calculate_end_date(
        self,
        result: ForecastResult,
        start_date: date,
        product: Literal["oil", "gas"]
    ) -> date:
        """Calculate forecast end date based on economic limit.

        Args:
            result: Forecast result with fitted model
            start_date: Forecast start date
            product: Product type for economic limit

        Returns:
            Forecast end date
        """
        econ_limit = self.economic_limit_oil if product == "oil" else self.economic_limit_gas
        model = result.model

        # Binary search for time to reach economic limit
        t_max = self.forecast_years * 12  # Max months
        t_low, t_high = 0, t_max

        while t_high - t_low > 1:
            t_mid = (t_low + t_high) // 2
            rate = model.rate(t_mid)
            if rate[0] > econ_limit:
                t_low = t_mid
            else:
                t_high = t_mid

        months_to_add = t_high

        # Calculate end date
        end_year = start_date.year + (start_date.month + months_to_add - 1) // 12
        end_month = (start_date.month + months_to_add - 1) % 12 + 1

        return date(end_year, end_month, 1)

    def _format_yyyymm(self, d: date) -> str:
        """Format date as YYYYMM string."""
        return f"{d.year:04d}{d.month:02d}"

    def export_well(
        self,
        well: Well,
        products: list[Literal["oil", "gas"]] | None = None
    ) -> list[dict]:
        """Export single well forecast to ARIES format rows.

        Args:
            well: Well with forecast results
            products: Products to export (default: both oil and gas if available)

        Returns:
            List of row dictionaries
        """
        if products is None:
            products = []
            if well.forecast_oil is not None:
                products.append("oil")
            if well.forecast_gas is not None:
                products.append("gas")

        rows = []
        propnum = well.identifier.propnum or well.identifier.api or well.identifier.primary_id

        for seq, product in enumerate(products, start=1):
            result = well.get_forecast(product)
            if result is None:
                continue

            model = result.model

            # Determine start date (after regime change, at last historical data point)
            if well.production.last_date:
                start_date = well.production.last_date
            else:
                start_date = date.today()

            end_date = self._calculate_end_date(result, start_date, product)

            # Economic limit for this product
            econ_limit = self.economic_limit_oil if product == "oil" else self.economic_limit_gas

            row = {
                "PROPNUM": propnum,
                "SEQUENCE": seq,
                "KEYWORD": product.upper(),
                "START_DATE": self._format_yyyymm(start_date),
                "END_DATE": self._format_yyyymm(end_date),
                "START_VALUE": round(model.qi, 2),
                "END_VALUE": round(econ_limit, 2),
                "DECLINE_TYPE": model.decline_type,
                "DECLINE_RATE": round(model.di * 12, 6),  # Convert to annual
                "B_FACTOR": round(model.b, 4),
            }
            rows.append(row)

        return rows

    def export_wells(
        self,
        wells: list[Well],
        products: list[Literal["oil", "gas"]] | None = None
    ) -> pd.DataFrame:
        """Export multiple wells to ARIES format DataFrame.

        Args:
            wells: List of wells with forecast results
            products: Products to export

        Returns:
            DataFrame in ARIES AC_FORECAST format
        """
        all_rows = []
        for well in wells:
            rows = self.export_well(well, products)
            all_rows.extend(rows)

        if not all_rows:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=[
                "PROPNUM", "SEQUENCE", "KEYWORD", "START_DATE", "END_DATE",
                "START_VALUE", "END_VALUE", "DECLINE_TYPE", "DECLINE_RATE", "B_FACTOR"
            ])

        return pd.DataFrame(all_rows)

    def save(
        self,
        wells: list[Well],
        output_path: Path | str,
        products: list[Literal["oil", "gas"]] | None = None
    ) -> Path:
        """Export wells and save to CSV file.

        Args:
            wells: List of wells with forecast results
            output_path: Output file path
            products: Products to export

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        df = self.export_wells(wells, products)
        df.to_csv(output_path, index=False)
        return output_path
