"""Export forecasts in JSON format."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from ..config import PyForecastConfig
from ..data.well import Well
from ..validation import ValidationResult


class JsonExporter:
    """Export decline forecasts in JSON format.

    Produces a structured JSON file containing configuration, well data,
    forecast parameters, and validation results.

    Note: Internal model uses daily rates. The JSON export includes both
    daily rates (qi) and monthly equivalents (qi_monthly) for convenience.
    """

    # Average days per month (365.25/12) for daily-to-monthly conversion
    DAYS_PER_MONTH = 30.4375

    def __init__(
        self,
        config: PyForecastConfig | None = None,
        forecast_months: int = 60,
    ):
        """Initialize exporter.

        Args:
            config: PyForecast configuration (included in export)
            forecast_months: Number of months to include in forecast arrays
        """
        self.config = config or PyForecastConfig()
        self.forecast_months = forecast_months

    def _format_date(self, d: date | None) -> str | None:
        """Format date as ISO string."""
        if d is None:
            return None
        return d.isoformat()

    def _format_month(self, d: date) -> str:
        """Format date as YYYY-MM string."""
        return f"{d.year:04d}-{d.month:02d}"

    def _export_product(
        self,
        well: Well,
        product: Literal["oil", "gas", "water"],
    ) -> dict[str, Any] | None:
        """Export forecast data for a single product.

        Args:
            well: Well with forecast result
            product: Product to export

        Returns:
            Product data dict or None if no forecast
        """
        result = well.get_forecast(product)
        if result is None:
            return None

        model = result.model

        # Generate forecast time series
        start_date = well.production.last_date or date.today()
        forecast_data = []

        for month_offset in range(self.forecast_months):
            # Calculate date
            total_months = start_date.month + month_offset
            year = start_date.year + (total_months - 1) // 12
            month = (total_months - 1) % 12 + 1
            forecast_date = date(year, month, 1)

            # Calculate rate
            rate = model.rate(month_offset)[0]
            if rate < 0.01:  # Below economic limit
                break

            forecast_data.append({
                "date": self._format_month(forecast_date),
                "rate": round(rate, 2),
            })

        # qi is daily rate; also provide monthly for convenience
        qi_monthly = model.qi * self.DAYS_PER_MONTH
        unit = "bbl/day" if product in ("oil", "water") else "mcf/day"
        unit_monthly = "bbl/month" if product in ("oil", "water") else "mcf/month"

        return {
            "qi": round(model.qi, 2),
            "qi_unit": unit,
            "qi_monthly": round(qi_monthly, 2),
            "qi_monthly_unit": unit_monthly,
            "di": round(model.di * 12, 6),  # Annual decline
            "b": round(model.b, 4),
            "dmin": round(model.dmin * 12, 6),  # Annual terminal decline
            "r_squared": round(result.r_squared, 4),
            "rmse": round(result.rmse, 2),
            "decline_type": model.decline_type,
            "regime_start_idx": result.regime_start_idx,
            "data_points_used": result.data_points_used,
            "forecast": forecast_data,
        }

    def _export_validation(
        self,
        validation_result: ValidationResult | None,
    ) -> dict[str, Any]:
        """Export validation results.

        Args:
            validation_result: Validation result for the well

        Returns:
            Validation data dict
        """
        if validation_result is None:
            return {
                "errors": 0,
                "warnings": 0,
                "issues": [],
            }

        issues = []
        for issue in validation_result.issues:
            issues.append({
                "code": issue.code,
                "severity": issue.severity.name.lower(),
                "message": issue.message,
                "guidance": issue.guidance,
            })

        return {
            "errors": validation_result.error_count,
            "warnings": validation_result.warning_count,
            "issues": issues,
        }

    def export_well(
        self,
        well: Well,
        products: list[Literal["oil", "gas", "water"]] | None = None,
        validation_result: ValidationResult | None = None,
    ) -> dict[str, Any]:
        """Export single well to JSON-compatible dict.

        Args:
            well: Well with forecast results
            products: Products to export (default: all available)
            validation_result: Optional validation result for the well

        Returns:
            Well data dict
        """
        if products is None:
            products = []
            if well.forecast_oil is not None:
                products.append("oil")
            if well.forecast_gas is not None:
                products.append("gas")
            if well.forecast_water is not None:
                products.append("water")

        # Export each product
        products_data = {}
        for product in products:
            product_data = self._export_product(well, product)
            if product_data is not None:
                products_data[product] = product_data

        return {
            "uwi": well.identifier.primary_id,
            "api": well.identifier.api,
            "propnum": well.identifier.propnum,
            "well_name": well.identifier.well_name,
            "first_production": self._format_date(well.production.first_date),
            "last_production": self._format_date(well.production.last_date),
            "months_of_data": well.production.n_months,
            "products": products_data,
            "validation": self._export_validation(validation_result),
        }

    def export_wells(
        self,
        wells: list[Well],
        products: list[Literal["oil", "gas", "water"]] | None = None,
        validation_results: dict[str, ValidationResult] | None = None,
    ) -> dict[str, Any]:
        """Export multiple wells to JSON-compatible dict.

        Args:
            wells: List of wells with forecast results
            products: Products to export
            validation_results: Dict mapping well_id to ValidationResult

        Returns:
            Complete export data dict
        """
        validation_results = validation_results or {}

        wells_data = []
        for well in wells:
            validation_result = validation_results.get(well.well_id)
            well_data = self.export_well(well, products, validation_result)
            wells_data.append(well_data)

        return {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "config": self.config.to_dict(),
            "well_count": len(wells),
            "wells": wells_data,
        }

    def save(
        self,
        wells: list[Well],
        output_path: Path | str,
        products: list[Literal["oil", "gas", "water"]] | None = None,
        validation_results: dict[str, ValidationResult] | None = None,
    ) -> Path:
        """Export wells and save to JSON file.

        Args:
            wells: List of wells with forecast results
            output_path: Output file path
            products: Products to export
            validation_results: Dict mapping well_id to ValidationResult

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        data = self.export_wells(wells, products, validation_results)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path
