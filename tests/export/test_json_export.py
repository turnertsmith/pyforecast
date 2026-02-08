"""Tests for JSON export."""

import json
import numpy as np
import pytest
from datetime import date

from pyforecast.export.json_export import JsonExporter
from pyforecast.config import PyForecastConfig
from pyforecast.core.models import HyperbolicModel, ForecastResult
from pyforecast.data.well import Well, WellIdentifier, ProductionData
from pyforecast.validation import ValidationResult, ValidationIssue, IssueSeverity, IssueCategory


def _make_well(propnum="WELL001", api="42-001-00001", n_months=24):
    """Create a test well with oil and gas forecasts."""
    dates = np.array(
        [np.datetime64(f"2022-{m:02d}-01") for m in range(1, 13)]
        + [np.datetime64(f"2023-{m:02d}-01") for m in range(1, min(n_months - 11, 13))]
    )[:n_months]
    oil = np.full(n_months, 3000.0)
    gas = np.full(n_months, 15000.0)

    well = Well(
        identifier=WellIdentifier(propnum=propnum, api=api, well_name="Test Well"),
        production=ProductionData(dates=dates, oil=oil, gas=gas),
    )

    oil_model = HyperbolicModel(qi=100.0, di=0.01, b=0.5, dmin=0.005)
    well.set_forecast("oil", ForecastResult(
        model=oil_model, r_squared=0.95, rmse=5.0,
        aic=100.0, bic=110.0, data_points_used=n_months,
        regime_start_idx=0,
    ))

    gas_model = HyperbolicModel(qi=500.0, di=0.015, b=0.6, dmin=0.005)
    well.set_forecast("gas", ForecastResult(
        model=gas_model, r_squared=0.92, rmse=50.0,
        aic=200.0, bic=210.0, data_points_used=n_months,
        regime_start_idx=0,
    ))

    return well


class TestJsonExporter:
    """Tests for JsonExporter."""

    def test_default_init(self):
        exporter = JsonExporter()
        assert exporter.forecast_months == 60
        assert exporter.config is not None

    def test_custom_config(self):
        config = PyForecastConfig()
        exporter = JsonExporter(config=config, forecast_months=120)
        assert exporter.forecast_months == 120

    def test_format_date(self):
        exporter = JsonExporter()
        assert exporter._format_date(date(2023, 6, 15)) == "2023-06-15"
        assert exporter._format_date(None) is None

    def test_format_month(self):
        exporter = JsonExporter()
        assert exporter._format_month(date(2023, 6, 1)) == "2023-06"
        assert exporter._format_month(date(2023, 12, 1)) == "2023-12"

    def test_export_product(self):
        well = _make_well()
        exporter = JsonExporter(forecast_months=12)
        product_data = exporter._export_product(well, "oil")

        assert product_data is not None
        assert product_data["qi"] == 100.0
        assert product_data["b"] == 0.5
        assert product_data["r_squared"] == 0.95
        assert "decline_type" in product_data
        assert len(product_data["forecast"]) > 0
        assert "date" in product_data["forecast"][0]
        assert "rate" in product_data["forecast"][0]

    def test_export_product_none(self):
        well = _make_well()
        exporter = JsonExporter()
        # Water has no forecast
        assert exporter._export_product(well, "water") is None

    def test_export_product_includes_monthly_qi(self):
        well = _make_well()
        exporter = JsonExporter()
        product_data = exporter._export_product(well, "oil")

        assert "qi_monthly" in product_data
        assert product_data["qi_monthly"] == pytest.approx(100.0 * 30.4375, rel=0.01)
        assert product_data["qi_unit"] == "bbl/day"
        assert product_data["qi_monthly_unit"] == "bbl/month"

    def test_export_product_gas_units(self):
        well = _make_well()
        exporter = JsonExporter()
        product_data = exporter._export_product(well, "gas")

        assert product_data["qi_unit"] == "mcf/day"
        assert product_data["qi_monthly_unit"] == "mcf/month"

    def test_export_validation_empty(self):
        exporter = JsonExporter()
        val_data = exporter._export_validation(None)

        assert val_data["errors"] == 0
        assert val_data["warnings"] == 0
        assert val_data["issues"] == []

    def test_export_validation_with_issues(self):
        exporter = JsonExporter()
        result = ValidationResult(well_id="W001")
        result.add_issue(ValidationIssue(
            code="DQ001",
            category=IssueCategory.DATA_QUALITY,
            severity=IssueSeverity.WARNING,
            message="Gap detected",
            guidance="Check data",
        ))
        val_data = exporter._export_validation(result)

        assert val_data["warnings"] == 1
        assert len(val_data["issues"]) == 1
        assert val_data["issues"][0]["code"] == "DQ001"
        assert val_data["issues"][0]["severity"] == "warning"

    def test_export_well(self):
        well = _make_well()
        exporter = JsonExporter(forecast_months=12)
        well_data = exporter.export_well(well)

        assert well_data["uwi"] == "42-001-00001"
        assert well_data["api"] == "42-001-00001"
        assert well_data["propnum"] == "WELL001"
        assert well_data["well_name"] == "Test Well"
        assert well_data["months_of_data"] == 24
        assert "oil" in well_data["products"]
        assert "gas" in well_data["products"]
        assert well_data["validation"]["errors"] == 0

    def test_export_well_specific_products(self):
        well = _make_well()
        exporter = JsonExporter()
        well_data = exporter.export_well(well, products=["gas"])

        assert "gas" in well_data["products"]
        assert "oil" not in well_data["products"]

    def test_export_wells(self):
        wells = [_make_well(propnum="W001"), _make_well(propnum="W002")]
        exporter = JsonExporter(forecast_months=6)
        data = exporter.export_wells(wells)

        assert data["well_count"] == 2
        assert len(data["wells"]) == 2
        assert "generated" in data
        assert "config" in data

    def test_export_wells_with_validation(self):
        well = _make_well()
        val_result = ValidationResult(well_id=well.well_id)
        val_result.add_issue(ValidationIssue(
            code="FR001",
            category=IssueCategory.FITTING_RESULT,
            severity=IssueSeverity.WARNING,
            message="Low R-squared",
            guidance="Review fit",
        ))

        exporter = JsonExporter()
        data = exporter.export_wells([well], validation_results={well.well_id: val_result})

        assert data["wells"][0]["validation"]["warnings"] == 1

    def test_save(self, tmp_path):
        well = _make_well()
        exporter = JsonExporter(forecast_months=6)
        output = tmp_path / "forecast.json"
        result_path = exporter.save([well], output)

        assert result_path.exists()
        with open(result_path) as f:
            data = json.load(f)
        assert data["well_count"] == 1
        assert len(data["wells"]) == 1

    def test_forecast_stops_at_economic_limit(self):
        """Forecast should stop when rate drops below 0.01."""
        # Very high decline rate to trigger economic limit quickly
        well = _make_well()
        model = HyperbolicModel(qi=1.0, di=0.5, b=0.01, dmin=0.1)
        well.set_forecast("oil", ForecastResult(
            model=model, r_squared=0.95, rmse=0.1,
            aic=10.0, bic=15.0, data_points_used=24,
            regime_start_idx=0,
        ))

        exporter = JsonExporter(forecast_months=120)
        product_data = exporter._export_product(well, "oil")

        # Should stop before 120 months due to economic limit
        assert len(product_data["forecast"]) < 120
