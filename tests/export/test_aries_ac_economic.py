"""Tests for ARIES AC_ECONOMIC export."""

import numpy as np
import pytest
from datetime import date

from pyforecast.export.aries_ac_economic import AriesAcEconomicExporter, _format_decimal
from pyforecast.core.models import HyperbolicModel, ForecastResult
from pyforecast.data.well import Well, WellIdentifier, ProductionData


def _make_well(
    propnum="WELL001",
    qi_oil=100.0,
    qi_gas=500.0,
    include_water=False,
    n_months=24,
):
    """Create a test well with forecasts."""
    dates = np.array(
        [np.datetime64(f"2022-{m:02d}-01") for m in range(1, min(n_months + 1, 13))]
        + [np.datetime64(f"2023-{m:02d}-01") for m in range(1, max(n_months - 11, 1))]
    )[:n_months]
    oil = np.full(n_months, 3000.0)
    gas = np.full(n_months, 15000.0)
    water = np.full(n_months, 500.0) if include_water else None

    well = Well(
        identifier=WellIdentifier(propnum=propnum),
        production=ProductionData(dates=dates, oil=oil, gas=gas, water=water),
    )

    oil_model = HyperbolicModel(qi=qi_oil, di=0.01, b=0.5, dmin=0.005)
    well.set_forecast("oil", ForecastResult(
        model=oil_model, r_squared=0.95, rmse=5.0,
        aic=100.0, bic=110.0, data_points_used=n_months,
        regime_start_idx=0,
    ))

    gas_model = HyperbolicModel(qi=qi_gas, di=0.015, b=0.6, dmin=0.005)
    well.set_forecast("gas", ForecastResult(
        model=gas_model, r_squared=0.92, rmse=50.0,
        aic=200.0, bic=210.0, data_points_used=n_months,
        regime_start_idx=0,
    ))

    if include_water:
        water_model = HyperbolicModel(qi=20.0, di=0.005, b=0.3, dmin=0.002)
        well.set_forecast("water", ForecastResult(
            model=water_model, r_squared=0.80, rmse=3.0,
            aic=50.0, bic=55.0, data_points_used=n_months,
            regime_start_idx=0,
        ))

    return well


class TestFormatDecimal:
    """Tests for _format_decimal helper."""

    def test_whole_number(self):
        assert _format_decimal(6.0) == "6"

    def test_decimal_number(self):
        assert _format_decimal(8.5) == "8.5"

    def test_trailing_zeros(self):
        assert _format_decimal(6.10) == "6.1"

    def test_two_decimal_places(self):
        assert _format_decimal(0.12) == "0.12"


class TestAriesAcEconomicExporter:
    """Tests for AriesAcEconomicExporter."""

    def test_default_qualifier(self):
        exporter = AriesAcEconomicExporter()
        assert exporter.qualifier.startswith("KA")
        assert len(exporter.qualifier) == 6

    def test_custom_qualifier(self):
        exporter = AriesAcEconomicExporter(qualifier="KA0625")
        assert exporter.qualifier == "KA0625"

    def test_format_expression(self):
        model = HyperbolicModel(qi=100.0, di=0.01, b=0.5, dmin=0.005)
        result = ForecastResult(
            model=model, r_squared=0.95, rmse=5.0,
            aic=100.0, bic=110.0, data_points_used=24,
            regime_start_idx=0,
        )
        exporter = AriesAcEconomicExporter()
        expr = exporter._format_expression(result, "oil")
        assert "X B/D" in expr
        assert "EXP" in expr
        assert "B/0.50" in expr

    def test_format_expression_gas(self):
        model = HyperbolicModel(qi=500.0, di=0.015, b=0.6, dmin=0.005)
        result = ForecastResult(
            model=model, r_squared=0.92, rmse=50.0,
            aic=200.0, bic=210.0, data_points_used=24,
            regime_start_idx=0,
        )
        exporter = AriesAcEconomicExporter()
        expr = exporter._format_expression(result, "gas")
        assert "X M/D" in expr

    def test_format_terminal_expression(self):
        model = HyperbolicModel(qi=100.0, di=0.01, b=0.5, dmin=0.005)
        result = ForecastResult(
            model=model, r_squared=0.95, rmse=5.0,
            aic=100.0, bic=110.0, data_points_used=24,
            regime_start_idx=0,
        )
        exporter = AriesAcEconomicExporter()
        expr = exporter._format_terminal_expression(result, "oil")
        assert "X 1 B/D X YRS EXP" in expr

    def test_di_greater_than_dmin(self):
        """Test that di is adjusted to be > dmin when they're equal."""
        model = HyperbolicModel(qi=100.0, di=0.005, b=0.5, dmin=0.005)
        result = ForecastResult(
            model=model, r_squared=0.95, rmse=5.0,
            aic=100.0, bic=110.0, data_points_used=24,
            regime_start_idx=0,
        )
        exporter = AriesAcEconomicExporter()
        expr = exporter._format_expression(result, "oil")
        # Parse di and dmin from expression
        parts = expr.split()
        dmin_pct = float(parts[3])
        di_pct = float(parts[-1])
        assert di_pct > dmin_pct

    def test_export_well_basic(self):
        well = _make_well()
        exporter = AriesAcEconomicExporter(qualifier="KA0125")
        rows = exporter.export_well(well)

        # Should have CUMS, START, OIL, OIL continuation, GAS, GAS continuation
        assert len(rows) == 6

        # Check CUMS row
        cums = rows[0]
        assert cums["KEYWORD"] == "CUMS"
        assert cums["PROPNUM"] == "WELL001"
        assert cums["SEQUENCE"] == 1

        # Check START row
        start = rows[1]
        assert start["KEYWORD"] == "START"

        # Check OIL row
        oil = rows[2]
        assert oil["KEYWORD"] == "OIL"
        assert oil["SEQUENCE"] == 100
        assert "X B/D" in oil["EXPRESSION"]

        # Check oil continuation
        oil_cont = rows[3]
        assert oil_cont["KEYWORD"] == '"'
        assert oil_cont["SEQUENCE"] == 200

        # Check GAS row
        gas = rows[4]
        assert gas["KEYWORD"] == "GAS"
        assert gas["SEQUENCE"] == 300

    def test_export_well_with_water(self):
        well = _make_well(include_water=True)
        exporter = AriesAcEconomicExporter()
        rows = exporter.export_well(well)

        # CUMS + START + (OIL + cont) + (GAS + cont) + (WATER + cont) = 8
        assert len(rows) == 8
        keywords = [r["KEYWORD"] for r in rows]
        assert "WATER" in keywords

    def test_export_well_specific_products(self):
        well = _make_well()
        exporter = AriesAcEconomicExporter()
        rows = exporter.export_well(well, products=["oil"])

        # CUMS + START + OIL + OIL continuation = 4
        assert len(rows) == 4

    def test_export_well_skips_low_qi(self):
        """Wells with qi < 0.3 should be skipped."""
        well = _make_well(qi_oil=0.1)
        exporter = AriesAcEconomicExporter()
        rows = exporter.export_well(well, products=["oil"])

        # Only CUMS + START (oil skipped due to low qi)
        assert len(rows) == 2

    def test_export_wells_dataframe(self):
        wells = [_make_well(propnum="W001"), _make_well(propnum="W002")]
        exporter = AriesAcEconomicExporter()
        df = exporter.export_wells(wells)

        assert len(df) == 12  # 6 rows per well
        assert list(df.columns) == [
            "PROPNUM", "SECTION", "SEQUENCE", "QUALIFIER", "KEYWORD", "EXPRESSION"
        ]

    def test_export_wells_empty(self):
        exporter = AriesAcEconomicExporter()
        df = exporter.export_wells([])
        assert len(df) == 0
        assert "PROPNUM" in df.columns

    def test_save(self, tmp_path):
        well = _make_well()
        exporter = AriesAcEconomicExporter()
        output = tmp_path / "forecast.csv"
        result_path = exporter.save([well], output)

        assert result_path.exists()
        content = result_path.read_text()
        assert "PROPNUM" in content
        assert "WELL001" in content

    def test_start_date_december_rollover(self):
        """Test START date calculation when last production is December."""
        dates = np.array([np.datetime64("2023-12-01")])
        well = Well(
            identifier=WellIdentifier(propnum="W001"),
            production=ProductionData(
                dates=dates,
                oil=np.array([1000.0]),
                gas=np.array([5000.0]),
            ),
        )
        model = HyperbolicModel(qi=100.0, di=0.01, b=0.5, dmin=0.005)
        well.set_forecast("oil", ForecastResult(
            model=model, r_squared=0.95, rmse=5.0,
            aic=100.0, bic=110.0, data_points_used=1,
            regime_start_idx=0,
        ))

        exporter = AriesAcEconomicExporter()
        rows = exporter.export_well(well)
        start_row = [r for r in rows if r["KEYWORD"] == "START"][0]
        assert start_row["EXPRESSION"] == "01/2024"
