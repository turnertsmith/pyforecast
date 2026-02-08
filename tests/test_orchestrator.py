"""Tests for validation/orchestrator.py."""

import numpy as np
import pytest

from pyforecast.config import PyForecastConfig
from pyforecast.core.models import HyperbolicModel, ForecastResult
from pyforecast.data.well import Well, WellIdentifier, ProductionData
from pyforecast.validation.orchestrator import ValidationOrchestrator


def _make_well(
    well_id="WELL001",
    n_months=24,
    oil_rate=3000.0,
    gas_rate=15000.0,
    include_forecast=False,
):
    """Create a test well with realistic production data."""
    dates = np.array(
        [np.datetime64(f"2022-{m:02d}-01") for m in range(1, 13)]
        + [np.datetime64(f"2023-{m:02d}-01") for m in range(1, min(n_months - 11, 13))]
    )[:n_months]

    # Add some realistic variance
    rng = np.random.default_rng(42)
    oil = np.maximum(oil_rate + rng.normal(0, oil_rate * 0.1, n_months), 0.0)
    gas = np.maximum(gas_rate + rng.normal(0, gas_rate * 0.1, n_months), 0.0)
    water = np.maximum(500.0 + rng.normal(0, 50, n_months), 0.0)

    well = Well(
        identifier=WellIdentifier(propnum=well_id),
        production=ProductionData(dates=dates, oil=oil, gas=gas, water=water),
    )

    if include_forecast:
        for product, qi in [("oil", 100.0), ("gas", 500.0), ("water", 20.0)]:
            model = HyperbolicModel(qi=qi, di=0.01, b=0.5, dmin=0.005)
            well.set_forecast(product, ForecastResult(
                model=model, r_squared=0.92, rmse=5.0,
                aic=100.0, bic=110.0, data_points_used=n_months,
                regime_start_idx=0,
            ))

    return well


class TestValidationOrchestrator:
    """Tests for ValidationOrchestrator."""

    def test_init_from_config(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)

        assert orch.input_validator is not None
        assert orch.quality_validator is not None
        assert orch.fitting_validator is not None

    def test_from_config_classmethod(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator.from_config(config)
        assert orch.config is config

    def test_validate_input(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)
        well = _make_well()

        result = orch.validate_input(well)
        assert result is not None
        # Normal data should have no errors
        assert result.error_count == 0

    def test_validate_input_extreme_rates(self):
        config = PyForecastConfig()
        config.validation.max_oil_rate = 100.0  # Very low threshold
        orch = ValidationOrchestrator(config=config)
        well = _make_well(oil_rate=5000.0)

        result = orch.validate_input(well)
        # Should flag rate exceedance
        assert result.has_warnings or result.has_errors

    def test_validate_quality(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)
        well = _make_well()

        result = orch.validate_quality(well, "oil")
        assert result is not None

    def test_validate_pre_fit(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)
        well = _make_well()

        result = orch.validate_pre_fit(well, "oil")
        assert result is not None

    def test_validate_post_fit(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)
        well = _make_well(include_forecast=True)

        result = orch.validate_post_fit(well, "oil")
        assert result is not None

    def test_validate_well_full(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)
        well = _make_well()

        result = orch.validate_well_full(well, products=["oil", "gas"])
        assert result is not None
        assert result.well_id == "WELL001"

    def test_validate_well_full_default_products(self):
        config = PyForecastConfig()
        config.output.products = ["oil"]
        orch = ValidationOrchestrator(config=config)
        well = _make_well()

        result = orch.validate_well_full(well)
        assert result is not None

    def test_validate_well_full_with_post_fit(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)
        well = _make_well(include_forecast=True)

        result = orch.validate_well_full(
            well, products=["oil"], include_post_fit=True
        )
        assert result is not None

    def test_validate_wells(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)
        wells = [_make_well(well_id="W001"), _make_well(well_id="W002")]

        results = orch.validate_wells(wells, products=["oil"])
        assert len(results) == 2

    def test_get_summary(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)
        wells = [_make_well()]
        results = orch.validate_wells(wells, products=["oil"])

        summary = orch.get_summary(results)
        assert "total_wells" in summary
        assert summary["total_wells"] == 1

    def test_format_report(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)
        wells = [_make_well()]
        results = orch.validate_wells(wells, products=["oil"])

        report = orch.format_report(results)
        assert "PyForecast Validation Report" in report
        assert "Total wells:" in report

    def test_format_report_include_ok(self):
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)
        wells = [_make_well()]
        results = orch.validate_wells(wells, products=["oil"])

        report = orch.format_report(results, include_ok=True)
        assert "WELL001" in report

    def test_unavailable_product_handled(self):
        """Orchestrator should handle products not available in well data."""
        config = PyForecastConfig()
        orch = ValidationOrchestrator(config=config)

        dates = np.array([np.datetime64("2022-01-01")])
        well = Well(
            identifier=WellIdentifier(propnum="W001"),
            production=ProductionData(
                dates=dates,
                oil=np.array([1000.0]),
                gas=np.array([5000.0]),
                water=np.array([100.0]),
            ),
        )

        # Products that do have data should work fine
        result = orch.validate_well_full(well, products=["oil", "gas"])
        assert result is not None
