"""Integration tests for end-to-end workflows."""

from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pytest

from pyforecast.data.base import load_wells, detect_parser, DataParser
from pyforecast.data.enverus import EnverusParser
from pyforecast.data.aries import AriesParser
from pyforecast.core.fitting import DeclineFitter, FittingConfig
from pyforecast.export.aries_ac_economic import AriesAcEconomicExporter
from pyforecast.visualization.plots import DeclinePlotter


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestDataParsing:
    """Test data loading and parsing."""

    def test_load_enverus_csv(self):
        """Test loading Enverus format CSV."""
        wells = load_wells(FIXTURES_DIR / "sample_enverus.csv")

        assert len(wells) == 2
        assert wells[0].identifier.entity_id == "WELL001"
        assert wells[0].production.n_months == 12
        assert wells[0].production.oil[0] == 1000

    def test_load_aries_csv(self):
        """Test loading ARIES format CSV."""
        wells = load_wells(FIXTURES_DIR / "sample_aries.csv")

        assert len(wells) == 2
        assert wells[0].identifier.propnum == "PROP001"
        assert wells[0].production.n_months == 12

    def test_auto_detect_enverus(self):
        """Test auto-detection of Enverus format."""
        df = DataParser.load_file(FIXTURES_DIR / "sample_enverus.csv")
        parser = detect_parser(df)

        assert isinstance(parser, EnverusParser)

    def test_auto_detect_aries(self):
        """Test auto-detection of ARIES format."""
        df = DataParser.load_file(FIXTURES_DIR / "sample_aries.csv")
        parser = detect_parser(df)

        assert isinstance(parser, AriesParser)


class TestEndToEndWorkflow:
    """Test complete analysis workflow."""

    def test_fit_and_export(self):
        """Test loading, fitting, and exporting."""
        # Load wells
        wells = load_wells(FIXTURES_DIR / "sample_enverus.csv")
        assert len(wells) >= 1

        # Fit decline curves
        config = FittingConfig(b_min=0.01, b_max=1.5, dmin_annual=0.06)
        fitter = DeclineFitter(config)

        for well in wells:
            for product in ["oil", "gas"]:
                t = well.production.time_months
                q = well.production.get_product(product)

                result = fitter.fit(t, q)
                well.set_forecast(product, result)

                # Verify fit quality
                assert result.r_squared > 0.5
                assert result.model.qi > 0

        # Export to ARIES AC_ECONOMIC format
        exporter = AriesAcEconomicExporter()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = Path(f.name)

        try:
            exporter.save(wells, output_path, products=["oil", "gas"])

            # Verify export
            df = pd.read_csv(output_path)
            assert len(df) > 0
            assert "PROPNUM" in df.columns
            assert "KEYWORD" in df.columns
            assert "EXPRESSION" in df.columns

        finally:
            output_path.unlink()

    def test_fit_and_plot(self):
        """Test loading, fitting, and plotting."""
        wells = load_wells(FIXTURES_DIR / "sample_enverus.csv")
        well = wells[0]

        # Fit
        fitter = DeclineFitter()
        t = well.production.time_months
        q = well.production.oil

        result = fitter.fit(t, q)
        well.forecast_oil = result

        # Plot (just verify no errors)
        plotter = DeclinePlotter()
        fig = plotter.plot_well(well, "oil")

        assert fig is not None
        assert len(fig.data) >= 2  # Historical + forecast traces

    def test_batch_workflow(self):
        """Test batch processing workflow."""
        from pyforecast.batch.processor import BatchProcessor, BatchConfig

        config = BatchConfig(
            products=["oil"],
            min_points=6,
            workers=1,
            save_plots=False,
            save_batch_plot=False,
        )

        processor = BatchProcessor(config)
        wells = processor.load_files([FIXTURES_DIR / "sample_enverus.csv"])

        result = processor.process(wells, show_progress=False)

        assert result.successful >= 1
        assert result.skipped == 0
        assert len(result.wells) == 2


class TestSyntheticDataRecovery:
    """Test parameter recovery from synthetic data."""

    def test_recover_exponential_params(self):
        """Test recovery of known exponential parameters."""
        # Generate synthetic exponential decline
        qi_true, di_true = 1500, 0.06
        t = np.arange(36, dtype=float)
        q = qi_true * np.exp(-di_true * t)

        # Add small noise
        np.random.seed(42)
        q = q + np.random.normal(0, 15, len(t))
        q = np.maximum(q, 1)

        # Fit
        config = FittingConfig(b_min=0.01, b_max=0.1)
        fitter = DeclineFitter(config)
        result = fitter.fit(t, q)

        # Verify recovery
        assert result.model.qi == pytest.approx(qi_true, rel=0.1)
        assert result.model.di == pytest.approx(di_true, rel=0.15)
        assert result.model.b < 0.15
        assert result.r_squared > 0.95

    def test_recover_hyperbolic_params(self):
        """Test recovery of known hyperbolic parameters."""
        # Generate synthetic hyperbolic decline
        qi_true, di_true, b_true = 2000, 0.12, 0.7
        t = np.arange(48, dtype=float)
        q = qi_true / np.power(1 + b_true * di_true * t, 1 / b_true)

        # Add noise
        np.random.seed(123)
        q = q + np.random.normal(0, 20, len(t))
        q = np.maximum(q, 1)

        # Fit
        config = FittingConfig(b_min=0.01, b_max=1.5)
        fitter = DeclineFitter(config)
        result = fitter.fit(t, q)

        # Verify recovery (hyperbolic is harder to fit precisely)
        assert result.model.qi == pytest.approx(qi_true, rel=0.15)
        assert result.model.b == pytest.approx(b_true, rel=0.25)
        assert result.r_squared > 0.9

    def test_regime_change_recovery(self):
        """Test correct handling of regime change."""
        # Create data with regime change (refrac)
        t = np.arange(36, dtype=float)

        # First 18 months: declining
        q1 = 1000 * np.exp(-0.08 * t[:18])

        # After refrac: new higher rate, different decline
        q2 = 1200 * np.exp(-0.05 * (t[18:] - 18))

        q = np.concatenate([q1, q2])

        # Fit with regime detection
        config = FittingConfig(regime_threshold=1.0)
        fitter = DeclineFitter(config)
        result = fitter.fit(t, q)

        # Should detect regime change and fit to post-refrac data
        assert result.regime_start_idx > 0
        assert result.regime_start_idx >= 15  # Should be around month 18

        # qi should reflect post-refrac rate
        assert result.model.qi == pytest.approx(1200, rel=0.15)
