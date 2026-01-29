"""Tests for ground truth comparison against ARIES projections."""

import csv
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest
import numpy as np

from pyforecast.refinement.ground_truth import (
    GroundTruthValidator,
    GroundTruthConfig,
    GroundTruthSummary,
    summarize_ground_truth_results,
)
from pyforecast.refinement.schemas import GroundTruthResult
from pyforecast.import_.aries_forecast import (
    AriesForecastParams,
    AriesForecastImporter,
)
from pyforecast.core.models import HyperbolicModel, ForecastResult


class TestGroundTruthResult:
    """Tests for GroundTruthResult dataclass."""

    def test_is_good_match_all_criteria_met(self):
        """Test is_good_match returns True when all criteria are met."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=102.0,
            pyf_di=0.0105,
            pyf_b=0.52,
            qi_pct_diff=2.0,
            di_pct_diff=5.0,
            b_abs_diff=0.02,
            comparison_months=60,
            mape=15.0,     # < 20%
            correlation=0.97,  # > 0.95
            bias=0.05,
            cumulative_diff_pct=10.0,  # < 15%
        )

        assert result.is_good_match is True

    def test_is_good_match_high_mape(self):
        """Test is_good_match returns False with high MAPE."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=102.0,
            pyf_di=0.0105,
            pyf_b=0.52,
            qi_pct_diff=2.0,
            di_pct_diff=5.0,
            b_abs_diff=0.02,
            comparison_months=60,
            mape=25.0,     # > 20%
            correlation=0.97,
            bias=0.05,
            cumulative_diff_pct=10.0,
        )

        assert result.is_good_match is False

    def test_is_good_match_low_correlation(self):
        """Test is_good_match returns False with low correlation."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=102.0,
            pyf_di=0.0105,
            pyf_b=0.52,
            qi_pct_diff=2.0,
            di_pct_diff=5.0,
            b_abs_diff=0.02,
            comparison_months=60,
            mape=15.0,
            correlation=0.90,  # < 0.95
            bias=0.05,
            cumulative_diff_pct=10.0,
        )

        assert result.is_good_match is False

    def test_is_good_match_high_cumulative_diff(self):
        """Test is_good_match returns False with high cumulative diff."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=102.0,
            pyf_di=0.0105,
            pyf_b=0.52,
            qi_pct_diff=2.0,
            di_pct_diff=5.0,
            b_abs_diff=0.02,
            comparison_months=60,
            mape=15.0,
            correlation=0.97,
            bias=0.05,
            cumulative_diff_pct=20.0,  # > 15%
        )

        assert result.is_good_match is False

    def test_is_good_match_high_b_diff(self):
        """Test is_good_match returns False with high b-factor diff."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=102.0,
            pyf_di=0.0105,
            pyf_b=0.85,  # 0.35 diff
            qi_pct_diff=2.0,
            di_pct_diff=5.0,
            b_abs_diff=0.35,  # > 0.3
            comparison_months=60,
            mape=15.0,
            correlation=0.97,
            bias=0.05,
            cumulative_diff_pct=10.0,
        )

        assert result.is_good_match is False

    def test_match_grade_a(self):
        """Test match grade A for excellent match."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=100.0,
            pyf_di=0.01,
            pyf_b=0.5,
            qi_pct_diff=0.0,
            di_pct_diff=0.0,
            b_abs_diff=0.0,
            comparison_months=60,
            mape=5.0,     # < 10
            correlation=0.99,  # > 0.98
            bias=0.0,
            cumulative_diff_pct=2.0,  # < 5
        )

        assert result.match_grade == "A"

    def test_match_grade_d(self):
        """Test match grade D for poor match."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=200.0,
            pyf_di=0.02,
            pyf_b=1.0,
            qi_pct_diff=100.0,
            di_pct_diff=100.0,
            b_abs_diff=0.5,  # > 0.5, no points
            comparison_months=60,
            mape=50.0,     # > 30, no points
            correlation=0.80,  # < 0.90, no points
            bias=0.5,
            cumulative_diff_pct=50.0,  # > 25, no points
        )

        assert result.match_grade == "D"

    def test_summary(self):
        """Test summary method returns correct dict."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=102.0,
            pyf_di=0.0105,
            pyf_b=0.52,
            qi_pct_diff=2.0,
            di_pct_diff=5.0,
            b_abs_diff=0.02,
            comparison_months=60,
            mape=15.0,
            correlation=0.97,
            bias=0.05,
            cumulative_diff_pct=10.0,
        )

        summary = result.summary()

        assert summary["well_id"] == "test-well"
        assert summary["product"] == "oil"
        assert summary["mape"] == 15.0
        assert summary["is_good_match"] is True
        assert summary["match_grade"] in ["A", "B", "C", "D"]


class TestGroundTruthValidator:
    """Tests for GroundTruthValidator class."""

    def _create_mock_well(self, well_id, propnum, qi=100.0, di=0.01, b=0.5):
        """Create a mock well with forecast."""
        well = MagicMock()
        well.well_id = well_id
        well.identifier.propnum = propnum
        well.identifier.api = None
        well.identifier.entity_id = None

        # Create real HyperbolicModel
        model = HyperbolicModel(qi=qi, di=di, b=b, dmin=0.005)

        forecast = MagicMock()
        forecast.model = model

        well.get_forecast.return_value = forecast
        return well

    def _create_importer_with_params(self, propnum, product, qi=100.0, di=0.01, b=0.5):
        """Create an importer with specific parameters."""
        importer = AriesForecastImporter()
        params = AriesForecastParams(
            propnum=propnum,
            product=product,
            qi=qi,
            di=di,
            b=b,
            dmin=0.005,
            decline_type="HYP",
        )
        importer._forecasts[(propnum, product)] = params
        return importer

    def test_validate_matching_params(self):
        """Test validation with matching parameters returns good match."""
        importer = self._create_importer_with_params(
            "test-well", "oil", qi=100.0, di=0.01, b=0.5
        )
        well = self._create_mock_well(
            "test-well", "test-well", qi=100.0, di=0.01, b=0.5
        )

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is not None
        assert result.well_id == "test-well"
        assert result.product == "oil"
        assert result.mape < 1.0  # Near-perfect match
        assert result.correlation > 0.99
        assert result.is_good_match is True

    def test_validate_different_params(self):
        """Test validation with different parameters."""
        importer = self._create_importer_with_params(
            "test-well", "oil", qi=100.0, di=0.01, b=0.5
        )
        # pyforecast fit has different params
        well = self._create_mock_well(
            "test-well", "test-well", qi=120.0, di=0.015, b=0.7
        )

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is not None
        assert result.qi_pct_diff == pytest.approx(20.0, rel=0.01)  # 20% higher qi
        assert result.b_abs_diff == pytest.approx(0.2, abs=0.01)  # 0.2 higher b

    def test_validate_no_pyf_forecast(self):
        """Test returns None when well has no pyforecast fit."""
        importer = self._create_importer_with_params("test-well", "oil")

        well = MagicMock()
        well.well_id = "test-well"
        well.identifier.propnum = "test-well"
        well.get_forecast.return_value = None

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is None

    def test_validate_no_aries_data(self):
        """Test returns None when ARIES has no data for well."""
        importer = AriesForecastImporter()  # Empty
        well = self._create_mock_well("test-well", "test-well")

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is None

    def test_validate_custom_comparison_months(self):
        """Test validation with custom comparison months."""
        importer = self._create_importer_with_params("test-well", "oil")
        well = self._create_mock_well("test-well", "test-well")

        config = GroundTruthConfig(comparison_months=120)
        validator = GroundTruthValidator(importer, config)
        result = validator.validate(well, "oil")

        assert result is not None
        assert result.comparison_months == 120
        assert len(result.forecast_months) == 120

    def test_validate_arrays_populated(self):
        """Test that forecast arrays are populated for plotting."""
        importer = self._create_importer_with_params("test-well", "oil")
        well = self._create_mock_well("test-well", "test-well")

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is not None
        assert len(result.forecast_months) == 60
        assert len(result.aries_rates) == 60
        assert len(result.pyf_rates) == 60
        assert result.forecast_months[0] == 0
        assert result.forecast_months[-1] == 59

    def test_validate_product_mismatch(self):
        """Test returns None when product not in ARIES data."""
        importer = self._create_importer_with_params("test-well", "oil")
        well = self._create_mock_well("test-well", "test-well")

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "gas")  # Request gas, have oil

        assert result is None


class TestSummarizeGroundTruthResults:
    """Tests for summarize_ground_truth_results function."""

    def test_empty_results(self):
        """Test summary with empty results."""
        summary = summarize_ground_truth_results([])

        assert summary["count"] == 0
        assert summary["avg_mape"] is None
        assert summary["good_match_count"] == 0

    def test_single_result(self):
        """Test summary with single result."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=100.0,
            pyf_di=0.01,
            pyf_b=0.5,
            qi_pct_diff=0.0,
            di_pct_diff=0.0,
            b_abs_diff=0.0,
            comparison_months=60,
            mape=5.0,
            correlation=0.99,
            bias=0.0,
            cumulative_diff_pct=2.0,
        )

        summary = summarize_ground_truth_results([result])

        assert summary["count"] == 1
        assert summary["avg_mape"] == 5.0
        assert summary["good_match_count"] == 1
        assert summary["good_match_pct"] == 100.0

    def test_multiple_results(self):
        """Test summary with multiple results."""
        results = [
            GroundTruthResult(
                well_id=f"well-{i}",
                product="oil",
                aries_qi=100.0,
                aries_di=0.01,
                aries_b=0.5,
                aries_decline_type="HYP",
                pyf_qi=100.0,
                pyf_di=0.01,
                pyf_b=0.5,
                qi_pct_diff=0.0,
                di_pct_diff=0.0,
                b_abs_diff=0.0,
                comparison_months=60,
                mape=float(i * 5),  # 0, 5, 10, 15, 20
                correlation=0.99 - i * 0.01,  # 0.99, 0.98, 0.97, 0.96, 0.95
                bias=0.0,
                cumulative_diff_pct=float(i * 2),
            )
            for i in range(5)
        ]

        summary = summarize_ground_truth_results(results)

        assert summary["count"] == 5
        assert summary["avg_mape"] == 10.0  # (0+5+10+15+20)/5
        assert summary["median_mape"] == 10.0

    def test_by_product_breakdown(self):
        """Test summary includes by-product breakdown."""
        results = [
            GroundTruthResult(
                well_id="well-1",
                product="oil",
                aries_qi=100.0,
                aries_di=0.01,
                aries_b=0.5,
                aries_decline_type="HYP",
                pyf_qi=100.0,
                pyf_di=0.01,
                pyf_b=0.5,
                qi_pct_diff=0.0,
                di_pct_diff=0.0,
                b_abs_diff=0.0,
                comparison_months=60,
                mape=10.0,
                correlation=0.97,
                bias=0.0,
                cumulative_diff_pct=5.0,
            ),
            GroundTruthResult(
                well_id="well-1",
                product="gas",
                aries_qi=500.0,
                aries_di=0.015,
                aries_b=0.3,
                aries_decline_type="HYP",
                pyf_qi=500.0,
                pyf_di=0.015,
                pyf_b=0.3,
                qi_pct_diff=0.0,
                di_pct_diff=0.0,
                b_abs_diff=0.0,
                comparison_months=60,
                mape=20.0,
                correlation=0.96,
                bias=0.0,
                cumulative_diff_pct=10.0,
            ),
        ]

        summary = summarize_ground_truth_results(results)

        assert "by_product" in summary
        assert "oil" in summary["by_product"]
        assert "gas" in summary["by_product"]
        assert summary["by_product"]["oil"]["count"] == 1
        assert summary["by_product"]["gas"]["count"] == 1

    def test_grade_distribution(self):
        """Test summary includes grade distribution."""
        # Create results with different grades
        results = []
        for i, (mape, corr, cum) in enumerate([
            (5, 0.99, 2),   # A grade
            (15, 0.96, 10),  # B grade
            (25, 0.92, 20),  # C grade
            (40, 0.85, 40),  # D grade
        ]):
            results.append(GroundTruthResult(
                well_id=f"well-{i}",
                product="oil",
                aries_qi=100.0,
                aries_di=0.01,
                aries_b=0.5,
                aries_decline_type="HYP",
                pyf_qi=100.0,
                pyf_di=0.01,
                pyf_b=0.5,
                qi_pct_diff=0.0,
                di_pct_diff=0.0,
                b_abs_diff=0.0,
                comparison_months=60,
                mape=float(mape),
                correlation=corr,
                bias=0.0,
                cumulative_diff_pct=float(cum),
            ))

        summary = summarize_ground_truth_results(results)

        assert "grade_distribution" in summary
        assert summary["grade_distribution"]["A"] >= 1
        assert summary["grade_distribution"]["D"] >= 1


class TestGroundTruthResultMapeHandling:
    """Tests for MAPE edge case handling."""

    def test_mape_valid_property(self):
        """Test mape_valid property returns correct value."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=100.0,
            pyf_di=0.01,
            pyf_b=0.5,
            qi_pct_diff=0.0,
            di_pct_diff=0.0,
            b_abs_diff=0.0,
            comparison_months=60,
            mape=10.0,
            correlation=0.99,
            bias=0.0,
            cumulative_diff_pct=5.0,
        )
        assert result.mape_valid is True

    def test_mape_valid_false_when_none(self):
        """Test mape_valid returns False when MAPE is None."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=100.0,
            pyf_di=0.01,
            pyf_b=0.5,
            qi_pct_diff=0.0,
            di_pct_diff=0.0,
            b_abs_diff=0.0,
            comparison_months=60,
            mape=None,
            correlation=0.99,
            bias=0.0,
            cumulative_diff_pct=5.0,
        )
        assert result.mape_valid is False

    def test_is_good_match_false_when_mape_none(self):
        """Test is_good_match returns False when MAPE is None."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=100.0,
            pyf_di=0.01,
            pyf_b=0.5,
            qi_pct_diff=0.0,
            di_pct_diff=0.0,
            b_abs_diff=0.0,
            comparison_months=60,
            mape=None,
            correlation=0.99,
            bias=0.0,
            cumulative_diff_pct=5.0,
        )
        assert result.is_good_match is False

    def test_match_grade_x_when_mape_none(self):
        """Test match_grade returns X when MAPE is None."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=100.0,
            pyf_di=0.01,
            pyf_b=0.5,
            qi_pct_diff=0.0,
            di_pct_diff=0.0,
            b_abs_diff=0.0,
            comparison_months=60,
            mape=None,
            correlation=0.99,
            bias=0.0,
            cumulative_diff_pct=5.0,
        )
        assert result.match_grade == "X"


class TestSummaryWithNoneMape:
    """Tests for summary functions with None MAPE values."""

    def test_summary_filters_none_mapes(self):
        """Test that summary calculates averages excluding None MAPEs."""
        results = [
            GroundTruthResult(
                well_id="well-1",
                product="oil",
                aries_qi=100.0,
                aries_di=0.01,
                aries_b=0.5,
                aries_decline_type="HYP",
                pyf_qi=100.0,
                pyf_di=0.01,
                pyf_b=0.5,
                qi_pct_diff=0.0,
                di_pct_diff=0.0,
                b_abs_diff=0.0,
                comparison_months=60,
                mape=10.0,
                correlation=0.99,
                bias=0.0,
                cumulative_diff_pct=5.0,
            ),
            GroundTruthResult(
                well_id="well-2",
                product="oil",
                aries_qi=100.0,
                aries_di=0.01,
                aries_b=0.5,
                aries_decline_type="HYP",
                pyf_qi=100.0,
                pyf_di=0.01,
                pyf_b=0.5,
                qi_pct_diff=0.0,
                di_pct_diff=0.0,
                b_abs_diff=0.0,
                comparison_months=60,
                mape=None,  # Insufficient data
                correlation=0.99,
                bias=0.0,
                cumulative_diff_pct=5.0,
            ),
            GroundTruthResult(
                well_id="well-3",
                product="oil",
                aries_qi=100.0,
                aries_di=0.01,
                aries_b=0.5,
                aries_decline_type="HYP",
                pyf_qi=100.0,
                pyf_di=0.01,
                pyf_b=0.5,
                qi_pct_diff=0.0,
                di_pct_diff=0.0,
                b_abs_diff=0.0,
                comparison_months=60,
                mape=20.0,
                correlation=0.99,
                bias=0.0,
                cumulative_diff_pct=5.0,
            ),
        ]

        summary = summarize_ground_truth_results(results)

        # Average should be (10 + 20) / 2 = 15, not (10 + 0 + 20) / 3
        assert summary["avg_mape"] == 15.0
        assert summary["mape_unavailable_count"] == 1
        assert summary["grade_distribution"]["X"] == 1


class TestValidateBatch:
    """Tests for validate_batch method."""

    def _create_mock_well(self, well_id, propnum, qi=100.0, di=0.01, b=0.5):
        """Create a mock well with forecast."""
        well = MagicMock()
        well.well_id = well_id
        well.identifier.propnum = propnum
        well.identifier.api = None
        well.identifier.entity_id = None

        model = HyperbolicModel(qi=qi, di=di, b=b, dmin=0.005)
        forecast = MagicMock()
        forecast.model = model
        well.get_forecast.return_value = forecast
        return well

    def test_validate_batch_returns_summary(self):
        """Test that validate_batch returns GroundTruthSummary."""
        importer = AriesForecastImporter()
        params = AriesForecastParams(
            propnum="test-well",
            product="oil",
            qi=100.0,
            di=0.01,
            b=0.5,
            dmin=0.005,
            decline_type="HYP",
        )
        importer._forecasts[("test-well", "oil")] = params

        well = self._create_mock_well("test-well", "test-well")
        validator = GroundTruthValidator(importer)

        summary = validator.validate_batch([well], ["oil"])

        assert isinstance(summary, GroundTruthSummary)
        assert summary.wells_matched == 1
        assert len(summary.results) == 1

    def test_validate_batch_tracks_mismatches(self):
        """Test that validate_batch tracks ID mismatches."""
        importer = AriesForecastImporter()
        # Add ARIES data for a well NOT in pyforecast batch
        params = AriesForecastParams(
            propnum="aries-only-well",
            product="oil",
            qi=100.0,
            di=0.01,
            b=0.5,
            dmin=0.005,
            decline_type="HYP",
        )
        importer._forecasts[("aries-only-well", "oil")] = params

        # Create pyforecast well NOT in ARIES
        well = self._create_mock_well("pyf-only-well", "pyf-only-well")

        validator = GroundTruthValidator(importer)
        summary = validator.validate_batch([well], ["oil"])

        assert "pyf-only-well" in summary.wells_in_pyf_only
        assert "aries-only-well" in summary.wells_in_aries_only
        assert summary.wells_matched == 0


class TestRateValidation:
    """Tests for rate array validation."""

    def _create_importer_with_params(self, propnum, product, qi=100.0, di=0.01, b=0.5):
        """Create an importer with specific parameters."""
        importer = AriesForecastImporter()
        params = AriesForecastParams(
            propnum=propnum,
            product=product,
            qi=qi,
            di=di,
            b=b,
            dmin=0.005,
            decline_type="HYP",
        )
        importer._forecasts[(propnum, product)] = params
        return importer

    def test_validate_rates_handles_nan(self):
        """Test that _validate_rates replaces NaN with 0."""
        importer = self._create_importer_with_params("test", "oil")
        validator = GroundTruthValidator(importer)

        rates = np.array([100.0, np.nan, 80.0, np.nan])
        cleaned = validator._validate_rates(rates, "test")

        assert not np.any(np.isnan(cleaned))
        assert cleaned[1] == 0.0
        assert cleaned[3] == 0.0

    def test_validate_rates_handles_inf(self):
        """Test that _validate_rates clips infinite values."""
        importer = self._create_importer_with_params("test", "oil")
        validator = GroundTruthValidator(importer)

        rates = np.array([100.0, np.inf, 80.0, -np.inf])
        cleaned = validator._validate_rates(rates, "test")

        assert not np.any(np.isinf(cleaned))
        assert cleaned[1] <= 1e9
        assert cleaned[3] >= 0

    def test_validate_rates_handles_negative(self):
        """Test that _validate_rates clips negative values to 0."""
        importer = self._create_importer_with_params("test", "oil")
        validator = GroundTruthValidator(importer)

        rates = np.array([100.0, -50.0, 80.0, -10.0])
        cleaned = validator._validate_rates(rates, "test")

        assert not np.any(cleaned < 0)
        assert cleaned[1] == 0.0
        assert cleaned[3] == 0.0


class TestDateAlignment:
    """Tests for forecast start date alignment checking."""

    def _create_mock_well_with_production(
        self, well_id, propnum, last_prod_date, qi=100.0, di=0.01, b=0.5
    ):
        """Create a mock well with forecast and production data."""
        well = MagicMock()
        well.well_id = well_id
        well.identifier.propnum = propnum
        well.identifier.api = None
        well.identifier.entity_id = None

        # Mock production data with last_date
        well.production.last_date = last_prod_date

        # Create real HyperbolicModel
        model = HyperbolicModel(qi=qi, di=di, b=b, dmin=0.005)
        forecast = MagicMock()
        forecast.model = model
        well.get_forecast.return_value = forecast

        return well

    def _create_importer_with_start_date(
        self, propnum, product, start_date, qi=100.0, di=0.01, b=0.5
    ):
        """Create an importer with specific parameters including start_date."""
        importer = AriesForecastImporter()
        params = AriesForecastParams(
            propnum=propnum,
            product=product,
            qi=qi,
            di=di,
            b=b,
            dmin=0.005,
            decline_type="HYP",
            start_date=start_date,
        )
        importer._forecasts[(propnum, product)] = params
        return importer

    def test_aligned_dates_no_warning(self):
        """Test no warning when ARIES and pyforecast start dates match."""
        # ARIES start: Feb 2025
        # pyforecast: last production Jan 2025 -> forecast starts Feb 2025
        importer = self._create_importer_with_start_date(
            "test-well", "oil", date(2025, 2, 1)
        )
        well = self._create_mock_well_with_production(
            "test-well", "test-well", date(2025, 1, 15)  # Last prod mid-Jan
        )

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is not None
        assert result.aries_start_date == date(2025, 2, 1)
        assert result.pyf_start_date == date(2025, 2, 1)
        assert result.alignment_warning is None

    def test_misaligned_dates_warning(self):
        """Test warning generated when dates differ by 3 months."""
        # ARIES start: Feb 2025
        # pyforecast: last production Oct 2024 -> forecast starts Nov 2024 (3 month diff)
        importer = self._create_importer_with_start_date(
            "test-well", "oil", date(2025, 2, 1)
        )
        well = self._create_mock_well_with_production(
            "test-well", "test-well", date(2024, 10, 15)  # Last prod mid-Oct
        )

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is not None
        assert result.aries_start_date == date(2025, 2, 1)
        assert result.pyf_start_date == date(2024, 11, 1)
        assert result.alignment_warning is not None
        assert "3 month(s)" in result.alignment_warning
        assert "ARIES=2025-02" in result.alignment_warning
        assert "pyf=2024-11" in result.alignment_warning

    def test_pyf_start_date_december_rollover(self):
        """Test pyf start date calculation rolls over December to January."""
        importer = self._create_importer_with_start_date(
            "test-well", "oil", date(2025, 1, 1)
        )
        # Last production in December -> forecast starts January
        well = self._create_mock_well_with_production(
            "test-well", "test-well", date(2024, 12, 15)
        )

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is not None
        assert result.pyf_start_date == date(2025, 1, 1)
        assert result.alignment_warning is None  # Dates match

    def test_no_aries_start_date(self):
        """Test no warning when ARIES has no start date."""
        importer = self._create_importer_with_start_date(
            "test-well", "oil", None  # No start date
        )
        well = self._create_mock_well_with_production(
            "test-well", "test-well", date(2024, 10, 15)
        )

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is not None
        assert result.aries_start_date is None
        assert result.pyf_start_date == date(2024, 11, 1)
        assert result.alignment_warning is None  # Can't compare

    def test_no_pyf_production_date(self):
        """Test no warning when pyforecast has no production date."""
        importer = self._create_importer_with_start_date(
            "test-well", "oil", date(2025, 2, 1)
        )
        well = self._create_mock_well_with_production(
            "test-well", "test-well", None  # No last_date
        )

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is not None
        assert result.aries_start_date == date(2025, 2, 1)
        assert result.pyf_start_date is None
        assert result.alignment_warning is None  # Can't compare

    def test_one_month_difference_warning(self):
        """Test warning for 1 month difference."""
        importer = self._create_importer_with_start_date(
            "test-well", "oil", date(2025, 3, 1)
        )
        well = self._create_mock_well_with_production(
            "test-well", "test-well", date(2025, 1, 15)  # Feb start
        )

        validator = GroundTruthValidator(importer)
        result = validator.validate(well, "oil")

        assert result is not None
        assert result.alignment_warning is not None
        assert "1 month(s)" in result.alignment_warning

    def test_result_summary_includes_alignment(self):
        """Test that result.summary() includes alignment fields."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=100.0,
            pyf_di=0.01,
            pyf_b=0.5,
            qi_pct_diff=0.0,
            di_pct_diff=0.0,
            b_abs_diff=0.0,
            comparison_months=60,
            mape=5.0,
            correlation=0.99,
            bias=0.0,
            cumulative_diff_pct=2.0,
            aries_start_date=date(2025, 2, 1),
            pyf_start_date=date(2024, 11, 1),
            alignment_warning="Start dates differ by 3 month(s)",
        )

        summary = result.summary()

        assert summary["aries_start_date"] == "2025-02-01"
        assert summary["pyf_start_date"] == "2024-11-01"
        assert summary["alignment_warning"] == "Start dates differ by 3 month(s)"

    def test_result_summary_handles_none_dates(self):
        """Test that result.summary() handles None dates."""
        result = GroundTruthResult(
            well_id="test-well",
            product="oil",
            aries_qi=100.0,
            aries_di=0.01,
            aries_b=0.5,
            aries_decline_type="HYP",
            pyf_qi=100.0,
            pyf_di=0.01,
            pyf_b=0.5,
            qi_pct_diff=0.0,
            di_pct_diff=0.0,
            b_abs_diff=0.0,
            comparison_months=60,
            mape=5.0,
            correlation=0.99,
            bias=0.0,
            cumulative_diff_pct=2.0,
            aries_start_date=None,
            pyf_start_date=None,
            alignment_warning=None,
        )

        summary = result.summary()

        assert summary["aries_start_date"] is None
        assert summary["pyf_start_date"] is None
        assert summary["alignment_warning"] is None
