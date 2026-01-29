"""Tests for ground truth comparison against ARIES projections."""

import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest
import numpy as np

from pyforecast.refinement.ground_truth import (
    GroundTruthValidator,
    GroundTruthConfig,
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
