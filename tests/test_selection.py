"""Tests for core/selection.py model selection and fit quality evaluation."""

import pytest

from pyforecast.core.selection import (
    evaluate_fit_quality,
    _grade_fit,
    compare_fits,
    DEFAULT_GRADE_THRESHOLDS,
)
from pyforecast.core.models import HyperbolicModel, ForecastResult


def _make_result(r_squared=0.90, b=0.5, data_points=24, regime_start=0, dmin=0.005):
    """Create a ForecastResult with specified parameters."""
    model = HyperbolicModel(qi=100.0, di=0.01, b=b, dmin=dmin)
    return ForecastResult(
        model=model,
        r_squared=r_squared,
        rmse=5.0,
        aic=100.0,
        bic=110.0,
        data_points_used=data_points,
        regime_start_idx=regime_start,
    )


class TestGradeFit:
    """Tests for _grade_fit helper."""

    def test_grade_a(self):
        assert _grade_fit(0.96) == "A"

    def test_grade_b(self):
        assert _grade_fit(0.90) == "B"

    def test_grade_c(self):
        assert _grade_fit(0.75) == "C"

    def test_grade_d(self):
        assert _grade_fit(0.55) == "D"

    def test_grade_f(self):
        assert _grade_fit(0.40) == "F"

    def test_boundary_a(self):
        assert _grade_fit(0.95) == "A"

    def test_boundary_b(self):
        assert _grade_fit(0.85) == "B"

    def test_boundary_c(self):
        assert _grade_fit(0.70) == "C"

    def test_boundary_d(self):
        assert _grade_fit(0.50) == "D"

    def test_custom_thresholds(self):
        thresholds = {"A": 0.99, "B": 0.95, "C": 0.80, "D": 0.60}
        assert _grade_fit(0.97, thresholds) == "B"
        assert _grade_fit(0.85, thresholds) == "C"


class TestEvaluateFitQuality:
    """Tests for evaluate_fit_quality."""

    def test_good_fit(self):
        result = _make_result(r_squared=0.96)
        assessment = evaluate_fit_quality(result)

        assert assessment["acceptable"] is True
        assert assessment["quality_grade"] == "A"
        assert assessment["r_squared"] == 0.96
        assert len(assessment["warnings"]) == 0

    def test_marginal_fit_warning(self):
        result = _make_result(r_squared=0.65)
        assessment = evaluate_fit_quality(result)

        assert assessment["acceptable"] is False
        assert any("Marginal" in w or "Poor" in w for w in assessment["warnings"])

    def test_poor_fit_warning(self):
        result = _make_result(r_squared=0.40)
        assessment = evaluate_fit_quality(result)

        assert any("Poor fit" in w for w in assessment["warnings"])

    def test_low_b_warning(self):
        result = _make_result(b=0.05)
        assessment = evaluate_fit_quality(result)

        assert any("Near-exponential" in w for w in assessment["warnings"])

    def test_high_b_warning(self):
        result = _make_result(b=1.3)
        assessment = evaluate_fit_quality(result)

        assert any("High b-factor" in w for w in assessment["warnings"])

    def test_limited_data_warning(self):
        result = _make_result(data_points=6)
        assessment = evaluate_fit_quality(result)

        assert any("Limited data" in w for w in assessment["warnings"])

    def test_regime_change_warning(self):
        result = _make_result(regime_start=5)
        assessment = evaluate_fit_quality(result)

        assert any("Regime change" in w for w in assessment["warnings"])

    def test_custom_thresholds(self):
        thresholds = {"A": 0.99, "B": 0.95, "C": 0.80, "D": 0.60}
        result = _make_result(r_squared=0.96)
        assessment = evaluate_fit_quality(result, grade_thresholds=thresholds)

        assert assessment["quality_grade"] == "B"

    def test_includes_decline_type(self):
        result = _make_result()
        assessment = evaluate_fit_quality(result)
        assert "decline_type" in assessment


class TestCompareFits:
    """Tests for compare_fits."""

    def test_selects_best_by_bic(self):
        r1 = _make_result(r_squared=0.90)
        r2 = _make_result(r_squared=0.92)

        # Give r2 a lower BIC
        r2_model = HyperbolicModel(qi=100.0, di=0.01, b=0.5, dmin=0.005)
        r2 = ForecastResult(
            model=r2_model, r_squared=0.92, rmse=4.0,
            aic=90.0, bic=95.0, data_points_used=24,
            regime_start_idx=0,
        )
        best = compare_fits([r1, r2])
        assert best.bic == 95.0

    def test_filters_unacceptable(self):
        good = _make_result(r_squared=0.90)
        bad = _make_result(r_squared=0.40)
        # Give bad a lower BIC
        bad_model = HyperbolicModel(qi=100.0, di=0.01, b=0.5, dmin=0.005)
        bad = ForecastResult(
            model=bad_model, r_squared=0.40, rmse=50.0,
            aic=50.0, bic=55.0, data_points_used=24,
            regime_start_idx=0,
        )
        best = compare_fits([good, bad])
        # Should pick good despite bad having lower BIC
        assert best.r_squared == 0.90

    def test_no_acceptable_returns_best_available(self):
        r1 = _make_result(r_squared=0.40)
        r2 = _make_result(r_squared=0.50)
        best = compare_fits([r1, r2])
        assert best.r_squared == 0.50

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No fit results"):
            compare_fits([])

    def test_custom_acceptable_threshold(self):
        r1 = _make_result(r_squared=0.80)
        r2 = _make_result(r_squared=0.90)

        # With threshold=0.85, only r2 is acceptable
        best = compare_fits([r1, r2], acceptable_r_squared=0.85)
        assert best.r_squared == 0.90

    def test_single_result(self):
        r = _make_result(r_squared=0.95)
        best = compare_fits([r])
        assert best is r
