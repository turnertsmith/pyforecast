"""Tests for hindcast validation."""

import numpy as np
import pytest

from pyforecast.refinement.hindcast import (
    HindcastValidator,
    HindcastConfig,
    run_hindcast_batch,
    summarize_hindcast_results,
)
from pyforecast.refinement.schemas import HindcastResult
from pyforecast.core.fitting import DeclineFitter, FittingConfig


class TestHindcastConfig:
    """Tests for HindcastConfig."""

    def test_default_values(self):
        config = HindcastConfig()
        assert config.holdout_months == 6
        assert config.min_training_months == 12
        assert config.min_holdout_rate == 0.1


class TestHindcastValidator:
    """Tests for HindcastValidator."""

    def test_init_default(self):
        validator = HindcastValidator()
        assert validator.config.holdout_months == 6
        assert validator.config.min_training_months == 12

    def test_init_custom(self):
        validator = HindcastValidator(holdout_months=3, min_training_months=6)
        assert validator.config.holdout_months == 3
        assert validator.config.min_training_months == 6

    def test_can_validate_sufficient_data(self):
        validator = HindcastValidator(holdout_months=6, min_training_months=12)
        assert validator.can_validate(18) is True
        assert validator.can_validate(20) is True

    def test_can_validate_insufficient_data(self):
        validator = HindcastValidator(holdout_months=6, min_training_months=12)
        assert validator.can_validate(17) is False
        assert validator.can_validate(10) is False

    def test_validate_with_data_insufficient(self):
        """Test validation returns None for insufficient data."""
        validator = HindcastValidator(holdout_months=6, min_training_months=12)
        fitter = DeclineFitter()

        # Only 10 months of data
        t = np.arange(10, dtype=float)
        q = 100 * np.exp(-0.05 * t)

        result = validator.validate_with_data(t, q, fitter)
        assert result is None

    def test_validate_with_data_synthetic(self):
        """Test hindcast with synthetic declining data."""
        validator = HindcastValidator(holdout_months=6, min_training_months=12)
        fitter = DeclineFitter(FittingConfig(b_min=0.01, b_max=1.0))

        # 24 months of synthetic exponential decline
        t = np.arange(24, dtype=float)
        # Add some noise to make it realistic
        np.random.seed(42)
        noise = np.random.normal(0, 5, len(t))
        q = 100 * np.exp(-0.05 * t) + noise
        q = np.maximum(q, 1)  # Ensure positive

        result = validator.validate_with_data(t, q, fitter, "test_well", "oil")

        assert result is not None
        assert result.well_id == "test_well"
        assert result.product == "oil"
        assert result.training_months == 18
        assert result.holdout_months == 6
        assert result.training_r_squared > 0
        assert result.mape >= 0
        assert -1 <= result.correlation <= 1
        assert len(result.holdout_actual) == 6
        assert len(result.holdout_predicted) == 6

    def test_calculate_metrics(self):
        """Test metric calculation."""
        validator = HindcastValidator()

        actual = np.array([100, 90, 80, 70, 60, 50])
        predicted = np.array([95, 88, 82, 72, 58, 52])

        metrics = validator._calculate_metrics(actual, predicted)

        assert "mape" in metrics
        assert "correlation" in metrics
        assert "bias" in metrics
        assert metrics["mape"] > 0
        assert 0 < metrics["correlation"] <= 1


class TestHindcastResult:
    """Tests for HindcastResult dataclass."""

    def test_is_good_hindcast_true(self):
        result = HindcastResult(
            well_id="test",
            product="oil",
            training_months=18,
            holdout_months=6,
            training_r_squared=0.95,
            training_qi=100,
            training_di=0.05,
            training_b=0.5,
            mape=15.0,  # < 30%
            correlation=0.9,  # > 0.7
            bias=0.1,  # < 0.3
        )
        assert result.is_good_hindcast is True

    def test_is_good_hindcast_false_high_mape(self):
        result = HindcastResult(
            well_id="test",
            product="oil",
            training_months=18,
            holdout_months=6,
            training_r_squared=0.95,
            training_qi=100,
            training_di=0.05,
            training_b=0.5,
            mape=40.0,  # > 30%
            correlation=0.9,
            bias=0.1,
        )
        assert result.is_good_hindcast is False

    def test_is_good_hindcast_false_low_correlation(self):
        result = HindcastResult(
            well_id="test",
            product="oil",
            training_months=18,
            holdout_months=6,
            training_r_squared=0.95,
            training_qi=100,
            training_di=0.05,
            training_b=0.5,
            mape=15.0,
            correlation=0.5,  # < 0.7
            bias=0.1,
        )
        assert result.is_good_hindcast is False

    def test_summary(self):
        result = HindcastResult(
            well_id="test",
            product="oil",
            training_months=18,
            holdout_months=6,
            training_r_squared=0.95,
            training_qi=100,
            training_di=0.05,
            training_b=0.5,
            mape=15.0,
            correlation=0.9,
            bias=0.1,
        )
        summary = result.summary()

        assert summary["well_id"] == "test"
        assert summary["product"] == "oil"
        assert summary["mape"] == 15.0
        assert summary["is_good_hindcast"] is True


class TestSummarizeHindcastResults:
    """Tests for summarize_hindcast_results function."""

    def test_empty_results(self):
        summary = summarize_hindcast_results([])
        assert summary["count"] == 0
        assert summary["avg_mape"] is None

    def test_with_results(self):
        results = [
            HindcastResult(
                well_id="w1", product="oil", training_months=18, holdout_months=6,
                training_r_squared=0.9, training_qi=100, training_di=0.05, training_b=0.5,
                mape=10.0, correlation=0.95, bias=0.05,
            ),
            HindcastResult(
                well_id="w2", product="oil", training_months=18, holdout_months=6,
                training_r_squared=0.85, training_qi=80, training_di=0.04, training_b=0.6,
                mape=20.0, correlation=0.85, bias=0.1,
            ),
        ]

        summary = summarize_hindcast_results(results)

        assert summary["count"] == 2
        assert summary["avg_mape"] == 15.0
        assert summary["avg_correlation"] == pytest.approx(0.9, rel=0.01)
        assert summary["good_hindcast_count"] == 2
