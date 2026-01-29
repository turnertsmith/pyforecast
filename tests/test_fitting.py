"""Tests for decline curve fitting."""

import numpy as np
import pytest

from pyforecast.core.models import HyperbolicModel
from pyforecast.core.fitting import DeclineFitter, FittingConfig


class TestFittingConfig:
    """Tests for FittingConfig class."""

    def test_default_values(self):
        """Test default configuration values."""
        config = FittingConfig()

        assert config.b_min == 0.01
        assert config.b_max == 1.5
        assert config.dmin_annual == 0.06
        assert config.regime_threshold == 1.0
        assert config.recency_half_life == 12.0
        assert config.min_points == 6

    def test_dmin_monthly_conversion(self):
        """Test annual to monthly Dmin conversion."""
        config = FittingConfig(dmin_annual=0.12)
        assert config.dmin_monthly == pytest.approx(0.01, rel=0.01)


class TestDeclineFitter:
    """Tests for DeclineFitter class."""

    def test_regime_change_detection(self):
        """Test detection of production regime changes."""
        fitter = DeclineFitter(FittingConfig(regime_threshold=1.0))

        # Create data with regime change (100% increase)
        rates = np.array([100, 90, 80, 70, 60, 50, 150, 140, 130, 120])
        #                                       ^-- regime change at idx 6

        regime_start = fitter.detect_regime_change(rates)
        assert regime_start == 6

    def test_no_regime_change(self):
        """Test when no regime change is present."""
        fitter = DeclineFitter(FittingConfig(regime_threshold=1.0))

        # Monotonically declining data
        rates = np.array([100, 95, 90, 85, 80, 75, 70, 65, 60, 55])

        regime_start = fitter.detect_regime_change(rates)
        assert regime_start == 0

    def test_exponential_weighting(self):
        """Test exponential decay weights computation."""
        fitter = DeclineFitter(FittingConfig(recency_half_life=6.0))

        weights = fitter.compute_weights(12)

        # Most recent point should have highest weight
        assert weights[-1] > weights[0]

        # Weights should sum to n_points (normalized)
        assert np.sum(weights) == pytest.approx(12, rel=0.01)

        # Half-life check: weight at t-6 should be ~half of weight at t
        ratio = weights[-1] / weights[-7]  # 6 months apart
        assert ratio == pytest.approx(2.0, rel=0.1)

    def test_initial_guess(self):
        """Test initial parameter estimation."""
        fitter = DeclineFitter()

        # Generate exponential decline data
        t = np.arange(24, dtype=float)
        qi_true, di_true = 1000, 0.05
        q = qi_true * np.exp(-di_true * t)

        qi_guess, di_guess, b_guess = fitter.initial_guess(t, q)

        # Initial guess should be in reasonable range
        assert qi_guess > 0
        assert 0.001 < di_guess < 1.0
        assert 0.01 <= b_guess <= 1.5

    def test_fit_exponential_recovery(self):
        """Test recovery of known exponential parameters."""
        fitter = DeclineFitter(FittingConfig(b_min=0.01, b_max=0.05))

        # Generate exponential decline data
        t = np.arange(36, dtype=float)
        qi_true, di_true = 1000, 0.08
        q = qi_true * np.exp(-di_true * t) + np.random.normal(0, 10, len(t))
        q = np.maximum(q, 1)  # Ensure positive

        result = fitter.fit(t, q)

        # Should recover parameters reasonably well
        assert result.model.qi == pytest.approx(qi_true, rel=0.15)
        assert result.model.di == pytest.approx(di_true, rel=0.2)
        assert result.model.b < 0.1  # Should be near-exponential
        assert result.r_squared > 0.9

    def test_fit_hyperbolic_recovery(self):
        """Test recovery of known hyperbolic parameters."""
        fitter = DeclineFitter(FittingConfig(b_min=0.01, b_max=1.5))

        # Generate hyperbolic decline data
        t = np.arange(36, dtype=float)
        qi_true, di_true, b_true = 1000, 0.1, 0.5
        q = qi_true / np.power(1 + b_true * di_true * t, 1 / b_true)
        q = q + np.random.normal(0, 5, len(t))
        q = np.maximum(q, 1)

        result = fitter.fit(t, q)

        # Should recover parameters reasonably well
        assert result.model.qi == pytest.approx(qi_true, rel=0.15)
        assert result.model.b == pytest.approx(b_true, rel=0.3)
        assert result.r_squared > 0.9

    def test_fit_with_regime_change(self):
        """Test fitting with regime change detection."""
        fitter = DeclineFitter(FittingConfig(regime_threshold=1.0))

        # Create data with regime change
        t1 = np.arange(12, dtype=float)
        q1 = 1000 * np.exp(-0.1 * t1)

        t2 = np.arange(24, dtype=float) + 12
        q2 = 800 * np.exp(-0.05 * (t2 - 12))  # New decline after refrac

        t = np.concatenate([t1, t2])
        q = np.concatenate([q1, q2])

        result = fitter.fit(t, q, apply_regime_detection=True)

        # Should detect regime change and fit to recent data
        assert result.regime_start_idx > 0
        assert result.data_points_used < len(t)

    def test_fit_insufficient_data(self):
        """Test error handling for insufficient data."""
        fitter = DeclineFitter(FittingConfig(min_points=6))

        t = np.array([0, 1, 2], dtype=float)
        q = np.array([100, 95, 90], dtype=float)

        with pytest.raises(ValueError, match="Insufficient data"):
            fitter.fit(t, q)

    def test_fit_weights_favor_recent(self):
        """Test that weighting favors fit to recent data."""
        config = FittingConfig(recency_half_life=3.0)  # Aggressive weighting
        fitter = DeclineFitter(config)

        # Create data where early and late decline rates differ
        t = np.arange(24, dtype=float)
        # Early: steep decline, Late: shallow decline
        q = np.where(t < 12, 1000 * np.exp(-0.15 * t), 400 * np.exp(-0.03 * (t - 12)))

        result_weighted = fitter.fit(t, q, apply_weights=True)

        # With aggressive weighting, should fit closer to late decline rate
        fitter_no_weight = DeclineFitter(FittingConfig(recency_half_life=1000))
        result_unweighted = fitter_no_weight.fit(t, q, apply_weights=True)

        # Weighted fit should have lower decline rate (fitting to recent shallow decline)
        assert result_weighted.model.di < result_unweighted.model.di


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_production(self):
        """Test handling of zero production values."""
        fitter = DeclineFitter()

        t = np.arange(24, dtype=float)
        q = np.concatenate([
            1000 * np.exp(-0.1 * np.arange(12)),
            np.zeros(12)  # Shut-in period
        ])

        # Should handle zeros without crashing
        result = fitter.fit(t, q)
        assert result.model.qi > 0

    def test_noisy_data(self):
        """Test fitting with very noisy data."""
        fitter = DeclineFitter()

        np.random.seed(42)  # Reproducible noise
        t = np.arange(48, dtype=float)
        qi_true = 1000
        q = qi_true * np.exp(-0.05 * t)
        q = q + np.random.normal(0, 50, len(t))  # ~5% noise at start
        q = np.maximum(q, 1)

        # Disable regime detection since noise might trigger false positives
        result = fitter.fit(t, q, apply_regime_detection=False)

        # Should still get reasonable fit despite noise
        assert result.model.qi == pytest.approx(qi_true, rel=0.3)
        assert result.r_squared > 0.5
