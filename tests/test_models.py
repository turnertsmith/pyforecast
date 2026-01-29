"""Tests for decline curve models."""

import numpy as np
import pytest

from pyforecast.core.models import HyperbolicModel, ForecastResult


class TestHyperbolicModel:
    """Tests for HyperbolicModel class."""

    def test_exponential_decline(self):
        """Test exponential decline (b ~ 0)."""
        model = HyperbolicModel(qi=1000, di=0.1, b=0.01, dmin=0.005)

        # At t=0, rate should equal qi
        assert model.rate(0)[0] == pytest.approx(1000, rel=0.01)

        # Exponential decay: q(t) = qi * exp(-di * t)
        expected_10 = 1000 * np.exp(-0.1 * 10)
        assert model.rate(10)[0] == pytest.approx(expected_10, rel=0.01)

    def test_harmonic_decline(self):
        """Test harmonic decline (b = 1)."""
        model = HyperbolicModel(qi=1000, di=0.1, b=1.0, dmin=0.005)

        # At t=0
        assert model.rate(0)[0] == pytest.approx(1000, rel=0.01)

        # Harmonic: q(t) = qi / (1 + di*t)
        expected_10 = 1000 / (1 + 0.1 * 10)
        assert model.rate(10)[0] == pytest.approx(expected_10, rel=0.01)

    def test_hyperbolic_decline(self):
        """Test general hyperbolic decline (0 < b < 1)."""
        model = HyperbolicModel(qi=1000, di=0.1, b=0.5, dmin=0.005)

        # At t=0
        assert model.rate(0)[0] == pytest.approx(1000, rel=0.01)

        # Hyperbolic: q(t) = qi / (1 + b*di*t)^(1/b)
        expected_10 = 1000 / np.power(1 + 0.5 * 0.1 * 10, 1 / 0.5)
        assert model.rate(10)[0] == pytest.approx(expected_10, rel=0.01)

    def test_terminal_decline_switch(self):
        """Test switch to terminal exponential decline at Dmin."""
        model = HyperbolicModel(qi=1000, di=0.1, b=0.5, dmin=0.02)

        # Switch time should be calculated
        assert model.t_switch > 0
        assert model.t_switch < float('inf')

        # After switch, decline rate should be Dmin
        t_after_switch = model.t_switch + 10
        d = model.instantaneous_decline(t_after_switch)
        assert d[0] == pytest.approx(0.02, rel=0.01)

    def test_cumulative_production(self):
        """Test cumulative production calculation."""
        model = HyperbolicModel(qi=1000, di=0.1, b=0.01, dmin=0.005)

        # Cumulative at t=0 should be 0
        assert model.cumulative(0)[0] == pytest.approx(0, abs=0.1)

        # Cumulative should be positive and increasing
        cum_10 = model.cumulative(10)[0]
        cum_20 = model.cumulative(20)[0]
        assert cum_10 > 0
        assert cum_20 > cum_10

    def test_decline_type_classification(self):
        """Test decline type string classification."""
        # Exponential
        model_exp = HyperbolicModel(qi=1000, di=0.1, b=0.05, dmin=0.005)
        assert model_exp.decline_type == "EXP"

        # Hyperbolic
        model_hyp = HyperbolicModel(qi=1000, di=0.1, b=0.5, dmin=0.005)
        assert model_hyp.decline_type == "HYP"

        # Harmonic
        model_hrm = HyperbolicModel(qi=1000, di=0.1, b=1.0, dmin=0.005)
        assert model_hrm.decline_type == "HRM"

    def test_forecast_generation(self):
        """Test forecast time series generation."""
        model = HyperbolicModel(qi=1000, di=0.1, b=0.5, dmin=0.005)

        t, q = model.forecast(months=60, start_month=0)

        assert len(t) == 60
        assert len(q) == 60
        assert t[0] == 0
        assert t[-1] == 59
        assert q[0] == pytest.approx(1000, rel=0.01)
        assert q[-1] < q[0]  # Rate should decline

    def test_array_input(self):
        """Test model handles array input correctly."""
        model = HyperbolicModel(qi=1000, di=0.1, b=0.5, dmin=0.005)

        t = np.array([0, 5, 10, 15, 20])
        q = model.rate(t)

        assert len(q) == 5
        assert all(q[i] > q[i + 1] for i in range(len(q) - 1))


class TestForecastResult:
    """Tests for ForecastResult class."""

    def test_is_acceptable(self):
        """Test R² threshold for acceptability."""
        model = HyperbolicModel(qi=1000, di=0.1, b=0.5, dmin=0.005)

        # Good fit
        result_good = ForecastResult(
            model=model,
            r_squared=0.85,
            rmse=50,
            aic=100,
            bic=105,
            regime_start_idx=0,
            data_points_used=24
        )
        assert result_good.is_acceptable is True

        # Poor fit
        result_poor = ForecastResult(
            model=model,
            r_squared=0.50,
            rmse=150,
            aic=200,
            bic=210,
            regime_start_idx=0,
            data_points_used=24
        )
        assert result_poor.is_acceptable is False

    def test_summary(self):
        """Test summary dictionary generation."""
        model = HyperbolicModel(qi=1000, di=0.1, b=0.5, dmin=0.005)
        result = ForecastResult(
            model=model,
            r_squared=0.85,
            rmse=50,
            aic=100,
            bic=105,
            regime_start_idx=5,
            data_points_used=24
        )

        summary = result.summary()

        assert "qi" in summary
        assert "di" in summary
        assert "b" in summary
        assert "r_squared" in summary
        assert summary["qi"] == 1000
        assert summary["b"] == 0.5
        assert summary["regime_start_idx"] == 5
