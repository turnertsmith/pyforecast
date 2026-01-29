"""Tests for residual analysis."""

import numpy as np
import pytest

from pyforecast.refinement.residual_analysis import (
    ResidualAnalyzer,
    ResidualAnalysisConfig,
    compute_residuals,
    summarize_residual_results,
)
from pyforecast.refinement.schemas import ResidualDiagnostics


class TestResidualDiagnostics:
    """Tests for ResidualDiagnostics dataclass."""

    def test_compute_basic(self):
        actual = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        predicted = np.array([98, 88, 82, 72, 58, 52, 42, 32, 18, 12])

        diag = ResidualDiagnostics.compute(actual, predicted)

        assert len(diag.residuals) == 10
        assert isinstance(diag.mean, float)
        assert isinstance(diag.std, float)
        assert isinstance(diag.durbin_watson, float)
        assert isinstance(diag.early_bias, float)
        assert isinstance(diag.late_bias, float)
        assert isinstance(diag.has_systematic_pattern, bool)

    def test_compute_perfect_fit(self):
        actual = np.array([100, 90, 80, 70, 60])
        predicted = actual.copy()

        diag = ResidualDiagnostics.compute(actual, predicted)

        assert np.allclose(diag.mean, 0)
        assert np.allclose(diag.std, 0)
        assert diag.early_bias == 0
        assert diag.late_bias == 0

    def test_compute_with_bias(self):
        actual = np.array([100, 90, 80, 70, 60, 50, 40, 30])
        # Systematic over-prediction
        predicted = actual + 10

        diag = ResidualDiagnostics.compute(actual, predicted)

        # Residuals are actual - predicted, so negative for over-prediction
        assert diag.mean < 0
        assert diag.early_bias < 0
        assert diag.late_bias < 0

    def test_durbin_watson_no_autocorrelation(self):
        """Random residuals should have DW near 2."""
        np.random.seed(42)
        actual = np.arange(100, dtype=float)
        predicted = actual + np.random.normal(0, 1, 100)

        diag = ResidualDiagnostics.compute(actual, predicted)

        # DW should be near 2 for random residuals
        assert 1.5 < diag.durbin_watson < 2.5

    def test_durbin_watson_positive_autocorrelation(self):
        """Systematically correlated residuals should have low DW."""
        actual = np.arange(20, dtype=float)
        # Residuals that slowly change (positive autocorrelation)
        predicted = actual + np.linspace(-5, 5, 20)

        diag = ResidualDiagnostics.compute(actual, predicted)

        # DW should be low for positive autocorrelation
        assert diag.durbin_watson < 2.0

    def test_has_systematic_pattern_detection(self):
        # Create data with clear bias pattern
        actual = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
        # Early under-prediction, late over-prediction
        predicted = np.array([90, 82, 75, 70, 60, 52, 45, 38, 28, 18])

        diag = ResidualDiagnostics.compute(actual, predicted)

        # Should detect systematic pattern from early/late bias difference
        # Early: actual > predicted (positive residuals)
        # Late: actual < predicted (negative residuals)
        assert abs(diag.early_bias - diag.late_bias) > 0.1

    def test_summary(self):
        actual = np.array([100, 90, 80, 70, 60])
        predicted = np.array([98, 88, 82, 72, 58])

        diag = ResidualDiagnostics.compute(actual, predicted)
        summary = diag.summary()

        assert "mean" in summary
        assert "std" in summary
        assert "durbin_watson" in summary
        assert "has_systematic_pattern" in summary


class TestResidualAnalyzer:
    """Tests for ResidualAnalyzer class."""

    def test_init_default(self):
        analyzer = ResidualAnalyzer()
        assert analyzer.config.dw_low_threshold == 1.5
        assert analyzer.config.dw_high_threshold == 2.5
        assert analyzer.config.bias_threshold == 0.15

    def test_init_custom_config(self):
        config = ResidualAnalysisConfig(
            dw_low_threshold=1.0,
            bias_threshold=0.2,
        )
        analyzer = ResidualAnalyzer(config)
        assert analyzer.config.dw_low_threshold == 1.0
        assert analyzer.config.bias_threshold == 0.2

    def test_analyze(self):
        analyzer = ResidualAnalyzer()
        actual = np.array([100, 90, 80, 70, 60])
        predicted = np.array([98, 88, 82, 72, 58])

        diag = analyzer.analyze(actual, predicted)

        assert isinstance(diag, ResidualDiagnostics)
        assert len(diag.residuals) == 5

    def test_get_validation_issues_no_issues(self):
        """Test that good fit produces no warnings."""
        analyzer = ResidualAnalyzer()

        # Create diagnostics with no systematic patterns
        diag = ResidualDiagnostics(
            residuals=np.array([1, -1, 2, -2, 1]),
            mean=0.2,
            std=1.5,
            autocorr_lag1=0.1,
            durbin_watson=2.0,  # Perfect
            early_bias=0.05,  # Small
            late_bias=-0.05,  # Small
            has_systematic_pattern=False,
        )

        result = analyzer.get_validation_issues(diag, "test_well", "oil")

        # Should have no error/warning issues
        errors_and_warnings = [
            i for i in result.issues
            if i.severity.name in ("ERROR", "WARNING")
        ]
        assert len(errors_and_warnings) == 0

    def test_get_validation_issues_autocorrelation(self):
        """Test that low DW produces warning."""
        analyzer = ResidualAnalyzer()

        diag = ResidualDiagnostics(
            residuals=np.array([1, 2, 3, 4, 5]),
            mean=3.0,
            std=1.5,
            autocorr_lag1=0.8,
            durbin_watson=1.0,  # Low - positive autocorrelation
            early_bias=0.05,
            late_bias=0.05,
            has_systematic_pattern=True,
        )

        result = analyzer.get_validation_issues(diag, "test_well", "oil")

        # Should have RD001 warning
        codes = [i.code for i in result.issues]
        assert "RD001" in codes

    def test_get_validation_issues_bias(self):
        """Test that large bias produces warning."""
        analyzer = ResidualAnalyzer()

        diag = ResidualDiagnostics(
            residuals=np.array([10, 8, 6, 4, 2]),
            mean=6.0,
            std=3.0,
            autocorr_lag1=0.1,
            durbin_watson=2.0,
            early_bias=0.25,  # > 0.15 threshold
            late_bias=0.05,
            has_systematic_pattern=True,
        )

        result = analyzer.get_validation_issues(diag, "test_well", "oil")

        # Should have RD002 warning for early bias
        codes = [i.code for i in result.issues]
        assert "RD002" in codes


class TestSummarizeResidualResults:
    """Tests for summarize_residual_results function."""

    def test_empty_list(self):
        summary = summarize_residual_results([])
        assert summary["count"] == 0

    def test_with_diagnostics(self):
        diag1 = ResidualDiagnostics(
            residuals=np.array([1, 2, 3]),
            mean=2.0, std=1.0,
            autocorr_lag1=0.2,
            durbin_watson=1.8,
            early_bias=0.1,
            late_bias=0.05,
            has_systematic_pattern=False,
        )
        diag2 = ResidualDiagnostics(
            residuals=np.array([2, 3, 4]),
            mean=3.0, std=1.0,
            autocorr_lag1=0.3,
            durbin_watson=2.2,
            early_bias=0.2,
            late_bias=0.1,
            has_systematic_pattern=True,
        )

        summary = summarize_residual_results([diag1, diag2])

        assert summary["count"] == 2
        assert "durbin_watson" in summary
        assert summary["durbin_watson"]["mean"] == pytest.approx(2.0, rel=0.01)
        assert summary["systematic_pattern_pct"] == 50.0
