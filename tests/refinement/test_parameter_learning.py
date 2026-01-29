"""Tests for parameter learning."""

import tempfile
from pathlib import Path

import pytest

from pyforecast.refinement.parameter_learning import (
    ParameterLearner,
    LearningConfig,
    apply_suggestion,
)
from pyforecast.refinement.storage import FitLogStorage
from pyforecast.refinement.schemas import FitLogRecord, ParameterSuggestion
from pyforecast.core.fitting import FittingConfig


class TestLearningConfig:
    """Tests for LearningConfig dataclass."""

    def test_default_values(self):
        config = LearningConfig()
        assert config.min_samples == 10
        assert config.min_samples_high_confidence == 100
        assert config.target_mape == 20.0


class TestParameterSuggestion:
    """Tests for ParameterSuggestion dataclass."""

    def test_confidence_low(self):
        suggestion = ParameterSuggestion(
            grouping="global",
            sample_count=5,
            product="oil",
            suggested_recency_half_life=12.0,
            suggested_regime_threshold=1.0,
            suggested_regime_window=6,
            suggested_regime_sustained_months=2,
            avg_r_squared=0.9,
        )
        assert suggestion.confidence == "low"

    def test_confidence_medium(self):
        suggestion = ParameterSuggestion(
            grouping="global",
            sample_count=50,
            product="oil",
            suggested_recency_half_life=12.0,
            suggested_regime_threshold=1.0,
            suggested_regime_window=6,
            suggested_regime_sustained_months=2,
            avg_r_squared=0.9,
        )
        assert suggestion.confidence == "medium"

    def test_confidence_high(self):
        suggestion = ParameterSuggestion(
            grouping="global",
            sample_count=150,
            product="oil",
            suggested_recency_half_life=12.0,
            suggested_regime_threshold=1.0,
            suggested_regime_window=6,
            suggested_regime_sustained_months=2,
            avg_r_squared=0.9,
        )
        assert suggestion.confidence == "high"

    def test_summary(self):
        suggestion = ParameterSuggestion(
            grouping="Permian/Wolfcamp",
            sample_count=50,
            product="oil",
            suggested_recency_half_life=10.0,
            suggested_regime_threshold=0.8,
            suggested_regime_window=6,
            suggested_regime_sustained_months=2,
            avg_r_squared=0.9,
            avg_hindcast_mape=15.0,
        )
        summary = suggestion.summary()

        assert summary["grouping"] == "Permian/Wolfcamp"
        assert summary["suggested_recency_half_life"] == 10.0
        assert summary["confidence"] == "medium"


class TestParameterLearner:
    """Tests for ParameterLearner class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database with sample data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_fits.db"
            storage = FitLogStorage(db_path)

            # Insert sample fit logs with hindcast data
            records = []
            for i in range(20):
                records.append(FitLogRecord(
                    well_id=f"well_{i}",
                    product="oil",
                    basin="Permian",
                    formation="Wolfcamp",
                    recency_half_life=10.0 + (i % 5),  # Vary half-life
                    regime_threshold=0.8 + (i % 3) * 0.2,  # Vary threshold
                    r_squared=0.85 + (i % 3) * 0.05,
                    hindcast_mape=15.0 + (i % 5) * 2,  # Lower MAPE for lower half-life
                    hindcast_correlation=0.9,
                    hindcast_bias=0.05,
                ))
            storage.insert_batch(records)

            yield db_path

    def test_init(self, temp_db):
        learner = ParameterLearner(temp_db)
        assert learner.storage.db_path == temp_db

    def test_suggest_with_data(self, temp_db):
        learner = ParameterLearner(temp_db)

        suggestion = learner.suggest(product="oil", basin="Permian")

        assert suggestion is not None
        assert suggestion.sample_count == 20
        assert suggestion.product == "oil"
        assert suggestion.suggested_recency_half_life > 0
        assert suggestion.suggested_regime_threshold > 0

    def test_suggest_insufficient_data(self):
        """Test that suggest returns None when insufficient data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "empty.db"
            storage = FitLogStorage(db_path)

            # Insert only 5 records (below min_samples=10)
            records = [
                FitLogRecord(
                    well_id=f"w{i}",
                    product="oil",
                    hindcast_mape=20.0,
                )
                for i in range(5)
            ]
            storage.insert_batch(records)

            learner = ParameterLearner(db_path)
            suggestion = learner.suggest(product="oil")

            assert suggestion is None

    def test_suggest_fallback_to_global(self, temp_db):
        """Test that suggestion falls back to broader grouping."""
        learner = ParameterLearner(temp_db)

        # Request for formation that doesn't exist
        suggestion = learner.suggest(
            product="oil",
            basin="Eagle Ford",  # No data for this basin
            formation="Austin Chalk",
        )

        # Should fall back to global
        assert suggestion is not None
        assert suggestion.grouping == "global"

    def test_update_suggestions(self, temp_db):
        learner = ParameterLearner(temp_db)

        count = learner.update_suggestions(product="oil")

        # Should create suggestions for global, Permian, and Permian/Wolfcamp
        assert count >= 1

    def test_get_all_suggestions(self, temp_db):
        learner = ParameterLearner(temp_db)

        # First update to create suggestions
        learner.update_suggestions(product="oil")

        suggestions = learner.get_all_suggestions(product="oil")
        assert len(suggestions) >= 1
        assert all(s.product == "oil" for s in suggestions)

    def test_export_suggestions(self, temp_db):
        learner = ParameterLearner(temp_db)
        learner.update_suggestions(product="oil")

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            count = learner.export_suggestions(csv_path)
            assert count >= 1
            assert csv_path.exists()

            # Check CSV content
            with open(csv_path) as f:
                lines = f.readlines()
            assert len(lines) > 1  # Header + at least one row
        finally:
            csv_path.unlink(missing_ok=True)


class TestApplySuggestion:
    """Tests for apply_suggestion function."""

    def test_apply_suggestion(self):
        suggestion = ParameterSuggestion(
            grouping="global",
            sample_count=100,
            product="oil",
            suggested_recency_half_life=10.0,
            suggested_regime_threshold=0.8,
            suggested_regime_window=8,
            suggested_regime_sustained_months=3,
            avg_r_squared=0.9,
        )

        base_config = FittingConfig(
            b_min=0.01,
            b_max=1.5,
            dmin_annual=0.06,
            recency_half_life=12.0,  # Will be overridden
            regime_threshold=1.0,  # Will be overridden
        )

        new_config = apply_suggestion(suggestion, base_config)

        # Suggestion values should be applied
        assert new_config.recency_half_life == 10.0
        assert new_config.regime_threshold == 0.8
        assert new_config.regime_window == 8
        assert new_config.regime_sustained_months == 3

        # Non-learned values should be preserved
        assert new_config.b_min == 0.01
        assert new_config.b_max == 1.5
        assert new_config.dmin_annual == 0.06

    def test_apply_suggestion_preserves_original(self):
        """Test that applying suggestion doesn't modify original config."""
        suggestion = ParameterSuggestion(
            grouping="global",
            sample_count=100,
            product="oil",
            suggested_recency_half_life=10.0,
            suggested_regime_threshold=0.8,
            suggested_regime_window=8,
            suggested_regime_sustained_months=3,
            avg_r_squared=0.9,
        )

        original_config = FittingConfig(recency_half_life=12.0)
        new_config = apply_suggestion(suggestion, original_config)

        # Original should be unchanged
        assert original_config.recency_half_life == 12.0
        assert new_config.recency_half_life == 10.0
