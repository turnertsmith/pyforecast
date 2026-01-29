"""Tests for fit logger with quality thresholds."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pyforecast.refinement.fit_logger import (
    DataQualityThresholds,
    FitLogger,
    FitLogAnalyzer,
)
from pyforecast.refinement.schemas import FitLogRecord, ResidualDiagnostics


class TestDataQualityThresholds:
    """Tests for DataQualityThresholds dataclass."""

    def test_default_values(self):
        thresholds = DataQualityThresholds()
        assert thresholds.min_data_points == 6
        assert thresholds.max_coefficient_of_variation == 0.0
        assert thresholds.min_r_squared == 0.0

    def test_passes_all_checks(self):
        thresholds = DataQualityThresholds(
            min_data_points=6,
            min_r_squared=0.5,
            max_coefficient_of_variation=2.0,
        )

        passes, reason = thresholds.passes(
            data_points=10,
            r_squared=0.9,
            coefficient_of_variation=1.0,
        )

        assert passes is True
        assert reason is None

    def test_fails_min_data_points(self):
        thresholds = DataQualityThresholds(min_data_points=12)

        passes, reason = thresholds.passes(
            data_points=6,
            r_squared=0.9,
        )

        assert passes is False
        assert "Insufficient data points" in reason
        assert "6 < 12" in reason

    def test_fails_min_r_squared(self):
        thresholds = DataQualityThresholds(min_r_squared=0.7)

        passes, reason = thresholds.passes(
            data_points=10,
            r_squared=0.5,
        )

        assert passes is False
        assert "R-squared too low" in reason
        assert "0.500" in reason

    def test_fails_max_cv(self):
        thresholds = DataQualityThresholds(max_coefficient_of_variation=1.5)

        passes, reason = thresholds.passes(
            data_points=10,
            r_squared=0.9,
            coefficient_of_variation=2.0,
        )

        assert passes is False
        assert "CV too high" in reason

    def test_cv_not_checked_if_none(self):
        """CV check is skipped if coefficient_of_variation is None."""
        thresholds = DataQualityThresholds(max_coefficient_of_variation=1.5)

        passes, reason = thresholds.passes(
            data_points=10,
            r_squared=0.9,
            coefficient_of_variation=None,
        )

        assert passes is True

    def test_cv_not_checked_if_limit_is_zero(self):
        """CV check is skipped if max_coefficient_of_variation is 0."""
        thresholds = DataQualityThresholds(max_coefficient_of_variation=0.0)

        passes, reason = thresholds.passes(
            data_points=10,
            r_squared=0.9,
            coefficient_of_variation=999.0,  # Would fail if checked
        )

        assert passes is True

    def test_r_squared_not_checked_if_limit_is_zero(self):
        """R-squared check is skipped if min_r_squared is 0."""
        thresholds = DataQualityThresholds(min_r_squared=0.0)

        passes, reason = thresholds.passes(
            data_points=10,
            r_squared=0.1,  # Would fail if checked
        )

        assert passes is True


class TestFitLoggerWithQualityThresholds:
    """Tests for FitLogger with quality thresholds."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_fits.db"
            yield db_path

    @pytest.fixture
    def mock_fit_result(self):
        """Create a mock ForecastResult."""
        result = MagicMock()
        result.data_points_used = 24
        result.regime_start_idx = 0
        result.model.qi = 100.0
        result.model.di = 0.05
        result.model.b = 0.5
        result.r_squared = 0.95
        result.rmse = 5.0
        result.aic = 100.0
        result.bic = 105.0
        return result

    @pytest.fixture
    def mock_fitting_config(self):
        """Create a mock FittingConfig."""
        config = MagicMock()
        config.b_min = 0.01
        config.b_max = 1.5
        config.dmin_annual = 0.06
        config.recency_half_life = 12.0
        config.regime_threshold = 1.0
        return config

    @pytest.fixture
    def mock_well(self):
        """Create a mock Well."""
        well = MagicMock()
        well.well_id = "test_well"
        well.metadata = {"basin": "Permian", "formation": "Wolfcamp"}
        well.production.time_months = list(range(24))
        return well

    def test_log_without_thresholds(
        self, temp_db, mock_fit_result, mock_fitting_config, mock_well
    ):
        """Test logging without quality thresholds (all records accepted)."""
        logger = FitLogger(storage_path=temp_db, quality_thresholds=None)

        record = logger.log(
            fit_result=mock_fit_result,
            well=mock_well,
            product="oil",
            fitting_config=mock_fitting_config,
        )

        assert record is not None
        assert record.well_id == "test_well"
        assert logger.pending_count == 1
        assert logger.skipped_count == 0

    def test_log_passes_quality_check(
        self, temp_db, mock_fit_result, mock_fitting_config, mock_well
    ):
        """Test logging when quality thresholds are met."""
        thresholds = DataQualityThresholds(
            min_data_points=6,
            min_r_squared=0.5,
        )
        logger = FitLogger(storage_path=temp_db, quality_thresholds=thresholds)

        record = logger.log(
            fit_result=mock_fit_result,
            well=mock_well,
            product="oil",
            fitting_config=mock_fitting_config,
        )

        assert record is not None
        assert logger.skipped_count == 0

    def test_log_fails_quality_check_data_points(
        self, temp_db, mock_fit_result, mock_fitting_config, mock_well
    ):
        """Test logging skipped when data points too low."""
        mock_fit_result.data_points_used = 5  # Below threshold

        thresholds = DataQualityThresholds(min_data_points=12)
        logger = FitLogger(storage_path=temp_db, quality_thresholds=thresholds)

        record = logger.log(
            fit_result=mock_fit_result,
            well=mock_well,
            product="oil",
            fitting_config=mock_fitting_config,
        )

        assert record is None
        assert logger.pending_count == 0
        assert logger.skipped_count == 1

    def test_log_fails_quality_check_r_squared(
        self, temp_db, mock_fit_result, mock_fitting_config, mock_well
    ):
        """Test logging skipped when R-squared too low."""
        mock_fit_result.r_squared = 0.4  # Below threshold

        thresholds = DataQualityThresholds(min_r_squared=0.7)
        logger = FitLogger(storage_path=temp_db, quality_thresholds=thresholds)

        record = logger.log(
            fit_result=mock_fit_result,
            well=mock_well,
            product="oil",
            fitting_config=mock_fitting_config,
        )

        assert record is None
        assert logger.skipped_count == 1

    def test_log_from_data_with_thresholds(
        self, temp_db, mock_fit_result, mock_fitting_config
    ):
        """Test log_from_data respects quality thresholds."""
        mock_fit_result.data_points_used = 5  # Below threshold

        thresholds = DataQualityThresholds(min_data_points=12)
        logger = FitLogger(storage_path=temp_db, quality_thresholds=thresholds)

        record = logger.log_from_data(
            fit_result=mock_fit_result,
            well_id="test_well",
            product="oil",
            fitting_config=mock_fitting_config,
            data_points_total=24,
        )

        assert record is None
        assert logger.skipped_count == 1

    def test_flush_writes_to_storage(
        self, temp_db, mock_fit_result, mock_fitting_config, mock_well
    ):
        """Test that flush writes pending records to storage."""
        logger = FitLogger(storage_path=temp_db, batch_size=100)

        logger.log(
            fit_result=mock_fit_result,
            well=mock_well,
            product="oil",
            fitting_config=mock_fitting_config,
        )

        assert logger.pending_count == 1

        count = logger.flush()
        assert count == 1
        assert logger.pending_count == 0

        # Verify record is in storage
        assert logger.storage.count() == 1

    def test_context_manager_flushes_on_exit(
        self, temp_db, mock_fit_result, mock_fitting_config, mock_well
    ):
        """Test that context manager flushes on exit."""
        with FitLogger(storage_path=temp_db) as logger:
            logger.log(
                fit_result=mock_fit_result,
                well=mock_well,
                product="oil",
                fitting_config=mock_fitting_config,
            )
            assert logger.pending_count == 1

        # After context exit, should be flushed
        # Verify by creating new logger and checking storage
        new_logger = FitLogger(storage_path=temp_db)
        assert new_logger.storage.count() == 1

    def test_auto_flush_on_batch_size(
        self, temp_db, mock_fit_result, mock_fitting_config, mock_well
    ):
        """Test automatic flush when batch size is reached."""
        logger = FitLogger(storage_path=temp_db, batch_size=3)

        # Log 3 records
        for i in range(3):
            mock_well.well_id = f"well_{i}"
            logger.log(
                fit_result=mock_fit_result,
                well=mock_well,
                product="oil",
                fitting_config=mock_fitting_config,
            )

        # Should have auto-flushed
        assert logger.pending_count == 0
        assert logger.storage.count() == 3

    def test_multiple_skips_counted(
        self, temp_db, mock_fit_result, mock_fitting_config, mock_well
    ):
        """Test that multiple skipped records are counted correctly."""
        mock_fit_result.data_points_used = 5  # Below threshold

        thresholds = DataQualityThresholds(min_data_points=12)
        logger = FitLogger(storage_path=temp_db, quality_thresholds=thresholds)

        for i in range(5):
            mock_well.well_id = f"well_{i}"
            logger.log(
                fit_result=mock_fit_result,
                well=mock_well,
                product="oil",
                fitting_config=mock_fitting_config,
            )

        assert logger.skipped_count == 5
        assert logger.pending_count == 0


class TestFitLogAnalyzer:
    """Tests for FitLogAnalyzer class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_fits.db"
            yield db_path

    def test_get_summary_empty_db(self, temp_db):
        """Test get_summary on empty database."""
        analyzer = FitLogAnalyzer(storage_path=temp_db)
        summary = analyzer.get_summary()

        assert summary["count"] == 0

    def test_get_summary_with_data(self, temp_db):
        """Test get_summary with data."""
        # Insert some test records
        from pyforecast.refinement.storage import FitLogStorage

        storage = FitLogStorage(temp_db)
        records = [
            FitLogRecord(well_id="w1", product="oil", r_squared=0.9, b=0.5),
            FitLogRecord(well_id="w2", product="oil", r_squared=0.85, b=0.6),
        ]
        storage.insert_batch(records)

        analyzer = FitLogAnalyzer(storage_path=temp_db)
        summary = analyzer.get_summary(product="oil")

        assert summary["count"] == 2
        assert summary["avg_r_squared"] == pytest.approx(0.875, rel=0.01)
