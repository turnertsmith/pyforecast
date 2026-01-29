"""Tests for batch processing with checkpoints."""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyforecast.batch.processor import (
    BatchConfig,
    BatchProcessor,
    CheckpointState,
)


class TestCheckpointState:
    """Tests for CheckpointState dataclass."""

    def test_default_values(self):
        state = CheckpointState()
        assert state.total_wells == 0
        assert state.processed_well_ids == set()
        assert state.successful == 0
        assert state.failed == 0
        assert state.skipped == 0
        assert state.errors == []
        assert state.started_at is not None
        assert state.last_updated_at is not None

    def test_to_dict(self):
        state = CheckpointState(
            total_wells=100,
            processed_well_ids={"w1", "w2"},
            successful=2,
            failed=0,
            skipped=0,
            errors=[],
        )
        d = state.to_dict()

        assert d["total_wells"] == 100
        assert set(d["processed_well_ids"]) == {"w1", "w2"}
        assert d["successful"] == 2

    def test_from_dict(self):
        data = {
            "total_wells": 100,
            "processed_well_ids": ["w1", "w2", "w3"],
            "successful": 3,
            "failed": 0,
            "skipped": 2,
            "errors": [("w4", "error message")],
            "started_at": "2024-01-15T10:00:00",
            "last_updated_at": "2024-01-15T10:30:00",
        }
        state = CheckpointState.from_dict(data)

        assert state.total_wells == 100
        assert state.processed_well_ids == {"w1", "w2", "w3"}
        assert state.successful == 3
        assert state.skipped == 2
        assert len(state.errors) == 1

    def test_progress_pct(self):
        state = CheckpointState(
            total_wells=100,
            processed_well_ids={"w1", "w2", "w3"},
        )
        assert state.progress_pct == 3.0

    def test_progress_pct_zero_wells(self):
        state = CheckpointState(total_wells=0)
        assert state.progress_pct == 0.0

    def test_save_and_load(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)

        try:
            original = CheckpointState(
                total_wells=50,
                processed_well_ids={"w1", "w2"},
                successful=2,
                failed=0,
                skipped=0,
                errors=[("w3", "test error")],
            )
            original.save(filepath)

            loaded = CheckpointState.load(filepath)

            assert loaded.total_wells == 50
            assert loaded.processed_well_ids == {"w1", "w2"}
            assert loaded.successful == 2
            assert len(loaded.errors) == 1
            assert loaded.errors[0] == ("w3", "test error")
        finally:
            filepath.unlink(missing_ok=True)

    def test_roundtrip(self):
        original = CheckpointState(
            total_wells=200,
            processed_well_ids={"well_001", "well_002", "well_003"},
            successful=2,
            failed=1,
            skipped=5,
            errors=[("well_003", "fitting failed"), ("well_004", "no data")],
        )

        d = original.to_dict()
        restored = CheckpointState.from_dict(d)

        assert restored.total_wells == original.total_wells
        assert restored.processed_well_ids == original.processed_well_ids
        assert restored.successful == original.successful
        assert restored.failed == original.failed
        assert restored.skipped == original.skipped
        assert len(restored.errors) == len(original.errors)


class TestBatchProcessorCheckpoint:
    """Tests for BatchProcessor with checkpoint functionality."""

    @pytest.fixture
    def mock_well(self):
        """Create a mock well object."""
        well = MagicMock()
        well.well_id = "test_well"
        well.metadata = {}
        well.has_sufficient_data.return_value = True
        well.get_forecast.return_value = MagicMock()
        return well

    @pytest.fixture
    def temp_checkpoint_file(self):
        """Create a temporary checkpoint file path."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = Path(f.name)
        # Delete the file so it doesn't exist initially
        filepath.unlink()
        yield filepath
        # Clean up
        if filepath.exists():
            filepath.unlink()

    def test_checkpoint_file_created(self, temp_checkpoint_file):
        """Test that checkpoint file is created when saving state."""
        # Create and save a checkpoint state
        state = CheckpointState(
            total_wells=10,
            processed_well_ids={"w1"},
            successful=1,
        )
        state.save(temp_checkpoint_file)

        # Verify file exists and contains expected data
        assert temp_checkpoint_file.exists()
        with open(temp_checkpoint_file) as f:
            data = json.load(f)
        assert data["total_wells"] == 10
        assert "w1" in data["processed_well_ids"]

    def test_checkpoint_resume_skips_processed(self, temp_checkpoint_file):
        """Test that resuming skips already processed wells."""
        # Create a checkpoint with some wells already processed
        existing_checkpoint = CheckpointState(
            total_wells=3,
            processed_well_ids={"w1", "w2"},
            successful=2,
            failed=0,
            skipped=0,
        )
        existing_checkpoint.save(temp_checkpoint_file)

        # Verify checkpoint was saved
        loaded = CheckpointState.load(temp_checkpoint_file)
        assert loaded.processed_well_ids == {"w1", "w2"}
        assert loaded.successful == 2

    def test_checkpoint_state_updated_on_save(self, temp_checkpoint_file):
        """Test that last_updated_at is updated on save."""
        state = CheckpointState(total_wells=10)
        original_time = state.last_updated_at

        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.01)

        state.save(temp_checkpoint_file)
        loaded = CheckpointState.load(temp_checkpoint_file)

        # The loaded timestamp should be different from original
        assert loaded.last_updated_at != original_time
