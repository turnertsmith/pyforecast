"""Tests for fit log storage."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from pyforecast.refinement.storage import FitLogStorage, get_default_storage_path
from pyforecast.refinement.schemas import FitLogRecord, ParameterSuggestion


class TestGetDefaultStoragePath:
    """Tests for get_default_storage_path function."""

    def test_returns_path_in_home(self):
        path = get_default_storage_path()
        assert isinstance(path, Path)
        assert ".pyforecast" in str(path)
        assert "fit_logs.db" in str(path)


class TestFitLogRecord:
    """Tests for FitLogRecord dataclass."""

    def test_default_values(self):
        record = FitLogRecord()
        assert record.fit_id  # UUID should be generated
        assert isinstance(record.timestamp, datetime)
        assert record.well_id == ""
        assert record.product == ""

    def test_to_dict(self):
        record = FitLogRecord(
            well_id="test_well",
            product="oil",
            qi=100.0,
            di=0.05,
            b=0.5,
            r_squared=0.95,
        )
        d = record.to_dict()

        assert d["well_id"] == "test_well"
        assert d["product"] == "oil"
        assert d["qi"] == 100.0
        assert d["r_squared"] == 0.95

    def test_from_dict(self):
        data = {
            "fit_id": "test-id",
            "timestamp": "2024-01-15T10:30:00",
            "well_id": "test_well",
            "product": "oil",
            "qi": 100.0,
            "r_squared": 0.95,
        }
        record = FitLogRecord.from_dict(data)

        assert record.fit_id == "test-id"
        assert record.well_id == "test_well"
        assert record.qi == 100.0

    def test_roundtrip(self):
        original = FitLogRecord(
            well_id="test_well",
            product="oil",
            basin="Permian",
            formation="Wolfcamp",
            qi=100.0,
            di=0.05,
            b=0.5,
            r_squared=0.95,
            hindcast_mape=15.0,
        )

        d = original.to_dict()
        restored = FitLogRecord.from_dict(d)

        assert restored.well_id == original.well_id
        assert restored.basin == original.basin
        assert restored.qi == original.qi
        assert restored.hindcast_mape == original.hindcast_mape


class TestFitLogStorage:
    """Tests for FitLogStorage class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_fits.db"
            yield db_path

    def test_init_creates_db(self, temp_db):
        storage = FitLogStorage(temp_db)
        assert temp_db.exists()

    def test_insert_and_query(self, temp_db):
        storage = FitLogStorage(temp_db)

        record = FitLogRecord(
            well_id="well_1",
            product="oil",
            qi=100.0,
            r_squared=0.95,
        )
        storage.insert(record)

        results = storage.query(well_id="well_1")
        assert len(results) == 1
        assert results[0].well_id == "well_1"
        assert results[0].qi == 100.0

    def test_insert_batch(self, temp_db):
        storage = FitLogStorage(temp_db)

        records = [
            FitLogRecord(well_id=f"well_{i}", product="oil", qi=100.0 + i)
            for i in range(5)
        ]
        count = storage.insert_batch(records)

        assert count == 5
        assert storage.count() == 5

    def test_query_with_filters(self, temp_db):
        storage = FitLogStorage(temp_db)

        # Insert records with different products and basins
        records = [
            FitLogRecord(well_id="w1", product="oil", basin="Permian", r_squared=0.9),
            FitLogRecord(well_id="w2", product="gas", basin="Permian", r_squared=0.85),
            FitLogRecord(well_id="w3", product="oil", basin="Eagle Ford", r_squared=0.95),
        ]
        storage.insert_batch(records)

        # Query by product
        oil_results = storage.query(product="oil")
        assert len(oil_results) == 2

        # Query by basin
        permian_results = storage.query(basin="Permian")
        assert len(permian_results) == 2

        # Query by both
        permian_oil = storage.query(product="oil", basin="Permian")
        assert len(permian_oil) == 1

        # Query with min_r_squared
        high_r2 = storage.query(min_r_squared=0.9)
        assert len(high_r2) == 2

    def test_count(self, temp_db):
        storage = FitLogStorage(temp_db)

        records = [
            FitLogRecord(well_id="w1", product="oil", basin="Permian"),
            FitLogRecord(well_id="w2", product="gas", basin="Permian"),
            FitLogRecord(well_id="w3", product="oil", basin="Eagle Ford"),
        ]
        storage.insert_batch(records)

        assert storage.count() == 3
        assert storage.count(product="oil") == 2
        assert storage.count(basin="Permian") == 2

    def test_get_statistics(self, temp_db):
        storage = FitLogStorage(temp_db)

        records = [
            FitLogRecord(well_id="w1", product="oil", r_squared=0.9, b=0.5),
            FitLogRecord(well_id="w2", product="oil", r_squared=0.8, b=0.6),
            FitLogRecord(well_id="w3", product="oil", r_squared=0.95, b=0.4),
        ]
        storage.insert_batch(records)

        stats = storage.get_statistics(product="oil")

        assert stats["count"] == 3
        assert stats["avg_r_squared"] == pytest.approx(0.883, rel=0.01)
        assert stats["avg_b"] == pytest.approx(0.5, rel=0.01)

    def test_iterate_all(self, temp_db):
        storage = FitLogStorage(temp_db)

        records = [FitLogRecord(well_id=f"w{i}", product="oil") for i in range(10)]
        storage.insert_batch(records)

        # Iterate with small batch size
        iterated = list(storage.iterate_all(batch_size=3))
        assert len(iterated) == 10

    def test_export_and_import_csv(self, temp_db):
        storage = FitLogStorage(temp_db)

        records = [
            FitLogRecord(well_id="w1", product="oil", qi=100.0, r_squared=0.9),
            FitLogRecord(well_id="w2", product="gas", qi=200.0, r_squared=0.85),
        ]
        storage.insert_batch(records)

        # Export to CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            csv_path = Path(f.name)

        try:
            count = storage.export_to_csv(csv_path)
            assert count == 2

            # Create new storage and import
            with tempfile.TemporaryDirectory() as tmpdir2:
                new_db = Path(tmpdir2) / "new.db"
                new_storage = FitLogStorage(new_db)
                imported = new_storage.import_from_csv(csv_path)
                assert imported == 2
                assert new_storage.count() == 2
        finally:
            csv_path.unlink(missing_ok=True)

    def test_save_and_get_suggestion(self, temp_db):
        storage = FitLogStorage(temp_db)

        suggestion = ParameterSuggestion(
            grouping="Permian",
            sample_count=50,
            product="oil",
            suggested_recency_half_life=10.0,
            suggested_regime_threshold=0.8,
            suggested_regime_window=6,
            suggested_regime_sustained_months=2,
            avg_r_squared=0.9,
            avg_hindcast_mape=15.0,
        )
        storage.save_suggestion(suggestion)

        # Retrieve suggestion
        retrieved = storage.get_suggestion("oil", basin="Permian")

        assert retrieved is not None
        assert retrieved.grouping == "Permian"
        assert retrieved.suggested_recency_half_life == 10.0
        assert retrieved.avg_hindcast_mape == 15.0

    def test_get_suggestion_fallback(self, temp_db):
        """Test that suggestion lookup falls back to less specific groupings."""
        storage = FitLogStorage(temp_db)

        # Only save global suggestion
        global_suggestion = ParameterSuggestion(
            grouping="global",
            sample_count=100,
            product="oil",
            suggested_recency_half_life=12.0,
            suggested_regime_threshold=1.0,
            suggested_regime_window=6,
            suggested_regime_sustained_months=2,
            avg_r_squared=0.85,
        )
        storage.save_suggestion(global_suggestion)

        # Request specific basin/formation - should fall back to global
        retrieved = storage.get_suggestion("oil", basin="Permian", formation="Wolfcamp")

        assert retrieved is not None
        assert retrieved.grouping == "global"

    def test_clear(self, temp_db):
        storage = FitLogStorage(temp_db)

        records = [FitLogRecord(well_id=f"w{i}", product="oil") for i in range(5)]
        storage.insert_batch(records)

        assert storage.count() == 5

        storage.clear()
        assert storage.count() == 0
