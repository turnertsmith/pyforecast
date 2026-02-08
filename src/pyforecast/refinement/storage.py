"""Storage abstraction for fit logs and parameter suggestions.

Provides SQLite storage for persistent accumulation of fit metadata
across all projects, enabling learning from historical fits.
"""

from datetime import datetime
from pathlib import Path
import sqlite3
from typing import Iterator, Literal
import csv
import logging

from .schemas import FitLogRecord, ParameterSuggestion

logger = logging.getLogger(__name__)


def get_default_storage_path() -> Path:
    """Get default storage path in user's home directory."""
    return Path.home() / ".pyforecast" / "fit_logs.db"


class FitLogStorage:
    """SQLite storage for fit logs and parameter suggestions.

    Stores fit metadata in a SQLite database for analysis and learning.
    Default location is ~/.pyforecast/fit_logs.db.

    Attributes:
        db_path: Path to SQLite database file
    """

    # SQL schema for fit_logs table
    SCHEMA_FIT_LOGS = """
    CREATE TABLE IF NOT EXISTS fit_logs (
        fit_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        well_id TEXT NOT NULL,
        product TEXT NOT NULL,
        basin TEXT,
        formation TEXT,

        -- Input characteristics
        data_points_total INTEGER,
        data_points_used INTEGER,
        regime_start_idx INTEGER,

        -- Parameters used
        b_min REAL,
        b_max REAL,
        dmin_annual REAL,
        recency_half_life REAL,
        regime_threshold REAL,

        -- Fit results
        qi REAL,
        di REAL,
        b REAL,
        r_squared REAL,
        rmse REAL,
        aic REAL,
        bic REAL,

        -- Residual diagnostics
        residual_mean REAL,
        residual_std REAL,
        durbin_watson REAL,
        early_bias REAL,
        late_bias REAL,

        -- Hindcast results
        hindcast_mape REAL,
        hindcast_correlation REAL,
        hindcast_bias REAL
    );
    """

    SCHEMA_INDEXES = """
    CREATE INDEX IF NOT EXISTS idx_fit_logs_well_id ON fit_logs(well_id);
    CREATE INDEX IF NOT EXISTS idx_fit_logs_basin ON fit_logs(basin);
    CREATE INDEX IF NOT EXISTS idx_fit_logs_formation ON fit_logs(formation);
    CREATE INDEX IF NOT EXISTS idx_fit_logs_product ON fit_logs(product);
    CREATE INDEX IF NOT EXISTS idx_fit_logs_timestamp ON fit_logs(timestamp);
    """

    SCHEMA_SUGGESTIONS = """
    CREATE TABLE IF NOT EXISTS parameter_suggestions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        basin TEXT,
        formation TEXT,
        product TEXT NOT NULL,
        sample_count INTEGER,
        suggested_recency_half_life REAL,
        suggested_regime_threshold REAL,
        suggested_regime_window INTEGER,
        suggested_regime_sustained_months INTEGER,
        avg_r_squared REAL,
        avg_hindcast_mape REAL,
        updated_at TEXT,
        UNIQUE(basin, formation, product)
    );
    """

    def __init__(self, db_path: Path | str | None = None):
        """Initialize storage.

        Args:
            db_path: Path to SQLite database. If None, uses default location.
        """
        if db_path is None:
            self.db_path = get_default_storage_path()
        else:
            self.db_path = Path(db_path)

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA_FIT_LOGS)
            conn.executescript(self.SCHEMA_INDEXES)
            conn.executescript(self.SCHEMA_SUGGESTIONS)
            conn.commit()

    def insert(
        self,
        record: FitLogRecord,
        on_conflict: Literal["replace", "update", "skip"] = "replace",
    ) -> bool:
        """Insert a fit log record.

        Args:
            record: FitLogRecord to insert
            on_conflict: Behavior when record with same (well_id, product, timestamp) exists:
                - "replace": Replace entire record (default, original behavior)
                - "update": Update existing record, keeping fit_id
                - "skip": Skip if duplicate exists

        Returns:
            True if record was inserted/updated, False if skipped
        """
        data = record.to_dict()

        if on_conflict == "update":
            return self._upsert_record(data)
        elif on_conflict == "skip":
            return self._insert_if_not_exists(data)
        else:
            # Original behavior: INSERT OR REPLACE by fit_id
            columns = list(data.keys())
            placeholders = ["?" for _ in columns]

            sql = f"""
            INSERT OR REPLACE INTO fit_logs ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            """

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(sql, list(data.values()))
                conn.commit()
            return True

    def _upsert_record(self, data: dict) -> bool:
        """Insert or update record by (well_id, product, timestamp) composite key.

        If a record with the same well_id, product, and timestamp (within 1 second)
        already exists, update it instead of creating a duplicate.

        Args:
            data: Record data dictionary

        Returns:
            True if record was inserted/updated
        """
        well_id = data["well_id"]
        product = data["product"]
        timestamp = data["timestamp"]

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Check for existing record with same well_id, product, and similar timestamp
            # Use date comparison (same day) to handle slight timestamp variations
            existing_sql = """
            SELECT fit_id FROM fit_logs
            WHERE well_id = ? AND product = ?
            AND date(timestamp) = date(?)
            """
            cursor = conn.execute(existing_sql, (well_id, product, timestamp))
            existing = cursor.fetchone()

            if existing:
                # Update existing record, keeping original fit_id
                existing_fit_id = existing["fit_id"]
                update_data = {k: v for k, v in data.items() if k != "fit_id"}

                set_clause = ", ".join(f"{k} = ?" for k in update_data.keys())
                update_sql = f"""
                UPDATE fit_logs SET {set_clause}
                WHERE fit_id = ?
                """
                conn.execute(update_sql, list(update_data.values()) + [existing_fit_id])
            else:
                # Insert new record
                columns = list(data.keys())
                placeholders = ["?" for _ in columns]
                insert_sql = f"""
                INSERT INTO fit_logs ({', '.join(columns)})
                VALUES ({', '.join(placeholders)})
                """
                conn.execute(insert_sql, list(data.values()))

            conn.commit()
        return True

    def _insert_if_not_exists(self, data: dict) -> bool:
        """Insert record only if no duplicate exists.

        Args:
            data: Record data dictionary

        Returns:
            True if record was inserted, False if duplicate exists
        """
        well_id = data["well_id"]
        product = data["product"]
        timestamp = data["timestamp"]

        with sqlite3.connect(self.db_path) as conn:
            # Check for existing record
            existing_sql = """
            SELECT 1 FROM fit_logs
            WHERE well_id = ? AND product = ?
            AND date(timestamp) = date(?)
            """
            cursor = conn.execute(existing_sql, (well_id, product, timestamp))

            if cursor.fetchone():
                return False  # Duplicate exists, skip

            # Insert new record
            columns = list(data.keys())
            placeholders = ["?" for _ in columns]
            insert_sql = f"""
            INSERT INTO fit_logs ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
            """
            conn.execute(insert_sql, list(data.values()))
            conn.commit()
        return True

    def insert_batch(
        self,
        records: list[FitLogRecord],
        on_conflict: Literal["replace", "update", "skip"] = "replace",
    ) -> int:
        """Insert multiple fit log records.

        Args:
            records: List of FitLogRecords to insert
            on_conflict: Behavior when record with same (well_id, product, timestamp) exists:
                - "replace": Replace entire record (default, original behavior)
                - "update": Update existing record, keeping fit_id
                - "skip": Skip if duplicate exists

        Returns:
            Number of records inserted/updated (excludes skipped records)
        """
        if not records:
            return 0

        if on_conflict in ("update", "skip"):
            # Use individual upsert logic for each record
            count = 0
            for record in records:
                if self.insert(record, on_conflict=on_conflict):
                    count += 1
            return count

        # Original behavior: INSERT OR REPLACE by fit_id
        data = records[0].to_dict()
        columns = list(data.keys())
        placeholders = ["?" for _ in columns]

        sql = f"""
        INSERT OR REPLACE INTO fit_logs ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """

        with sqlite3.connect(self.db_path) as conn:
            for record in records:
                conn.execute(sql, list(record.to_dict().values()))
            conn.commit()

        return len(records)

    @staticmethod
    def _build_where(
        **filters: tuple[str, object] | None,
    ) -> tuple[str, list]:
        """Build WHERE clause from keyword filters.

        Each keyword maps to (sql_operator, value). None values are skipped.
        Datetime values are converted to isoformat strings.

        Returns:
            Tuple of (where_clause_str, params_list)
        """
        conditions: list[str] = []
        params: list = []
        for col_op, value in filters.values():
            if value is None:
                continue
            conditions.append(col_op)
            params.append(value.isoformat() if isinstance(value, datetime) else value)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        return where, params

    def query(
        self,
        well_id: str | None = None,
        product: str | None = None,
        basin: str | None = None,
        formation: str | None = None,
        min_r_squared: float | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int | None = None,
    ) -> list[FitLogRecord]:
        """Query fit log records with optional filters.

        Args:
            well_id: Filter by well ID
            product: Filter by product
            basin: Filter by basin
            formation: Filter by formation
            min_r_squared: Minimum R-squared threshold
            start_date: Filter records on or after this date
            end_date: Filter records on or before this date
            limit: Maximum number of records to return

        Returns:
            List of matching FitLogRecords
        """
        where, params = self._build_where(
            well_id=("well_id = ?", well_id),
            product=("product = ?", product),
            basin=("basin = ?", basin),
            formation=("formation = ?", formation),
            min_r_squared=("r_squared >= ?", min_r_squared),
            start_date=("timestamp >= ?", start_date),
            end_date=("timestamp <= ?", end_date),
        )

        sql = f"SELECT * FROM fit_logs{where} ORDER BY timestamp DESC"
        if limit is not None:
            sql += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        return [FitLogRecord.from_dict(dict(row)) for row in rows]

    def iterate_all(
        self,
        batch_size: int = 1000,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> Iterator[FitLogRecord]:
        """Iterate over all records in batches.

        Args:
            batch_size: Number of records per batch
            start_date: Filter records on or after this date
            end_date: Filter records on or before this date

        Yields:
            FitLogRecord objects
        """
        where, params = self._build_where(
            start_date=("timestamp >= ?", start_date),
            end_date=("timestamp <= ?", end_date),
        )

        offset = 0
        while True:
            sql = f"SELECT * FROM fit_logs{where} ORDER BY timestamp LIMIT {batch_size} OFFSET {offset}"
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()

            if not rows:
                break

            for row in rows:
                yield FitLogRecord.from_dict(dict(row))

            offset += batch_size

    def count(
        self,
        basin: str | None = None,
        formation: str | None = None,
        product: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> int:
        """Count records matching filters.

        Args:
            basin: Filter by basin
            formation: Filter by formation
            product: Filter by product
            start_date: Filter records on or after this date
            end_date: Filter records on or before this date

        Returns:
            Count of matching records
        """
        where, params = self._build_where(
            basin=("basin = ?", basin),
            formation=("formation = ?", formation),
            product=("product = ?", product),
            start_date=("timestamp >= ?", start_date),
            end_date=("timestamp <= ?", end_date),
        )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM fit_logs{where}", params)
            return cursor.fetchone()[0]

    def get_statistics(
        self,
        basin: str | None = None,
        formation: str | None = None,
        product: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict:
        """Get aggregate statistics for matching records.

        Args:
            basin: Filter by basin
            formation: Filter by formation
            product: Filter by product
            start_date: Filter records on or after this date
            end_date: Filter records on or before this date

        Returns:
            Dictionary with statistics (count, avg_r_squared, avg_mape, etc.)
        """
        where, params = self._build_where(
            basin=("basin = ?", basin),
            formation=("formation = ?", formation),
            product=("product = ?", product),
            start_date=("timestamp >= ?", start_date),
            end_date=("timestamp <= ?", end_date),
        )

        sql = f"""
        SELECT
            COUNT(*) as count,
            AVG(r_squared) as avg_r_squared,
            AVG(rmse) as avg_rmse,
            AVG(hindcast_mape) as avg_hindcast_mape,
            AVG(hindcast_correlation) as avg_hindcast_correlation,
            AVG(b) as avg_b,
            MIN(b) as min_b,
            MAX(b) as max_b,
            AVG(recency_half_life) as avg_recency_half_life,
            AVG(regime_threshold) as avg_regime_threshold
        FROM fit_logs
        {where}
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            row = cursor.fetchone()

        return dict(row) if row else {}

    def save_suggestion(self, suggestion: ParameterSuggestion) -> None:
        """Save or update a parameter suggestion.

        Args:
            suggestion: ParameterSuggestion to save
        """
        from datetime import datetime

        sql = """
        INSERT OR REPLACE INTO parameter_suggestions (
            basin, formation, product, sample_count,
            suggested_recency_half_life, suggested_regime_threshold,
            suggested_regime_window, suggested_regime_sustained_months,
            avg_r_squared, avg_hindcast_mape, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Extract basin/formation from grouping
        parts = suggestion.grouping.split("/")
        basin = parts[0] if parts[0] != "global" else None
        formation = parts[1] if len(parts) > 1 else None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql, (
                basin,
                formation,
                suggestion.product,
                suggestion.sample_count,
                suggestion.suggested_recency_half_life,
                suggestion.suggested_regime_threshold,
                suggestion.suggested_regime_window,
                suggestion.suggested_regime_sustained_months,
                suggestion.avg_r_squared,
                suggestion.avg_hindcast_mape,
                datetime.now().isoformat(),
            ))
            conn.commit()

    def get_suggestion(
        self,
        product: str,
        basin: str | None = None,
        formation: str | None = None,
    ) -> ParameterSuggestion | None:
        """Get parameter suggestion for a grouping.

        Args:
            product: Product type
            basin: Basin name (None for global)
            formation: Formation name (None for basin-level)

        Returns:
            ParameterSuggestion if found, None otherwise
        """
        # Try specific first, then fall back to less specific
        for b, f in [(basin, formation), (basin, None), (None, None)]:
            sql = """
            SELECT * FROM parameter_suggestions
            WHERE product = ?
            AND basin IS ? AND formation IS ?
            """
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql, (product, b, f))
                row = cursor.fetchone()

            if row:
                grouping = "global"
                if row["basin"]:
                    grouping = row["basin"]
                    if row["formation"]:
                        grouping += f"/{row['formation']}"

                return ParameterSuggestion(
                    grouping=grouping,
                    sample_count=row["sample_count"],
                    product=row["product"],
                    suggested_recency_half_life=row["suggested_recency_half_life"],
                    suggested_regime_threshold=row["suggested_regime_threshold"],
                    suggested_regime_window=row["suggested_regime_window"],
                    suggested_regime_sustained_months=row["suggested_regime_sustained_months"],
                    avg_r_squared=row["avg_r_squared"],
                    avg_hindcast_mape=row["avg_hindcast_mape"],
                )

        return None

    def export_to_csv(self, filepath: Path | str) -> int:
        """Export all fit logs to CSV.

        Args:
            filepath: Output CSV file path

        Returns:
            Number of records exported
        """
        filepath = Path(filepath)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM fit_logs ORDER BY timestamp")
            rows = cursor.fetchall()

        if not rows:
            return 0

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

        return len(rows)

    def import_from_csv(self, filepath: Path | str) -> int:
        """Import fit logs from CSV.

        Args:
            filepath: Input CSV file path

        Returns:
            Number of records imported
        """
        filepath = Path(filepath)

        records = []
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                for key in row:
                    if row[key] == "":
                        row[key] = None
                    elif key in ("data_points_total", "data_points_used", "regime_start_idx"):
                        row[key] = int(row[key]) if row[key] else 0
                    elif key not in ("fit_id", "timestamp", "well_id", "product", "basin", "formation"):
                        try:
                            row[key] = float(row[key]) if row[key] else None
                        except ValueError:
                            pass
                records.append(FitLogRecord.from_dict(row))

        return self.insert_batch(records)

    def clear(self) -> None:
        """Clear all data from database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM fit_logs")
            conn.execute("DELETE FROM parameter_suggestions")
            conn.commit()
