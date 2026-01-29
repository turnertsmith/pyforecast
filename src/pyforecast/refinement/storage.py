"""Storage abstraction for fit logs and parameter suggestions.

Provides SQLite storage for persistent accumulation of fit metadata
across all projects, enabling learning from historical fits.
"""

from pathlib import Path
import sqlite3
from typing import Iterator
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

    def insert(self, record: FitLogRecord) -> None:
        """Insert a fit log record.

        Args:
            record: FitLogRecord to insert
        """
        data = record.to_dict()

        columns = list(data.keys())
        placeholders = ["?" for _ in columns]

        sql = f"""
        INSERT OR REPLACE INTO fit_logs ({', '.join(columns)})
        VALUES ({', '.join(placeholders)})
        """

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(sql, list(data.values()))
            conn.commit()

    def insert_batch(self, records: list[FitLogRecord]) -> int:
        """Insert multiple fit log records.

        Args:
            records: List of FitLogRecords to insert

        Returns:
            Number of records inserted
        """
        if not records:
            return 0

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

    def query(
        self,
        well_id: str | None = None,
        product: str | None = None,
        basin: str | None = None,
        formation: str | None = None,
        min_r_squared: float | None = None,
        limit: int | None = None,
    ) -> list[FitLogRecord]:
        """Query fit log records with optional filters.

        Args:
            well_id: Filter by well ID
            product: Filter by product
            basin: Filter by basin
            formation: Filter by formation
            min_r_squared: Minimum R-squared threshold
            limit: Maximum number of records to return

        Returns:
            List of matching FitLogRecords
        """
        conditions = []
        params = []

        if well_id is not None:
            conditions.append("well_id = ?")
            params.append(well_id)
        if product is not None:
            conditions.append("product = ?")
            params.append(product)
        if basin is not None:
            conditions.append("basin = ?")
            params.append(basin)
        if formation is not None:
            conditions.append("formation = ?")
            params.append(formation)
        if min_r_squared is not None:
            conditions.append("r_squared >= ?")
            params.append(min_r_squared)

        sql = "SELECT * FROM fit_logs"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY timestamp DESC"
        if limit is not None:
            sql += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

        return [FitLogRecord.from_dict(dict(row)) for row in rows]

    def iterate_all(self, batch_size: int = 1000) -> Iterator[FitLogRecord]:
        """Iterate over all records in batches.

        Args:
            batch_size: Number of records per batch

        Yields:
            FitLogRecord objects
        """
        offset = 0
        while True:
            sql = f"SELECT * FROM fit_logs ORDER BY timestamp LIMIT {batch_size} OFFSET {offset}"
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql)
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
    ) -> int:
        """Count records matching filters.

        Args:
            basin: Filter by basin
            formation: Filter by formation
            product: Filter by product

        Returns:
            Count of matching records
        """
        conditions = []
        params = []

        if basin is not None:
            conditions.append("basin = ?")
            params.append(basin)
        if formation is not None:
            conditions.append("formation = ?")
            params.append(formation)
        if product is not None:
            conditions.append("product = ?")
            params.append(product)

        sql = "SELECT COUNT(*) FROM fit_logs"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchone()[0]

    def get_statistics(
        self,
        basin: str | None = None,
        formation: str | None = None,
        product: str | None = None,
    ) -> dict:
        """Get aggregate statistics for matching records.

        Args:
            basin: Filter by basin
            formation: Filter by formation
            product: Filter by product

        Returns:
            Dictionary with statistics (count, avg_r_squared, avg_mape, etc.)
        """
        conditions = []
        params = []

        if basin is not None:
            conditions.append("basin = ?")
            params.append(basin)
        if formation is not None:
            conditions.append("formation = ?")
            params.append(formation)
        if product is not None:
            conditions.append("product = ?")
            params.append(product)

        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

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
        {where_clause}
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
