"""Fit logging for capturing and persisting fit metadata.

Logs all fit parameters, metrics, and optional diagnostics to persistent
storage for analysis and learning.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import FitLogRecord, HindcastResult, ResidualDiagnostics
from .storage import FitLogStorage

if TYPE_CHECKING:
    from ..core.fitting import FittingConfig
    from ..core.models import ForecastResult
    from ..data.well import Well

logger = logging.getLogger(__name__)


class FitLogger:
    """Captures and persists fit metadata to storage.

    Logs all fit parameters, results, and optional diagnostics (residuals,
    hindcast) to SQLite storage for later analysis and learning.

    Example:
        logger = FitLogger()  # Uses default storage path

        # Log a basic fit
        logger.log(fit_result, well, "oil", fitting_config)

        # Log with hindcast results
        logger.log(fit_result, well, "oil", fitting_config, hindcast=hindcast_result)

        # Flush batch to storage
        logger.flush()
    """

    def __init__(
        self,
        storage_path: Path | str | None = None,
        batch_size: int = 100,
    ):
        """Initialize fit logger.

        Args:
            storage_path: Path to storage file (None = default location)
            batch_size: Number of records to batch before writing
        """
        self.storage = FitLogStorage(storage_path)
        self.batch_size = batch_size
        self._batch: list[FitLogRecord] = []

    def log(
        self,
        fit_result: "ForecastResult",
        well: "Well",
        product: str,
        fitting_config: "FittingConfig",
        hindcast: HindcastResult | None = None,
        residuals: ResidualDiagnostics | None = None,
        basin: str | None = None,
        formation: str | None = None,
    ) -> FitLogRecord:
        """Log a fit result.

        Args:
            fit_result: ForecastResult from fitting
            well: Well that was fit
            product: Product type (oil, gas, water)
            fitting_config: FittingConfig used for fitting
            hindcast: Optional HindcastResult
            residuals: Optional ResidualDiagnostics
            basin: Optional basin name from well metadata
            formation: Optional formation name from well metadata

        Returns:
            FitLogRecord that was logged
        """
        # Extract basin/formation from well metadata if not provided
        if basin is None:
            basin = well.metadata.get("basin")
        if formation is None:
            formation = well.metadata.get("formation")

        # Get total data points
        t = well.production.time_months
        data_points_total = len(t)

        # Create record
        record = FitLogRecord(
            timestamp=datetime.now(),
            well_id=well.well_id,
            product=product,
            basin=basin,
            formation=formation,
            data_points_total=data_points_total,
            data_points_used=fit_result.data_points_used,
            regime_start_idx=fit_result.regime_start_idx,
            b_min=fitting_config.b_min,
            b_max=fitting_config.b_max,
            dmin_annual=fitting_config.dmin_annual,
            recency_half_life=fitting_config.recency_half_life,
            regime_threshold=fitting_config.regime_threshold,
            qi=fit_result.model.qi,
            di=fit_result.model.di,
            b=fit_result.model.b,
            r_squared=fit_result.r_squared,
            rmse=fit_result.rmse,
            aic=fit_result.aic,
            bic=fit_result.bic,
        )

        # Add hindcast results if available
        if hindcast is not None:
            record.hindcast_mape = hindcast.mape
            record.hindcast_correlation = hindcast.correlation
            record.hindcast_bias = hindcast.bias

        # Add residual diagnostics if available
        if residuals is not None:
            record.residual_mean = residuals.mean
            record.residual_std = residuals.std
            record.durbin_watson = residuals.durbin_watson
            record.early_bias = residuals.early_bias
            record.late_bias = residuals.late_bias

        # Add to batch
        self._batch.append(record)

        # Flush if batch is full
        if len(self._batch) >= self.batch_size:
            self.flush()

        return record

    def log_from_data(
        self,
        fit_result: "ForecastResult",
        well_id: str,
        product: str,
        fitting_config: "FittingConfig",
        data_points_total: int,
        hindcast: HindcastResult | None = None,
        residuals: ResidualDiagnostics | None = None,
        basin: str | None = None,
        formation: str | None = None,
    ) -> FitLogRecord:
        """Log a fit result without a Well object.

        Useful for logging from batch processing where Well may not be available.

        Args:
            fit_result: ForecastResult from fitting
            well_id: Well identifier
            product: Product type
            fitting_config: FittingConfig used
            data_points_total: Total data points available
            hindcast: Optional HindcastResult
            residuals: Optional ResidualDiagnostics
            basin: Optional basin name
            formation: Optional formation name

        Returns:
            FitLogRecord that was logged
        """
        record = FitLogRecord(
            timestamp=datetime.now(),
            well_id=well_id,
            product=product,
            basin=basin,
            formation=formation,
            data_points_total=data_points_total,
            data_points_used=fit_result.data_points_used,
            regime_start_idx=fit_result.regime_start_idx,
            b_min=fitting_config.b_min,
            b_max=fitting_config.b_max,
            dmin_annual=fitting_config.dmin_annual,
            recency_half_life=fitting_config.recency_half_life,
            regime_threshold=fitting_config.regime_threshold,
            qi=fit_result.model.qi,
            di=fit_result.model.di,
            b=fit_result.model.b,
            r_squared=fit_result.r_squared,
            rmse=fit_result.rmse,
            aic=fit_result.aic,
            bic=fit_result.bic,
        )

        # Add hindcast results if available
        if hindcast is not None:
            record.hindcast_mape = hindcast.mape
            record.hindcast_correlation = hindcast.correlation
            record.hindcast_bias = hindcast.bias

        # Add residual diagnostics if available
        if residuals is not None:
            record.residual_mean = residuals.mean
            record.residual_std = residuals.std
            record.durbin_watson = residuals.durbin_watson
            record.early_bias = residuals.early_bias
            record.late_bias = residuals.late_bias

        # Add to batch
        self._batch.append(record)

        # Flush if batch is full
        if len(self._batch) >= self.batch_size:
            self.flush()

        return record

    def flush(self) -> int:
        """Write batched records to storage.

        Returns:
            Number of records written
        """
        if not self._batch:
            return 0

        count = self.storage.insert_batch(self._batch)
        logger.debug(f"Flushed {count} fit log records to storage")
        self._batch = []
        return count

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush remaining records."""
        self.flush()
        return False

    @property
    def pending_count(self) -> int:
        """Number of records waiting to be flushed."""
        return len(self._batch)


class FitLogAnalyzer:
    """Analyzes accumulated fit logs to generate reports.

    Provides analysis capabilities for fit logs including:
    - Summary statistics by grouping (global, basin, formation)
    - Parameter distribution analysis
    - Hindcast performance analysis
    """

    def __init__(self, storage_path: Path | str | None = None):
        """Initialize analyzer.

        Args:
            storage_path: Path to storage file (None = default location)
        """
        self.storage = FitLogStorage(storage_path)

    def get_summary(
        self,
        basin: str | None = None,
        formation: str | None = None,
        product: str | None = None,
    ) -> dict:
        """Get summary statistics for matching records.

        Args:
            basin: Filter by basin
            formation: Filter by formation
            product: Filter by product

        Returns:
            Dictionary with summary statistics
        """
        return self.storage.get_statistics(basin, formation, product)

    def get_hindcast_performance(
        self,
        basin: str | None = None,
        formation: str | None = None,
        product: str | None = None,
    ) -> dict:
        """Get hindcast performance statistics.

        Args:
            basin: Filter by basin
            formation: Filter by formation
            product: Filter by product

        Returns:
            Dictionary with hindcast performance metrics
        """
        records = self.storage.query(
            basin=basin,
            formation=formation,
            product=product,
        )

        # Filter to records with hindcast data
        with_hindcast = [r for r in records if r.hindcast_mape is not None]

        if not with_hindcast:
            return {
                "count": 0,
                "avg_mape": None,
                "avg_correlation": None,
            }

        import numpy as np

        mapes = [r.hindcast_mape for r in with_hindcast]
        correlations = [r.hindcast_correlation for r in with_hindcast if r.hindcast_correlation is not None]

        return {
            "count": len(with_hindcast),
            "avg_mape": float(np.mean(mapes)),
            "median_mape": float(np.median(mapes)),
            "std_mape": float(np.std(mapes)),
            "avg_correlation": float(np.mean(correlations)) if correlations else None,
            "good_hindcast_pct": sum(1 for m in mapes if m < 30) / len(mapes) * 100,
        }

    def get_parameter_distribution(
        self,
        basin: str | None = None,
        formation: str | None = None,
        product: str | None = None,
    ) -> dict:
        """Get distribution of fitted parameters.

        Args:
            basin: Filter by basin
            formation: Filter by formation
            product: Filter by product

        Returns:
            Dictionary with parameter distribution statistics
        """
        records = self.storage.query(
            basin=basin,
            formation=formation,
            product=product,
        )

        if not records:
            return {"count": 0}

        import numpy as np

        b_values = [r.b for r in records]
        qi_values = [r.qi for r in records]
        di_values = [r.di for r in records]

        return {
            "count": len(records),
            "b": {
                "mean": float(np.mean(b_values)),
                "std": float(np.std(b_values)),
                "min": float(np.min(b_values)),
                "max": float(np.max(b_values)),
                "p25": float(np.percentile(b_values, 25)),
                "p50": float(np.percentile(b_values, 50)),
                "p75": float(np.percentile(b_values, 75)),
            },
            "qi": {
                "mean": float(np.mean(qi_values)),
                "std": float(np.std(qi_values)),
                "p25": float(np.percentile(qi_values, 25)),
                "p50": float(np.percentile(qi_values, 50)),
                "p75": float(np.percentile(qi_values, 75)),
            },
            "di_annual": {
                "mean": float(np.mean(di_values)) * 12,
                "std": float(np.std(di_values)) * 12,
                "p25": float(np.percentile(di_values, 25)) * 12,
                "p50": float(np.percentile(di_values, 50)) * 12,
                "p75": float(np.percentile(di_values, 75)) * 12,
            },
        }

    def export_report(
        self,
        output_path: Path | str,
        basin: str | None = None,
        formation: str | None = None,
        product: str | None = None,
    ) -> None:
        """Export analysis report to CSV.

        Args:
            output_path: Path for output CSV
            basin: Filter by basin
            formation: Filter by formation
            product: Filter by product
        """
        import csv

        output_path = Path(output_path)

        # Get all statistics
        summary = self.get_summary(basin, formation, product)
        hindcast = self.get_hindcast_performance(basin, formation, product)
        params = self.get_parameter_distribution(basin, formation, product)

        # Write report
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            writer.writerow(["PyForecast Fit Analysis Report"])
            writer.writerow([])

            # Filters
            writer.writerow(["Filters"])
            writer.writerow(["Basin", basin or "All"])
            writer.writerow(["Formation", formation or "All"])
            writer.writerow(["Product", product or "All"])
            writer.writerow([])

            # Summary
            writer.writerow(["Summary Statistics"])
            for key, value in summary.items():
                writer.writerow([key, value])
            writer.writerow([])

            # Hindcast
            writer.writerow(["Hindcast Performance"])
            for key, value in hindcast.items():
                writer.writerow([key, value])
            writer.writerow([])

            # Parameters
            writer.writerow(["Parameter Distribution"])
            if "b" in params:
                writer.writerow(["B Factor"])
                for key, value in params["b"].items():
                    writer.writerow([f"  {key}", value])
            if "qi" in params:
                writer.writerow(["Qi"])
                for key, value in params["qi"].items():
                    writer.writerow([f"  {key}", value])
            if "di_annual" in params:
                writer.writerow(["Di (Annual)"])
                for key, value in params["di_annual"].items():
                    writer.writerow([f"  {key}", f"{value:.1%}"])

        logger.info(f"Exported fit analysis report to {output_path}")
