"""Batch export functionality for forecast results.

This module provides the BatchExporter class which handles exporting
forecast results in various formats (AC_ECONOMIC, JSON).

Extracted from BatchProcessor to follow single responsibility principle.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..data.well import Well
from ..export.aries_ac_economic import AriesAcEconomicExporter
from ..export.json_export import JsonExporter
from ..validation import ValidationResult

if TYPE_CHECKING:
    from ..config import PyForecastConfig

logger = logging.getLogger(__name__)


class BatchExporter:
    """Export batch processing results to files.

    Handles exporting wells with forecasts to various output formats,
    with support for validation results inclusion.

    Attributes:
        export_format: Export format ('ac_economic' or 'json')
        pyforecast_config: Optional PyForecastConfig for JSON export metadata

    Example:
        >>> exporter = BatchExporter(export_format="ac_economic")
        >>> exporter.export_forecasts(
        ...     wells=wells,
        ...     output_dir=Path("output"),
        ...     products=["oil", "gas"],
        ... )
    """

    def __init__(
        self,
        export_format: Literal["ac_economic", "json"] = "ac_economic",
        pyforecast_config: "PyForecastConfig | None" = None,
    ):
        """Initialize batch exporter.

        Args:
            export_format: Output format - 'ac_economic' for ARIES or 'json'
            pyforecast_config: Configuration for JSON export metadata
        """
        self.export_format = export_format
        self.pyforecast_config = pyforecast_config

    def export_forecasts(
        self,
        wells: list[Well],
        output_dir: Path,
        products: list[Literal["oil", "gas", "water"]],
        validation_results: dict[str, ValidationResult] | None = None,
    ) -> Path:
        """Export forecasts to file.

        Args:
            wells: List of wells with forecasts
            output_dir: Output directory
            products: Products to include in export
            validation_results: Optional validation results for JSON export

        Returns:
            Path to the created export file
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.export_format == "json":
            return self._export_json(wells, output_dir, products, validation_results)
        else:
            return self._export_ac_economic(wells, output_dir, products)

    def _export_ac_economic(
        self,
        wells: list[Well],
        output_dir: Path,
        products: list[Literal["oil", "gas", "water"]],
    ) -> Path:
        """Export to AC_ECONOMIC format.

        Args:
            wells: List of wells with forecasts
            output_dir: Output directory
            products: Products to include

        Returns:
            Path to created CSV file
        """
        exporter = AriesAcEconomicExporter()
        forecast_path = output_dir / "ac_economic.csv"
        exporter.save(wells, forecast_path, products)
        logger.info(f"Exported {len(wells)} wells to {forecast_path}")
        return forecast_path

    def _export_json(
        self,
        wells: list[Well],
        output_dir: Path,
        products: list[Literal["oil", "gas", "water"]],
        validation_results: dict[str, ValidationResult] | None = None,
    ) -> Path:
        """Export to JSON format.

        Args:
            wells: List of wells with forecasts
            output_dir: Output directory
            products: Products to include
            validation_results: Optional validation results

        Returns:
            Path to created JSON file
        """
        exporter = JsonExporter(config=self.pyforecast_config)
        forecast_path = output_dir / "forecasts.json"
        exporter.save(
            wells,
            forecast_path,
            products,
            validation_results or {},
        )
        logger.info(f"Exported {len(wells)} wells to {forecast_path}")
        return forecast_path

    def export_errors(
        self,
        errors: list[tuple[str, str]],
        output_dir: Path,
    ) -> Path | None:
        """Export error log.

        Args:
            errors: List of (well_id, error_message) tuples
            output_dir: Output directory

        Returns:
            Path to error file, or None if no errors
        """
        if not errors:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        error_path = output_dir / "errors.txt"

        with open(error_path, "w") as f:
            for well_id, error in errors:
                f.write(f"{well_id}: {error}\n")

        logger.info(f"Saved {len(errors)} errors to {error_path}")
        return error_path

    def export_validation_report(
        self,
        validation_results: dict[str, ValidationResult],
        output_dir: Path,
    ) -> Path:
        """Export validation report.

        Args:
            validation_results: Dict of well_id -> ValidationResult
            output_dir: Output directory

        Returns:
            Path to validation report file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "validation_report.txt"

        # Calculate summary statistics
        summary = self._get_validation_summary(validation_results)

        with open(report_path, "w") as f:
            f.write("PyForecast Validation Report\n")
            f.write("=" * 40 + "\n\n")

            f.write("Summary:\n")
            f.write(f"  Wells with errors: {summary['wells_with_errors']}\n")
            f.write(f"  Wells with warnings: {summary['wells_with_warnings']}\n")
            f.write(f"  Total errors: {summary['total_errors']}\n")
            f.write(f"  Total warnings: {summary['total_warnings']}\n\n")

            if summary["by_category"]:
                f.write("Issues by category:\n")
                for cat, count in sorted(summary["by_category"].items()):
                    f.write(f"  {cat}: {count}\n")
                f.write("\n")

            # Write detailed issues per well
            f.write("Detailed Issues:\n")
            f.write("-" * 40 + "\n")

            for well_id, val_result in sorted(validation_results.items()):
                if val_result.issues:
                    f.write(f"\n{well_id}:\n")
                    for issue in val_result.issues:
                        f.write(f"  [{issue.code}] {issue.severity.name}: {issue.message}\n")
                        f.write(f"    Guidance: {issue.guidance}\n")

        logger.info(f"Saved validation report to {report_path}")
        return report_path

    @staticmethod
    def _get_validation_summary(
        validation_results: dict[str, ValidationResult],
    ) -> dict:
        """Calculate validation summary statistics.

        Args:
            validation_results: Dict of well_id -> ValidationResult

        Returns:
            Summary statistics dict
        """
        from ..validation.result import summarize_validation
        return summarize_validation(validation_results)
