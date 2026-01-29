"""Batch processing for multiple wells with parallel execution."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TYPE_CHECKING
import logging

from tqdm import tqdm

from ..data.well import Well
from ..data.base import load_wells
from ..core.fitting import DeclineFitter, FittingConfig
from ..core.models import ForecastResult
from ..export.aries_export import AriesExporter
from ..export.aries_ac_economic import AriesAcEconomicExporter
from ..export.json_export import JsonExporter
from ..visualization.plots import DeclinePlotter
from ..validation import (
    ValidationResult,
    InputValidator,
    DataQualityValidator,
    FittingValidator,
    IssueCategory,
    IssueSeverity,
)

if TYPE_CHECKING:
    from ..config import PyForecastConfig, ValidationConfig

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        products: Products to forecast (oil, gas, water)
        min_points: Minimum data points required for fitting
        workers: Number of parallel workers (None = auto)
        fitting_config: Legacy single config for all products (deprecated)
        pyforecast_config: Full config with per-product settings
        output_dir: Output directory for results
        save_plots: Whether to save individual well plots
        save_batch_plot: Whether to save multi-well overlay plot
        export_format: Export format (ac_forecast, ac_economic, or json)
    """
    products: list[Literal["oil", "gas", "water"]]
    min_points: int = 6
    workers: int | None = None
    fitting_config: FittingConfig | None = None
    pyforecast_config: "PyForecastConfig | None" = None
    output_dir: Path | None = None
    save_plots: bool = True
    save_batch_plot: bool = True
    export_format: Literal["ac_forecast", "ac_economic", "json"] = "ac_economic"

    def get_fitting_config(self, product: str) -> FittingConfig:
        """Get fitting config for a specific product.

        Args:
            product: Product name (oil, gas, water)

        Returns:
            FittingConfig for the product
        """
        if self.pyforecast_config is not None:
            return FittingConfig.from_pyforecast_config(self.pyforecast_config, product)
        elif self.fitting_config is not None:
            return self.fitting_config
        else:
            return FittingConfig()


@dataclass
class BatchResult:
    """Results from batch processing.

    Attributes:
        wells: List of wells with forecasts
        successful: Count of successfully fitted wells
        failed: Count of failed fits
        skipped: Count of wells skipped (insufficient data)
        errors: List of (well_id, error_message) tuples
        validation_results: Dict of well_id -> ValidationResult
    """
    wells: list[Well]
    successful: int
    failed: int
    skipped: int
    errors: list[tuple[str, str]]
    validation_results: dict[str, ValidationResult] = field(default_factory=dict)

    def get_validation_summary(self) -> dict:
        """Get summary of validation results.

        Returns:
            Dict with counts by severity and category
        """
        summary = {
            "wells_with_errors": 0,
            "wells_with_warnings": 0,
            "total_errors": 0,
            "total_warnings": 0,
            "by_category": {},
        }

        for result in self.validation_results.values():
            if result.has_errors:
                summary["wells_with_errors"] += 1
            if result.has_warnings:
                summary["wells_with_warnings"] += 1
            summary["total_errors"] += result.error_count
            summary["total_warnings"] += result.warning_count

            for issue in result.issues:
                cat_name = issue.category.name
                if cat_name not in summary["by_category"]:
                    summary["by_category"][cat_name] = 0
                summary["by_category"][cat_name] += 1

        return summary


def _fit_single_well(
    well: Well,
    products: list[str],
    product_configs: dict[str, FittingConfig],
    validation_config: "ValidationConfig | None" = None,
) -> tuple[Well, list[str], ValidationResult]:
    """Fit decline curves for a single well (worker function).

    Args:
        well: Well object with production data
        products: Products to fit
        product_configs: Dict of product -> FittingConfig
        validation_config: Optional validation configuration

    Returns:
        Tuple of (well with forecasts, list of error messages, validation result)
    """
    errors = []
    validation_result = ValidationResult(well_id=well.well_id)

    # Create validators if config provided
    if validation_config is not None:
        input_validator = InputValidator(
            max_oil_rate=validation_config.max_oil_rate,
            max_gas_rate=validation_config.max_gas_rate,
            max_water_rate=validation_config.max_water_rate,
        )
        quality_validator = DataQualityValidator(
            gap_threshold_months=validation_config.gap_threshold_months,
            outlier_sigma=validation_config.outlier_sigma,
            shutin_threshold=validation_config.shutin_threshold,
            min_cv=validation_config.min_cv,
        )
        # Fitting validator uses product-specific b bounds
        fitting_validators = {}
        for product in products:
            config = product_configs.get(product, FittingConfig())
            fitting_validators[product] = FittingValidator(
                min_points=config.min_points,
                min_r_squared=validation_config.min_r_squared,
                b_min=config.b_min,
                b_max=config.b_max,
                max_annual_decline=validation_config.max_annual_decline,
            )

        # Run input validation
        validation_result = validation_result.merge(input_validator.validate(well))
    else:
        input_validator = None
        quality_validator = None
        fitting_validators = {}

    for product in products:
        try:
            config = product_configs.get(product, FittingConfig())
            fitter = DeclineFitter(config)

            t = well.production.time_months
            q = well.production.get_product(product)

            # Skip if no significant production
            if q.max() < 1.0:
                continue

            # Run data quality validation
            if quality_validator is not None:
                validation_result = validation_result.merge(
                    quality_validator.validate(well, product)
                )

            # Run pre-fit validation
            if product in fitting_validators:
                validation_result = validation_result.merge(
                    fitting_validators[product].validate_pre_fit(well, product)
                )

            result = fitter.fit(t, q)
            well.set_forecast(product, result)

            # Run post-fit validation
            if product in fitting_validators:
                validation_result = validation_result.merge(
                    fitting_validators[product].validate_fit_result(
                        result, well.well_id, product
                    )
                )

        except ValueError as e:
            errors.append(f"{product}: {str(e)}")
        except Exception as e:
            errors.append(f"{product}: Unexpected error - {str(e)}")

    return well, errors, validation_result


class BatchProcessor:
    """Process multiple wells with parallel curve fitting."""

    def __init__(self, config: BatchConfig | None = None):
        """Initialize batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig(products=["oil", "gas"])

    def load_files(self, filepaths: list[Path | str]) -> list[Well]:
        """Load wells from multiple files.

        Args:
            filepaths: List of input file paths

        Returns:
            Combined list of wells from all files
        """
        all_wells = []

        for filepath in filepaths:
            try:
                wells = load_wells(filepath)
                all_wells.extend(wells)
                logger.info(f"Loaded {len(wells)} wells from {filepath}")
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")

        return all_wells

    def filter_wells(self, wells: list[Well]) -> tuple[list[Well], int]:
        """Filter wells with insufficient data.

        Args:
            wells: List of wells to filter

        Returns:
            Tuple of (filtered wells, count skipped)
        """
        valid = []
        skipped = 0

        for well in wells:
            if well.has_sufficient_data(self.config.min_points):
                valid.append(well)
            else:
                skipped += 1
                logger.debug(f"Skipping {well.well_id}: insufficient data")

        return valid, skipped

    def process(
        self,
        wells: list[Well],
        show_progress: bool = True
    ) -> BatchResult:
        """Process wells with parallel curve fitting.

        Args:
            wells: List of wells to process
            show_progress: Whether to show progress bar

        Returns:
            BatchResult with processed wells and statistics
        """
        # Filter wells
        filtered_wells, skipped = self.filter_wells(wells)

        if not filtered_wells:
            return BatchResult(
                wells=[],
                successful=0,
                failed=0,
                skipped=skipped,
                errors=[]
            )

        # Build per-product configs
        product_configs = {
            product: self.config.get_fitting_config(product)
            for product in self.config.products
        }

        # Get validation config if available
        validation_config = None
        if self.config.pyforecast_config is not None:
            validation_config = self.config.pyforecast_config.validation

        successful = 0
        failed = 0
        all_errors = []
        processed_wells = []
        all_validation_results = {}

        # Process in parallel
        workers = self.config.workers
        if workers is None:
            import os
            workers = min(os.cpu_count() or 4, len(filtered_wells))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _fit_single_well,
                    well,
                    self.config.products,
                    product_configs,
                    validation_config,
                ): well.well_id
                for well in filtered_wells
            }

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator,
                    total=len(futures),
                    desc="Fitting wells"
                )

            for future in iterator:
                well_id = futures[future]
                try:
                    well, errors, validation_result = future.result()
                    processed_wells.append(well)
                    all_validation_results[well_id] = validation_result

                    if errors:
                        all_errors.extend((well_id, e) for e in errors)
                        has_forecast = any(
                            well.get_forecast(p) is not None
                            for p in self.config.products
                        )
                        if has_forecast:
                            successful += 1
                        else:
                            failed += 1
                    else:
                        successful += 1

                except Exception as e:
                    failed += 1
                    all_errors.append((well_id, str(e)))
                    logger.error(f"Failed to process {well_id}: {e}")

        return BatchResult(
            wells=processed_wells,
            successful=successful,
            failed=failed,
            skipped=skipped,
            errors=all_errors,
            validation_results=all_validation_results,
        )

    def run(
        self,
        input_files: list[Path | str],
        output_dir: Path | str | None = None,
        show_progress: bool = True
    ) -> BatchResult:
        """Run complete batch processing pipeline.

        Args:
            input_files: Input file paths
            output_dir: Output directory (overrides config)
            show_progress: Whether to show progress bars

        Returns:
            BatchResult with processed wells
        """
        output_dir = Path(output_dir) if output_dir else self.config.output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Load wells
        logger.info(f"Loading wells from {len(input_files)} file(s)")
        wells = self.load_files(input_files)
        logger.info(f"Loaded {len(wells)} total wells")

        # Process wells
        result = self.process(wells, show_progress=show_progress)
        logger.info(
            f"Processing complete: {result.successful} successful, "
            f"{result.failed} failed, {result.skipped} skipped"
        )

        if output_dir:
            self._save_outputs(result, output_dir)

        return result

    def _save_outputs(self, result: BatchResult, output_dir: Path) -> None:
        """Save all outputs to directory.

        Args:
            result: Batch processing result
            output_dir: Output directory
        """
        # Export forecasts based on format
        if self.config.export_format == "json":
            exporter = JsonExporter(
                config=self.config.pyforecast_config,
            )
            forecast_path = output_dir / "forecasts.json"
            exporter.save(
                result.wells,
                forecast_path,
                self.config.products,
                result.validation_results,
            )
        elif self.config.export_format == "ac_economic":
            exporter = AriesAcEconomicExporter()
            forecast_path = output_dir / "ac_economic.csv"
            exporter.save(result.wells, forecast_path, self.config.products)
        else:
            exporter = AriesExporter()
            forecast_path = output_dir / "forecasts.csv"
            exporter.save(result.wells, forecast_path, self.config.products)

        logger.info(f"Saved forecast to {forecast_path}")

        # Save plots
        if self.config.save_plots or self.config.save_batch_plot:
            plotter = DeclinePlotter()
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)

            # Individual plots
            if self.config.save_plots:
                for well in result.wells:
                    for product in self.config.products:
                        if well.get_forecast(product) is not None:
                            try:
                                fig = plotter.plot_well(well, product)
                                filename = f"{well.well_id}_{product}.html".replace("/", "_")
                                plotter.save(fig, plots_dir / filename)
                            except Exception as e:
                                logger.warning(f"Failed to plot {well.well_id}: {e}")

            # Batch overlay plot
            if self.config.save_batch_plot and result.wells:
                for product in self.config.products:
                    wells_with_forecast = [
                        w for w in result.wells if w.get_forecast(product) is not None
                    ]
                    if wells_with_forecast:
                        try:
                            fig = plotter.plot_multiple_wells(
                                wells_with_forecast[:20],
                                product
                            )
                            plotter.save(fig, plots_dir / f"batch_{product}.html")
                        except Exception as e:
                            logger.warning(f"Failed to create batch plot: {e}")

        # Save error log
        if result.errors:
            error_path = output_dir / "errors.txt"
            with open(error_path, "w") as f:
                for well_id, error in result.errors:
                    f.write(f"{well_id}: {error}\n")
            logger.info(f"Saved error log to {error_path}")

        # Save validation report
        if result.validation_results:
            self._save_validation_report(result, output_dir)

    def _save_validation_report(
        self,
        result: BatchResult,
        output_dir: Path,
    ) -> None:
        """Save validation report to file.

        Args:
            result: Batch processing result
            output_dir: Output directory
        """
        report_path = output_dir / "validation_report.txt"
        summary = result.get_validation_summary()

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

            for well_id, val_result in sorted(result.validation_results.items()):
                if val_result.issues:
                    f.write(f"\n{well_id}:\n")
                    for issue in val_result.issues:
                        f.write(f"  [{issue.code}] {issue.severity.name}: {issue.message}\n")
                        f.write(f"    Guidance: {issue.guidance}\n")

        logger.info(f"Saved validation report to {report_path}")
