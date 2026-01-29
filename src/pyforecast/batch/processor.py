"""Batch processing for multiple wells with parallel execution."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import logging

from tqdm import tqdm

from ..data.well import Well
from ..data.base import load_wells
from ..core.fitting import DeclineFitter, FittingConfig
from ..core.models import ForecastResult
from ..export.aries_export import AriesExporter
from ..visualization.plots import DeclinePlotter

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for batch processing.

    Attributes:
        products: Products to forecast (oil, gas, or both)
        min_points: Minimum data points required for fitting
        workers: Number of parallel workers (None = auto)
        fitting_config: Configuration for curve fitting
        output_dir: Output directory for results
        save_plots: Whether to save individual well plots
        save_batch_plot: Whether to save multi-well overlay plot
    """
    products: list[Literal["oil", "gas"]]
    min_points: int = 6
    workers: int | None = None
    fitting_config: FittingConfig | None = None
    output_dir: Path | None = None
    save_plots: bool = True
    save_batch_plot: bool = True


@dataclass
class BatchResult:
    """Results from batch processing.

    Attributes:
        wells: List of wells with forecasts
        successful: Count of successfully fitted wells
        failed: Count of failed fits
        skipped: Count of wells skipped (insufficient data)
        errors: List of (well_id, error_message) tuples
    """
    wells: list[Well]
    successful: int
    failed: int
    skipped: int
    errors: list[tuple[str, str]]


def _fit_single_well(
    well: Well,
    products: list[str],
    config: FittingConfig
) -> tuple[Well, list[str]]:
    """Fit decline curves for a single well (worker function).

    Args:
        well: Well object with production data
        products: Products to fit
        config: Fitting configuration

    Returns:
        Tuple of (well with forecasts, list of error messages)
    """
    fitter = DeclineFitter(config)
    errors = []

    for product in products:
        try:
            t = well.production.time_months
            q = well.production.get_product(product)

            # Skip if no significant production
            if q.max() < 1.0:
                continue

            result = fitter.fit(t, q)
            well.set_forecast(product, result)

        except ValueError as e:
            errors.append(f"{product}: {str(e)}")
        except Exception as e:
            errors.append(f"{product}: Unexpected error - {str(e)}")

    return well, errors


class BatchProcessor:
    """Process multiple wells with parallel curve fitting."""

    def __init__(self, config: BatchConfig | None = None):
        """Initialize batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config or BatchConfig(products=["oil", "gas"])
        self.fitting_config = self.config.fitting_config or FittingConfig()

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

        successful = 0
        failed = 0
        all_errors = []
        processed_wells = []

        # Process in parallel
        workers = self.config.workers
        if workers is None:
            # Default to reasonable number
            import os
            workers = min(os.cpu_count() or 4, len(filtered_wells))

        # Use ProcessPoolExecutor for CPU-bound curve fitting
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _fit_single_well,
                    well,
                    self.config.products,
                    self.fitting_config
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
                    well, errors = future.result()
                    processed_wells.append(well)

                    if errors:
                        all_errors.extend((well_id, e) for e in errors)
                        # Still count as successful if at least one product worked
                        has_forecast = (
                            well.forecast_oil is not None or
                            well.forecast_gas is not None
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
            errors=all_errors
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
        # Export ARIES forecast
        exporter = AriesExporter()
        forecast_path = output_dir / "forecasts.csv"
        exporter.save(result.wells, forecast_path, self.config.products)
        logger.info(f"Saved ARIES forecast to {forecast_path}")

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
                                wells_with_forecast[:20],  # Limit for readability
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
