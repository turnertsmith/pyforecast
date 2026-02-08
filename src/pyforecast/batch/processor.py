"""Batch processing for multiple wells with parallel execution."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Literal, TYPE_CHECKING
import json
import logging

from tqdm import tqdm

from ..data.well import Well
from ..data.base import load_wells
from ..core.fitting import DeclineFitter, FittingConfig
from ..core.models import ForecastResult
from ..validation import (
    ValidationResult,
    InputValidator,
    DataQualityValidator,
    FittingValidator,
    IssueCategory,
    IssueSeverity,
)

# Lazy imports for decomposed modules to avoid circular imports
# These are imported in methods that need them

if TYPE_CHECKING:
    from ..config import PyForecastConfig, ValidationConfig, RefinementConfig

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
        export_format: Export format (ac_economic or json)
    """
    products: list[Literal["oil", "gas", "water"]]
    min_points: int = 6
    workers: int | None = None
    fitting_config: FittingConfig | None = None
    pyforecast_config: "PyForecastConfig | None" = None
    output_dir: Path | None = None
    save_plots: bool = True
    save_batch_plot: bool = True
    export_format: Literal["ac_economic", "json"] = "ac_economic"

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
class RefinementResults:
    """Results from refinement analysis.

    Attributes:
        hindcast_results: Dict of (well_id, product) -> HindcastResult
        residual_diagnostics: Dict of (well_id, product) -> ResidualDiagnostics
        ground_truth_results: Dict of (well_id, product) -> GroundTruthResult
        fit_logs_count: Number of fit logs recorded
    """
    hindcast_results: dict = field(default_factory=dict)
    residual_diagnostics: dict = field(default_factory=dict)
    ground_truth_results: dict = field(default_factory=dict)
    fit_logs_count: int = 0

    def get_hindcast_summary(self) -> dict:
        """Get summary of hindcast results."""
        if not self.hindcast_results:
            return {"count": 0}

        import numpy as np
        mapes = [r.mape for r in self.hindcast_results.values()]
        correlations = [r.correlation for r in self.hindcast_results.values()]
        good_count = sum(1 for r in self.hindcast_results.values() if r.is_good_hindcast)

        return {
            "count": len(mapes),
            "avg_mape": float(np.mean(mapes)),
            "median_mape": float(np.median(mapes)),
            "avg_correlation": float(np.mean(correlations)),
            "good_hindcast_pct": good_count / len(mapes) * 100 if mapes else 0,
        }

    def get_ground_truth_summary(self) -> dict:
        """Get summary of ground truth comparison results."""
        if not self.ground_truth_results:
            return {"count": 0}

        from ..refinement.ground_truth import summarize_ground_truth_results
        results = list(self.ground_truth_results.values())
        return summarize_ground_truth_results(results)


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
        refinement_results: Optional refinement analysis results
    """
    wells: list[Well]
    successful: int
    failed: int
    skipped: int
    errors: list[tuple[str, str]]
    validation_results: dict[str, ValidationResult] = field(default_factory=dict)
    refinement_results: RefinementResults | None = None

    def get_validation_summary(self) -> dict:
        """Get summary of validation results.

        Returns:
            Dict with counts by severity and category
        """
        from ..validation.result import summarize_validation
        return summarize_validation(self.validation_results)


@dataclass
class RefinementOptions:
    """Options for refinement in single-well fitting.

    Attributes:
        enable_hindcast: Run hindcast validation
        enable_residuals: Capture residuals for analysis
        hindcast_holdout_months: Months to hold out for hindcast
        min_training_months: Minimum training months for hindcast
    """
    enable_hindcast: bool = False
    enable_residuals: bool = False
    hindcast_holdout_months: int = 6
    min_training_months: int = 12


@dataclass
class CheckpointState:
    """State for resumable batch processing.

    Captures the progress of a batch job so it can be resumed after failure.

    Attributes:
        total_wells: Total number of wells to process
        processed_well_ids: Set of well IDs that have been processed
        successful: Count of successfully fitted wells
        failed: Count of failed fits
        skipped: Count of wells skipped (insufficient data)
        errors: List of (well_id, error_message) tuples
        started_at: Timestamp when processing started
        last_updated_at: Timestamp of last checkpoint update
    """
    total_wells: int = 0
    processed_well_ids: set[str] = field(default_factory=set)
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_wells": self.total_wells,
            "processed_well_ids": list(self.processed_well_ids),
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "started_at": self.started_at,
            "last_updated_at": self.last_updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointState":
        """Create from dictionary."""
        return cls(
            total_wells=data.get("total_wells", 0),
            processed_well_ids=set(data.get("processed_well_ids", [])),
            successful=data.get("successful", 0),
            failed=data.get("failed", 0),
            skipped=data.get("skipped", 0),
            errors=[(e[0], e[1]) for e in data.get("errors", [])],
            started_at=data.get("started_at", datetime.now().isoformat()),
            last_updated_at=data.get("last_updated_at", datetime.now().isoformat()),
        )

    def save(self, filepath: Path) -> None:
        """Save checkpoint to JSON file."""
        self.last_updated_at = datetime.now().isoformat()
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "CheckpointState":
        """Load checkpoint from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @property
    def progress_pct(self) -> float:
        """Percentage of wells processed."""
        if self.total_wells == 0:
            return 0.0
        return len(self.processed_well_ids) / self.total_wells * 100


@dataclass
class SingleWellRefinementResult:
    """Refinement results for a single well.

    Attributes:
        hindcast_results: Dict of product -> HindcastResult
        residual_diagnostics: Dict of product -> ResidualDiagnostics
    """
    hindcast_results: dict = field(default_factory=dict)
    residual_diagnostics: dict = field(default_factory=dict)


def _fit_single_well(
    well: Well,
    products: list[str],
    product_configs: dict[str, FittingConfig],
    validation_config: "ValidationConfig | None" = None,
    refinement_options: RefinementOptions | None = None,
) -> tuple[Well, list[str], ValidationResult, SingleWellRefinementResult | None]:
    """Fit decline curves for a single well (worker function).

    Args:
        well: Well object with production data
        products: Products to fit
        product_configs: Dict of product -> FittingConfig
        validation_config: Optional validation configuration
        refinement_options: Optional refinement options

    Returns:
        Tuple of (well with forecasts, list of error messages, validation result, refinement result)
    """
    errors = []
    validation_result = ValidationResult(well_id=well.well_id)
    refinement_result = None

    # Initialize refinement components if enabled
    hindcast_validator = None
    residual_analyzer = None
    if refinement_options is not None:
        if refinement_options.enable_hindcast:
            from ..refinement.hindcast import HindcastValidator
            hindcast_validator = HindcastValidator(
                holdout_months=refinement_options.hindcast_holdout_months,
                min_training_months=refinement_options.min_training_months,
            )
        if refinement_options.enable_residuals:
            from ..refinement.residual_analysis import ResidualAnalyzer
            residual_analyzer = ResidualAnalyzer()

        refinement_result = SingleWellRefinementResult()

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
            # Use daily rates for fitting to normalize for varying month lengths
            q = well.production.get_product_daily(product)

            # Skip if no significant production (daily rate threshold)
            if q.max() < 0.1:
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

            # Determine if we need residuals
            capture_residuals = residual_analyzer is not None

            # Fit the decline curve
            result = fitter.fit(t, q, capture_residuals=capture_residuals)
            well.set_forecast(product, result)

            # Run post-fit validation
            if product in fitting_validators:
                validation_result = validation_result.merge(
                    fitting_validators[product].validate_fit_result(
                        result, well.well_id, product
                    )
                )

            # Run hindcast validation if enabled
            if hindcast_validator is not None and refinement_result is not None:
                hindcast = hindcast_validator.validate(well, product, fitter)
                if hindcast is not None:
                    refinement_result.hindcast_results[product] = hindcast

            # Run residual analysis if enabled
            if residual_analyzer is not None and refinement_result is not None:
                diagnostics, residual_validation = residual_analyzer.analyze_and_validate(
                    t, q, result, well.well_id, product
                )
                refinement_result.residual_diagnostics[product] = diagnostics
                validation_result = validation_result.merge(residual_validation)

        except ValueError as e:
            errors.append(f"{product}: {str(e)}")
        except (RuntimeError, TypeError, ArithmeticError) as e:
            errors.append(f"{product}: Unexpected error - {str(e)}")

    return well, errors, validation_result, refinement_result


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
            except (ValueError, OSError, KeyError) as e:
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

    def _build_processing_context(self) -> tuple[
        dict[str, FittingConfig],
        "ValidationConfig | None",
        RefinementOptions | None,
        bool,
    ]:
        """Build shared processing context from config.

        Returns:
            Tuple of (product_configs, validation_config, refinement_options, refinement_enabled)
        """
        product_configs = {
            product: self.config.get_fitting_config(product)
            for product in self.config.products
        }

        validation_config = None
        if self.config.pyforecast_config is not None:
            validation_config = self.config.pyforecast_config.validation

        refinement_options = None
        refinement_enabled = False
        if self.config.pyforecast_config is not None:
            ref_config = self.config.pyforecast_config.refinement
            if ref_config.enable_hindcast or ref_config.enable_residual_analysis:
                refinement_enabled = True
                refinement_options = RefinementOptions(
                    enable_hindcast=ref_config.enable_hindcast,
                    enable_residuals=ref_config.enable_residual_analysis,
                    hindcast_holdout_months=ref_config.hindcast_holdout_months,
                    min_training_months=ref_config.min_training_months,
                )

        return product_configs, validation_config, refinement_options, refinement_enabled

    def _resolve_workers(self, n_wells: int) -> int:
        """Resolve number of parallel workers.

        Args:
            n_wells: Number of wells to process

        Returns:
            Number of workers to use
        """
        if self.config.workers is not None:
            return self.config.workers
        import os
        return min(os.cpu_count() or 4, n_wells)

    def _submit_wells(
        self,
        executor: ProcessPoolExecutor,
        wells: list[Well],
        product_configs: dict[str, FittingConfig],
        validation_config,
        refinement_options: RefinementOptions | None,
    ) -> dict:
        """Submit wells to executor for parallel processing.

        Args:
            executor: ProcessPoolExecutor instance
            wells: Wells to process
            product_configs: Per-product fitting configs
            validation_config: Validation configuration
            refinement_options: Refinement options

        Returns:
            Dict mapping futures to well IDs
        """
        return {
            executor.submit(
                _fit_single_well,
                well,
                self.config.products,
                product_configs,
                validation_config,
                refinement_options,
            ): well.well_id
            for well in wells
        }

    @staticmethod
    def _collect_refinement(
        refinement_result: SingleWellRefinementResult | None,
        all_refinement_results: RefinementResults | None,
        well_id: str,
    ) -> None:
        """Collect refinement results from a single well into the batch results.

        Args:
            refinement_result: Single well refinement result
            all_refinement_results: Batch-level refinement results accumulator
            well_id: Well identifier
        """
        if refinement_result is not None and all_refinement_results is not None:
            for product, hindcast in refinement_result.hindcast_results.items():
                all_refinement_results.hindcast_results[(well_id, product)] = hindcast
            for product, diagnostics in refinement_result.residual_diagnostics.items():
                all_refinement_results.residual_diagnostics[(well_id, product)] = diagnostics

    def _tally_future_result(
        self,
        future,
        well_id: str,
        processed_wells: list,
        all_validation_results: dict,
        all_refinement_results: RefinementResults | None,
        errors_list: list,
    ) -> tuple[bool, bool]:
        """Process a completed future and tally the result.

        Args:
            future: Completed Future object
            well_id: Well identifier
            processed_wells: List to append processed well to
            all_validation_results: Dict to store validation results
            all_refinement_results: Optional RefinementResults to collect into
            errors_list: List to append errors to

        Returns:
            Tuple of (is_success, is_failure) - exactly one will be True
        """
        try:
            well, errors, validation_result, refinement_result = future.result()
            processed_wells.append(well)
            all_validation_results[well_id] = validation_result
            self._collect_refinement(refinement_result, all_refinement_results, well_id)

            if errors:
                errors_list.extend((well_id, e) for e in errors)
                has_forecast = any(
                    well.get_forecast(p) is not None
                    for p in self.config.products
                )
                return has_forecast, not has_forecast
            return True, False

        except Exception as e:
            errors_list.append((well_id, str(e)))
            logger.error(f"Failed to process {well_id}: {e}")
            return False, True

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
        filtered_wells, skipped = self.filter_wells(wells)

        if not filtered_wells:
            return BatchResult(
                wells=[], successful=0, failed=0, skipped=skipped, errors=[],
            )

        product_configs, validation_config, refinement_options, refinement_enabled = (
            self._build_processing_context()
        )

        successful = 0
        failed = 0
        all_errors: list = []
        processed_wells: list = []
        all_validation_results: dict = {}
        all_refinement_results = RefinementResults() if refinement_enabled else None

        workers = self._resolve_workers(len(filtered_wells))

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = self._submit_wells(
                executor, filtered_wells, product_configs,
                validation_config, refinement_options,
            )

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Fitting wells")

            for future in iterator:
                well_id = futures[future]
                is_success, is_failure = self._tally_future_result(
                    future, well_id, processed_wells,
                    all_validation_results, all_refinement_results, all_errors,
                )
                successful += is_success
                failed += is_failure

        return BatchResult(
            wells=processed_wells,
            successful=successful,
            failed=failed,
            skipped=skipped,
            errors=all_errors,
            validation_results=all_validation_results,
            refinement_results=all_refinement_results,
        )

    def process_with_checkpoint(
        self,
        wells: list[Well],
        checkpoint_file: Path | str,
        show_progress: bool = True,
        progress_callback: Callable[[int, int, str], None] | None = None,
        checkpoint_interval: int = 10,
    ) -> BatchResult:
        """Process wells with checkpoint/resume capability.

        If processing fails partway through, can resume from the last checkpoint
        by passing the same checkpoint file.

        Args:
            wells: List of wells to process
            checkpoint_file: Path to checkpoint file (JSON)
            show_progress: Whether to show progress bar
            progress_callback: Optional callback(processed, total, well_id) for progress
            checkpoint_interval: Save checkpoint every N wells processed

        Returns:
            BatchResult with processed wells and statistics
        """
        checkpoint_file = Path(checkpoint_file)

        # Load or create checkpoint
        if checkpoint_file.exists():
            checkpoint = CheckpointState.load(checkpoint_file)
            logger.info(
                f"Resuming from checkpoint: {len(checkpoint.processed_well_ids)}/{checkpoint.total_wells} "
                f"wells already processed ({checkpoint.progress_pct:.1f}%)"
            )
        else:
            checkpoint = CheckpointState(total_wells=len(wells))

        # Filter wells - skip already processed and insufficient data
        remaining_wells = []
        for well in wells:
            if well.well_id in checkpoint.processed_well_ids:
                continue
            if well.has_sufficient_data(self.config.min_points):
                remaining_wells.append(well)
            else:
                checkpoint.skipped += 1
                checkpoint.processed_well_ids.add(well.well_id)
                logger.debug(f"Skipping {well.well_id}: insufficient data")

        if not remaining_wells:
            return BatchResult(
                wells=[], successful=checkpoint.successful, failed=checkpoint.failed,
                skipped=checkpoint.skipped, errors=checkpoint.errors,
            )

        product_configs, validation_config, refinement_options, refinement_enabled = (
            self._build_processing_context()
        )

        processed_wells: list = []
        all_validation_results: dict = {}
        all_refinement_results = RefinementResults() if refinement_enabled else None
        wells_since_checkpoint = 0

        workers = self._resolve_workers(len(remaining_wells))

        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = self._submit_wells(
                    executor, remaining_wells, product_configs,
                    validation_config, refinement_options,
                )

                iterator = as_completed(futures)
                if show_progress:
                    iterator = tqdm(
                        iterator, total=len(futures), desc="Fitting wells",
                        initial=len(checkpoint.processed_well_ids) - checkpoint.skipped,
                    )

                for future in iterator:
                    well_id = futures[future]
                    is_success, is_failure = self._tally_future_result(
                        future, well_id, processed_wells,
                        all_validation_results, all_refinement_results,
                        checkpoint.errors,
                    )
                    checkpoint.processed_well_ids.add(well_id)
                    checkpoint.successful += is_success
                    checkpoint.failed += is_failure

                    if progress_callback:
                        progress_callback(
                            len(checkpoint.processed_well_ids),
                            checkpoint.total_wells,
                            well_id,
                        )

                    wells_since_checkpoint += 1
                    if wells_since_checkpoint >= checkpoint_interval or is_failure:
                        checkpoint.save(checkpoint_file)
                        wells_since_checkpoint = 0

        except KeyboardInterrupt:
            logger.warning("Processing interrupted. Saving checkpoint...")
            checkpoint.save(checkpoint_file)
            raise

        # Final checkpoint save
        checkpoint.save(checkpoint_file)

        return BatchResult(
            wells=processed_wells,
            successful=checkpoint.successful,
            failed=checkpoint.failed,
            skipped=checkpoint.skipped,
            errors=checkpoint.errors,
            validation_results=all_validation_results,
            refinement_results=all_refinement_results,
        )

    def _run_pipeline(
        self,
        input_files: list[Path | str],
        output_dir: Path | str | None,
        process_fn,
        **process_kwargs,
    ) -> BatchResult:
        """Shared pipeline for run() and run_with_checkpoint().

        Args:
            input_files: Input file paths
            output_dir: Output directory (overrides config)
            process_fn: Processing method to call with wells
            **process_kwargs: Additional kwargs for process_fn

        Returns:
            BatchResult with processed wells
        """
        output_dir = Path(output_dir) if output_dir else self.config.output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading wells from {len(input_files)} file(s)")
        wells = self.load_files(input_files)
        logger.info(f"Loaded {len(wells)} total wells")

        result = process_fn(wells, **process_kwargs)
        logger.info(
            f"Processing complete: {result.successful} successful, "
            f"{result.failed} failed, {result.skipped} skipped"
        )

        if output_dir:
            self._save_outputs(result, output_dir)

        return result

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
        return self._run_pipeline(
            input_files, output_dir, self.process,
            show_progress=show_progress,
        )

    def run_with_checkpoint(
        self,
        input_files: list[Path | str],
        output_dir: Path | str | None = None,
        checkpoint_file: Path | str = "checkpoint.json",
        show_progress: bool = True,
    ) -> BatchResult:
        """Run batch processing with checkpoint/resume capability.

        Args:
            input_files: Input file paths
            output_dir: Output directory (overrides config)
            checkpoint_file: Path to checkpoint file (JSON)
            show_progress: Whether to show progress bars

        Returns:
            BatchResult with processed wells
        """
        return self._run_pipeline(
            input_files, output_dir, self.process_with_checkpoint,
            checkpoint_file=checkpoint_file,
            show_progress=show_progress,
        )

    def _save_outputs(self, result: BatchResult, output_dir: Path) -> None:
        """Save all outputs to directory.

        Uses BatchExporter for forecast export and BatchVisualizer for plots.
        This method orchestrates the output process while delegating actual
        work to specialized classes.

        Args:
            result: Batch processing result
            output_dir: Output directory
        """
        from .exporter import BatchExporter
        from .visualizer import BatchVisualizer

        # Export forecasts using BatchExporter
        exporter = BatchExporter(
            export_format=self.config.export_format,
            pyforecast_config=self.config.pyforecast_config,
        )
        exporter.export_forecasts(
            wells=result.wells,
            output_dir=output_dir,
            products=self.config.products,
            validation_results=result.validation_results,
        )

        # Save plots using BatchVisualizer
        if self.config.save_plots or self.config.save_batch_plot:
            visualizer = BatchVisualizer()
            visualizer.save_all_plots(
                wells=result.wells,
                output_dir=output_dir,
                products=self.config.products,
                save_individual=self.config.save_plots,
                save_batch=self.config.save_batch_plot,
            )

        # Save error log
        if result.errors:
            exporter.export_errors(result.errors, output_dir)

        # Save validation report
        if result.validation_results:
            exporter.export_validation_report(result.validation_results, output_dir)

        # Save refinement report if enabled
        if result.refinement_results is not None:
            self._save_refinement_report(result.refinement_results, output_dir)

    def _save_refinement_report(
        self,
        refinement_results: RefinementResults,
        output_dir: Path,
    ) -> None:
        """Save refinement analysis report to file.

        Args:
            refinement_results: Refinement results
            output_dir: Output directory
        """
        report_path = output_dir / "refinement_report.txt"

        with open(report_path, "w") as f:
            f.write("PyForecast Refinement Analysis Report\n")
            f.write("=" * 40 + "\n\n")

            if refinement_results.hindcast_results:
                self._write_hindcast_section(f, refinement_results)

            if refinement_results.residual_diagnostics:
                self._write_residual_section(f, refinement_results)

            if refinement_results.ground_truth_results:
                self._write_ground_truth_section(f, refinement_results)

        logger.info(f"Saved refinement report to {report_path}")

    @staticmethod
    def _write_hindcast_section(f, refinement_results: RefinementResults) -> None:
        """Write hindcast validation section to report file."""
        summary = refinement_results.get_hindcast_summary()
        f.write("Hindcast Validation Summary:\n")
        f.write(f"  Wells with hindcast: {summary['count']}\n")
        f.write(f"  Average MAPE: {summary['avg_mape']:.1f}%\n")
        f.write(f"  Median MAPE: {summary['median_mape']:.1f}%\n")
        f.write(f"  Average correlation: {summary['avg_correlation']:.3f}\n")
        f.write(f"  Good hindcast rate: {summary['good_hindcast_pct']:.1f}%\n\n")

        f.write("Hindcast Details:\n")
        f.write("-" * 40 + "\n")
        for (well_id, product), hindcast in sorted(refinement_results.hindcast_results.items()):
            status = "GOOD" if hindcast.is_good_hindcast else "POOR"
            f.write(
                f"  {well_id}/{product}: MAPE={hindcast.mape:.1f}%, "
                f"corr={hindcast.correlation:.3f}, bias={hindcast.bias:.1%} [{status}]\n"
            )
        f.write("\n")

    @staticmethod
    def _write_residual_section(f, refinement_results: RefinementResults) -> None:
        """Write residual diagnostics section to report file."""
        systematic_count = sum(
            1 for d in refinement_results.residual_diagnostics.values()
            if d.has_systematic_pattern
        )
        total = len(refinement_results.residual_diagnostics)
        f.write("Residual Analysis Summary:\n")
        f.write(f"  Total analyzed: {total}\n")
        pct = systematic_count / total * 100 if total > 0 else 0.0
        f.write(f"  With systematic patterns: {systematic_count} ({pct:.1f}%)\n\n")

        f.write("Wells with systematic residual patterns:\n")
        f.write("-" * 40 + "\n")
        for (well_id, product), diag in sorted(refinement_results.residual_diagnostics.items()):
            if diag.has_systematic_pattern:
                f.write(
                    f"  {well_id}/{product}: DW={diag.durbin_watson:.2f}, "
                    f"early_bias={diag.early_bias:.1%}, late_bias={diag.late_bias:.1%}\n"
                )
        f.write("\n")

    @staticmethod
    def _write_ground_truth_section(f, refinement_results: RefinementResults) -> None:
        """Write ground truth comparison section to report file."""
        gt_summary = refinement_results.get_ground_truth_summary()
        f.write("Ground Truth Comparison Summary:\n")
        f.write(f"  Wells with ARIES data: {gt_summary['count']}\n")
        avg_mape = gt_summary['avg_mape']
        f.write(f"  Average MAPE: {f'{avg_mape:.1f}%' if avg_mape is not None else 'N/A'}\n")
        f.write(f"  Average correlation: {gt_summary['avg_correlation']:.3f}\n")
        f.write(f"  Good match rate: {gt_summary['good_match_pct']:.1f}%\n\n")

        grades = gt_summary.get("grade_distribution", {})
        f.write("  Grade distribution:\n")
        for grade in ["A", "B", "C", "D"]:
            count = grades.get(grade, 0)
            f.write(f"    {grade}: {count}\n")
        f.write("\n")

        f.write("Ground Truth Details:\n")
        f.write("-" * 40 + "\n")
        for (well_id, product), gt_result in sorted(refinement_results.ground_truth_results.items()):
            status = "GOOD" if gt_result.is_good_match else "----"
            mape_str = f"{gt_result.mape:.1f}%" if gt_result.mape is not None else "N/A"
            f.write(
                f"  {well_id}/{product}: [{gt_result.match_grade}] {status} "
                f"MAPE={mape_str}, corr={gt_result.correlation:.3f}, "
                f"cum_diff={gt_result.cumulative_diff_pct:+.1f}%\n"
            )
