"""CLI commands for PyForecast."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from ..config import PyForecastConfig, generate_default_config

app = typer.Typer(
    name="pyforecast",
    help="Oil & Gas DCA Auto-Forecasting Tool",
    add_completion=False,
)


@app.command()
def process(
    input_files: Annotated[
        list[Path],
        typer.Argument(
            help="Input CSV/Excel file(s) with production data",
            exists=True,
        )
    ],
    output: Annotated[
        Path,
        typer.Option(
            "-o", "--output",
            help="Output directory for forecasts and plots",
        )
    ] = Path("output"),
    config: Annotated[
        Optional[Path],
        typer.Option(
            "-c", "--config",
            help="YAML config file (use 'pyforecast init' to generate template)",
            exists=True,
        )
    ] = None,
    product: Annotated[
        Optional[list[str]],
        typer.Option(
            "-p", "--product",
            help="Product(s) to forecast: oil, gas, water (overrides config)",
        )
    ] = None,
    workers: Annotated[
        Optional[int],
        typer.Option(
            "-w", "--workers",
            help="Number of parallel workers (default: auto)",
        )
    ] = None,
    no_plots: Annotated[
        bool,
        typer.Option(
            "--no-plots",
            help="Skip generating individual well plots",
        )
    ] = False,
    no_batch_plot: Annotated[
        bool,
        typer.Option(
            "--no-batch-plot",
            help="Skip generating batch overlay plot",
        )
    ] = False,
    export_format: Annotated[
        Optional[str],
        typer.Option(
            "--format",
            help="Export format: ac_economic or json (overrides config)",
        )
    ] = None,
    hindcast: Annotated[
        bool,
        typer.Option(
            "--hindcast",
            help="Run hindcast validation to measure forecast accuracy",
        )
    ] = False,
    log_fits: Annotated[
        bool,
        typer.Option(
            "--log-fits",
            help="Log fit metadata to persistent storage for analysis",
        )
    ] = False,
    residuals: Annotated[
        bool,
        typer.Option(
            "--residuals",
            help="Compute residual diagnostics for fit quality analysis",
        )
    ] = False,
    ground_truth: Annotated[
        Optional[Path],
        typer.Option(
            "--ground-truth",
            help="ARIES AC_ECONOMIC CSV file for ground truth comparison",
            exists=True,
        )
    ] = None,
    gt_months: Annotated[
        int,
        typer.Option(
            "--gt-months",
            help="Months to compare forecasts against ground truth",
        )
    ] = 60,
    gt_plots: Annotated[
        bool,
        typer.Option(
            "--gt-plots",
            help="Generate comparison plots for each well (requires matplotlib)",
        )
    ] = False,
    gt_lazy: Annotated[
        bool,
        typer.Option(
            "--gt-lazy",
            help="Stream ARIES file instead of loading into memory (for large files)",
        )
    ] = False,
    gt_workers: Annotated[
        int,
        typer.Option(
            "--gt-workers",
            help="Number of parallel workers for ground truth validation (default: 1 = sequential)",
        )
    ] = 1,
    checkpoint: Annotated[
        Optional[Path],
        typer.Option(
            "--checkpoint",
            help="Checkpoint file for resume capability. If the file exists, resumes from last checkpoint.",
        )
    ] = None,
) -> None:
    """Process production data and generate decline forecasts.

    Reads production data from CSV/Excel files (Enverus or ARIES format),
    fits hyperbolic decline curves with per-product parameters, and exports
    ARIES-compatible forecasts.

    Use a config file for per-product b-factor and dmin settings:
        pyforecast process data.csv --config settings.yaml

    Example:
        pyforecast process data.csv -o forecasts/ --product oil
    """
    from ..batch.processor import BatchProcessor, BatchConfig
    from ..core.fitting import FittingConfig

    # Load config file or use defaults
    if config:
        typer.echo(f"Loading config from {config}")
        pf_config = PyForecastConfig.from_yaml(config)
    else:
        pf_config = PyForecastConfig()

    # CLI overrides
    if product:
        valid_products = {"oil", "gas", "water"}
        products = [p.lower() for p in product]
        for p in products:
            if p not in valid_products:
                typer.echo(f"Error: Invalid product '{p}'. Must be oil, gas, or water.", err=True)
                raise typer.Exit(1)
        pf_config.output.products = products  # type: ignore

    if no_plots:
        pf_config.output.plots = False
    if no_batch_plot:
        pf_config.output.batch_plot = False
    if export_format:
        if export_format not in ("ac_economic", "json"):
            typer.echo(f"Error: Invalid format '{export_format}'. Must be ac_economic or json.", err=True)
            raise typer.Exit(1)
        pf_config.output.format = export_format  # type: ignore

    # Refinement flags override config
    if hindcast:
        pf_config.refinement.enable_hindcast = True
    if residuals:
        pf_config.refinement.enable_residual_analysis = True
    if log_fits:
        pf_config.refinement.enable_logging = True

    # Create batch config with per-product fitting configs
    batch_config = BatchConfig(
        products=pf_config.output.products,
        min_points=pf_config.fitting.min_points,
        workers=workers,
        fitting_config=None,  # Will use per-product configs
        pyforecast_config=pf_config,
        output_dir=output,
        save_plots=pf_config.output.plots,
        save_batch_plot=pf_config.output.batch_plot,
        export_format=pf_config.output.format,
    )

    # Run batch processing
    typer.echo(f"Processing {len(input_files)} file(s)...")
    typer.echo(f"Products: {', '.join(pf_config.output.products)}")
    processor = BatchProcessor(batch_config)

    if checkpoint:
        typer.echo(f"Checkpoint file: {checkpoint}")
        if checkpoint.exists():
            typer.echo("  Resuming from existing checkpoint...")
        result = processor.run_with_checkpoint(input_files, output, checkpoint_file=checkpoint)
    else:
        result = processor.run(input_files, output)

    # Post-processing steps
    if log_fits and result.wells:
        _handle_fit_logging(result, batch_config, pf_config)

    gt_results = _handle_ground_truth(
        result, pf_config, ground_truth, gt_months, gt_lazy, gt_workers,
    )

    _report_results(result, gt_results, output, gt_plots)


def _handle_fit_logging(result, batch_config, pf_config) -> None:
    """Log fit results to persistent storage."""
    from ..refinement.fit_logger import FitLogger

    typer.echo("Logging fit results...")
    with FitLogger() as fit_logger:
        for well in result.wells:
            for prod in pf_config.output.products:
                forecast = well.get_forecast(prod)
                if forecast is not None:
                    fitting_config = batch_config.get_fitting_config(prod)

                    hindcast_result = None
                    if result.refinement_results and result.refinement_results.hindcast_results:
                        hindcast_result = result.refinement_results.hindcast_results.get(
                            (well.well_id, prod)
                        )

                    residual_diag = None
                    if result.refinement_results and result.refinement_results.residual_diagnostics:
                        residual_diag = result.refinement_results.residual_diagnostics.get(
                            (well.well_id, prod)
                        )

                    fit_logger.log(
                        forecast, well, prod, fitting_config,
                        hindcast=hindcast_result, residuals=residual_diag,
                    )

    typer.echo(f"  Logged {fit_logger.storage.count()} fits to storage")


def _handle_ground_truth(
    result, pf_config, ground_truth, gt_months, gt_lazy, gt_workers,
) -> list:
    """Run ground truth comparison if configured."""
    gt_results: list = []

    gt_file = ground_truth or (
        Path(pf_config.refinement.ground_truth_file)
        if pf_config.refinement.ground_truth_file
        else None
    )
    if not gt_file or not result.wells:
        return gt_results

    gt_comparison_months = gt_months if ground_truth else pf_config.refinement.ground_truth_months

    from ..import_.aries_forecast import AriesForecastImporter
    from ..refinement.ground_truth import GroundTruthValidator, GroundTruthConfig

    typer.echo("Running ground truth comparison...")
    aries_importer = AriesForecastImporter(lazy=gt_lazy)
    try:
        count = aries_importer.load(gt_file)
        mode_str = " (lazy mode)" if gt_lazy else ""
        typer.echo(f"  Loaded {count} ARIES forecasts from {gt_file}{mode_str}")
    except (OSError, ValueError, KeyError) as e:
        typer.echo(f"  Error loading ARIES file: {e}", err=True)
        return gt_results

    gt_config = GroundTruthConfig(comparison_months=gt_comparison_months)
    gt_validator = GroundTruthValidator(aries_importer, gt_config)

    if gt_workers > 1:
        typer.echo(f"  Using {gt_workers} parallel workers")
        gt_summary_result = gt_validator.validate_batch_parallel(
            result.wells, pf_config.output.products, max_workers=gt_workers,
        )
        gt_results = gt_summary_result.results
    else:
        for well in result.wells:
            for prod in pf_config.output.products:
                gt_result = gt_validator.validate(well, prod)
                if gt_result is not None:
                    gt_results.append(gt_result)

    return gt_results


def _report_results(result, gt_results, output, gt_plots) -> None:
    """Report processing results to console."""
    typer.echo("")
    typer.echo("Results:")
    typer.echo(f"  Successful: {result.successful}")
    typer.echo(f"  Failed: {result.failed}")
    typer.echo(f"  Skipped (insufficient data): {result.skipped}")

    if result.errors:
        typer.echo(f"\n{len(result.errors)} error(s) occurred. See {output}/errors.txt")

    # Validation summary
    if result.validation_results:
        summary = result.get_validation_summary()
        typer.echo("")
        typer.echo("Validation Summary:")
        typer.echo(f"  Wells with errors: {summary['wells_with_errors']}")
        typer.echo(f"  Wells with warnings: {summary['wells_with_warnings']}")

        if summary["by_category"]:
            typer.echo("")
            typer.echo("  Issues by category:")
            for cat, count in sorted(summary["by_category"].items()):
                typer.echo(f"    {cat}: {count}")

        if summary['wells_with_errors'] > 0 or summary['wells_with_warnings'] > 0:
            typer.echo(f"\n  See {output}/validation_report.txt for details")

    # Refinement summary
    if result.refinement_results is not None:
        ref_results = result.refinement_results
        typer.echo("")
        typer.echo("Refinement Analysis:")

        if ref_results.hindcast_results:
            hindcast_summary = ref_results.get_hindcast_summary()
            typer.echo(f"  Hindcast validation: {hindcast_summary['count']} wells")
            typer.echo(f"    Average MAPE: {hindcast_summary['avg_mape']:.1f}%")
            typer.echo(f"    Good hindcast rate: {hindcast_summary['good_hindcast_pct']:.1f}%")

        if ref_results.residual_diagnostics:
            systematic = sum(
                1 for d in ref_results.residual_diagnostics.values()
                if d.has_systematic_pattern
            )
            total = len(ref_results.residual_diagnostics)
            typer.echo(f"  Residual analysis: {total} fits analyzed")
            pct = systematic / total * 100 if total > 0 else 0.0
            typer.echo(f"    With systematic patterns: {systematic} ({pct:.1f}%)")

        typer.echo(f"\n  See {output}/refinement_report.txt for details")

    # Ground truth summary
    if gt_results:
        from ..refinement.ground_truth import summarize_ground_truth_results

        gt_summary = summarize_ground_truth_results(gt_results)
        typer.echo("")
        typer.echo("Ground Truth Comparison:")
        typer.echo(f"  Wells with ARIES data: {gt_summary['count']} of {result.successful}")
        avg_mape = gt_summary['avg_mape']
        typer.echo(f"  Average MAPE: {f'{avg_mape:.1f}%' if avg_mape is not None else 'N/A'}")
        typer.echo(f"  Average correlation: {gt_summary['avg_correlation']:.3f}")
        typer.echo(f"  Good match rate: {gt_summary['good_match_pct']:.1f}%")

        _save_ground_truth_report(gt_results, gt_summary, output)
        _save_ground_truth_csv(gt_results, output)
        _save_ground_truth_timeseries(gt_results, output)
        typer.echo(f"\n  See {output}/ground_truth_report.txt for details")
        typer.echo(f"  CSV export: {output}/ground_truth_results.csv")
        typer.echo(f"  Time-series: {output}/ground_truth_timeseries.csv")

        if gt_plots:
            try:
                from ..refinement.plotting import plot_all_comparisons
                plot_count = plot_all_comparisons(gt_results, output)
                typer.echo(f"  Generated {plot_count} comparison plots in {output}/ground_truth_plots/")
            except ImportError:
                typer.echo("  Warning: matplotlib not installed, skipping plot generation", err=True)

    typer.echo(f"\nOutput saved to: {output}/")


@app.command()
def plot(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input CSV/Excel file with production data",
            exists=True,
        )
    ],
    well_id: Annotated[
        Optional[str],
        typer.Option(
            "--well-id", "--api",
            help="Specific well ID/API to plot (plots first well if not specified)",
        )
    ] = None,
    product: Annotated[
        str,
        typer.Option(
            "-p", "--product",
            help="Product to plot: oil, gas, or water",
        )
    ] = "oil",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o", "--output",
            help="Output file path (default: show in browser)",
        )
    ] = None,
    config: Annotated[
        Optional[Path],
        typer.Option(
            "-c", "--config",
            help="YAML config file",
            exists=True,
        )
    ] = None,
) -> None:
    """Plot decline curve for a single well.

    Loads production data, fits a decline curve using product-specific
    parameters from config, and displays an interactive semi-log plot.

    Example:
        pyforecast plot data.csv --well-id "42-001-00001" --product oil
    """
    from ..data.base import load_wells
    from ..core.fitting import DeclineFitter, FittingConfig
    from ..visualization.plots import DeclinePlotter

    # Load config
    if config:
        pf_config = PyForecastConfig.from_yaml(config)
    else:
        pf_config = PyForecastConfig()

    # Validate product
    product = product.lower()
    if product not in ("oil", "gas", "water"):
        typer.echo(f"Error: Invalid product '{product}'. Must be oil, gas, or water.", err=True)
        raise typer.Exit(1)

    # Load wells
    typer.echo(f"Loading data from {input_file}...")
    wells = load_wells(input_file)

    if not wells:
        typer.echo("Error: No wells found in file.", err=True)
        raise typer.Exit(1)

    # Find specified well or use first
    well = None
    if well_id:
        for w in wells:
            if (w.identifier.api == well_id or
                w.identifier.propnum == well_id or
                w.identifier.entity_id == well_id or
                w.identifier.well_name == well_id):
                well = w
                break
        if well is None:
            typer.echo(f"Error: Well '{well_id}' not found.", err=True)
            typer.echo(f"Available wells: {[w.well_id for w in wells[:10]]}...")
            raise typer.Exit(1)
    else:
        well = wells[0]
        typer.echo(f"Using first well: {well.well_id}")

    # Get product-specific fitting config
    fitting_config = FittingConfig.from_pyforecast_config(pf_config, product)

    # Fit decline curve
    typer.echo(f"Fitting {product} decline curve...")
    typer.echo(f"  b range: [{fitting_config.b_min}, {fitting_config.b_max}]")
    typer.echo(f"  Dmin: {fitting_config.dmin_annual:.1%}/year")

    fitter = DeclineFitter(fitting_config)

    try:
        t = well.production.time_months
        # Use daily rates for fitting to normalize for varying month lengths
        q = well.production.get_product_daily(product)  # type: ignore
        result = fitter.fit(t, q)
        well.set_forecast(product, result)  # type: ignore
    except ValueError as e:
        typer.echo(f"Error fitting curve: {e}", err=True)
        raise typer.Exit(1)

    # Report fit quality (qi is daily rate)
    unit = "bbl/day" if product in ("oil", "water") else "mcf/day"
    typer.echo(f"\nFit Results:")
    typer.echo(f"  qi: {result.model.qi:.1f} {unit}")
    typer.echo(f"  Di: {result.model.di * 12:.1%}/year")
    typer.echo(f"  b: {result.model.b:.3f}")
    typer.echo(f"  R²: {result.r_squared:.3f}")
    if result.regime_start_idx > 0:
        typer.echo(f"  Regime change detected at month {result.regime_start_idx}")

    # Create plot
    plotter = DeclinePlotter()
    fig = plotter.plot_well(well, product)  # type: ignore

    if output:
        plotter.save(fig, output)
        typer.echo(f"\nPlot saved to: {output}")
    else:
        fig.show()


@app.command()
def init(
    output: Annotated[
        Path,
        typer.Option(
            "-o", "--output",
            help="Output file path",
        )
    ] = Path("pyforecast.yaml"),
    profile: Annotated[
        Optional[str],
        typer.Option(
            "--profile",
            help="Configuration profile: quick, production, or research",
        )
    ] = None,
) -> None:
    """Generate a default configuration file.

    Creates a YAML config file with all available settings and their defaults.
    Edit this file to customize per-product b-factor and dmin settings.

    Profiles:
        quick      - Fast processing, minimal validation, no plots
        production - Balanced settings for daily operations (default)
        research   - All features enabled, maximum diagnostics

    Examples:
        pyforecast init -o my_config.yaml
        pyforecast init --profile quick -o quick_config.yaml
    """
    import shutil

    if output.exists():
        overwrite = typer.confirm(f"{output} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit(0)

    if profile:
        # Use profile template
        valid_profiles = {"quick", "production", "research"}
        if profile.lower() not in valid_profiles:
            typer.echo(f"Error: Invalid profile '{profile}'. Must be one of: {', '.join(valid_profiles)}", err=True)
            raise typer.Exit(1)

        # Find profile file - check multiple locations
        profile_name = profile.lower()
        profile_paths = [
            Path(__file__).parent.parent.parent.parent / "configs" / f"{profile_name}.yaml",
            Path(__file__).parent.parent / "configs" / f"{profile_name}.yaml",
        ]

        profile_path = None
        for p in profile_paths:
            if p.exists():
                profile_path = p
                break

        if profile_path is None:
            # Generate inline if profile files not found
            typer.echo(f"Using {profile_name} profile settings...")
            _generate_profile_config(output, profile_name)
        else:
            shutil.copy(profile_path, output)

        typer.echo(f"Config file created: {output} (profile: {profile_name})")
    else:
        generate_default_config(output)
        typer.echo(f"Config file created: {output}")

    typer.echo("\nEdit this file to customize settings, then use:")
    typer.echo(f"  pyforecast process data.csv --config {output}")


def _generate_profile_config(filepath: Path, profile: str) -> None:
    """Generate a profile-specific configuration file.

    Args:
        filepath: Output file path
        profile: Profile name (quick, production, research)
    """
    # Profile-specific settings
    profiles = {
        "quick": {
            "plots": "false",
            "batch_plot": "false",
            "gap_threshold_months": "3",
            "outlier_sigma": "4.0",
            "min_cv": "0.03",
            "min_r_squared": "0.4",
            "max_annual_decline": "1.5",
            "refinement_logging": "false",
            "refinement_hindcast": "false",
            "refinement_residuals": "false",
        },
        "production": {
            "plots": "true",
            "batch_plot": "true",
            "gap_threshold_months": "2",
            "outlier_sigma": "3.0",
            "min_cv": "0.05",
            "min_r_squared": "0.5",
            "max_annual_decline": "1.0",
            "refinement_logging": "false",
            "refinement_hindcast": "false",
            "refinement_residuals": "false",
        },
        "research": {
            "plots": "true",
            "batch_plot": "true",
            "gap_threshold_months": "2",
            "outlier_sigma": "3.0",
            "min_cv": "0.05",
            "min_r_squared": "0.5",
            "max_annual_decline": "1.0",
            "refinement_logging": "true",
            "refinement_hindcast": "true",
            "refinement_residuals": "true",
        },
    }

    settings = profiles[profile]

    content = f"""# PyForecast Configuration File
# Profile: {profile}
# Per-product decline curve fitting parameters

oil:
  b_min: 0.01      # Minimum b-factor (0.01 = near-exponential)
  b_max: 1.5       # Maximum b-factor (1.0 = harmonic, >1 = super-harmonic)
  dmin: 0.06       # Terminal decline rate (annual, 0.06 = 6%)

gas:
  b_min: 0.01
  b_max: 1.5
  dmin: 0.06

water:
  b_min: 0.01
  b_max: 1.5
  dmin: 0.06

# Regime change detection (RTP, refrac)
regime:
  threshold: 1.0          # Minimum increase to trigger (1.0 = 100%)
  window: 6               # Months of trend data to fit
  sustained_months: 2     # Months elevation must persist

# General fitting parameters
fitting:
  recency_half_life: 12.0  # Half-life for recent data weighting (months)
  min_points: 6            # Minimum months of data required

# Output options
output:
  products:               # Products to forecast
    - oil
    - gas
    - water
  plots: {settings['plots']}             # Generate individual well plots
  batch_plot: {settings['batch_plot']}        # Generate multi-well overlay plot
  format: ac_economic     # Export format: ac_economic or json

# Data validation settings
validation:
  max_oil_rate: 50000     # Max expected oil rate (bbl/mo) - IV002
  max_gas_rate: 500000    # Max expected gas rate (mcf/mo) - IV002
  max_water_rate: 100000  # Max expected water rate (bbl/mo) - IV002
  gap_threshold_months: {settings['gap_threshold_months']} # Min gap size to flag - DQ001
  outlier_sigma: {settings['outlier_sigma']}      # Std devs for outlier detection - DQ002
  shutin_threshold: 1.0   # Rate below = shut-in - DQ003
  min_cv: {settings['min_cv']}            # Min coefficient of variation - DQ004
  min_r_squared: {settings['min_r_squared']}      # Min acceptable R² - FR001
  max_annual_decline: {settings['max_annual_decline']} # Max annual decline rate - FR005
  strict_mode: false      # Treat warnings as errors

# Refinement settings (fit quality analysis)
refinement:
  enable_logging: {settings['refinement_logging']}           # Log fit metadata to storage
  log_storage: sqlite             # Storage type: sqlite or csv
  log_path: null                  # null = ~/.pyforecast/fit_logs.db
  enable_hindcast: {settings['refinement_hindcast']}          # Run hindcast validation
  hindcast_holdout_months: 6      # Months to hold out for validation
  min_training_months: 12         # Minimum training data required
  enable_residual_analysis: {settings['refinement_residuals']} # Compute residual diagnostics
  known_events_file: null         # CSV with known regime events
  enable_learning: false          # Enable parameter learning
  min_data_points_for_logging: 6  # Min data points to log fit
  max_coefficient_of_variation: 0 # Max CV for logging (0 = no limit)
  min_r_squared_for_logging: 0    # Min R² for logging (0 = no limit)
  ground_truth_file: null         # ARIES AC_ECONOMIC CSV for comparison
  ground_truth_months: 60         # Months to compare forecasts
  ground_truth_lazy: false        # Stream file instead of loading into memory
  ground_truth_workers: 1         # Parallel workers (1 = sequential)
"""

    with open(filepath, "w") as f:
        f.write(content)


@app.command()
def validate(
    input_files: Annotated[
        list[Path],
        typer.Argument(
            help="Input CSV/Excel file(s) with production data",
            exists=True,
        )
    ],
    config: Annotated[
        Optional[Path],
        typer.Option(
            "-c", "--config",
            help="YAML config file with validation settings",
            exists=True,
        )
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o", "--output",
            help="Output file for detailed validation report",
        )
    ] = None,
) -> None:
    """Validate production data without running forecasts.

    Loads wells from input files and runs input validation and data quality
    checks. Prints a summary to console and optionally writes a detailed
    report to file.

    Exit codes:
        0: No errors found (warnings may be present)
        1: Validation errors found

    Example:
        pyforecast validate data.csv --output report.txt
    """
    from ..data.base import load_wells
    from ..validation import (
        InputValidator,
        DataQualityValidator,
        ValidationResult,
        merge_results,
    )

    # Load config
    if config:
        typer.echo(f"Loading config from {config}")
        pf_config = PyForecastConfig.from_yaml(config)
    else:
        pf_config = PyForecastConfig()

    # Create validators from config
    input_validator = InputValidator(
        max_oil_rate=pf_config.validation.max_oil_rate,
        max_gas_rate=pf_config.validation.max_gas_rate,
        max_water_rate=pf_config.validation.max_water_rate,
    )
    quality_validator = DataQualityValidator(
        gap_threshold_months=pf_config.validation.gap_threshold_months,
        outlier_sigma=pf_config.validation.outlier_sigma,
        shutin_threshold=pf_config.validation.shutin_threshold,
        min_cv=pf_config.validation.min_cv,
    )

    # Load and validate all wells
    all_results: list[ValidationResult] = []
    total_wells = 0

    for input_file in input_files:
        typer.echo(f"Loading {input_file}...")
        try:
            wells = load_wells(input_file)
        except Exception as e:
            typer.echo(f"  Error loading file: {e}", err=True)
            continue

        typer.echo(f"  Found {len(wells)} wells")
        total_wells += len(wells)

        for well in wells:
            # Run input validation
            result = input_validator.validate(well)

            # Run data quality checks for each product
            for product in pf_config.output.products:
                try:
                    quality_result = quality_validator.validate(well, product)
                    result = result.merge(quality_result)
                except Exception:
                    pass  # Product may not be available

            all_results.append(result)

    # Summarize results
    total_errors = sum(r.error_count for r in all_results)
    total_warnings = sum(r.warning_count for r in all_results)
    wells_with_errors = sum(1 for r in all_results if r.has_errors)
    wells_with_warnings = sum(1 for r in all_results if r.has_warnings)

    typer.echo("")
    typer.echo("Validation Summary:")
    typer.echo(f"  Total wells: {total_wells}")
    typer.echo(f"  Wells with errors: {wells_with_errors}")
    typer.echo(f"  Wells with warnings: {wells_with_warnings}")
    typer.echo(f"  Total errors: {total_errors}")
    typer.echo(f"  Total warnings: {total_warnings}")

    # Count issues by code
    issue_counts: dict[str, int] = {}
    for result in all_results:
        for issue in result.issues:
            issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1

    if issue_counts:
        typer.echo("")
        typer.echo("Issues by code:")
        for code, count in sorted(issue_counts.items()):
            typer.echo(f"  {code}: {count}")

    # Write detailed report if requested
    if output:
        with open(output, "w") as f:
            f.write("PyForecast Validation Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Files: {[str(p) for p in input_files]}\n")
            f.write(f"Total wells: {total_wells}\n")
            f.write(f"Total errors: {total_errors}\n")
            f.write(f"Total warnings: {total_warnings}\n\n")

            for result in all_results:
                if result.issues:
                    f.write(f"\n{result.well_id}\n")
                    f.write("-" * 30 + "\n")
                    for issue in result.issues:
                        f.write(f"  [{issue.code}] {issue.severity.name}: {issue.message}\n")
                        f.write(f"    Guidance: {issue.guidance}\n")

        typer.echo(f"\nDetailed report written to: {output}")

    # Exit with error code if errors found
    if total_errors > 0:
        raise typer.Exit(1)


@app.command()
def info(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input CSV/Excel file to inspect",
            exists=True,
        )
    ],
) -> None:
    """Display information about a production data file.

    Shows detected format, well count, date range, and column names.
    """
    from ..data.base import DataParser, detect_parser

    typer.echo(f"Inspecting: {input_file}")
    typer.echo("")

    # Load file
    df = DataParser.load_file(input_file)
    typer.echo(f"Rows: {len(df)}")
    typer.echo(f"Columns: {list(df.columns)}")

    # Detect format
    try:
        parser = detect_parser(df)
        parser_name = type(parser).__name__
        typer.echo(f"Detected format: {parser_name}")

        # Parse and show summary
        wells = parser.parse(df)
        typer.echo(f"Wells found: {len(wells)}")

        if wells:
            typer.echo("\nSample wells:")
            for well in wells[:5]:
                prod = well.production
                typer.echo(
                    f"  {well.well_id}: "
                    f"{prod.n_months} months, "
                    f"{prod.first_date} to {prod.last_date}"
                )
            if len(wells) > 5:
                typer.echo(f"  ... and {len(wells) - 5} more")

    except ValueError as e:
        typer.echo(f"Format detection failed: {e}", err=True)
        raise typer.Exit(1)


@app.command("analyze-fits")
def analyze_fits(
    storage_path: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to fit logs database (default: ~/.pyforecast/fit_logs.db)",
        )
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o", "--output",
            help="Output CSV file for analysis report",
        )
    ] = None,
    basin: Annotated[
        Optional[str],
        typer.Option(
            "--basin",
            help="Filter by basin name",
        )
    ] = None,
    formation: Annotated[
        Optional[str],
        typer.Option(
            "--formation",
            help="Filter by formation name",
        )
    ] = None,
    product: Annotated[
        Optional[str],
        typer.Option(
            "-p", "--product",
            help="Filter by product (oil, gas, water)",
        )
    ] = None,
) -> None:
    """Analyze accumulated fit logs from previous runs.

    Reads fit log database and generates summary statistics, parameter
    distributions, and hindcast performance analysis.

    Example:
        pyforecast analyze-fits --basin "Permian" -o analysis.csv
    """
    from ..refinement.fit_logger import FitLogAnalyzer
    from ..refinement.storage import get_default_storage_path

    # Use default storage if not specified
    db_path = storage_path or get_default_storage_path()

    if not db_path.exists():
        typer.echo(f"Error: No fit logs found at {db_path}", err=True)
        typer.echo("Run 'pyforecast process --log-fits' to create fit logs.")
        raise typer.Exit(1)

    typer.echo(f"Analyzing fit logs from {db_path}")

    analyzer = FitLogAnalyzer(db_path)

    # Get summary statistics
    summary = analyzer.get_summary(basin=basin, formation=formation, product=product)

    if summary.get("count", 0) == 0:
        typer.echo("No matching fit logs found.")
        raise typer.Exit(0)

    typer.echo("")
    typer.echo("Fit Log Summary:")
    typer.echo(f"  Total fits: {summary.get('count', 0)}")
    if summary.get('avg_r_squared'):
        typer.echo(f"  Average R²: {summary['avg_r_squared']:.3f}")
    if summary.get('avg_rmse'):
        typer.echo(f"  Average RMSE: {summary['avg_rmse']:.2f}")

    # Get hindcast performance
    hindcast = analyzer.get_hindcast_performance(basin=basin, formation=formation, product=product)
    if hindcast.get("count", 0) > 0:
        typer.echo("")
        typer.echo("Hindcast Performance:")
        typer.echo(f"  Fits with hindcast: {hindcast['count']}")
        typer.echo(f"  Average MAPE: {hindcast['avg_mape']:.1f}%")
        typer.echo(f"  Good hindcast rate: {hindcast['good_hindcast_pct']:.1f}%")

    # Get parameter distribution
    params = analyzer.get_parameter_distribution(basin=basin, formation=formation, product=product)
    if "b" in params:
        typer.echo("")
        typer.echo("B-Factor Distribution:")
        typer.echo(f"  Mean: {params['b']['mean']:.3f}")
        typer.echo(f"  Std: {params['b']['std']:.3f}")
        typer.echo(f"  Range: [{params['b']['min']:.3f}, {params['b']['max']:.3f}]")

    # Export to CSV if requested
    if output:
        analyzer.export_report(output, basin=basin, formation=formation, product=product)
        typer.echo(f"\nAnalysis report saved to: {output}")


@app.command("suggest-params")
def suggest_params(
    storage_path: Annotated[
        Optional[Path],
        typer.Argument(
            help="Path to fit logs database (default: ~/.pyforecast/fit_logs.db)",
        )
    ] = None,
    basin: Annotated[
        Optional[str],
        typer.Option(
            "--basin",
            help="Get suggestion for specific basin",
        )
    ] = None,
    formation: Annotated[
        Optional[str],
        typer.Option(
            "--formation",
            help="Get suggestion for specific formation",
        )
    ] = None,
    product: Annotated[
        str,
        typer.Option(
            "-p", "--product",
            help="Product to get suggestions for",
        )
    ] = "oil",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o", "--output",
            help="Export all suggestions to CSV",
        )
    ] = None,
    update: Annotated[
        bool,
        typer.Option(
            "--update",
            help="Recompute all suggestions from current fit logs",
        )
    ] = False,
) -> None:
    """Get parameter suggestions learned from historical fits.

    Analyzes accumulated fit logs to suggest optimal fitting parameters
    (recency_half_life, regime_threshold, etc.) based on hindcast performance.

    Example:
        pyforecast suggest-params --basin "Permian" -p oil
    """
    from ..refinement.parameter_learning import ParameterLearner
    from ..refinement.storage import get_default_storage_path

    # Use default storage if not specified
    db_path = storage_path or get_default_storage_path()

    if not db_path.exists():
        typer.echo(f"Error: No fit logs found at {db_path}", err=True)
        typer.echo("Run 'pyforecast process --log-fits --hindcast' to create fit logs with hindcast data.")
        raise typer.Exit(1)

    learner = ParameterLearner(db_path)

    # Update suggestions if requested
    if update:
        typer.echo("Updating parameter suggestions from fit logs...")
        count = learner.update_suggestions(product=product)
        typer.echo(f"Updated {count} suggestions")
        typer.echo("")

    # Export all suggestions if requested
    if output:
        count = learner.export_suggestions(output)
        typer.echo(f"Exported {count} suggestions to {output}")
        return

    # Get specific suggestion
    suggestion = learner.suggest(product=product, basin=basin, formation=formation)

    if suggestion is None:
        typer.echo("No parameter suggestions available.")
        typer.echo("Need at least 10 fits with hindcast data to generate suggestions.")
        typer.echo("Run 'pyforecast process --log-fits --hindcast' on more wells.")
        raise typer.Exit(0)

    typer.echo(f"Parameter Suggestion for {suggestion.grouping}/{product}")
    typer.echo(f"  Based on {suggestion.sample_count} fits (confidence: {suggestion.confidence})")
    typer.echo("")
    typer.echo("Suggested values:")
    typer.echo(f"  recency_half_life: {suggestion.suggested_recency_half_life:.1f}")
    typer.echo(f"  regime_threshold: {suggestion.suggested_regime_threshold:.2f}")
    typer.echo(f"  regime_window: {suggestion.suggested_regime_window}")
    typer.echo(f"  regime_sustained_months: {suggestion.suggested_regime_sustained_months}")
    typer.echo("")
    typer.echo("Historical performance with these parameters:")
    typer.echo(f"  Average R²: {suggestion.avg_r_squared:.3f}")
    if suggestion.avg_hindcast_mape:
        typer.echo(f"  Average hindcast MAPE: {suggestion.avg_hindcast_mape:.1f}%")


@app.command("calibrate-regime")
def calibrate_regime(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input CSV/Excel file with production data",
            exists=True,
        )
    ],
    events_file: Annotated[
        Path,
        typer.Option(
            "--events",
            help="CSV file with known events (well_id, event_date, event_type)",
            exists=True,
        )
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o", "--output",
            help="Output JSON file for calibration results",
        )
    ] = None,
    product: Annotated[
        str,
        typer.Option(
            "-p", "--product",
            help="Product to calibrate",
        )
    ] = "oil",
) -> None:
    """Calibrate regime detection from known refrac/workover events.

    Reads a CSV of known events (refracs, workovers) and evaluates
    different regime detection thresholds to find optimal settings.

    Events CSV format:
        well_id,event_date,event_type
        42-001-00001,2022-06-15,refrac
        42-001-00002,2023-01-20,workover

    Example:
        pyforecast calibrate-regime data.csv --events known_events.csv -o calibration.json
    """
    import json
    import csv
    from datetime import datetime

    import pandas as pd

    from ..data.base import load_wells
    from ..core.fitting import DeclineFitter, FittingConfig

    typer.echo(f"Loading production data from {input_file}...")
    wells = load_wells(input_file)
    wells_by_id = {w.well_id: w for w in wells}

    typer.echo(f"Loading known events from {events_file}...")
    events = []
    with open(events_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append({
                "well_id": row["well_id"],
                "event_date": row["event_date"],
                "event_type": row.get("event_type", "regime_change"),
            })

    typer.echo(f"Found {len(events)} known events")

    # Test different threshold values
    thresholds = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    results = []

    typer.echo("")
    typer.echo("Testing regime detection thresholds...")

    for threshold in thresholds:
        config = FittingConfig(regime_threshold=threshold)
        fitter = DeclineFitter(config)

        true_positives = 0
        false_negatives = 0
        total_events = 0

        for event in events:
            well_id = event["well_id"]
            if well_id not in wells_by_id:
                continue

            well = wells_by_id[well_id]
            t = well.production.time_months
            q = well.production.get_product_daily(product)

            if len(q) < 12 or q.max() < 0.1:
                continue

            total_events += 1

            # Find event month index
            event_date = datetime.strptime(event["event_date"], "%Y-%m-%d")
            dates = well.production.dates
            event_idx = None
            for i, d in enumerate(dates):
                dt = pd.Timestamp(d)
                if dt.year == event_date.year and dt.month == event_date.month:
                    event_idx = i
                    break

            if event_idx is None:
                continue

            # Check if regime detection finds it
            regime_start = fitter.detect_regime_change(q)

            # Consider it detected if within 3 months of actual event
            if abs(regime_start - event_idx) <= 3:
                true_positives += 1
            else:
                false_negatives += 1

        detection_rate = true_positives / total_events * 100 if total_events > 0 else 0
        results.append({
            "threshold": threshold,
            "total_events": total_events,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "detection_rate": detection_rate,
        })

        typer.echo(f"  threshold={threshold:.2f}: {detection_rate:.1f}% detection rate "
                   f"({true_positives}/{total_events})")

    # Find best threshold
    best = max(results, key=lambda r: r["detection_rate"])
    typer.echo("")
    typer.echo(f"Recommended threshold: {best['threshold']:.2f} "
               f"({best['detection_rate']:.1f}% detection rate)")

    # Save results
    if output:
        with open(output, "w") as f:
            json.dump({
                "recommended_threshold": best["threshold"],
                "results": results,
            }, f, indent=2)
        typer.echo(f"\nCalibration results saved to: {output}")


def _save_ground_truth_timeseries(
    results: list,
    output_dir: Path,
) -> None:
    """Save ground truth time-series data for plotting.

    Exports the forecast arrays (ARIES vs pyforecast) to CSV
    for external visualization.

    Args:
        results: List of GroundTruthResult objects
        output_dir: Output directory
    """
    import csv

    csv_path = output_dir / "ground_truth_timeseries.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "well_id",
            "product",
            "month",
            "aries_rate",
            "pyf_rate",
            "diff",
            "pct_diff",
        ])

        for r in results:
            # Skip if arrays are empty
            if len(r.forecast_months) == 0:
                continue

            for i, month in enumerate(r.forecast_months):
                aries_rate = r.aries_rates[i]
                pyf_rate = r.pyf_rates[i]
                diff = pyf_rate - aries_rate
                pct_diff = (diff / aries_rate * 100) if aries_rate > 0.1 else 0.0

                writer.writerow([
                    r.well_id,
                    r.product,
                    int(month),
                    round(aries_rate, 4),
                    round(pyf_rate, 4),
                    round(diff, 4),
                    round(pct_diff, 2),
                ])


def _save_ground_truth_csv(
    results: list,
    output_dir: Path,
) -> None:
    """Save ground truth comparison results to CSV file.

    Args:
        results: List of GroundTruthResult objects
        output_dir: Output directory
    """
    import csv

    csv_path = output_dir / "ground_truth_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "well_id",
            "product",
            "match_grade",
            "is_good_match",
            "mape",
            "correlation",
            "bias",
            "cumulative_diff_pct",
            "aries_qi",
            "aries_di_annual",
            "aries_b",
            "aries_decline_type",
            "pyf_qi",
            "pyf_di_annual",
            "pyf_b",
            "qi_pct_diff",
            "di_pct_diff",
            "b_abs_diff",
            "comparison_months",
            "aries_start_date",
            "pyf_start_date",
            "alignment_warning",
        ])

        for r in results:
            writer.writerow([
                r.well_id,
                r.product,
                r.match_grade,
                r.is_good_match,
                round(r.mape, 2) if r.mape is not None else "",
                round(r.correlation, 4),
                round(r.bias, 4),
                round(r.cumulative_diff_pct, 2),
                round(r.aries_qi, 2),
                round(r.aries_di * 12, 4),  # Convert to annual
                round(r.aries_b, 3),
                r.aries_decline_type,
                round(r.pyf_qi, 2),
                round(r.pyf_di * 12, 4),  # Convert to annual
                round(r.pyf_b, 3),
                round(r.qi_pct_diff, 2),
                round(r.di_pct_diff, 2),
                round(r.b_abs_diff, 3),
                r.comparison_months,
                r.aries_start_date.isoformat() if r.aries_start_date else "",
                r.pyf_start_date.isoformat() if r.pyf_start_date else "",
                r.alignment_warning or "",
            ])


def _save_ground_truth_report(
    results: list,
    summary: dict,
    output_dir: Path,
) -> None:
    """Save ground truth comparison report to file.

    Args:
        results: List of GroundTruthResult objects
        summary: Summary statistics dictionary
        output_dir: Output directory
    """
    report_path = output_dir / "ground_truth_report.txt"

    # Count alignment warnings
    alignment_warnings = [r for r in results if r.alignment_warning]

    with open(report_path, "w") as f:
        f.write("PyForecast Ground Truth Comparison Report\n")
        f.write("=" * 50 + "\n\n")

        # Overall summary
        f.write("Summary:\n")
        f.write(f"  Total comparisons: {summary['count']}\n")
        f.write(f"  Average MAPE: {summary['avg_mape']:.1f}%\n")
        f.write(f"  Median MAPE: {summary['median_mape']:.1f}%\n")
        f.write(f"  Average correlation: {summary['avg_correlation']:.3f}\n")
        f.write(f"  Average cumulative diff: {summary['avg_cumulative_diff_pct']:.1f}%\n")
        f.write(f"  Good match rate: {summary['good_match_pct']:.1f}%\n")
        if alignment_warnings:
            f.write(f"  Wells with date misalignment: {len(alignment_warnings)} of {summary['count']}\n")
        f.write("\n")

        # Grade distribution
        f.write("Grade Distribution:\n")
        grades = summary.get("grade_distribution", {})
        for grade in ["A", "B", "C", "D"]:
            count = grades.get(grade, 0)
            pct = count / summary['count'] * 100 if summary['count'] > 0 else 0
            f.write(f"  {grade}: {count} ({pct:.1f}%)\n")
        f.write("\n")

        # By product
        by_product = summary.get("by_product", {})
        if by_product:
            f.write("By Product:\n")
            for product, stats in sorted(by_product.items()):
                f.write(f"  {product}: {stats['count']} wells, "
                       f"avg MAPE {stats['avg_mape']:.1f}%, "
                       f"good match {stats['good_match_pct']:.1f}%\n")
            f.write("\n")

        # Detailed results
        f.write("Detailed Results:\n")
        f.write("-" * 50 + "\n")

        # Sort by match grade then MAPE (handle None MAPE)
        sorted_results = sorted(
            results,
            key=lambda r: (r.match_grade, r.mape if r.mape is not None else float('inf'))
        )

        for r in sorted_results:
            status = "GOOD" if r.is_good_match else "----"
            f.write(f"\n{r.well_id}/{r.product} [{r.match_grade}] {status}\n")
            f.write(f"  ARIES:      qi={r.aries_qi:.1f}, di={r.aries_di*12:.1%}/yr, b={r.aries_b:.3f}\n")
            f.write(f"  pyforecast: qi={r.pyf_qi:.1f}, di={r.pyf_di*12:.1%}/yr, b={r.pyf_b:.3f}\n")
            f.write(f"  Differences: qi={r.qi_pct_diff:+.1f}%, di={r.di_pct_diff:+.1f}%, b={r.b_abs_diff:+.3f}\n")
            mape_str = f"{r.mape:.1f}%" if r.mape is not None else "N/A"
            f.write(f"  Metrics: MAPE={mape_str}, corr={r.correlation:.3f}, "
                   f"bias={r.bias:.1%}, cum_diff={r.cumulative_diff_pct:+.1f}%\n")
            if r.alignment_warning:
                f.write(f"  Warning: {r.alignment_warning}\n")


if __name__ == "__main__":
    app()
