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
            help="Export format: ac_forecast or ac_economic (overrides config)",
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
        if export_format not in ("ac_forecast", "ac_economic"):
            typer.echo(f"Error: Invalid format '{export_format}'.", err=True)
            raise typer.Exit(1)
        pf_config.output.format = export_format  # type: ignore

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
    result = processor.run(input_files, output)

    # Report results
    typer.echo("")
    typer.echo("Results:")
    typer.echo(f"  Successful: {result.successful}")
    typer.echo(f"  Failed: {result.failed}")
    typer.echo(f"  Skipped (insufficient data): {result.skipped}")

    if result.errors:
        typer.echo(f"\n{len(result.errors)} error(s) occurred. See {output}/errors.txt")

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
        q = well.production.get_product(product)  # type: ignore
        result = fitter.fit(t, q)
        well.set_forecast(product, result)  # type: ignore
    except ValueError as e:
        typer.echo(f"Error fitting curve: {e}", err=True)
        raise typer.Exit(1)

    # Report fit quality
    typer.echo(f"\nFit Results:")
    typer.echo(f"  qi: {result.model.qi:.1f}")
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
) -> None:
    """Generate a default configuration file.

    Creates a YAML config file with all available settings and their defaults.
    Edit this file to customize per-product b-factor and dmin settings.

    Example:
        pyforecast init -o my_config.yaml
    """
    if output.exists():
        overwrite = typer.confirm(f"{output} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit(0)

    generate_default_config(output)
    typer.echo(f"Config file created: {output}")
    typer.echo("\nEdit this file to customize settings, then use:")
    typer.echo(f"  pyforecast process data.csv --config {output}")


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


if __name__ == "__main__":
    app()
