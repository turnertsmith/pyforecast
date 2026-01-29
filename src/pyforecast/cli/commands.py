"""CLI commands for PyForecast."""

from pathlib import Path
from typing import Annotated, Optional

import typer

from ..core.fitting import FittingConfig
from ..batch.processor import BatchProcessor, BatchConfig

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
    product: Annotated[
        list[str],
        typer.Option(
            "-p", "--product",
            help="Product(s) to forecast: oil, gas, or both",
        )
    ] = ["oil", "gas"],
    min_points: Annotated[
        int,
        typer.Option(
            "--min-points",
            help="Minimum months of production data required",
        )
    ] = 6,
    workers: Annotated[
        Optional[int],
        typer.Option(
            "-w", "--workers",
            help="Number of parallel workers (default: auto)",
        )
    ] = None,
    b_min: Annotated[
        float,
        typer.Option(
            "--b-min",
            help="Minimum b-factor for hyperbolic decline",
        )
    ] = 0.01,
    b_max: Annotated[
        float,
        typer.Option(
            "--b-max",
            help="Maximum b-factor for hyperbolic decline",
        )
    ] = 1.5,
    dmin: Annotated[
        float,
        typer.Option(
            "--dmin",
            help="Terminal decline rate (annual fraction, e.g., 0.06 = 6%)",
        )
    ] = 0.06,
    regime_threshold: Annotated[
        float,
        typer.Option(
            "--regime-threshold",
            help="Threshold for regime change detection (fraction, e.g., 1.0 = 100%)",
        )
    ] = 1.0,
    recency_half_life: Annotated[
        float,
        typer.Option(
            "--recency-half-life",
            help="Half-life (months) for exponential decay weighting of recent data",
        )
    ] = 12.0,
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
) -> None:
    """Process production data and generate decline forecasts.

    Reads production data from CSV/Excel files (Enverus or ARIES format),
    fits hyperbolic decline curves, and exports ARIES-compatible forecasts.

    Example:
        pyforecast process data.csv -o forecasts/ --product oil
    """
    # Validate products
    valid_products = {"oil", "gas"}
    products = [p.lower() for p in product]
    for p in products:
        if p not in valid_products:
            typer.echo(f"Error: Invalid product '{p}'. Must be 'oil' or 'gas'.", err=True)
            raise typer.Exit(1)

    # Create configurations
    fitting_config = FittingConfig(
        b_min=b_min,
        b_max=b_max,
        dmin_annual=dmin,
        regime_threshold=regime_threshold,
        recency_half_life=recency_half_life,
        min_points=min_points,
    )

    batch_config = BatchConfig(
        products=products,  # type: ignore
        min_points=min_points,
        workers=workers,
        fitting_config=fitting_config,
        output_dir=output,
        save_plots=not no_plots,
        save_batch_plot=not no_batch_plot,
    )

    # Run batch processing
    typer.echo(f"Processing {len(input_files)} file(s)...")
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
            help="Product to plot: oil or gas",
        )
    ] = "oil",
    output: Annotated[
        Optional[Path],
        typer.Option(
            "-o", "--output",
            help="Output file path (default: show in browser)",
        )
    ] = None,
    b_min: Annotated[
        float,
        typer.Option("--b-min", help="Minimum b-factor")
    ] = 0.01,
    b_max: Annotated[
        float,
        typer.Option("--b-max", help="Maximum b-factor")
    ] = 1.5,
    dmin: Annotated[
        float,
        typer.Option("--dmin", help="Terminal decline rate (annual)")
    ] = 0.06,
    regime_threshold: Annotated[
        float,
        typer.Option("--regime-threshold", help="Regime change threshold")
    ] = 1.0,
    recency_half_life: Annotated[
        float,
        typer.Option("--recency-half-life", help="Recency weighting half-life (months)")
    ] = 12.0,
) -> None:
    """Plot decline curve for a single well.

    Loads production data, fits a decline curve, and displays an interactive
    semi-log plot with historical production and forecast.

    Example:
        pyforecast plot data.csv --well-id "42-001-00001" --product oil
    """
    from ..data.base import load_wells
    from ..core.fitting import DeclineFitter, FittingConfig
    from ..visualization.plots import DeclinePlotter

    # Validate product
    if product.lower() not in ("oil", "gas"):
        typer.echo(f"Error: Invalid product '{product}'. Must be 'oil' or 'gas'.", err=True)
        raise typer.Exit(1)

    product = product.lower()

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

    # Fit decline curve
    typer.echo(f"Fitting {product} decline curve...")
    config = FittingConfig(
        b_min=b_min,
        b_max=b_max,
        dmin_annual=dmin,
        regime_threshold=regime_threshold,
        recency_half_life=recency_half_life,
    )
    fitter = DeclineFitter(config)

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
