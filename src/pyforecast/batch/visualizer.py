"""Batch visualization functionality for forecast results.

This module provides the BatchVisualizer class which handles generating
plots for individual wells and batch overview plots.

Extracted from BatchProcessor to follow single responsibility principle.
"""

import logging
from pathlib import Path
from typing import Literal

from ..data.well import Well
from ..visualization.plots import DeclinePlotter

logger = logging.getLogger(__name__)


class BatchVisualizer:
    """Generate visualizations for batch processing results.

    Handles creating individual well plots and batch overlay plots
    for production data and fitted decline curves.

    Attributes:
        plotter: DeclinePlotter instance for creating plots
        max_wells_batch: Maximum wells to include in batch overlay plot

    Example:
        >>> visualizer = BatchVisualizer()
        >>> visualizer.save_individual_plots(
        ...     wells=wells,
        ...     output_dir=Path("output/plots"),
        ...     products=["oil", "gas"],
        ... )
        >>> visualizer.save_batch_plot(
        ...     wells=wells,
        ...     output_dir=Path("output/plots"),
        ...     products=["oil"],
        ... )
    """

    def __init__(
        self,
        plotter: DeclinePlotter | None = None,
        max_wells_batch: int = 20,
    ):
        """Initialize batch visualizer.

        Args:
            plotter: Optional DeclinePlotter instance (creates new if None)
            max_wells_batch: Maximum wells to include in batch overlay plot
        """
        self.plotter = plotter or DeclinePlotter()
        self.max_wells_batch = max_wells_batch

    def save_individual_plots(
        self,
        wells: list[Well],
        output_dir: Path,
        products: list[Literal["oil", "gas", "water"]],
    ) -> list[Path]:
        """Save individual well plots.

        Creates one HTML plot per well/product combination where a forecast
        exists.

        Args:
            wells: List of wells with forecasts
            output_dir: Output directory for plots
            products: Products to plot

        Returns:
            List of paths to created plot files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        created_plots = []

        for well in wells:
            for product in products:
                if well.get_forecast(product) is not None:
                    try:
                        fig = self.plotter.plot_well(well, product)
                        # Sanitize filename (replace / with _)
                        filename = f"{well.well_id}_{product}.html".replace("/", "_")
                        filepath = output_dir / filename
                        self.plotter.save(fig, filepath)
                        created_plots.append(filepath)
                    except Exception as e:
                        logger.warning(f"Failed to plot {well.well_id}/{product}: {e}")

        logger.info(f"Created {len(created_plots)} individual plots in {output_dir}")
        return created_plots

    def save_batch_plot(
        self,
        wells: list[Well],
        output_dir: Path,
        products: list[Literal["oil", "gas", "water"]],
    ) -> list[Path]:
        """Save batch overlay plots.

        Creates one plot per product showing multiple wells overlaid
        for comparison. Limited to max_wells_batch wells per plot.

        Args:
            wells: List of wells with forecasts
            output_dir: Output directory for plots
            products: Products to plot

        Returns:
            List of paths to created plot files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        created_plots = []

        for product in products:
            # Filter to wells with forecast for this product
            wells_with_forecast = [
                w for w in wells if w.get_forecast(product) is not None
            ]

            if not wells_with_forecast:
                continue

            try:
                # Limit to max wells
                wells_to_plot = wells_with_forecast[:self.max_wells_batch]
                fig = self.plotter.plot_multiple_wells(wells_to_plot, product)
                filepath = output_dir / f"batch_{product}.html"
                self.plotter.save(fig, filepath)
                created_plots.append(filepath)

                if len(wells_with_forecast) > self.max_wells_batch:
                    logger.info(
                        f"Batch plot for {product} shows {self.max_wells_batch} of "
                        f"{len(wells_with_forecast)} wells"
                    )

            except Exception as e:
                logger.warning(f"Failed to create batch plot for {product}: {e}")

        logger.info(f"Created {len(created_plots)} batch plots in {output_dir}")
        return created_plots

    def save_all_plots(
        self,
        wells: list[Well],
        output_dir: Path,
        products: list[Literal["oil", "gas", "water"]],
        save_individual: bool = True,
        save_batch: bool = True,
    ) -> dict[str, list[Path]]:
        """Save all plots (individual and batch).

        Convenience method to generate all visualization outputs.

        Args:
            wells: List of wells with forecasts
            output_dir: Output directory for plots
            products: Products to plot
            save_individual: Whether to save individual well plots
            save_batch: Whether to save batch overlay plots

        Returns:
            Dict with 'individual' and 'batch' keys mapping to lists of paths
        """
        plots_dir = output_dir / "plots"
        result = {"individual": [], "batch": []}

        if save_individual:
            result["individual"] = self.save_individual_plots(
                wells, plots_dir, products
            )

        if save_batch:
            result["batch"] = self.save_batch_plot(
                wells, plots_dir, products
            )

        return result
