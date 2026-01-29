"""Plotting utilities for ground truth comparison.

Generates overlay plots comparing ARIES vs pyforecast decline curves.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import GroundTruthResult


def plot_ground_truth_comparison(
    result: "GroundTruthResult",
    output_path: Path,
    show_metrics: bool = True,
) -> None:
    """Generate overlay plot of ARIES vs pyforecast curves.

    Creates a plot showing both forecasts side-by-side with metrics
    displayed in a text box.

    Args:
        result: GroundTruthResult with forecast arrays
        output_path: Path to save the plot image
        show_metrics: Whether to display metrics text box

    Raises:
        ImportError: If matplotlib is not installed
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both curves
    ax.plot(
        result.forecast_months,
        result.aries_rates,
        'b-',
        label='ARIES',
        linewidth=2,
    )
    ax.plot(
        result.forecast_months,
        result.pyf_rates,
        'r--',
        label='pyforecast',
        linewidth=2,
    )

    # Configure axes
    ax.set_xlabel('Months')
    ax.set_ylabel('Rate')
    ax.set_title(f'{result.well_id} - {result.product.upper()}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add metrics text box if requested
    if show_metrics:
        mape_str = f"{result.mape:.1f}%" if result.mape is not None else "N/A"
        metrics_text = (
            f"MAPE: {mape_str}\n"
            f"Corr: {result.correlation:.3f}\n"
            f"Grade: {result.match_grade}"
        )
        ax.text(
            0.95, 0.95,
            metrics_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontfamily='monospace',
            fontsize=10,
        )

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_all_comparisons(
    results: list["GroundTruthResult"],
    output_dir: Path,
) -> int:
    """Generate plots for all comparisons.

    Creates a subdirectory with individual plots for each well/product
    combination.

    Args:
        results: List of GroundTruthResult objects
        output_dir: Base output directory

    Returns:
        Number of plots generated
    """
    plots_dir = output_dir / "ground_truth_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for r in results:
        # Skip results without forecast arrays
        if len(r.forecast_months) == 0:
            continue

        # Create safe filename from well ID
        safe_well_id = r.well_id.replace("/", "_").replace("\\", "_")
        filename = f"{safe_well_id}_{r.product}.png"

        plot_ground_truth_comparison(r, plots_dir / filename)
        count += 1

    return count
