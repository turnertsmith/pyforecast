#!/usr/bin/env python3
"""Generate sample ground truth comparison plots with actual production data."""

import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "-q"])
    import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
    import matplotlib.pyplot as plt


def hyperbolic_decline(t, qi, di, b):
    """Calculate hyperbolic decline rate at time t."""
    if b < 0.001:  # Near-exponential
        return qi * np.exp(-di * t)
    return qi / (1 + b * di * t) ** (1 / b)


def generate_sample_plot(
    well_id: str,
    product: str,
    aries_qi: float,
    aries_di: float,
    aries_b: float,
    pyf_qi: float,
    pyf_di: float,
    pyf_b: float,
    mape: float,
    correlation: float,
    grade: str,
    actual_months: int,
    noise_factor: float,
    output_dir: Path,
):
    """Generate a comparison plot with actual production data."""
    # Forecast months
    forecast_months = np.arange(0, 60)

    # Generate forecast curves
    aries_rates = hyperbolic_decline(forecast_months, aries_qi, aries_di, aries_b)
    pyf_rates = hyperbolic_decline(forecast_months, pyf_qi, pyf_di, pyf_b)

    # Generate actual production data (historical)
    # Use the pyforecast parameters as "truth" with some noise
    actual_t = np.arange(0, actual_months)

    # Add noise to simulate real production
    np.random.seed(hash(well_id) % 2**32)
    noise = 1 + noise_factor * np.random.randn(actual_months)
    noise = np.clip(noise, 0.5, 1.5)  # Limit noise range

    # Actuals based on a blend - simulate wells declining from historical peak
    actual_qi = pyf_qi * 1.1  # Production was slightly higher at start
    actual_di = pyf_di * 0.95  # Decline slightly different
    actual_b = pyf_b
    actual_rates = hyperbolic_decline(actual_t, actual_qi, actual_di, actual_b) * noise

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot actual production as scatter points
    ax.scatter(actual_t, actual_rates, c='#6B7280', s=30, alpha=0.7,
               label='Actual Production', zorder=3, edgecolors='white', linewidths=0.5)

    # Plot ARIES forecast
    ax.plot(forecast_months, aries_rates, 'b-', linewidth=2.5, label='ARIES Forecast', zorder=2)

    # Plot pyforecast
    ax.plot(forecast_months, pyf_rates, 'r--', linewidth=2.5, label='PyForecast', zorder=2)

    # Styling
    ax.set_xlabel('Months', fontsize=12)
    ax.set_ylabel(f'{product.title()} Rate (bbl/mo)' if product == 'oil' else f'{product.title()} Rate (mcf/mo)',
                  fontsize=12)
    ax.set_title(f'{well_id} - {product.upper()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, max(max(aries_rates), max(pyf_rates), max(actual_rates)) * 1.1)

    # Add metrics box in upper right
    mape_str = f"{mape:.1f}%" if mape is not None else "N/A"
    metrics_text = f"MAPE: {mape_str}\nCorr: {correlation:.3f}\nGrade: {grade}"

    # Color based on grade
    grade_colors = {'A': '#10B981', 'B': '#3B82F6', 'C': '#F59E0B', 'D': '#EF4444', 'X': '#6B7280'}
    box_color = grade_colors.get(grade, '#6B7280')

    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=box_color, linewidth=2)
    ax.text(0.97, 0.97, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=props, fontfamily='monospace')

    # Legend in lower right to avoid overlap with metrics box
    ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

    # Add vertical line showing end of historical data
    ax.axvline(x=actual_months - 1, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(actual_months - 1, ax.get_ylim()[1] * 0.95, ' History',
            fontsize=9, color='gray', alpha=0.7)

    plt.tight_layout()

    # Save
    output_path = output_dir / f"{well_id}_{product}.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"  Generated: {output_path.name}")
    return output_path


def main():
    """Generate sample plots with actual production data."""
    output_dir = Path("/Users/turnersmith/pyforecast/drive/sample_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating sample comparison plots with actual production...")
    print()

    # Sample 1: Grade A - Excellent match
    generate_sample_plot(
        well_id="WELL-001",
        product="oil",
        aries_qi=1000.0,
        aries_di=0.08,
        aries_b=0.50,
        pyf_qi=1012.0,
        pyf_di=0.082,
        pyf_b=0.52,
        mape=5.2,
        correlation=0.998,
        grade="A",
        actual_months=24,
        noise_factor=0.08,
        output_dir=output_dir,
    )

    # Sample 2: Grade B - Good match
    generate_sample_plot(
        well_id="WELL-002",
        product="oil",
        aries_qi=800.0,
        aries_di=0.10,
        aries_b=0.60,
        pyf_qi=845.0,
        pyf_di=0.095,
        pyf_b=0.55,
        mape=12.8,
        correlation=0.985,
        grade="B",
        actual_months=18,
        noise_factor=0.12,
        output_dir=output_dir,
    )

    # Sample 3: Grade C - Fair match (gas well)
    generate_sample_plot(
        well_id="WELL-003",
        product="gas",
        aries_qi=5000.0,
        aries_di=0.12,
        aries_b=0.40,
        pyf_qi=5500.0,
        pyf_di=0.10,
        pyf_b=0.55,
        mape=22.5,
        correlation=0.945,
        grade="C",
        actual_months=20,
        noise_factor=0.15,
        output_dir=output_dir,
    )

    # Sample 4: Grade D - Poor match
    generate_sample_plot(
        well_id="WELL-004",
        product="oil",
        aries_qi=600.0,
        aries_di=0.15,
        aries_b=0.30,
        pyf_qi=750.0,
        pyf_di=0.08,
        pyf_b=0.70,
        mape=35.2,
        correlation=0.892,
        grade="D",
        actual_months=15,
        noise_factor=0.18,
        output_dir=output_dir,
    )

    print()
    print(f"All plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
