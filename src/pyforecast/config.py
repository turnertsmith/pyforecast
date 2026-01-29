"""Configuration file support for PyForecast.

Supports YAML config files with per-product fitting parameters.
CLI flags override config file values.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class ProductConfig:
    """Fitting parameters for a single product.

    Attributes:
        b_min: Minimum b-factor (default 0.01)
        b_max: Maximum b-factor (default 1.5)
        dmin: Terminal decline rate, annual fraction (default 0.06 = 6%)
    """
    b_min: float = 0.01
    b_max: float = 1.5
    dmin: float = 0.06


@dataclass
class RegimeConfig:
    """Regime detection parameters.

    Attributes:
        threshold: Minimum fractional increase to detect regime change (default 1.0 = 100%)
        window: Months of data for trend fitting (default 6)
        sustained_months: Months elevation must persist to confirm (default 2)
    """
    threshold: float = 1.0
    window: int = 6
    sustained_months: int = 2


@dataclass
class FittingDefaults:
    """Default fitting parameters applied to all products.

    Attributes:
        recency_half_life: Half-life in months for exponential decay weighting (default 12)
        min_points: Minimum data points required for fitting (default 6)
    """
    recency_half_life: float = 12.0
    min_points: int = 6


@dataclass
class OutputConfig:
    """Output configuration.

    Attributes:
        products: Products to forecast (default: oil, gas)
        plots: Generate individual well plots (default: True)
        batch_plot: Generate batch overlay plot (default: True)
        format: Export format - 'ac_forecast' or 'ac_economic' (default: ac_economic)
    """
    products: list[Literal["oil", "gas", "water"]] = field(default_factory=lambda: ["oil", "gas"])
    plots: bool = True
    batch_plot: bool = True
    format: Literal["ac_forecast", "ac_economic"] = "ac_economic"


@dataclass
class PyForecastConfig:
    """Complete PyForecast configuration.

    Attributes:
        oil: Oil-specific fitting parameters
        gas: Gas-specific fitting parameters
        water: Water-specific fitting parameters
        regime: Regime detection parameters
        fitting: General fitting parameters
        output: Output configuration
    """
    oil: ProductConfig = field(default_factory=ProductConfig)
    gas: ProductConfig = field(default_factory=ProductConfig)
    water: ProductConfig = field(default_factory=ProductConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    fitting: FittingDefaults = field(default_factory=FittingDefaults)
    output: OutputConfig = field(default_factory=OutputConfig)

    def get_product_config(self, product: Literal["oil", "gas", "water"]) -> ProductConfig:
        """Get configuration for a specific product."""
        if product == "oil":
            return self.oil
        elif product == "gas":
            return self.gas
        elif product == "water":
            return self.water
        else:
            raise ValueError(f"Unknown product: {product}")

    @classmethod
    def from_yaml(cls, filepath: Path | str) -> "PyForecastConfig":
        """Load configuration from YAML file.

        Args:
            filepath: Path to YAML config file

        Returns:
            PyForecastConfig instance
        """
        filepath = Path(filepath)
        with open(filepath) as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "PyForecastConfig":
        """Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            PyForecastConfig instance
        """
        config = cls()

        # Product configs
        if "oil" in data:
            config.oil = ProductConfig(**data["oil"])
        if "gas" in data:
            config.gas = ProductConfig(**data["gas"])
        if "water" in data:
            config.water = ProductConfig(**data["water"])

        # Regime config
        if "regime" in data:
            config.regime = RegimeConfig(**data["regime"])

        # Fitting defaults
        if "fitting" in data:
            config.fitting = FittingDefaults(**data["fitting"])

        # Output config
        if "output" in data:
            output_data = data["output"].copy()
            # Handle products list
            if "products" in output_data:
                output_data["products"] = list(output_data["products"])
            config.output = OutputConfig(**output_data)

        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "oil": {
                "b_min": self.oil.b_min,
                "b_max": self.oil.b_max,
                "dmin": self.oil.dmin,
            },
            "gas": {
                "b_min": self.gas.b_min,
                "b_max": self.gas.b_max,
                "dmin": self.gas.dmin,
            },
            "water": {
                "b_min": self.water.b_min,
                "b_max": self.water.b_max,
                "dmin": self.water.dmin,
            },
            "regime": {
                "threshold": self.regime.threshold,
                "window": self.regime.window,
                "sustained_months": self.regime.sustained_months,
            },
            "fitting": {
                "recency_half_life": self.fitting.recency_half_life,
                "min_points": self.fitting.min_points,
            },
            "output": {
                "products": self.output.products,
                "plots": self.output.plots,
                "batch_plot": self.output.batch_plot,
                "format": self.output.format,
            },
        }

    def to_yaml(self, filepath: Path | str) -> None:
        """Save configuration to YAML file.

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def generate_default_config(filepath: Path | str) -> Path:
    """Generate a default configuration file.

    Args:
        filepath: Output file path

    Returns:
        Path to created file
    """
    filepath = Path(filepath)
    config = PyForecastConfig()

    # Add comments by writing manually
    content = """# PyForecast Configuration File
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
  plots: true             # Generate individual well plots
  batch_plot: true        # Generate multi-well overlay plot
  format: ac_economic     # Export format: ac_forecast or ac_economic
"""

    with open(filepath, "w") as f:
        f.write(content)

    return filepath
