"""Configuration file support for PyForecast.

Supports YAML config files with per-product fitting parameters.
CLI flags override config file values.
"""

from dataclasses import asdict, dataclass, field, fields as dataclass_fields
import logging
from pathlib import Path
from typing import ClassVar, Literal

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ProductConfig:
    """Fitting parameters for a single product.

    Attributes:
        b_min: Minimum b-factor (default 0.01)
        b_max: Maximum b-factor (default 1.5)
        dmin: Terminal decline rate, annual fraction (default 0.06 = 6%)
        recency_half_life: Product-specific half-life for recency weighting (months).
            If None, uses FittingDefaults.recency_half_life. Suggested defaults:
            Oil=12, Gas=9, Water=18 months.
    """
    b_min: float = 0.01
    b_max: float = 1.5
    dmin: float = 0.06
    recency_half_life: float | None = None


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
        model_selection: Model selection strategy - 'hyperbolic', 'exponential',
            'harmonic', or 'auto' to try all and select best by BIC (default 'hyperbolic')
        estimate_dmin: Estimate terminal decline from late-time data (default False)
        dmin_min_annual: Minimum Dmin when estimating (default 0.024 = 2.4%/yr)
        dmin_max_annual: Maximum Dmin when estimating (default 0.24 = 24%/yr)
        adaptive_regime_detection: Use CV-adaptive thresholds for regime detection (default False)
    """
    recency_half_life: float = 12.0
    min_points: int = 6
    model_selection: str = "hyperbolic"
    estimate_dmin: bool = False
    dmin_min_annual: float = 0.024
    dmin_max_annual: float = 0.24
    adaptive_regime_detection: bool = False


@dataclass
class OutputConfig:
    """Output configuration.

    Attributes:
        products: Products to forecast (default: oil, gas)
        plots: Generate individual well plots (default: True)
        batch_plot: Generate batch overlay plot (default: True)
        format: Export format - 'ac_economic' or 'json' (default: ac_economic)
    """
    products: list[Literal["oil", "gas", "water"]] = field(default_factory=lambda: ["oil", "gas", "water"])
    plots: bool = True
    batch_plot: bool = True
    format: Literal["ac_economic", "json"] = "ac_economic"


@dataclass
class RefinementConfig:
    """Refinement configuration for fit quality analysis and learning.

    All refinement features are disabled by default - zero impact on
    existing workflows. Enable features as needed for analysis.

    Attributes:
        enable_logging: Log fit metadata to persistent storage
        log_storage: Storage type - "sqlite" or "csv"
        log_path: Path to storage file (None = ~/.pyforecast/fit_logs.db)
        enable_hindcast: Run hindcast validation during fitting
        hindcast_holdout_months: Months to hold out for hindcast validation
        min_training_months: Minimum training data required for hindcast
        enable_residual_analysis: Compute residual diagnostics
        known_events_file: CSV file with known regime events for calibration
        enable_learning: Enable parameter learning/suggestions
        min_data_points_for_logging: Minimum data points required to log a fit
        max_coefficient_of_variation: Maximum CV to accept fit for logging (0 = no limit)
        min_r_squared_for_logging: Minimum R-squared to accept fit for logging (0 = no limit)
        ground_truth_file: ARIES AC_ECONOMIC CSV file for ground truth comparison
        ground_truth_months: Number of months to compare forecasts (default 60)
    """

    # Logging settings
    enable_logging: bool = False
    log_storage: Literal["sqlite", "csv"] = "sqlite"
    log_path: str | None = None  # None = ~/.pyforecast/fit_logs.db

    # Hindcast settings
    enable_hindcast: bool = False
    hindcast_holdout_months: int = 6
    min_training_months: int = 12

    # Residual analysis
    enable_residual_analysis: bool = False

    # Regime calibration
    known_events_file: str | None = None

    # Parameter learning
    enable_learning: bool = False

    # Data quality thresholds for logging
    min_data_points_for_logging: int = 6
    max_coefficient_of_variation: float = 0.0  # 0 = no limit
    min_r_squared_for_logging: float = 0.0  # 0 = no limit

    # Ground truth comparison
    ground_truth_file: str | None = None  # ARIES AC_ECONOMIC CSV
    ground_truth_months: int = 60  # Months to compare
    ground_truth_lazy: bool = False  # Stream file instead of loading into memory
    ground_truth_workers: int = 1  # Parallel workers (1 = sequential)


@dataclass
class ValidationConfig:
    """Validation configuration.

    Attributes:
        max_oil_rate: Maximum expected oil rate (bbl/mo) - IV002 threshold
        max_gas_rate: Maximum expected gas rate (mcf/mo) - IV002 threshold
        max_water_rate: Maximum expected water rate (bbl/mo) - IV002 threshold
        gap_threshold_months: Minimum gap size to flag (months) - DQ001
        outlier_sigma: Number of std devs for outlier detection - DQ002
        shutin_threshold: Rate below which is considered shut-in - DQ003
        min_cv: Minimum coefficient of variation - DQ004
        min_r_squared: Minimum acceptable R² value - FR001
        max_annual_decline: Maximum acceptable annual decline rate - FR005
        strict_mode: If True, treat warnings as errors
        acceptable_r_squared: Threshold for ForecastResult.is_acceptable (default 0.7)
    """
    max_oil_rate: float = 50000.0
    max_gas_rate: float = 500000.0
    max_water_rate: float = 100000.0
    gap_threshold_months: int = 2
    outlier_sigma: float = 3.0
    shutin_threshold: float = 1.0
    min_cv: float = 0.05
    min_r_squared: float = 0.5
    max_annual_decline: float = 1.0
    strict_mode: bool = False
    acceptable_r_squared: float = 0.7


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
        validation: Validation configuration
        refinement: Refinement configuration (fit quality analysis)
    """
    oil: ProductConfig = field(default_factory=ProductConfig)
    gas: ProductConfig = field(default_factory=ProductConfig)
    water: ProductConfig = field(default_factory=ProductConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    fitting: FittingDefaults = field(default_factory=FittingDefaults)
    output: OutputConfig = field(default_factory=OutputConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)

    def get_product_config(self, product: Literal["oil", "gas", "water"]) -> ProductConfig:
        """Get configuration for a specific product."""
        if product not in ("oil", "gas", "water"):
            raise ValueError(f"Unknown product: {product}")
        return getattr(self, product)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid
        """
        errors = []

        # Validate product configs
        for product_name in ("oil", "gas", "water"):
            product_config = self.get_product_config(product_name)
            if product_config.b_min >= product_config.b_max:
                errors.append(
                    f"{product_name}: b_min ({product_config.b_min}) must be less than "
                    f"b_max ({product_config.b_max})"
                )
            if product_config.dmin <= 0:
                errors.append(
                    f"{product_name}: dmin ({product_config.dmin}) must be greater than 0"
                )

        # Validate fitting defaults
        if self.fitting.min_points < 3:
            errors.append(
                f"fitting.min_points ({self.fitting.min_points}) must be at least 3"
            )
        if self.fitting.recency_half_life <= 0:
            errors.append(
                f"fitting.recency_half_life ({self.fitting.recency_half_life}) must be greater than 0"
            )

        # Validate regime config
        if self.regime.threshold <= 0:
            errors.append(
                f"regime.threshold ({self.regime.threshold}) must be greater than 0"
            )
        if self.regime.window < 2:
            errors.append(
                f"regime.window ({self.regime.window}) must be at least 2"
            )
        if self.regime.sustained_months < 1:
            errors.append(
                f"regime.sustained_months ({self.regime.sustained_months}) must be at least 1"
            )

        if errors:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    @classmethod
    def from_yaml(cls, filepath: Path | str) -> "PyForecastConfig":
        """Load configuration from YAML file.

        Args:
            filepath: Path to YAML config file

        Returns:
            PyForecastConfig instance

        Raises:
            ValueError: If configuration values are invalid
        """
        filepath = Path(filepath)
        with open(filepath) as f:
            data = yaml.safe_load(f) or {}

        config = cls.from_dict(data)
        config.validate()
        return config

    @staticmethod
    def _filter_unknown_keys(
        section_data: dict,
        dataclass_type: type,
        section_name: str,
    ) -> dict:
        """Filter unknown keys from a config section and warn about them.

        Args:
            section_data: Raw config dictionary for a section
            dataclass_type: The dataclass type to validate against
            section_name: Section name for error messages

        Returns:
            Filtered dictionary with only known keys
        """
        known_keys = {f.name for f in dataclass_fields(dataclass_type)}
        unknown_keys = set(section_data) - known_keys
        if unknown_keys:
            logger.warning(
                f"Unknown key(s) in '{section_name}' config section: "
                f"{', '.join(sorted(unknown_keys))}. "
                f"Valid keys: {', '.join(sorted(known_keys))}"
            )
        return {k: v for k, v in section_data.items() if k in known_keys}

    # Mapping of section name -> dataclass type for from_dict iteration
    _SECTION_TYPES: ClassVar[dict[str, type]] = {
        "oil": ProductConfig, "gas": ProductConfig, "water": ProductConfig,
        "regime": RegimeConfig, "fitting": FittingDefaults, "output": OutputConfig,
        "validation": ValidationConfig, "refinement": RefinementConfig,
    }

    @classmethod
    def from_dict(cls, data: dict) -> "PyForecastConfig":
        """Create configuration from dictionary.

        Unknown keys in any section are logged as warnings and ignored,
        rather than causing opaque TypeErrors.

        Args:
            data: Configuration dictionary

        Returns:
            PyForecastConfig instance
        """
        config = cls()

        unknown_sections = set(data) - set(cls._SECTION_TYPES)
        if unknown_sections:
            logger.warning(
                f"Unknown top-level config section(s): {', '.join(sorted(unknown_sections))}. "
                f"Valid sections: {', '.join(sorted(cls._SECTION_TYPES))}"
            )

        for section, dtype in cls._SECTION_TYPES.items():
            if section in data:
                section_data = cls._filter_unknown_keys(data[section], dtype, section)
                if section == "output" and "products" in section_data:
                    section_data["products"] = list(section_data["products"])
                setattr(config, section, dtype(**section_data))

        return config

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)

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
  # recency_half_life: 12  # Optional: product-specific half-life (months)

gas:
  b_min: 0.01
  b_max: 1.5
  dmin: 0.06
  # recency_half_life: 9   # Gas typically has faster decline, weight recent data more

water:
  b_min: 0.01
  b_max: 1.5
  dmin: 0.06
  # recency_half_life: 18  # Water production more stable, weight history more

# Regime change detection (RTP, refrac)
regime:
  threshold: 1.0          # Minimum increase to trigger (1.0 = 100%)
  window: 6               # Months of trend data to fit
  sustained_months: 2     # Months elevation must persist

# General fitting parameters
fitting:
  recency_half_life: 12.0  # Half-life for recent data weighting (months)
  min_points: 6            # Minimum months of data required
  model_selection: hyperbolic  # auto, hyperbolic, exponential, or harmonic
  estimate_dmin: false     # Estimate terminal decline from late-time data
  dmin_min_annual: 0.024   # Min Dmin when estimating (2.4%/yr)
  dmin_max_annual: 0.24    # Max Dmin when estimating (24%/yr)
  adaptive_regime_detection: false  # Use CV-adaptive regime thresholds

# Output options
output:
  products:               # Products to forecast
    - oil
    - gas
    - water
  plots: true             # Generate individual well plots
  batch_plot: true        # Generate multi-well overlay plot
  format: ac_economic     # Export format: ac_economic or json

# Data validation settings
validation:
  max_oil_rate: 50000     # Max expected oil rate (bbl/mo) - IV002
  max_gas_rate: 500000    # Max expected gas rate (mcf/mo) - IV002
  max_water_rate: 100000  # Max expected water rate (bbl/mo) - IV002
  gap_threshold_months: 2 # Min gap size to flag - DQ001
  outlier_sigma: 3.0      # Std devs for outlier detection - DQ002
  shutin_threshold: 1.0   # Rate below = shut-in - DQ003
  min_cv: 0.05            # Min coefficient of variation - DQ004
  min_r_squared: 0.5      # Min acceptable R² - FR001
  max_annual_decline: 1.0 # Max annual decline rate - FR005
  strict_mode: false      # Treat warnings as errors
  acceptable_r_squared: 0.7  # Threshold for ForecastResult.is_acceptable

# Refinement settings (fit quality analysis - all disabled by default)
refinement:
  enable_logging: false           # Log fit metadata to storage
  log_storage: sqlite             # Storage type: sqlite or csv
  log_path: null                  # null = ~/.pyforecast/fit_logs.db
  enable_hindcast: false          # Run hindcast validation
  hindcast_holdout_months: 6      # Months to hold out for validation
  min_training_months: 12         # Minimum training data required
  enable_residual_analysis: false # Compute residual diagnostics
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

    return filepath
