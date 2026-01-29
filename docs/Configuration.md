# PyForecast Configuration Reference

Complete reference for all PyForecast configuration options.

## Table of Contents

1. [Configuration File](#configuration-file)
2. [Product Settings](#product-settings)
3. [Regime Detection](#regime-detection)
4. [Fitting Parameters](#fitting-parameters)
5. [Output Settings](#output-settings)
6. [Validation Settings](#validation-settings)
7. [Refinement Settings](#refinement-settings)
8. [Configuration Profiles](#configuration-profiles)

---

## Configuration File

PyForecast uses YAML configuration files. Generate a template with:

```bash
pyforecast init -o pyforecast.yaml
```

Or use a preset profile:

```bash
pyforecast init --profile production -o pyforecast.yaml
```

### File Structure

```yaml
# Per-product decline parameters
oil:
  b_min: 0.01
  b_max: 1.5
  dmin: 0.06

gas:
  b_min: 0.01
  b_max: 1.5
  dmin: 0.06

water:
  b_min: 0.01
  b_max: 1.5
  dmin: 0.06

# Regime change detection
regime:
  threshold: 1.0
  window: 6
  sustained_months: 2

# General fitting parameters
fitting:
  recency_half_life: 12.0
  min_points: 6

# Output options
output:
  products: [oil, gas, water]
  plots: true
  batch_plot: true
  format: ac_economic

# Validation thresholds
validation:
  # ... (see Validation Settings)

# Refinement features
refinement:
  # ... (see Refinement Settings)
```

### CLI Override

Command-line flags override config file values:

```bash
# Config says plots: true, but CLI disables them
pyforecast process data.csv -c config.yaml --no-plots
```

---

## Product Settings

Per-product decline curve parameters. Configure separately for oil, gas, and water.

### `oil`, `gas`, `water`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `b_min` | float | 0.01 | Minimum b-factor bound |
| `b_max` | float | 1.5 | Maximum b-factor bound |
| `dmin` | float | 0.06 | Terminal decline rate (annual fraction) |

#### `b_min` / `b_max`

The hyperbolic b-factor is constrained to this range during optimization.

| B-Factor Range | Decline Behavior | Typical Use Case |
|----------------|------------------|------------------|
| 0.01 - 0.1 | Near-exponential | Conventional wells, water drive |
| 0.3 - 0.8 | Hyperbolic | Typical unconventional (tight oil/gas) |
| 0.8 - 1.0 | Near-harmonic | Transient flow, high-permeability |
| 1.0 - 1.5 | Super-harmonic | Early-time transient, rare |

**Recommendations by play type:**

| Play Type | b_min | b_max |
|-----------|-------|-------|
| Conventional | 0.01 | 0.5 |
| Permian (Wolfcamp) | 0.3 | 1.2 |
| Bakken | 0.4 | 1.0 |
| Eagle Ford | 0.3 | 1.0 |
| Marcellus (gas) | 0.5 | 1.2 |

#### `dmin`

Terminal decline rate at which the model switches from hyperbolic to exponential decline. Expressed as annual fraction.

| Value | Annual % | Description |
|-------|----------|-------------|
| 0.03 | 3% | Aggressive (long well life) |
| 0.05 | 5% | Conservative standard |
| 0.06 | 6% | **Default** - industry standard |
| 0.08 | 8% | Conservative for tight plays |
| 0.10 | 10% | Very conservative |

**Example:**

```yaml
oil:
  b_min: 0.3      # Tight oil minimum
  b_max: 1.2      # Allow super-harmonic
  dmin: 0.06      # 6% terminal decline

gas:
  b_min: 0.5      # Gas typically higher b
  b_max: 1.2
  dmin: 0.06
```

---

## Regime Detection

Controls detection of production regime changes (refracs, workovers, RTPs).

### `regime`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `threshold` | float | 1.0 | Minimum fractional increase to trigger detection |
| `window` | int | 6 | Months of trend data for fitting |
| `sustained_months` | int | 2 | Consecutive months to confirm change |

#### `threshold`

Minimum production increase (as fraction) above projected trend to trigger detection.

| Value | Increase Required | Sensitivity |
|-------|-------------------|-------------|
| 0.5 | 50% | High - catches smaller events |
| 1.0 | 100% | **Default** - standard refracs |
| 1.5 | 150% | Low - only major events |
| 2.0 | 200% | Very low - significant refracs only |

#### `window`

Months of prior production used to fit the decline trend for projection.

- **Lower values (3-4)**: More responsive, but trend estimates less stable
- **Higher values (8-12)**: More stable trends, but may miss changes in shorter declines
- **Default (6)**: Balanced

#### `sustained_months`

Consecutive months that production must exceed the threshold to confirm a regime change. Filters out single-month outliers.

- **1 month**: Very sensitive, may catch false positives
- **2 months**: **Default** - good balance
- **3+ months**: Conservative, may miss some events

**Example:**

```yaml
regime:
  threshold: 0.75      # More sensitive (75% increase)
  window: 6
  sustained_months: 3  # More robust confirmation
```

---

## Fitting Parameters

General parameters applied to all products.

### `fitting`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `recency_half_life` | float | 12.0 | Half-life for exponential decay weighting (months) |
| `min_points` | int | 6 | Minimum data points required for fitting |

#### `recency_half_life`

Controls how aggressively recent data is weighted. Uses exponential decay where data `half_life` months old has weight 0.5.

| Value | Effect | Use Case |
|-------|--------|----------|
| 6-9 | Strong recent weighting | Rapidly changing wells, recent workovers |
| 12 | **Default** - balanced | Most scenarios |
| 18-24 | Smoother fits | Stable wells, noisy data |
| 36+ | Nearly equal weighting | Long-term trend analysis |

**Weight decay example (half_life = 12):**

| Months Ago | Weight |
|------------|--------|
| 0 (current) | 1.000 |
| 6 | 0.707 |
| 12 | 0.500 |
| 24 | 0.250 |
| 36 | 0.125 |

#### `min_points`

Minimum months of production data required before attempting to fit a decline curve.

- **Minimum**: 3 (absolute minimum for fitting)
- **Recommended minimum**: 6 (more reliable)
- **For noisy data**: 12+

**Example:**

```yaml
fitting:
  recency_half_life: 9.0   # Weight recent data more
  min_points: 12           # Require 1 year of data
```

---

## Output Settings

Controls what PyForecast produces.

### `output`

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `products` | list | [oil, gas, water] | Products to forecast |
| `plots` | bool | true | Generate individual well plots |
| `batch_plot` | bool | true | Generate multi-well overlay plot |
| `format` | string | ac_economic | Export format: `ac_economic` or `json` |

#### `products`

List of products to forecast. Wells without data for a product are skipped for that product.

```yaml
output:
  products:
    - oil
    - gas
    # water omitted - won't forecast water
```

#### `plots` / `batch_plot`

- `plots`: Creates individual HTML plot for each well in `plots/` subdirectory
- `batch_plot`: Creates `batch_plot.html` with all wells overlaid

Set both to `false` for faster processing when plots aren't needed.

#### `format`

| Format | Description | Use Case |
|--------|-------------|----------|
| `ac_economic` | ARIES-compatible CSV | Import into ARIES economic software |
| `json` | Structured JSON | API integration, custom tools |

See [Export Formats](Export_Formats.md) for detailed format specifications.

**Example:**

```yaml
output:
  products: [oil, gas]
  plots: false          # Skip individual plots
  batch_plot: true      # Keep batch overview
  format: json          # JSON for API
```

---

## Validation Settings

Controls data validation thresholds and behavior.

### `validation`

| Option | Type | Default | Code | Description |
|--------|------|---------|------|-------------|
| `max_oil_rate` | float | 50000 | IV002 | Max expected oil rate (bbl/mo) |
| `max_gas_rate` | float | 500000 | IV002 | Max expected gas rate (mcf/mo) |
| `max_water_rate` | float | 100000 | IV002 | Max expected water rate (bbl/mo) |
| `gap_threshold_months` | int | 2 | DQ001 | Minimum gap size to flag |
| `outlier_sigma` | float | 3.0 | DQ002 | Std devs for outlier detection |
| `shutin_threshold` | float | 1.0 | DQ003 | Rate below which is shut-in |
| `min_cv` | float | 0.05 | DQ004 | Minimum coefficient of variation |
| `min_r_squared` | float | 0.5 | FR001 | Minimum acceptable R² |
| `max_annual_decline` | float | 1.0 | FR005 | Maximum annual decline rate |
| `strict_mode` | bool | false | - | Treat warnings as errors |

#### Rate Limits (`max_oil_rate`, etc.)

Values exceeding these limits trigger IV002 warnings. Adjust based on your wells:

| Well Type | max_oil_rate | max_gas_rate |
|-----------|--------------|--------------|
| Conventional | 10,000 | 100,000 |
| Unconventional | 50,000 | 500,000 |
| High-rate | 100,000 | 1,000,000 |

#### `gap_threshold_months`

Minimum gap (consecutive months of zero/missing data) to flag as DQ001 warning.

#### `outlier_sigma`

Number of standard deviations from mean to flag as outlier (DQ002).

- **3.0**: Default - flags ~0.3% of normal data
- **4.0**: Permissive - fewer false positives
- **2.5**: Sensitive - catches more anomalies

#### `min_r_squared`

Minimum coefficient of determination (R²) for a fit to be considered acceptable.

| Value | Meaning | Use Case |
|-------|---------|----------|
| 0.3 | Very permissive | Exploratory, noisy data |
| 0.5 | **Default** | Most scenarios |
| 0.7 | Strict | High-quality fits only |
| 0.9 | Very strict | Near-perfect fits only |

#### `strict_mode`

When `true`, warnings are treated as errors and will fail processing.

**Example:**

```yaml
validation:
  max_oil_rate: 100000    # High-rate wells
  outlier_sigma: 4.0      # More permissive outlier detection
  min_r_squared: 0.6      # Slightly higher quality threshold
  strict_mode: false
```

---

## Refinement Settings

Controls advanced fit quality analysis features. All disabled by default.

### `refinement`

#### Logging Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_logging` | bool | false | Log fit metadata to storage |
| `log_storage` | string | sqlite | Storage type: `sqlite` or `csv` |
| `log_path` | string | null | Path to storage (null = ~/.pyforecast/fit_logs.db) |
| `min_data_points_for_logging` | int | 6 | Minimum points to log a fit |
| `max_coefficient_of_variation` | float | 0 | Max CV to log (0 = no limit) |
| `min_r_squared_for_logging` | float | 0 | Min R² to log (0 = no limit) |

#### Hindcast Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_hindcast` | bool | false | Run hindcast validation |
| `hindcast_holdout_months` | int | 6 | Months to hold out for validation |
| `min_training_months` | int | 12 | Minimum training data required |

#### Analysis Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_residual_analysis` | bool | false | Compute residual diagnostics |
| `enable_learning` | bool | false | Enable parameter learning |
| `known_events_file` | string | null | CSV with known regime events |

#### Ground Truth Settings

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ground_truth_file` | string | null | ARIES AC_ECONOMIC CSV for comparison |
| `ground_truth_months` | int | 60 | Months to compare forecasts |
| `ground_truth_lazy` | bool | false | Stream file instead of loading |
| `ground_truth_workers` | int | 1 | Parallel workers (1 = sequential) |

**Example - Research Mode:**

```yaml
refinement:
  enable_logging: true
  enable_hindcast: true
  hindcast_holdout_months: 6
  min_training_months: 12
  enable_residual_analysis: true
  enable_learning: true
```

**Example - Ground Truth Comparison:**

```yaml
refinement:
  ground_truth_file: aries_forecasts.csv
  ground_truth_months: 120    # 10-year comparison
  ground_truth_lazy: true     # For large files
  ground_truth_workers: 4     # Parallel validation
```

---

## Configuration Profiles

PyForecast includes preset profiles for common scenarios.

### Available Profiles

| Profile | Description | Key Settings |
|---------|-------------|--------------|
| `quick` | Fast initial runs | No plots, permissive validation |
| `production` | Daily operations | Balanced accuracy/speed |
| `research` | Deep analysis | All refinement features enabled |

### Using Profiles

```bash
# Generate config with profile
pyforecast init --profile quick -o config.yaml

# Use directly
pyforecast process data.csv --config config.yaml
```

### Profile Comparison

| Setting | quick | production | research |
|---------|-------|------------|----------|
| `plots` | false | true | true |
| `batch_plot` | false | true | true |
| `min_r_squared` | 0.4 | 0.5 | 0.5 |
| `outlier_sigma` | 4.0 | 3.0 | 3.0 |
| `enable_logging` | false | false | true |
| `enable_hindcast` | false | false | true |
| `enable_residual_analysis` | false | false | true |

---

## Validation Rules

Configuration is validated when loaded. Invalid configurations raise errors:

| Rule | Error |
|------|-------|
| b_min >= b_max | "b_min must be less than b_max" |
| dmin <= 0 | "dmin must be greater than 0" |
| min_points < 3 | "min_points must be at least 3" |
| recency_half_life <= 0 | "recency_half_life must be greater than 0" |
| threshold <= 0 | "threshold must be greater than 0" |
| window < 2 | "window must be at least 2" |
| sustained_months < 1 | "sustained_months must be at least 1" |

---

## Programmatic Usage

```python
from pyforecast.config import PyForecastConfig, ProductConfig

# Load from file
config = PyForecastConfig.from_yaml("pyforecast.yaml")

# Create programmatically
config = PyForecastConfig(
    oil=ProductConfig(b_min=0.3, b_max=1.2, dmin=0.06),
    gas=ProductConfig(b_min=0.5, b_max=1.2, dmin=0.06),
)

# Validate
config.validate()  # Raises ValueError if invalid

# Access settings
print(config.oil.b_max)  # 1.2
print(config.fitting.recency_half_life)  # 12.0

# Get product-specific config
oil_config = config.get_product_config("oil")

# Save to file
config.to_yaml("output_config.yaml")
```
