# PyForecast

Automated decline curve analysis for oil and gas production forecasting.

PyForecast fits hyperbolic decline curves to historical production data and exports ARIES-compatible forecasts. It includes automatic regime change detection, comprehensive data validation, and tools for measuring forecast accuracy.

## Quick Start

```bash
# Install
pip install pyforecast

# Process production data
pyforecast process production.csv -o forecasts/

# View results
ls forecasts/
```

## Installation

**Requirements:** Python 3.10+

```bash
# From PyPI (when published)
pip install pyforecast

# From source
git clone https://github.com/your-org/pyforecast.git
cd pyforecast
pip install -e .
```

## Features

- **Hyperbolic Decline Fitting** - Arps decline with automatic b-factor optimization
- **Terminal Decline Switch** - Automatic switch to exponential at Dmin threshold
- **Regime Change Detection** - Detects refracs, workovers, and production changes
- **Recency Weighting** - Weights recent data more heavily for better fits
- **ARIES AC_ECONOMIC Export** - Direct import into ARIES economic software
- **JSON Export** - API-friendly format for custom integrations
- **Data Validation** - Comprehensive checks with actionable error codes
- **Ground Truth Comparison** - Compare auto-fits against expert projections
- **Hindcast Validation** - Measure actual forecast accuracy through backtesting

## Basic Usage

### Process Production Data

```bash
# Process with default settings
pyforecast process data.csv -o output/

# Specify products to forecast
pyforecast process data.csv -o output/ --product oil --product gas

# Use a configuration file
pyforecast process data.csv -o output/ --config pyforecast.yaml
```

### Generate Configuration

```bash
# Create default config file
pyforecast init -o pyforecast.yaml

# Create config with preset profile
pyforecast init --profile quick -o pyforecast.yaml
```

### Validate Data Only

```bash
# Check data quality without running forecasts
pyforecast validate data.csv -o validation_report.txt
```

### Interactive Plotting

```bash
# Plot single well
pyforecast plot data.csv --well-id "42-001-00001" --product oil
```

## Configuration

PyForecast uses YAML configuration files. Generate a template with `pyforecast init`:

```yaml
# Key settings (see docs/Configuration.md for full reference)
oil:
  b_min: 0.01      # Minimum b-factor
  b_max: 1.5       # Maximum b-factor
  dmin: 0.06       # Terminal decline (6%/year)

fitting:
  recency_half_life: 12.0  # Weight recent data (months)
  min_points: 6            # Minimum data points

output:
  products: [oil, gas, water]
  format: ac_economic      # or json
```

### Configuration Profiles

For common use cases, use preset profiles:

| Profile | Use Case |
|---------|----------|
| `quick` | Fast processing, minimal validation |
| `production` | Balanced accuracy and performance |
| `research` | All features enabled, maximum diagnostics |

```bash
pyforecast init --profile production -o pyforecast.yaml
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `process` | Process production data and generate forecasts |
| `init` | Generate configuration file |
| `validate` | Validate data without running forecasts |
| `plot` | Plot decline curve for a single well |
| `info` | Display information about a data file |
| `analyze-fits` | Analyze accumulated fit logs |
| `suggest-params` | Get parameter suggestions from historical fits |
| `calibrate-regime` | Calibrate regime detection from known events |

Run `pyforecast --help` or `pyforecast <command> --help` for details.

## Input Formats

PyForecast automatically detects these input formats:

| Format | Description |
|--------|-------------|
| Enverus | Standard Enverus production export |
| ARIES | ARIES production data format |

Required columns (names are flexible):
- Well identifier (API, PROPNUM, or well name)
- Production date
- Oil, gas, and/or water volumes

## Output Formats

### AC_ECONOMIC (Default)

ARIES-compatible CSV with decline curve expressions:

```csv
PROPNUM,SECTION,SEQUENCE,QUALIFIER,KEYWORD,EXPRESSION
WELL001,4,1,KA0126,CUMS,9.310 47.940
WELL001,4,2,KA0126,START,01/2023
WELL001,4,100,KA0126,OIL,33.5 X B/D 6 EXP B/0.48 68.11
```

### JSON

API-friendly format with full model parameters:

```json
{
  "well_id": "WELL001",
  "forecasts": {
    "oil": {
      "qi": 33.5,
      "di_annual": 0.6811,
      "b": 0.48,
      "dmin_annual": 0.06,
      "r_squared": 0.92
    }
  }
}
```

## Refinement Features

Enable advanced features for forecast quality analysis:

```bash
# Hindcast validation (backtest accuracy)
pyforecast process data.csv --hindcast -o output/

# Log fits for analysis
pyforecast process data.csv --log-fits -o output/

# Compare against expert ARIES projections
pyforecast process data.csv --ground-truth aries_forecasts.csv -o output/
```

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/User_Guide.md) | Comprehensive usage guide with examples |
| [Configuration](docs/Configuration.md) | All configuration options explained |
| [Export Formats](docs/Export_Formats.md) | AC_ECONOMIC and JSON format specifications |
| [Algorithms](docs/Algorithms.md) | Technical details on decline curve fitting |
| [Validation Guide](docs/PyForecast_Validation_Guide.md) | Data validation and error codes |
| [Refinement Guide](docs/PyForecast_Refinement_Guide.md) | Hindcast, logging, and parameter learning |

## Example Workflow

```bash
# 1. Check your data
pyforecast info production.csv
pyforecast validate production.csv

# 2. Create configuration for your basin
pyforecast init --profile production -o permian.yaml
# Edit permian.yaml to adjust b-factor ranges for your play

# 3. Process with validation
pyforecast process production.csv -c permian.yaml -o forecasts/

# 4. Review results
cat forecasts/validation_report.txt
# Open forecasts/batch_plot.html in browser

# 5. (Optional) Compare against existing ARIES forecasts
pyforecast process production.csv -c permian.yaml \
    --ground-truth existing_aries.csv --gt-plots -o forecasts/
```

## License

[Your License Here]

## Contributing

[Contribution guidelines]
