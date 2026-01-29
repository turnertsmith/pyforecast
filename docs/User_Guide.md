# PyForecast User Guide

A practical guide to using PyForecast for oil and gas production forecasting.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Common Workflows](#common-workflows)
3. [Configuration Guide](#configuration-guide)
4. [Working with Data](#working-with-data)
5. [Understanding Results](#understanding-results)
6. [Ground Truth Comparison](#ground-truth-comparison)
7. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

```bash
# From source
git clone https://github.com/your-org/pyforecast.git
cd pyforecast
pip install -e .

# Verify installation
pyforecast --help
```

### First Run

```bash
# Check your data file
pyforecast info production.csv

# Run with defaults
pyforecast process production.csv -o output/

# View results
ls output/
cat output/validation_report.txt
```

### Output Files

After processing, you'll find these files in your output directory:

| File | Description |
|------|-------------|
| `forecasts.csv` | AC_ECONOMIC export for ARIES |
| `validation_report.txt` | Data quality and fit validation |
| `batch_plot.html` | Interactive multi-well plot |
| `plots/` | Individual well plots (if enabled) |
| `errors.txt` | Processing errors (if any) |

---

## Common Workflows

### Workflow 1: Basic Batch Processing

Process a batch of wells with default settings:

```bash
pyforecast process production.csv -o forecasts/
```

Review results:
```bash
cat forecasts/validation_report.txt
open forecasts/batch_plot.html  # macOS
```

### Workflow 2: Configured Processing

Create a configuration file for your specific basin/play:

```bash
# Generate template
pyforecast init -o permian.yaml

# Edit to customize (see Configuration Guide below)
# Then process with config
pyforecast process production.csv -c permian.yaml -o forecasts/
```

### Workflow 3: Validate Before Processing

Check data quality first:

```bash
# Validate only (no forecasts)
pyforecast validate production.csv -o validation_report.txt

# Review issues
cat validation_report.txt

# If OK, proceed
pyforecast process production.csv -o forecasts/
```

### Workflow 4: Compare Against Expert Forecasts

Validate auto-fits against existing ARIES projections:

```bash
# Process with ground truth comparison
pyforecast process production.csv \
    --ground-truth existing_aries.csv \
    --gt-plots \
    -o forecasts/

# Review comparison
cat forecasts/ground_truth_report.txt
open forecasts/ground_truth_plots/  # View comparison plots
```

### Workflow 5: Hindcast Validation

Measure actual forecast accuracy through backtesting:

```bash
# Process with hindcast enabled
pyforecast process production.csv --hindcast -o forecasts/

# Review hindcast results
cat forecasts/refinement_report.txt
```

### Workflow 6: Iterative Improvement

Build up parameter suggestions over time:

```bash
# Process multiple batches with logging
pyforecast process batch1.csv --log-fits --hindcast -o output1/
pyforecast process batch2.csv --log-fits --hindcast -o output2/

# Analyze accumulated fits
pyforecast analyze-fits

# Get parameter suggestions
pyforecast suggest-params --basin "Permian" -p oil
```

---

## Configuration Guide

### Top 10 Most Important Options

These are the settings you'll most often need to adjust:

#### 1. `oil.b_min` / `oil.b_max` (Per-Product B-Factor Bounds)

Controls the range of hyperbolic exponent values.

```yaml
oil:
  b_min: 0.01   # Near-exponential (conventional wells)
  b_max: 1.5    # Super-harmonic (unconventional tight oil)
```

| Play Type | Typical b_min | Typical b_max |
|-----------|---------------|---------------|
| Conventional | 0.01 | 0.5 |
| Tight Oil/Gas | 0.3 | 1.2 |
| Ultra-tight | 0.5 | 1.5 |

#### 2. `oil.dmin` (Terminal Decline Rate)

Annual decline rate at which the curve switches to exponential.

```yaml
oil:
  dmin: 0.06   # 6% annual (typical default)
```

Common values:
- 5-6%: Standard assumption
- 8-10%: Conservative for tight plays
- 3-4%: Water drive reservoirs

#### 3. `fitting.recency_half_life` (Recent Data Weighting)

How aggressively to weight recent production data.

```yaml
fitting:
  recency_half_life: 12.0  # 12 months (default)
```

- Lower values (6-9): More weight on recent data, captures rapid changes
- Higher values (18-24): Smoother fits, less reactive to recent noise
- 12: Balanced default

#### 4. `fitting.min_points` (Minimum Data Required)

Minimum months of production needed to fit a curve.

```yaml
fitting:
  min_points: 6   # 6 months minimum
```

- 6: Minimum for reasonable fits
- 12: More reliable for noisy data
- 24: Very conservative

#### 5. `regime.threshold` (Regime Change Detection)

Fractional increase that triggers regime change detection.

```yaml
regime:
  threshold: 1.0   # 100% increase required
```

- 0.5: Sensitive (catches smaller workovers)
- 1.0: Default (detects major refracs)
- 2.0: Very conservative (only major events)

#### 6. `output.products` (Products to Forecast)

Which products to fit and export.

```yaml
output:
  products:
    - oil
    - gas
    - water
```

#### 7. `output.format` (Export Format)

```yaml
output:
  format: ac_economic  # or json
```

#### 8. `validation.min_r_squared` (Fit Quality Threshold)

Minimum R² to accept a fit.

```yaml
validation:
  min_r_squared: 0.5   # 50% minimum
```

- 0.3: Very permissive
- 0.5: Default
- 0.7: Strict

#### 9. `validation.strict_mode` (Treat Warnings as Errors)

```yaml
validation:
  strict_mode: false  # true to fail on warnings
```

#### 10. `refinement.enable_hindcast` (Enable Backtesting)

```yaml
refinement:
  enable_hindcast: true
  hindcast_holdout_months: 6
```

### Configuration Profiles

Use preset profiles for common scenarios:

```bash
# Quick processing (fast, minimal checks)
pyforecast init --profile quick -o config.yaml

# Production processing (balanced)
pyforecast init --profile production -o config.yaml

# Research mode (all features, maximum diagnostics)
pyforecast init --profile research -o config.yaml
```

| Profile | Use Case | Features |
|---------|----------|----------|
| `quick` | Fast initial runs | Minimal validation, no plots |
| `production` | Daily operations | Balanced accuracy/speed |
| `research` | Deep analysis | All refinement features enabled |

### Full Configuration Example

```yaml
# pyforecast.yaml - Full configuration example

# Per-product decline parameters
oil:
  b_min: 0.01      # Minimum b-factor
  b_max: 1.5       # Maximum b-factor
  dmin: 0.06       # Terminal decline (6%/year)

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
  threshold: 1.0          # 100% increase required
  window: 6               # Months of trend data
  sustained_months: 2     # Months to confirm

# Fitting parameters
fitting:
  recency_half_life: 12.0  # Recent data weighting
  min_points: 6            # Minimum data points

# Output options
output:
  products: [oil, gas, water]
  plots: true              # Individual well plots
  batch_plot: true         # Multi-well overlay
  format: ac_economic      # or json

# Validation thresholds
validation:
  max_oil_rate: 50000      # Max expected bbl/mo
  max_gas_rate: 500000     # Max expected mcf/mo
  gap_threshold_months: 2  # Min gap to flag
  outlier_sigma: 3.0       # Outlier detection
  min_r_squared: 0.5       # Fit quality threshold
  strict_mode: false       # Treat warnings as errors

# Refinement features (optional)
refinement:
  enable_logging: false
  enable_hindcast: false
  enable_residual_analysis: false
```

---

## Working with Data

### Supported Input Formats

PyForecast automatically detects:

1. **Enverus Format** - Standard Enverus production export
2. **ARIES Format** - ARIES production data

### Required Columns

At minimum, your data needs:
- Well identifier (API, PROPNUM, Entity ID, or Well Name)
- Production date
- At least one product volume (oil, gas, or water)

### Check Your Data

```bash
# See what PyForecast detects
pyforecast info production.csv
```

Output:
```
Inspecting: production.csv

Rows: 5400
Columns: ['API', 'Date', 'Oil', 'Gas', 'Water']
Detected format: EnverusParser
Wells found: 45

Sample wells:
  42-123-45678: 120 months, 2015-01-01 to 2024-12-01
  42-123-45679: 108 months, 2016-01-01 to 2024-12-01
  ...
```

### Data Quality Considerations

For best results:
- Use monthly production data (not daily)
- Ensure dates are consistent (all same format)
- Remove forecast/projected data from historical records
- Check for negative values (should be zero or positive)

### Validation Codes

See [Validation Guide](PyForecast_Validation_Guide.md) for complete code reference.

| Category | Examples | Action |
|----------|----------|--------|
| `IV*` | Input format issues | Fix data before processing |
| `DQ*` | Data quality issues | Review, may be OK |
| `FP*` | Fitting prerequisites | May need more data |
| `FR*` | Fit result issues | Review fit quality |

---

## Understanding Results

### Reading the Validation Report

```
Validation Summary:
  Wells with errors: 3
  Wells with warnings: 12

  Issues by category:
    DATA_QUALITY: 8
    FITTING_RESULT: 7
```

- **Errors**: Should be fixed before trusting results
- **Warnings**: Review but may be acceptable
- **Info**: Informational only

### Interpreting Fit Parameters

| Parameter | Description | Good Range |
|-----------|-------------|------------|
| `qi` | Initial rate | Varies by well |
| `di` | Initial decline | 5-80%/year typical |
| `b` | Hyperbolic exponent | 0.3-1.2 for unconventional |
| `R²` | Fit quality | > 0.7 ideal, > 0.5 acceptable |
| `RMSE` | Prediction error | Lower is better |

### Decline Type Interpretation

| Type | B-Factor | Meaning |
|------|----------|---------|
| EXP | b ≤ 0.1 | Exponential (conventional wells) |
| HYP | 0.1 < b < 0.95 | Hyperbolic (most unconventional) |
| HRM | b ≥ 0.95 | Harmonic (rare, transient flow) |

### Regime Detection

When a regime change is detected:
- `regime_start_idx > 0` in results
- Only data after the regime change is used for fitting
- This handles refracs, workovers, and production changes

---

## Ground Truth Comparison

### Purpose

Compare auto-fitted curves against expert/approved ARIES projections to:
- Validate that auto-fitting produces reasonable results
- Identify wells needing manual review
- Measure overall fitting accuracy

### Setup

You need an existing ARIES AC_ECONOMIC file with forecasts:

```csv
PROPNUM,SECTION,SEQUENCE,QUALIFIER,KEYWORD,EXPRESSION
42-123-45678,4,100,KA0125,OIL,1000 X B/D 6 EXP B/0.50 8.5
42-123-45678,4,300,KA0125,GAS,5000 X M/D 6 EXP B/0.75 12
```

### Running Comparison

```bash
# Basic comparison
pyforecast process production.csv \
    --ground-truth aries_projections.csv \
    -o output/

# With comparison plots
pyforecast process production.csv \
    --ground-truth aries_projections.csv \
    --gt-plots \
    -o output/

# Custom comparison period (default 60 months)
pyforecast process production.csv \
    --ground-truth aries_projections.csv \
    --gt-months 120 \
    -o output/

# For large files, use lazy loading
pyforecast process production.csv \
    --ground-truth large_aries.csv \
    --gt-lazy \
    -o output/
```

### Understanding Results

```
Ground Truth Comparison:
  Wells with ARIES data: 45 of 50
  Average MAPE: 12.3%
  Average correlation: 0.987
  Good match rate: 82.2%
```

| Metric | Good Value | Description |
|--------|------------|-------------|
| MAPE | < 20% | Mean Absolute Percentage Error |
| Correlation | > 0.95 | How well curves track together |
| Cumulative Diff | < 15% | Total production difference |

### Match Grades

| Grade | Meaning | Action |
|-------|---------|--------|
| A | Excellent match | No review needed |
| B | Good match | Minor differences |
| C | Fair match | Review recommended |
| D | Poor match | Manual review needed |

### Output Files

| File | Description |
|------|-------------|
| `ground_truth_report.txt` | Detailed comparison report |
| `ground_truth_results.csv` | Summary metrics per well |
| `ground_truth_timeseries.csv` | Monthly rate comparisons |
| `ground_truth_plots/` | Overlay plots (if `--gt-plots`) |

---

## Troubleshooting

### Common Issues

#### "No wells found in file"

**Cause:** Format not detected or columns don't match expected patterns.

**Solution:**
```bash
# Check what PyForecast sees
pyforecast info data.csv

# Verify column names include date and well identifier
head -1 data.csv
```

#### "Insufficient data points"

**Cause:** Well has fewer months than `min_points` setting.

**Solution:**
- Check data for missing months
- Lower `min_points` in config (minimum 6 recommended)
- Skip well if truly insufficient data

#### "Poor fit quality (R² < 0.5)"

**Causes:**
1. Noisy data
2. Wrong b-factor bounds for play type
3. Missed regime change
4. Data quality issues

**Solution:**
```bash
# Check validation
cat output/validation_report.txt

# Plot the well to visualize
pyforecast plot data.csv --well-id "problem_well" --product oil

# Adjust b-factor bounds in config
# Try adjusting regime detection threshold
```

#### "B-factor at upper bound"

**Cause:** B-factor hit the `b_max` limit.

**Solutions:**
1. If common for your play, increase `b_max` in config
2. If unusual, check for early-time transient data
3. Review for data quality issues

#### "Very high decline rate (>100%/yr)"

**Causes:**
1. Data is daily rates, not monthly
2. Early months only (before stabilization)
3. Data quality issue

**Solution:**
```bash
# Verify data units
pyforecast info data.csv

# Check for unit conversion issues
head -20 data.csv
```

#### "Wells with ARIES data: 0 of N"

**Cause:** Well identifiers don't match between files.

**Solution:**
1. Check identifier format (API vs PROPNUM)
2. Verify normalization (leading zeros, dashes)
3. Use `pyforecast info` on both files

### Getting Help

```bash
# Command help
pyforecast --help
pyforecast process --help

# View validation codes
cat docs/PyForecast_Validation_Guide.md
```

### Debug Mode

For detailed debugging:

```bash
# Increase verbosity (coming soon)
pyforecast process data.csv -o output/ -v

# Check specific well
pyforecast plot data.csv --well-id "problem_well" --product oil
```

---

## Appendix: CLI Reference

### Main Commands

```bash
pyforecast process INPUT [OPTIONS]     # Process production data
pyforecast init [OPTIONS]              # Generate config file
pyforecast validate INPUT [OPTIONS]    # Validate data only
pyforecast plot INPUT [OPTIONS]        # Plot single well
pyforecast info INPUT                  # Show file info
pyforecast analyze-fits [OPTIONS]      # Analyze fit logs
pyforecast suggest-params [OPTIONS]    # Get parameter suggestions
pyforecast calibrate-regime [OPTIONS]  # Calibrate regime detection
```

### Process Options

```bash
pyforecast process INPUT [OPTIONS]

Options:
  -o, --output PATH          Output directory (default: output/)
  -c, --config PATH          YAML configuration file
  -p, --product TEXT         Products to forecast (can repeat)
  -w, --workers INT          Parallel workers
  --no-plots                 Skip individual well plots
  --no-batch-plot           Skip batch overlay plot
  --format TEXT             Export format: ac_economic or json
  --hindcast                Enable hindcast validation
  --log-fits                Log fit metadata
  --residuals               Enable residual analysis
  --ground-truth PATH       ARIES file for comparison
  --gt-months INT           Comparison period (default: 60)
  --gt-plots                Generate comparison plots
  --gt-lazy                 Stream ARIES file
  --gt-workers INT          Parallel validation workers
```

### Init Options

```bash
pyforecast init [OPTIONS]

Options:
  -o, --output PATH          Output file (default: pyforecast.yaml)
  --profile TEXT            Profile: quick, production, research
```
