---
title: "PyForecast Data Validation & Error Handling"
subtitle: "Comprehensive Guide to Production Data Quality Assurance"
date: "January 2026"
geometry: margin=1in
fontsize: 11pt
toc: true
toc-depth: 3
numbersections: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{xcolor}
  - \definecolor{errorred}{RGB}{220,53,69}
  - \definecolor{warningyellow}{RGB}{255,193,7}
  - \definecolor{infogreen}{RGB}{40,167,69}
---

\newpage

# Executive Summary

PyForecast now includes a comprehensive data validation and error handling system designed to catch data quality issues before they impact decline curve analysis. This system provides:

- **Automatic detection** of common production data problems
- **Structured error codes** for easy issue identification
- **Configurable thresholds** to match your operational parameters
- **Actionable guidance** for resolving each issue type
- **Integrated reporting** in batch processing workflows

The validation system operates at three levels:

1. **Input Validation** - Verifies data format and value ranges
2. **Data Quality Validation** - Detects gaps, outliers, and anomalies
3. **Fitting Validation** - Ensures curve fitting prerequisites and quality

\newpage

# Getting Started

## Basic Usage

Validation runs automatically during batch processing:

```bash
pyforecast process production_data.csv -o output/
```

After processing, you'll see a validation summary:

```
Results:
  Successful: 45
  Failed: 3
  Skipped (insufficient data): 2

Validation Summary:
  Wells with errors: 3
  Wells with warnings: 12

  Issues by category:
    DATA_QUALITY: 8
    FITTING_RESULT: 7

  See output/validation_report.txt for details
```

## Configuration

Generate a configuration file with validation settings:

```bash
pyforecast init -o pyforecast.yaml
```

Then customize the validation section:

```yaml
validation:
  max_oil_rate: 50000     # Max expected oil rate (bbl/mo)
  max_gas_rate: 500000    # Max expected gas rate (mcf/mo)
  max_water_rate: 100000  # Max expected water rate (bbl/mo)
  gap_threshold_months: 2 # Min gap size to flag
  outlier_sigma: 3.0      # Std devs for outlier detection
  shutin_threshold: 1.0   # Rate below = shut-in
  min_cv: 0.05            # Min coefficient of variation
  min_r_squared: 0.5      # Min acceptable R-squared
  max_annual_decline: 1.0 # Max annual decline rate (1.0 = 100%)
  strict_mode: false      # Treat warnings as errors
```

Run with your configuration:

```bash
pyforecast process data.csv -o output/ --config pyforecast.yaml
```

\newpage

# Validation Categories

## Overview

All validation issues are categorized for easy filtering and reporting:

| Category | Description | When Checked |
|----------|-------------|--------------|
| `DATA_FORMAT` | Column, date, and value format issues | Before fitting |
| `DATA_QUALITY` | Gaps, outliers, shut-ins, variability | Before fitting |
| `FITTING_PREREQ` | Pre-fit requirements not met | Before fitting |
| `FITTING_RESULT` | Post-fit quality concerns | After fitting |

## Severity Levels

Each issue has a severity level indicating required action:

| Severity | Meaning | Action Required |
|----------|---------|-----------------|
| **ERROR** | Cannot proceed safely | Must resolve before trusting results |
| **WARNING** | Can proceed with caution | Review recommended |
| **INFO** | Informational only | No action required |

\newpage

# Input Validation (IV Codes)

Input validation checks that production data is properly formatted and within expected ranges.

## IV001: Negative Production Values

**Severity:** ERROR

**Description:** Production values must be non-negative. Negative values indicate data entry errors or unit conversion problems.

**Example Issue:**
```
[IV001] ERROR: Found 3 negative oil values
  Guidance: Production values must be non-negative;
            check data source for errors
  Details:
    - Indices: [5, 12, 23]
    - Values: [-50.0, -12.5, -3.2]
```

**Resolution:**

1. Check source data for typos or sign errors
2. Verify unit conversions are correct
3. Replace negative values with zero or interpolated values

---

## IV002: Values Exceed Threshold

**Severity:** WARNING

**Description:** Production values exceed configured maximum rates. Very high values may indicate unit conversion errors (e.g., daily rates instead of monthly).

**Default Thresholds:**

| Product | Default Max | Unit |
|---------|-------------|------|
| Oil | 50,000 | bbl/month |
| Gas | 500,000 | mcf/month |
| Water | 100,000 | bbl/month |

**Example Issue:**
```
[IV002] WARNING: Found 2 oil values exceeding 50,000
  Guidance: Very high production values may indicate
            unit conversion issues or data errors
  Details:
    - Threshold: 50,000
    - Max value: 125,000
    - Indices: [0, 1]
```

**Resolution:**

1. Verify units are monthly (not daily or annual)
2. Check for decimal point errors
3. Adjust `max_oil_rate` in config if values are legitimate

---

## IV003: Date Parsing Failed

**Severity:** ERROR

**Description:** Production dates could not be parsed. This typically indicates an unsupported date format.

**Supported Formats:**

- `YYYY-MM-DD` (ISO format)
- `MM/DD/YYYY` (US format)
- `DD-Mon-YYYY` (e.g., 01-Jan-2024)
- Excel date serial numbers

**Resolution:**

1. Check date column format in source data
2. Ensure consistent date formatting throughout file
3. Convert dates to ISO format (YYYY-MM-DD) if needed

---

## IV004: Future Dates in Data

**Severity:** WARNING

**Description:** Production dates are in the future, which may indicate data entry errors or placeholder values.

**Example Issue:**
```
[IV004] WARNING: Found 2 future dates in production data
  Guidance: Verify data dates are correct; future dates
            may indicate data entry errors
  Details:
    - First future date: 2026-03-01
    - Indices: [35, 36]
```

**Resolution:**

1. Verify dates are correct in source system
2. Remove forecast/projected data from historical production
3. Check for year entry errors (e.g., 2026 vs 2024)

\newpage

# Data Quality Validation (DQ Codes)

Data quality validation detects patterns that may affect fitting accuracy.

## DQ001: Data Gaps Detected

**Severity:** WARNING

**Description:** Consecutive months of zero or near-zero production surrounded by non-zero production. Gaps may represent missing data or operational shut-ins.

**Default Threshold:** 2+ months

**Example Issue:**
```
[DQ001] WARNING: Found 2 data gaps (>=2 months)
  Guidance: Gaps may indicate shut-ins or missing data;
            consider excluding or interpolating
  Details:
    - Gaps: [(5, 7, 3), (15, 17, 3)]  # (start, end, length)
    - Total gap months: 6
```

**Resolution:**

1. Determine if gaps are operational (shut-ins) or data issues
2. For missing data: obtain from source or interpolate
3. For shut-ins: regime detection will handle automatically
4. Adjust `gap_threshold_months` to change sensitivity

---

## DQ002: Outliers Detected

**Severity:** WARNING

**Description:** Values significantly different from the typical production pattern. Uses Modified Z-Score with Median Absolute Deviation (MAD) for robust detection.

**Detection Method:**
$$\text{Modified Z-Score} = \frac{0.6745 \times (x - \text{median})}{\text{MAD}}$$

Values with |Modified Z-Score| > threshold are flagged.

**Default Threshold:** 3.0 sigma

**Example Issue:**
```
[DQ002] WARNING: Found 2 potential outliers in oil data
  Guidance: Review outlier values for data errors;
            consider excluding from fit
  Details:
    - Outlier count: 2
    - Indices: [8, 22]
    - Values: [5230.5, 4890.2]
    - Median: 850.0
    - MAD: 125.3
```

**Resolution:**

1. Investigate if outliers are real production events
2. Check for data entry errors or unit issues
3. If erroneous: correct or exclude from dataset
4. Adjust `outlier_sigma` to change sensitivity

---

## DQ003: Shut-in Periods Detected

**Severity:** INFO

**Description:** Periods where production drops to near-zero then resumes. These trigger regime detection, which will fit only the most recent decline period.

**Default Threshold:** < 1.0 bbl/month (or mcf/month)

**Example Issue:**
```
[DQ003] INFO: Found 1 shut-in periods in oil data
  Guidance: Shut-in periods may trigger regime detection;
            verify expected behavior
  Details:
    - Periods: [(12, 14, 3)]  # (start, end, length)
    - Threshold: 1.0
```

**Resolution:**

- Usually no action required
- Regime detection automatically identifies post-shut-in decline
- Review fit to ensure correct regime was selected

---

## DQ004: Low Data Variability

**Severity:** WARNING

**Description:** Production data has very low variability (near-constant values). This may indicate synthetic data, heavily smoothed data, or allocation issues.

**Detection Method:** Coefficient of Variation (CV)
$$CV = \frac{\sigma}{\mu}$$

**Default Threshold:** CV < 0.05

**Example Issue:**
```
[DQ004] WARNING: Very low variability in oil data (CV=0.0234)
  Guidance: Flat production data may be synthetic or averaged;
            decline curve may not be appropriate
  Details:
    - CV: 0.0234
    - Threshold: 0.05
    - Mean: 1250.5
    - Std: 29.3
```

**Resolution:**

1. Verify data is actual metered production (not allocated)
2. Check if data has been smoothed or averaged
3. Decline curves may not be appropriate for flat production
4. Adjust `min_cv` threshold if needed

\newpage

# Fitting Prerequisite Validation (FP Codes)

Pre-fit validation ensures data is suitable for decline curve analysis.

## FP001: Insufficient Data Points

**Severity:** ERROR

**Description:** Not enough data points to perform reliable curve fitting.

**Default Minimum:** 6 months

**Example Issue:**
```
[FP001] ERROR: Insufficient oil data points: 4 < 6
  Guidance: Need at least 6 months of data for reliable fitting
  Details:
    - Point count: 4
    - Min required: 6
```

**Resolution:**

1. Obtain additional production history if available
2. Adjust `min_points` in config (not recommended below 6)
3. Skip fitting for this well/product combination

---

## FP002: Increasing Trend

**Severity:** WARNING

**Description:** Production shows an increasing trend rather than decline. Decline curve analysis assumes production is declining.

**Detection Method:** Log-linear regression slope > 0.01

**Example Issue:**
```
[FP002] WARNING: Increasing oil trend detected (not declining)
  Guidance: Decline curve fitting requires declining production;
            data may be early-time or ramping
  Details:
    - Monthly slope: 0.025
    - Annual rate: +30%
    - R-squared: 0.85
```

**Resolution:**

1. Verify well is in decline phase (not ramp-up)
2. Check for recent workover or stimulation
3. Wait for production to stabilize before fitting
4. Use regime detection to isolate declining segment

---

## FP003: Flat Trend

**Severity:** WARNING

**Description:** Production shows minimal decline, which hyperbolic models may not fit well.

**Example Issue:**
```
[FP003] WARNING: Flat oil trend detected (minimal decline)
  Guidance: Very stable production may not fit hyperbolic
            model well
  Details:
    - Monthly slope: -0.0005
    - R-squared: 0.92
```

**Resolution:**

1. Verify production pattern is accurate
2. Flat production may indicate artificial lift or water drive
3. Consider if decline forecasting is appropriate
4. Results may show near-exponential decline (low b-factor)

\newpage

# Fitting Result Validation (FR Codes)

Post-fit validation assesses the quality and reasonableness of fitted parameters.

## FR001: Poor Fit Quality

**Severity:** WARNING (R² 0.3-0.5) or ERROR (R² < 0.3)

**Description:** The fitted curve does not match the production data well.

**Default Threshold:** R² < 0.5

**Example Issue:**
```
[FR001] WARNING: Poor oil fit quality: R²=0.42
  Guidance: Low R² suggests poor model fit; consider
            data quality or alternative model
  Details:
    - R-squared: 0.42
    - Threshold: 0.50
    - RMSE: 125.3
```

**Resolution:**

1. Review data quality issues (outliers, gaps)
2. Check if regime detection selected appropriate segment
3. Consider if production pattern fits hyperbolic model
4. Manually review and adjust if needed

---

## FR003: B-Factor at Lower Bound

**Severity:** INFO

**Description:** The fitted b-factor is at the configured lower bound, suggesting near-exponential decline.

**Default Lower Bound:** 0.01

**Example Issue:**
```
[FR003] INFO: oil b-factor at lower bound (0.010)
  Guidance: B at lower bound suggests near-exponential decline;
            may be constrained by bound
  Details:
    - b: 0.010
    - b_min: 0.01
    - b_max: 1.50
```

**Resolution:**

- Usually acceptable (exponential decline is valid)
- If consistently hitting bound, may lower `b_min` in config
- Review if decline is truly exponential vs data artifact

---

## FR004: B-Factor at Upper Bound

**Severity:** WARNING

**Description:** The fitted b-factor is at the configured upper bound. Very high b-factors may indicate transient flow or data quality issues.

**Default Upper Bound:** 1.5

**Example Issue:**
```
[FR004] WARNING: oil b-factor at upper bound (1.500)
  Guidance: B at upper bound may indicate transient flow
            or data issues; review fit
  Details:
    - b: 1.500
    - b_min: 0.01
    - b_max: 1.50
```

**Resolution:**

1. Review production plot for unusual patterns
2. Check for early-time transient behavior
3. Verify data quality (outliers, gaps)
4. Consider increasing `b_max` if justified (super-harmonic)

---

## FR005: Very High Decline Rate

**Severity:** WARNING

**Description:** The fitted initial decline rate exceeds 100% per year, which is unusual and may indicate fitting issues.

**Default Threshold:** > 100%/year

**Example Issue:**
```
[FR005] WARNING: Very high oil decline rate: 156%/year
  Guidance: Decline >100%/year is unusual; verify data
            quality and fit
  Details:
    - Annual decline: 1.56
    - Monthly decline: 0.13
    - Threshold: 1.0
```

**Resolution:**

1. Verify data is monthly (not daily production)
2. Check for data quality issues in early months
3. Review if regime detection captured correct segment
4. High declines may be valid for tight oil/gas plays

\newpage

# Error Code Quick Reference

| Code | Severity | Category | Description |
|------|----------|----------|-------------|
| IV001 | ERROR | DATA_FORMAT | Negative production values |
| IV002 | WARNING | DATA_FORMAT | Values exceed threshold |
| IV003 | ERROR | DATA_FORMAT | Date parsing failed |
| IV004 | WARNING | DATA_FORMAT | Future dates in data |
| DQ001 | WARNING | DATA_QUALITY | Data gaps detected |
| DQ002 | WARNING | DATA_QUALITY | Outliers detected |
| DQ003 | INFO | DATA_QUALITY | Shut-in periods detected |
| DQ004 | WARNING | DATA_QUALITY | Low data variability |
| FP001 | ERROR | FITTING_PREREQ | Insufficient data points |
| FP002 | WARNING | FITTING_PREREQ | Increasing trend |
| FP003 | WARNING | FITTING_PREREQ | Flat trend |
| FR001 | WARN/ERR | FITTING_RESULT | Poor fit (R² < threshold) |
| FR003 | INFO | FITTING_RESULT | B-factor at lower bound |
| FR004 | WARNING | FITTING_RESULT | B-factor at upper bound |
| FR005 | WARNING | FITTING_RESULT | Very high decline rate |

\newpage

# Programmatic Usage

## Basic Validation

```python
from pyforecast.validation import (
    InputValidator,
    DataQualityValidator,
    FittingValidator,
)

# Create validators with custom thresholds
input_validator = InputValidator(
    max_oil_rate=75000,  # Higher threshold for prolific wells
    max_gas_rate=1000000,
)

quality_validator = DataQualityValidator(
    gap_threshold_months=3,  # Less sensitive to gaps
    outlier_sigma=2.5,       # More sensitive to outliers
)

fitting_validator = FittingValidator(
    min_r_squared=0.6,  # Stricter quality requirement
)

# Validate a well
input_result = input_validator.validate(well)
quality_result = quality_validator.validate(well, "oil")
fit_result = fitting_validator.validate_post_fit(well, "oil")

# Check results
if input_result.has_errors:
    print("Input validation failed!")
    for error in input_result.errors():
        print(f"  {error.code}: {error.message}")
```

## Working with ValidationResult

```python
from pyforecast.validation import (
    ValidationResult,
    IssueCategory,
    IssueSeverity,
    merge_results,
)

# Filter by category
quality_issues = result.by_category(IssueCategory.DATA_QUALITY)

# Filter by severity
errors = result.by_severity(IssueSeverity.ERROR)
warnings = result.warnings()

# Merge multiple results
combined = merge_results([input_result, quality_result, fit_result])

# Check status
print(f"Valid: {combined.is_valid}")
print(f"Errors: {combined.error_count}")
print(f"Warnings: {combined.warning_count}")
```

## Accessing Issue Details

```python
for issue in result.issues:
    print(f"Code: {issue.code}")
    print(f"Message: {issue.message}")
    print(f"Guidance: {issue.guidance}")

    # Access structured details
    if issue.code == "DQ002":  # Outliers
        print(f"Outlier indices: {issue.details['indices']}")
        print(f"Outlier values: {issue.details['values']}")
        print(f"Median: {issue.details['median']}")
```

\newpage

# Configuration Reference

## Complete Validation Configuration

```yaml
validation:
  # Input validation thresholds
  max_oil_rate: 50000      # bbl/month - IV002
  max_gas_rate: 500000     # mcf/month - IV002
  max_water_rate: 100000   # bbl/month - IV002

  # Data quality thresholds
  gap_threshold_months: 2  # Months - DQ001
  outlier_sigma: 3.0       # Std deviations - DQ002
  shutin_threshold: 1.0    # Rate threshold - DQ003
  min_cv: 0.05             # Coefficient of variation - DQ004

  # Fitting quality thresholds
  min_r_squared: 0.5       # R² threshold - FR001
  max_annual_decline: 1.0  # Annual fraction - FR005

  # Behavior
  strict_mode: false       # If true, treat warnings as errors
```

## Parameter Descriptions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_oil_rate` | 50,000 | Maximum expected oil rate (bbl/mo) |
| `max_gas_rate` | 500,000 | Maximum expected gas rate (mcf/mo) |
| `max_water_rate` | 100,000 | Maximum expected water rate (bbl/mo) |
| `gap_threshold_months` | 2 | Minimum consecutive zero months to flag |
| `outlier_sigma` | 3.0 | Modified Z-score threshold for outliers |
| `shutin_threshold` | 1.0 | Rate below which is considered shut-in |
| `min_cv` | 0.05 | Minimum coefficient of variation |
| `min_r_squared` | 0.5 | Minimum acceptable R² value |
| `max_annual_decline` | 1.0 | Maximum annual decline rate (1.0 = 100%) |
| `strict_mode` | false | Treat warnings as errors |

\newpage

# Validation Report Format

The validation report (`validation_report.txt`) is generated automatically in the output directory.

## Sample Report

```
PyForecast Validation Report
========================================

Summary:
  Wells with errors: 3
  Wells with warnings: 12
  Total errors: 5
  Total warnings: 18

Issues by category:
  DATA_FORMAT: 2
  DATA_QUALITY: 8
  FITTING_RESULT: 13

Detailed Issues:
----------------------------------------

WELL-001:
  [IV001] ERROR: Found 2 negative oil values
    Guidance: Production values must be non-negative;
              check data source for errors
  [DQ002] WARNING: Found 3 potential outliers in oil data
    Guidance: Review outlier values for data errors;
              consider excluding from fit

WELL-002:
  [FR001] WARNING: Poor oil fit quality: R²=0.45
    Guidance: Low R² suggests poor model fit; consider
              data quality or alternative model
  [FR004] WARNING: oil b-factor at upper bound (1.500)
    Guidance: B at upper bound may indicate transient
              flow or data issues; review fit

WELL-003:
  [DQ001] WARNING: Found 1 data gaps (>=2 months)
    Guidance: Gaps may indicate shut-ins or missing data;
              consider excluding or interpolating
```

\newpage

# Best Practices

## Data Preparation Checklist

Before running PyForecast:

- [ ] Verify all dates are in a consistent format
- [ ] Confirm production values are monthly (not daily/annual)
- [ ] Check for negative values and correct
- [ ] Remove forecast/projected data from historical records
- [ ] Verify units match expected (bbl, mcf)

## Interpreting Results

1. **Address ERRORs first** - These prevent reliable analysis
2. **Review WARNINGs** - Understand the cause before accepting
3. **Note INFOs** - Generally informational, no action needed

## Adjusting Thresholds

- **Prolific wells**: Increase `max_oil_rate` and `max_gas_rate`
- **Tight formations**: May have high decline - increase `max_annual_decline`
- **Noisy data**: Increase `outlier_sigma` to reduce false positives
- **Strict QC**: Set `strict_mode: true` to fail on any warning

## Common Workflows

### Initial Data Load
```bash
# First run - check for issues
pyforecast process data.csv -o output/ --no-plots

# Review validation report
cat output/validation_report.txt
```

### Production Processing
```bash
# With custom config for your basin/play
pyforecast process data.csv -o output/ --config basin_config.yaml
```

### Troubleshooting Poor Fits
1. Check `validation_report.txt` for data quality issues
2. Review individual well plots
3. Adjust config thresholds if needed
4. Re-run with corrected data

\newpage

# Appendix: Technical Details

## Outlier Detection Algorithm

PyForecast uses the Modified Z-Score method with Median Absolute Deviation (MAD) for robust outlier detection:

1. Calculate median of non-zero production values
2. Calculate MAD: `median(|xi - median|)`
3. Calculate Modified Z-Score: `0.6745 * (x - median) / MAD`
4. Flag values where `|Modified Z-Score| > outlier_sigma`

This method is robust to outliers in the data itself, unlike standard deviation-based methods.

## Trend Detection Algorithm

Pre-fit trend analysis uses log-linear regression:

1. Filter to non-zero production values
2. Apply natural log transformation: `ln(q)`
3. Fit linear regression: `ln(q) = a + b*t`
4. Interpret slope `b`:
   - `b > 0.01`: Increasing trend (FP002)
   - `|b| < 0.001` with high R²: Flat trend (FP003)
   - `b < -0.01`: Normal declining (no issue)

## Validation Execution Order

1. **Input Validation** (all products)
   - Date validation
   - Value range validation per product

2. **Data Quality Validation** (per product)
   - Gap detection
   - Outlier detection
   - Shut-in detection
   - Variability check

3. **Pre-Fit Validation** (per product)
   - Data sufficiency
   - Trend direction

4. **Fitting** (performed by DeclineFitter)

5. **Post-Fit Validation** (per product)
   - R² quality check
   - B-factor bounds check
   - Decline rate check
