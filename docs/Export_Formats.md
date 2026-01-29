# PyForecast Export Formats

This guide documents the output formats supported by PyForecast for exporting decline curve forecasts.

## Overview

PyForecast supports two export formats:

| Format | Use Case | CLI Flag |
|--------|----------|----------|
| **AC_ECONOMIC** | ARIES software import | `--format ac_economic` (default) |
| **JSON** | API integration, custom tools | `--format json` |

---

## AC_ECONOMIC Format

The AC_ECONOMIC format is a row-based CSV structure compatible with ARIES economic software. Each well has multiple rows with different KEYWORD values.

### Column Structure

| Column | Type | Description |
|--------|------|-------------|
| `PROPNUM` | string | Well identifier (API number or property number) |
| `SECTION` | int | Always `4` for forecasts |
| `SEQUENCE` | int | Row ordering (determines display order in ARIES) |
| `QUALIFIER` | string | Date-based qualifier (e.g., "KA0126" for Jan 2026) |
| `KEYWORD` | string | Row type: `CUMS`, `START`, `OIL`, `GAS`, `WATER`, or `"` |
| `EXPRESSION` | string | Data or formula for this row |

### Row Types

#### CUMS Row (SEQUENCE=1)

Cumulative production to date. Contains historical cumulative oil and gas volumes.

```
EXPRESSION: "{oil_mbbl} {gas_mmcf}"
```

| Field | Unit | Description |
|-------|------|-------------|
| oil_mbbl | MBbl | Cumulative oil in thousand barrels |
| gas_mmcf | MMcf | Cumulative gas in million cubic feet |

**Example:**
```csv
WELL001,4,1,KA0126,CUMS,150.234 89.456
```

#### START Row (SEQUENCE=2)

Forecast effective date. Computed as the month after the last production date.

```
EXPRESSION: "MM/YYYY"
```

**Example:**
```csv
WELL001,4,2,KA0126,START,02/2025
```

#### Product Rows (OIL/GAS/WATER)

Decline curve parameters for each product. Sequence numbers:
- OIL: 100
- GAS: 300
- WATER: 500

```
EXPRESSION: "{qi} X {unit} {dmin%} EXP B/{b} {di%}"
```

| Component | Description | Example |
|-----------|-------------|---------|
| `{qi}` | Initial rate (daily) | `85.3` |
| `X` | Literal separator | `X` |
| `{unit}` | Unit code: `B/D` (oil/water) or `M/D` (gas) | `B/D` |
| `{dmin%}` | Terminal decline rate (annual %) | `6` |
| `EXP` | Decline type (always EXP in ARIES) | `EXP` |
| `B/{b}` | Hyperbolic b-factor | `B/0.50` |
| `{di%}` | Initial decline rate (annual %) | `8.5` |

**Example:**
```csv
WELL001,4,100,KA0126,OIL,85.3 X B/D 6 EXP B/0.50 8.5
```

This means:
- Initial rate: 85.3 barrels/day
- Terminal decline: 6%/year
- B-factor: 0.50
- Initial decline: 8.5%/year

#### Continuation Rows (Terminal Decline)

Terminal decline phase after switching from hyperbolic. Uses `"` as keyword.
Sequence numbers: 200 (oil), 400 (gas), 600 (water)

```
EXPRESSION: "X 1 {unit} X YRS EXP {dmin%}"
```

**Example:**
```csv
WELL001,4,200,KA0126,",X 1 B/D X YRS EXP 6
```

### Complete Example

```csv
PROPNUM,SECTION,SEQUENCE,QUALIFIER,KEYWORD,EXPRESSION
WELL001,4,1,KA0126,CUMS,9.310 47.940
WELL001,4,2,KA0126,START,01/2023
WELL001,4,100,KA0126,OIL,33.5 X B/D 6 EXP B/0.48 68.11
WELL001,4,200,KA0126,",X 1 B/D X YRS EXP 6
WELL001,4,300,KA0126,GAS,168.1 X M/D 6 EXP B/0.56 61.88
WELL001,4,400,KA0126,",X 1 M/D X YRS EXP 6
WELL002,4,1,KA0126,CUMS,5.120 23.450
WELL002,4,2,KA0126,START,02/2023
WELL002,4,100,KA0126,OIL,45.2 X B/D 6 EXP B/0.35 52.30
WELL002,4,200,KA0126,",X 1 B/D X YRS EXP 6
```

### Unit Specifications

| Product | Unit Code | Unit | Rate Type |
|---------|-----------|------|-----------|
| Oil | `B/D` | Barrels per day | Daily |
| Gas | `M/D` | Mcf per day | Daily |
| Water | `B/D` | Barrels per day | Daily |

**Important:** PyForecast exports use daily rates. The model internally stores daily rates (qi is in bbl/day or mcf/day), which matches ARIES conventions.

### Key Points

1. **Always EXP**: ARIES convention uses `EXP` for all decline types. The b-factor is stored but the decline keyword is always `EXP`.

2. **Daily rates**: Output uses B/D (barrels/day) and M/D (mcf/day), not monthly rates.

3. **START date**: Computed as the month after the last production date.

4. **CUMS**: Cumulative production in MBbl and MMcf, calculated from historical data.

5. **Qualifier format**: Generated as "KA" + month (2 digits) + year (2 digits). Example: "KA0126" for January 2026.

---

## JSON Format

The JSON export provides a structured format suitable for API integration and custom tooling.

### Structure

```json
{
  "metadata": {
    "generated_at": "2026-01-29T10:30:00Z",
    "pyforecast_version": "0.1.0",
    "config": {
      "products": ["oil", "gas"],
      "b_min": 0.01,
      "b_max": 1.5,
      "dmin_annual": 0.06
    }
  },
  "wells": [
    {
      "well_id": "WELL001",
      "identifier": {
        "api": "42-123-45678",
        "propnum": "WELL001",
        "well_name": "Smith 1H"
      },
      "production_summary": {
        "first_date": "2020-01-01",
        "last_date": "2024-12-01",
        "n_months": 60,
        "cumulative_oil_bbl": 150234,
        "cumulative_gas_mcf": 89456
      },
      "forecasts": {
        "oil": {
          "qi": 33.5,
          "qi_unit": "bbl/day",
          "di": 0.0568,
          "di_unit": "fraction/month",
          "di_annual": 0.6811,
          "b": 0.48,
          "dmin": 0.005,
          "dmin_annual": 0.06,
          "decline_type": "HYP",
          "t_switch_months": 45.2,
          "r_squared": 0.923,
          "rmse": 15.3,
          "aic": 125.4,
          "bic": 128.7,
          "regime_start_idx": 0,
          "data_points_used": 48
        },
        "gas": {
          "qi": 168.1,
          "qi_unit": "mcf/day",
          "di": 0.0516,
          "di_annual": 0.6188,
          "b": 0.56,
          "dmin": 0.005,
          "dmin_annual": 0.06,
          "decline_type": "HYP",
          "t_switch_months": 52.8,
          "r_squared": 0.945,
          "rmse": 45.2,
          "aic": 312.1,
          "bic": 315.4,
          "regime_start_idx": 0,
          "data_points_used": 48
        }
      },
      "validation": {
        "has_errors": false,
        "has_warnings": true,
        "issues": [
          {
            "code": "DQ002",
            "severity": "WARNING",
            "message": "Found 2 potential outliers in oil data",
            "category": "DATA_QUALITY"
          }
        ]
      }
    }
  ]
}
```

### Field Descriptions

#### Metadata Section

| Field | Type | Description |
|-------|------|-------------|
| `generated_at` | string | ISO 8601 timestamp |
| `pyforecast_version` | string | Version of PyForecast |
| `config` | object | Configuration used for this run |

#### Well Object

| Field | Type | Description |
|-------|------|-------------|
| `well_id` | string | Primary well identifier |
| `identifier` | object | All available identifiers |
| `production_summary` | object | Historical production statistics |
| `forecasts` | object | Forecast results by product |
| `validation` | object | Validation results |

#### Forecast Object

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `qi` | float | bbl/day or mcf/day | Initial rate at t=0 |
| `qi_unit` | string | - | Unit label for qi |
| `di` | float | fraction/month | Initial decline rate (monthly) |
| `di_annual` | float | fraction/year | Initial decline rate (annual) |
| `b` | float | - | Hyperbolic b-factor (0-1.5 typical) |
| `dmin` | float | fraction/month | Terminal decline rate (monthly) |
| `dmin_annual` | float | fraction/year | Terminal decline rate (annual) |
| `decline_type` | string | - | `EXP`, `HYP`, or `HRM` |
| `t_switch_months` | float | months | Time to switch to terminal decline |
| `r_squared` | float | - | Coefficient of determination (0-1) |
| `rmse` | float | same as qi | Root mean squared error |
| `aic` | float | - | Akaike Information Criterion |
| `bic` | float | - | Bayesian Information Criterion |
| `regime_start_idx` | int | - | Index where current regime starts |
| `data_points_used` | int | - | Number of data points in fit |

#### Decline Type Values

| Value | Description | B-Factor Range |
|-------|-------------|----------------|
| `EXP` | Exponential decline | b ≤ 0.1 |
| `HYP` | Hyperbolic decline | 0.1 < b < 0.95 |
| `HRM` | Harmonic decline | b ≥ 0.95 |

### Validation Object

| Field | Type | Description |
|-------|------|-------------|
| `has_errors` | boolean | True if any ERROR-level issues |
| `has_warnings` | boolean | True if any WARNING-level issues |
| `issues` | array | List of validation issues |

Each issue contains:
- `code`: Error code (e.g., "DQ002")
- `severity`: ERROR, WARNING, or INFO
- `message`: Human-readable description
- `category`: DATA_FORMAT, DATA_QUALITY, FITTING_PREREQ, or FITTING_RESULT

---

## Generating Rate Forecasts

To generate actual rate forecasts (time series) from the exported parameters:

### From AC_ECONOMIC Parameters

```python
import numpy as np

# Parsed from expression: "85.3 X B/D 6 EXP B/0.50 8.5"
qi = 85.3        # bbl/day
di_annual = 0.085  # 8.5% annual
b = 0.50
dmin_annual = 0.06  # 6% annual

# Convert to monthly
di_monthly = di_annual / 12
dmin_monthly = dmin_annual / 12

# Calculate switch time (months)
if b > 0.01:
    t_switch = (di_monthly / dmin_monthly - 1) / (b * di_monthly)
else:
    t_switch = float('inf')

# Generate rates
def rate(t_months):
    q = np.zeros_like(t_months, dtype=float)

    # Hyperbolic phase
    mask = t_months <= t_switch
    if b <= 0.01:
        q[mask] = qi * np.exp(-di_monthly * t_months[mask])
    else:
        q[mask] = qi / np.power(1 + b * di_monthly * t_months[mask], 1/b)

    # Exponential terminal phase
    mask = t_months > t_switch
    if np.any(mask):
        q_at_switch = qi / np.power(1 + b * di_monthly * t_switch, 1/b)
        t_after = t_months[mask] - t_switch
        q[mask] = q_at_switch * np.exp(-dmin_monthly * t_after)

    return q

# Example: 60 months forecast
t = np.arange(60)
rates = rate(t)  # bbl/day
```

### From JSON Parameters

```python
import json

with open("forecasts.json") as f:
    data = json.load(f)

well = data["wells"][0]
oil = well["forecasts"]["oil"]

qi = oil["qi"]            # Already in bbl/day
di = oil["di"]            # Already monthly
b = oil["b"]
dmin = oil["dmin"]        # Already monthly
t_switch = oil["t_switch_months"]

# Use same rate function as above
```

---

## Converting Between Formats

### AC_ECONOMIC to JSON

```bash
# Process and export both formats
pyforecast process data.csv --format json -o output_json/
pyforecast process data.csv --format ac_economic -o output_aries/

# Or use Python
from pyforecast.export import AriesAcEconomicExporter, JsonExporter

wells = load_and_process(data)

aries_exporter = AriesAcEconomicExporter()
aries_exporter.save(wells, "forecasts.csv")

json_exporter = JsonExporter()
json_exporter.save(wells, "forecasts.json")
```

### Monthly vs Daily Rate Conversion

PyForecast internally uses **daily rates** for all calculations and exports:

```python
# Monthly to daily (approximate for 30.4-day month)
rate_daily = rate_monthly / 30.4

# Daily to monthly
rate_monthly = rate_daily * 30.4

# For more precision, use actual days in month
import calendar
days_in_month = calendar.monthrange(year, month)[1]
rate_daily = rate_monthly / days_in_month
```

---

## Importing AC_ECONOMIC Files

PyForecast can import existing ARIES AC_ECONOMIC files for ground truth comparison:

```python
from pyforecast.import_ import AriesForecastImporter

importer = AriesForecastImporter()
count = importer.load("existing_forecasts.csv")
print(f"Loaded {count} forecasts")

# Get forecast for specific well
params = importer.get("42-123-45678", "oil")
if params:
    print(f"qi={params.qi}, di={params.di_monthly}, b={params.b}")
```

The importer parses the expression syntax and extracts:
- `qi`: Initial rate (converted to daily)
- `di_monthly`: Initial decline (monthly fraction)
- `di_annual`: Initial decline (annual fraction)
- `b`: Hyperbolic b-factor
- `dmin_monthly`: Terminal decline (monthly)
- `dmin_annual`: Terminal decline (annual)
- `decline_type`: EXP, HYP, or HRM

---

## Best Practices

1. **Use AC_ECONOMIC for ARIES** - This is the native format and requires no conversion.

2. **Use JSON for APIs** - Structured data is easier to parse programmatically.

3. **Check validation** - Review `validation_report.txt` alongside exports.

4. **Verify units** - Ensure downstream systems expect daily rates.

5. **Store original data** - Keep both the export and the source production data.
