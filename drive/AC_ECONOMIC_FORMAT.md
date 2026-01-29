# ARIES AC_ECONOMIC Format

## Overview

AC_ECONOMIC is the row-based ARIES forecast format. Each well has multiple rows with different KEYWORD values.

## Columns

| Column | Description |
|--------|-------------|
| PROPNUM | Well identifier (API number or property number) |
| SECTION | Always 4 for forecasts |
| SEQUENCE | Row ordering (1=CUMS, 2=START, 100+=products) |
| QUALIFIER | Date-based qualifier (e.g., "KA0125" for Jan 2025) |
| KEYWORD | Row type: CUMS, START, OIL, GAS, WATER, or `"` for continuation |
| EXPRESSION | Data/formula for this row |

## Row Types

### CUMS Row (SEQUENCE=1)
Cumulative production to date.
```
EXPRESSION: "{oil_mbbl} {gas_mmcf}"
Example: "150.234 89.456"
```

### START Row (SEQUENCE=2)
Forecast effective date (month after last production).
```
EXPRESSION: "MM/YYYY"
Example: "02/2025"
```

### Product Rows (OIL/GAS/WATER, SEQUENCE=100/300/500)
Decline curve parameters.
```
EXPRESSION: "{qi} X {unit} {dmin%} {type} B/{b} {di%}"

Components:
- qi: Initial rate (daily)
- unit: B/D (oil/water) or M/D (gas)
- dmin%: Terminal decline rate (annual %)
- type: EXP, HYP, or HRM
- b: Hyperbolic exponent
- di%: Initial decline rate (annual %)

Example: "85.3 X B/D 6 EXP B/0.50 8.5"
         = 85.3 bbl/day, 6% terminal, exponential, b=0.50, 8.5% initial
```

### Continuation Rows (KEYWORD=", SEQUENCE=200/400/600)
Terminal decline phase.
```
EXPRESSION: "X 1 {unit} X YRS EXP {dmin%}"
Example: "X 1 B/D X YRS EXP 6"
```

## Sample Import File

```csv
PROPNUM,SECTION,SEQUENCE,QUALIFIER,KEYWORD,EXPRESSION
42-123-45678,4,1,KA0125,CUMS,150.234 89.456
42-123-45678,4,2,KA0125,START,02/2025
42-123-45678,4,100,KA0125,OIL,85.3 X B/D 6 EXP B/0.50 8.5
42-123-45678,4,200,KA0125,",X 1 B/D X YRS EXP 6
42-123-45678,4,300,KA0125,GAS,450 X M/D 6 HYP B/0.75 12
42-123-45678,4,400,KA0125,",X 1 M/D X YRS EXP 6
```

## Sample Export File

```csv
PROPNUM,SECTION,SEQUENCE,QUALIFIER,KEYWORD,EXPRESSION
WELL001,4,1,KA0126,CUMS,9.310 47.940
WELL001,4,2,KA0126,START,01/2023
WELL001,4,100,KA0126,OIL,33.5 X B/D 6 EXP B/0.48 68.11
WELL001,4,200,KA0126,",X 1 B/D X YRS EXP 6
WELL001,4,300,KA0126,GAS,168.1 X M/D 6 EXP B/0.56 61.88
WELL001,4,400,KA0126,",X 1 M/D X YRS EXP 6
```

## Key Differences: Import vs Export

| Aspect | Import (ARIES) | Export (pyforecast) |
|--------|---------------|---------------------|
| START date | Parsed from file | Computed from last_date + 1 month |
| CUMS | Typically from ARIES reserves | Computed from production history |
| qi | Parsed from expression | Model's fitted qi (daily) |
| Rates | Can be daily or monthly | Always daily (B/D, M/D) |
