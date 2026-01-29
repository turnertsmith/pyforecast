"""Parser for ARIES production data exports.

This module provides the AriesParser class for parsing CSV files exported
from ARIES (or ARIES-compatible) production databases.

Format Detection
----------------

The parser automatically detects ARIES format when a CSV file has:
1. A PROPNUM column (property number identifier), OR
2. An ARIES-style date column (P_DATE, YYYYMM format)
3. At least one production column (OIL, GAS, MONTHLY_OIL, etc.)

ARIES format is distinguished from Enverus by the presence of PROPNUM
(rather than Entity ID) and the YYYYMM date format.

Column Mapping
--------------

The parser maps common ARIES column names to standard internal names:

Well Identifiers:
    - propnum, prop_num, property, property number, well -> propnum
    - api -> api

Dates:
    - p_date, pdate, prod_date -> date
    - yyyymm, year_month, date -> date

Production Volumes:
    - oil, monthly_oil, monthlyoil, oil_prod, oil_volume -> oil
    - gas, monthly_gas, monthlygas, gas_prod, gas_volume -> gas
    - water, monthly_water, monthlywater, water_prod, water_volume -> water

Date Format Handling
--------------------

The parser automatically detects and handles multiple ARIES date formats:

1. YYYYMM (numeric): 202301 -> 2023-01-01
2. YYYY-MM (string): "2023-01" -> 2023-01-01
3. Standard date formats: Parsed via pandas.to_datetime()

Expected Units:
    - Oil: barrels (bbl) per month
    - Gas: thousand cubic feet (mcf) per month
    - Water: barrels (bbl) per month

Example Input Files
-------------------

Format 1 - YYYYMM dates:
```csv
PROPNUM,YYYYMM,MONTHLY_OIL,MONTHLY_GAS,MONTHLY_WATER
WELL001,202001,5000,25000,1000
WELL001,202002,4500,22000,1100
WELL002,202001,6000,30000,800
```

Format 2 - P_DATE dates:
```csv
PROPNUM,P_DATE,OIL,GAS,WATER
WELL001,2020-01-01,5000,25000,1000
WELL001,2020-02-01,4500,22000,1100
WELL002,2020-01-01,6000,30000,800
```
"""

import numpy as np
import pandas as pd

from .base import DataParser
from .well import Well, WellIdentifier, ProductionData


class AriesParser(DataParser):
    """Parser for ARIES production data CSV exports.

    This parser handles the standard ARIES production data export format,
    with flexible column name matching and multiple date format support.

    Format Detection Logic:
        The parser identifies ARIES format by checking for:
        1. A PROPNUM-style identifier column, OR
        2. An ARIES-style date column (P_DATE, YYYYMM)
        3. At least one production column (oil, gas)

        ARIES format takes precedence when both PROPNUM and Entity ID exist.

    Date Parsing:
        The parser auto-detects date format:
        - 6-digit numeric (YYYYMM): Parsed as year-month
        - 7-char with dash (YYYY-MM): Parsed as year-month
        - Other formats: Passed to pandas.to_datetime()

    Attributes:
        COLUMN_MAPPINGS: Dictionary mapping lowercase column names to standard
            internal names. Used for flexible column matching.

    Example:
        >>> parser = AriesParser()
        >>> if parser.can_parse(df):
        ...     wells = parser.parse(df)
        ...     for well in wells:
        ...         print(f"{well.well_id}: {len(well.production.oil)} months")

    See Also:
        EnverusParser: For Enverus-format production data
        DataParser: Abstract base class for data parsers
    """

    # Column name mappings (lowercase -> standard name)
    COLUMN_MAPPINGS = {
        # Identifiers
        'propnum': 'propnum',
        'prop_num': 'propnum',
        'property': 'propnum',
        'property number': 'propnum',
        'well': 'propnum',
        'api': 'api',
        # Dates
        'p_date': 'date',
        'pdate': 'date',
        'prod_date': 'date',
        'yyyymm': 'date',
        'year_month': 'date',
        'date': 'date',
        # Oil
        'oil': 'oil',
        'monthly_oil': 'oil',
        'monthlyoil': 'oil',
        'oil_prod': 'oil',
        'oil_volume': 'oil',
        # Gas
        'gas': 'gas',
        'monthly_gas': 'gas',
        'monthlygas': 'gas',
        'gas_prod': 'gas',
        'gas_volume': 'gas',
        # Water
        'water': 'water',
        'monthly_water': 'water',
        'monthlywater': 'water',
        'water_prod': 'water',
        'water_volume': 'water',
    }

    def can_parse(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has ARIES-style columns."""
        cols_lower = {c.lower().strip(): c for c in df.columns}

        # Must have propnum
        has_propnum = any(
            k in cols_lower for k in ['propnum', 'prop_num', 'property', 'property number']
        )

        # Or ARIES-style date format
        has_aries_date = any(
            k in cols_lower for k in ['p_date', 'pdate', 'yyyymm']
        )

        # Must have production
        has_production = any(
            k in cols_lower for k in ['oil', 'monthly_oil', 'gas', 'monthly_gas']
        )

        # ARIES typically has propnum + production
        # Also detect by yyyymm date format
        return (has_propnum or has_aries_date) and has_production

    def _map_columns(self, df: pd.DataFrame) -> dict[str, str]:
        """Map DataFrame columns to standard names."""
        cols_lower = {c.lower().strip(): c for c in df.columns}
        mapping = {}

        for col_lower, actual_col in cols_lower.items():
            if col_lower in self.COLUMN_MAPPINGS:
                standard_name = self.COLUMN_MAPPINGS[col_lower]
                if standard_name not in mapping:
                    mapping[standard_name] = actual_col

        return mapping

    def _parse_aries_date(self, date_series: pd.Series) -> pd.Series:
        """Parse ARIES date formats (YYYYMM or various date formats).

        Args:
            date_series: Series with date values

        Returns:
            Series with datetime values
        """
        # Try to detect format
        sample = str(date_series.iloc[0])

        # YYYYMM format (e.g., 202301)
        if sample.isdigit() and len(sample) == 6:
            return pd.to_datetime(date_series.astype(str), format='%Y%m')

        # YYYY-MM format
        if len(sample) == 7 and '-' in sample:
            return pd.to_datetime(date_series, format='%Y-%m')

        # Try general parsing
        return pd.to_datetime(date_series)

    def parse(self, df: pd.DataFrame) -> list[Well]:
        """Parse ARIES DataFrame into Well objects."""
        col_map = self._map_columns(df)

        # Determine ID column
        id_col = col_map.get('propnum') or col_map.get('api')
        if not id_col:
            raise ValueError("No identifier column (PROPNUM) found")

        date_col = col_map.get('date')
        if not date_col:
            raise ValueError("No date column found")

        oil_col = col_map.get('oil')
        gas_col = col_map.get('gas')
        water_col = col_map.get('water')

        # Convert date column
        df = df.copy()
        df[date_col] = self._parse_aries_date(df[date_col])

        # Group by well
        wells = []
        for well_id, group in df.groupby(id_col):
            group = group.sort_values(date_col)

            # Create identifier
            identifier = WellIdentifier(
                propnum=str(well_id),
            )

            # Extract production arrays
            dates = group[date_col].values
            oil = group[oil_col].fillna(0).values if oil_col else np.zeros(len(group))
            gas = group[gas_col].fillna(0).values if gas_col else np.zeros(len(group))
            water = group[water_col].fillna(0).values if water_col else None

            production = ProductionData(
                dates=dates,
                oil=oil.astype(float),
                gas=gas.astype(float),
                water=water.astype(float) if water is not None else None,
            )

            wells.append(Well(
                identifier=identifier,
                production=production,
            ))

        return wells
