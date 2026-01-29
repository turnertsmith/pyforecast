"""Parser for ARIES production data exports."""

import numpy as np
import pandas as pd

from .base import DataParser
from .well import Well, WellIdentifier, ProductionData


class AriesParser(DataParser):
    """Parser for ARIES production data CSV exports.

    Expected columns (case-insensitive):
    - PROPNUM: Property number (well identifier)
    - P_DATE or YYYYMM: Production date
    - OIL or MONTHLY_OIL: Oil production (bbl)
    - GAS or MONTHLY_GAS: Gas production (mcf)
    - WATER or MONTHLY_WATER: Optional water production
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
