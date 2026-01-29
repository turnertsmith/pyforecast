"""Parser for Enverus production data exports."""

import numpy as np
import pandas as pd

from .base import DataParser
from .well import Well, WellIdentifier, ProductionData


class EnverusParser(DataParser):
    """Parser for Enverus (formerly DrillingInfo) CSV exports.

    Expected columns (case-insensitive):
    - Entity ID or API Number: Well identifier
    - Well Name: Optional well name
    - Production Date or Date: Production month
    - Oil or Liquid (BBL): Oil production
    - Gas (MCF): Gas production
    - Water (BBL): Optional water production
    """

    # Column name mappings (lowercase -> standard name)
    COLUMN_MAPPINGS = {
        # Identifiers
        'entity id': 'entity_id',
        'entityid': 'entity_id',
        'entity_id': 'entity_id',
        'api': 'api',
        'api number': 'api',
        'api_number': 'api',
        'api14': 'api',
        'api 14': 'api',
        'well name': 'well_name',
        'wellname': 'well_name',
        'well_name': 'well_name',
        # Dates
        'production date': 'date',
        'productiondate': 'date',
        'production_date': 'date',
        'date': 'date',
        'prod date': 'date',
        'month': 'date',
        # Oil
        'oil': 'oil',
        'oil (bbl)': 'oil',
        'oil bbl': 'oil',
        'oil_bbl': 'oil',
        'liquid': 'oil',
        'liquid (bbl)': 'oil',
        'liq': 'oil',
        'monthly oil': 'oil',
        # Gas
        'gas': 'gas',
        'gas (mcf)': 'gas',
        'gas mcf': 'gas',
        'gas_mcf': 'gas',
        'monthly gas': 'gas',
        # Water
        'water': 'water',
        'water (bbl)': 'water',
        'water bbl': 'water',
        'water_bbl': 'water',
        'monthly water': 'water',
    }

    def can_parse(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame has Enverus-style columns."""
        cols_lower = {c.lower().strip(): c for c in df.columns}

        # Must have entity_id or api
        has_id = any(
            k in cols_lower for k in ['entity id', 'entityid', 'entity_id', 'api', 'api number', 'api14']
        )

        # Must have date column
        has_date = any(
            k in cols_lower for k in ['production date', 'productiondate', 'date', 'prod date', 'month']
        )

        # Must have at least oil or gas
        has_production = any(
            k in cols_lower for k in ['oil', 'oil (bbl)', 'gas', 'gas (mcf)', 'liquid']
        )

        return has_id and has_date and has_production

    def _map_columns(self, df: pd.DataFrame) -> dict[str, str]:
        """Map DataFrame columns to standard names.

        Returns:
            Dictionary mapping standard name -> actual column name
        """
        cols_lower = {c.lower().strip(): c for c in df.columns}
        mapping = {}

        for col_lower, actual_col in cols_lower.items():
            if col_lower in self.COLUMN_MAPPINGS:
                standard_name = self.COLUMN_MAPPINGS[col_lower]
                # Don't overwrite if already mapped (priority to first match)
                if standard_name not in mapping:
                    mapping[standard_name] = actual_col

        return mapping

    def parse(self, df: pd.DataFrame) -> list[Well]:
        """Parse Enverus DataFrame into Well objects."""
        col_map = self._map_columns(df)

        # Determine ID column
        id_col = col_map.get('entity_id') or col_map.get('api')
        if not id_col:
            raise ValueError("No identifier column found")

        date_col = col_map.get('date')
        if not date_col:
            raise ValueError("No date column found")

        oil_col = col_map.get('oil')
        gas_col = col_map.get('gas')
        water_col = col_map.get('water')
        name_col = col_map.get('well_name')

        # Convert date column
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        # Group by well
        wells = []
        for well_id, group in df.groupby(id_col):
            group = group.sort_values(date_col)

            # Create identifier
            identifier = WellIdentifier(
                entity_id=str(well_id) if 'entity_id' in col_map else None,
                api=str(well_id) if 'api' in col_map and 'entity_id' not in col_map else None,
                well_name=str(group[name_col].iloc[0]) if name_col and name_col in group.columns else None,
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
