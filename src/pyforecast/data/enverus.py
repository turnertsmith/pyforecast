"""Parser for Enverus production data exports.

This module provides the EnverusParser class for parsing CSV files exported
from Enverus (formerly DrillingInfo) production database.

Format Detection
----------------

The parser automatically detects Enverus format when a CSV file has:
1. A well identifier column (Entity ID, API Number, or similar)
2. A date column (Production Date, Date, etc.)
3. At least one production column (Oil, Gas, Liquid)

The detection is case-insensitive and tolerant of spacing variations.

Column Mapping
--------------

The parser maps common Enverus column names to standard internal names:

Well Identifiers:
    - entity id, entityid, entity_id -> entity_id
    - api, api number, api_number, api14, api 14 -> api
    - well name, wellname, well_name -> well_name

Dates:
    - production date, productiondate, production_date -> date
    - date, prod date, month -> date

Production Volumes:
    - oil, oil (bbl), oil bbl, oil_bbl, liquid, liquid (bbl), liq -> oil
    - gas, gas (mcf), gas mcf, gas_mcf, monthly gas -> gas
    - water, water (bbl), water bbl, water_bbl, monthly water -> water

Expected Units:
    - Oil: barrels (bbl) per month
    - Gas: thousand cubic feet (mcf) per month
    - Water: barrels (bbl) per month

Example Input File
------------------

```csv
Entity ID,Well Name,Production Date,Oil (BBL),Gas (MCF),Water (BBL)
12345678,Smith 1H,2020-01-01,5000,25000,1000
12345678,Smith 1H,2020-02-01,4500,22000,1100
12345679,Jones 2H,2020-01-01,6000,30000,800
```
"""

import pandas as pd

from .base import DataParser
from .well import WellIdentifier


class EnverusParser(DataParser):
    """Parser for Enverus (formerly DrillingInfo) CSV exports.

    This parser handles the standard Enverus production data export format,
    with flexible column name matching to accommodate variations in exports
    from different Enverus products and versions.

    Format Detection Logic:
        The parser identifies Enverus format by checking for:
        1. An identifier column (entity_id OR api number)
        2. A date column (production date, date, month)
        3. At least one production column (oil, gas, liquid)

        All checks are case-insensitive.

    Column Priority:
        When multiple columns match the same standard name, the first match
        (in the order columns appear in the file) takes priority.

    Attributes:
        COLUMN_MAPPINGS: Dictionary mapping lowercase column names to standard
            internal names. Used for flexible column matching.

    Example:
        >>> parser = EnverusParser()
        >>> if parser.can_parse(df):
        ...     wells = parser.parse(df)
        ...     for well in wells:
        ...         print(f"{well.well_id}: {len(well.production.oil)} months")

    See Also:
        AriesParser: For ARIES-format production data
        DataParser: Abstract base class for data parsers
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

    def _resolve_id_column(self, col_map: dict[str, str]) -> str:
        """Resolve Enverus ID column (entity_id or api)."""
        id_col = col_map.get('entity_id') or col_map.get('api')
        if not id_col:
            raise ValueError("No identifier column found")
        return id_col

    def _build_identifier(self, well_id: str, col_map: dict[str, str], group: pd.DataFrame) -> WellIdentifier:
        """Build identifier from Enverus columns."""
        name_col = col_map.get('well_name')
        return WellIdentifier(
            entity_id=well_id if 'entity_id' in col_map else None,
            api=well_id if 'api' in col_map and 'entity_id' not in col_map else None,
            well_name=str(group[name_col].iloc[0]) if name_col and name_col in group.columns else None,
        )
