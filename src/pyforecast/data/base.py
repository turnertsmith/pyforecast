"""Abstract base class for data parsers."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from .well import Well, WellIdentifier, ProductionData


class DataParser(ABC):
    """Abstract base class for production data parsers."""

    # Subclasses must define COLUMN_MAPPINGS: dict[str, str]
    COLUMN_MAPPINGS: dict[str, str] = {}

    @abstractmethod
    def can_parse(self, df: pd.DataFrame) -> bool:
        """Check if this parser can handle the given DataFrame.

        Args:
            df: DataFrame loaded from input file

        Returns:
            True if this parser recognizes the format
        """
        pass

    def _map_columns(self, df: pd.DataFrame) -> dict[str, str]:
        """Map DataFrame columns to standard names using COLUMN_MAPPINGS.

        Returns:
            Dictionary mapping standard name -> actual column name
        """
        cols_lower = {c.lower().strip(): c for c in df.columns}
        mapping: dict[str, str] = {}

        for col_lower, actual_col in cols_lower.items():
            if col_lower in self.COLUMN_MAPPINGS:
                standard_name = self.COLUMN_MAPPINGS[col_lower]
                # Don't overwrite if already mapped (priority to first match)
                if standard_name not in mapping:
                    mapping[standard_name] = actual_col

        return mapping

    def _parse_dates(self, date_series: pd.Series) -> pd.Series:
        """Parse date column into datetime. Subclasses can override for custom formats.

        Args:
            date_series: Series with date values

        Returns:
            Series with datetime values
        """
        return pd.to_datetime(date_series)

    def _build_identifier(self, well_id: str, col_map: dict[str, str], group: pd.DataFrame) -> WellIdentifier:
        """Build a WellIdentifier from parsed data. Subclasses should override.

        Args:
            well_id: The grouped well identifier value
            col_map: Column mapping dict
            group: DataFrame group for this well

        Returns:
            WellIdentifier instance
        """
        return WellIdentifier()

    def parse(self, df: pd.DataFrame) -> list[Well]:
        """Parse DataFrame into list of Well objects.

        Args:
            df: DataFrame with production data

        Returns:
            List of Well objects with production data populated
        """
        col_map = self._map_columns(df)

        id_col = self._resolve_id_column(col_map)
        date_col = col_map.get('date')
        if not date_col:
            raise ValueError("No date column found")

        oil_col = col_map.get('oil')
        gas_col = col_map.get('gas')
        water_col = col_map.get('water')

        # Convert date column
        df = df.copy()
        df[date_col] = self._parse_dates(df[date_col])

        # Group by well
        wells = []
        for well_id, group in df.groupby(id_col):
            group = group.sort_values(date_col)

            identifier = self._build_identifier(str(well_id), col_map, group)

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

    @abstractmethod
    def _resolve_id_column(self, col_map: dict[str, str]) -> str:
        """Resolve the well identifier column from the column mapping.

        Args:
            col_map: Standard name -> actual column name mapping

        Returns:
            Actual column name to group by

        Raises:
            ValueError: If no identifier column found
        """
        pass

    @classmethod
    def load_file(cls, filepath: Path | str) -> pd.DataFrame:
        """Load CSV or Excel file into DataFrame.

        Args:
            filepath: Path to input file

        Returns:
            pandas DataFrame

        Raises:
            ValueError: If file format not supported
        """
        filepath = Path(filepath)

        if filepath.suffix.lower() == '.csv':
            return pd.read_csv(filepath)
        elif filepath.suffix.lower() in ('.xlsx', '.xls'):
            return pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")


def detect_parser(df: pd.DataFrame) -> DataParser:
    """Auto-detect appropriate parser for DataFrame.

    Args:
        df: DataFrame loaded from input file

    Returns:
        Appropriate DataParser instance

    Raises:
        ValueError: If no parser can handle the format
    """
    # Import here to avoid circular imports
    from .enverus import EnverusParser
    from .aries import AriesParser

    parsers = [EnverusParser(), AriesParser()]

    for parser in parsers:
        if parser.can_parse(df):
            return parser

    # Show available columns to help diagnose
    cols = list(df.columns[:10])
    raise ValueError(
        f"Could not detect data format. Columns found: {cols}... "
        "Expected Enverus or ARIES format."
    )


def load_wells(filepath: Path | str) -> list[Well]:
    """Load wells from file with auto-detection.

    Args:
        filepath: Path to CSV or Excel file

    Returns:
        List of Well objects

    Raises:
        ValueError: If file format not supported or not recognized
    """
    df = DataParser.load_file(filepath)
    parser = detect_parser(df)
    return parser.parse(df)
