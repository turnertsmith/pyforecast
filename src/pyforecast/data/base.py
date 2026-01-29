"""Abstract base class for data parsers."""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .well import Well


class DataParser(ABC):
    """Abstract base class for production data parsers."""

    @abstractmethod
    def can_parse(self, df: pd.DataFrame) -> bool:
        """Check if this parser can handle the given DataFrame.

        Args:
            df: DataFrame loaded from input file

        Returns:
            True if this parser recognizes the format
        """
        pass

    @abstractmethod
    def parse(self, df: pd.DataFrame) -> list[Well]:
        """Parse DataFrame into list of Well objects.

        Args:
            df: DataFrame with production data

        Returns:
            List of Well objects with production data populated
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
