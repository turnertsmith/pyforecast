"""ARIES AC_ECONOMIC forecast parser.

Parses ARIES expression-based forecast format for ground truth comparison.

ARIES Expression Format:
    "{Qi} X {unit} {Dmin%} EXP B/{b} {Di%}"

Example:
    "1000 X B/M 6 EXP B/0.50 8.5"
    - qi = 1000 bbl/month
    - unit = B/M (barrels/month)
    - dmin = 6% annual terminal decline
    - b = 0.50 hyperbolic exponent
    - di = 8.5% annual initial decline

The parser converts all values to internal units (daily rates, monthly decline).
"""

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..core.models import HyperbolicModel

logger = logging.getLogger(__name__)


# Average days per month for conversion
DAYS_PER_MONTH = 30.4375


def normalize_well_id(well_id: str) -> str:
    """Normalize a well ID for consistent matching.

    For API numbers (XX-XXX-XXXXX format or 10+ digits with separators):
    - Remove dashes, spaces, and other separators
    - Result is a 10-14 digit string

    For PROPNUMs (non-API identifiers):
    - Just strip whitespace, preserve as-is

    Examples:
        "42-123-45678" -> "4212345678"  (API normalized)
        "42 123 45678" -> "4212345678"  (API normalized)
        "WELL001" -> "WELL001"          (PROPNUM unchanged)
        "Smith_Ranch_1" -> "Smith_Ranch_1"  (PROPNUM unchanged)

    Args:
        well_id: Raw well identifier

    Returns:
        Normalized well identifier
    """
    if not well_id:
        return ""

    stripped = well_id.strip()

    # Check if this looks like an API number
    # API format: XX-XXX-XXXXX (state-county-well) with optional suffixes
    # When separators removed, should be 10-14 digits
    digits_only = re.sub(r"[^0-9]", "", stripped)

    # If it's 10-14 digits and the original had separators typical of APIs,
    # treat it as an API and return just the digits
    if 10 <= len(digits_only) <= 14:
        # Check if original format looks like API (has dashes or spaces between digit groups)
        if re.match(r"^\d{2}[-\s]?\d{3}[-\s]?\d{5}", stripped):
            return digits_only

    # Otherwise return as-is (PROPNUM)
    return stripped


@dataclass
class AriesForecastParams:
    """ARIES forecast parameters converted to internal units.

    Attributes:
        propnum: Well property number identifier
        product: Product type (oil, gas, water)
        qi: Initial rate in daily units (bbl/day or mcf/day)
        di: Initial decline rate (fraction/month)
        b: Hyperbolic exponent (0 = exponential, 1 = harmonic)
        dmin: Terminal decline rate (fraction/month)
        decline_type: Decline type string (EXP, HYP, HRM)
    """
    propnum: str
    product: Literal["oil", "gas", "water"]
    qi: float  # daily rate
    di: float  # monthly decline (fraction)
    b: float
    dmin: float  # monthly decline (fraction)
    decline_type: str  # EXP, HYP, HRM

    @property
    def qi_monthly(self) -> float:
        """Initial rate in monthly units."""
        return self.qi * DAYS_PER_MONTH

    @property
    def di_annual(self) -> float:
        """Initial decline rate in annual units."""
        return self.di * 12

    @property
    def dmin_annual(self) -> float:
        """Terminal decline rate in annual units."""
        return self.dmin * 12


class AriesForecastImporter:
    """Import ARIES AC_ECONOMIC format forecasts.

    Parses CSV files with ARIES expression-based forecasts and provides
    lookup by well/product for ground truth comparison.

    Expected CSV columns:
        - PROPNUM: Well property number
        - {PRODUCT}_EXPRESSION or expression column with product indicator

    Example usage:
        importer = AriesForecastImporter()
        count = importer.load(Path("aries_forecasts.csv"))
        params = importer.get("42-001-00001", "oil")
        if params:
            model = importer.to_model(params)
    """

    # Expression pattern: "Qi X UNIT Dmin TYPE B/b Di"
    # Example: "1000 X B/M 6 EXP B/0.50 8.5"
    # Or: "500 X M/M 6 HYP B/0.75 12"
    EXPRESSION_PATTERN = re.compile(
        r"^\s*"
        r"(?P<qi>[\d.]+)\s+"  # Initial rate
        r"X\s+"
        r"(?P<unit>[BMG]/[MD])\s+"  # Unit (B/M, B/D, M/M, M/D, G/M, G/D)
        r"(?P<dmin>[\d.]+)\s+"  # Terminal decline %
        r"(?P<type>EXP|HYP|HRM)\s+"  # Decline type
        r"B/(?P<b>[\d.]+)\s+"  # b-factor
        r"(?P<di>[\d.]+)"  # Initial decline %
        r"\s*$",
        re.IGNORECASE,
    )

    # Alternative pattern without explicit type (infer from b)
    EXPRESSION_PATTERN_ALT = re.compile(
        r"^\s*"
        r"(?P<qi>[\d.]+)\s+"
        r"X\s+"
        r"(?P<unit>[BMG]/[MD])\s+"
        r"(?P<dmin>[\d.]+)\s+"
        r"B/(?P<b>[\d.]+)\s+"
        r"(?P<di>[\d.]+)"
        r"\s*$",
        re.IGNORECASE,
    )

    # Unit mapping: ARIES unit -> (product, is_daily)
    UNIT_MAP = {
        "B/M": ("oil", False),   # barrels/month
        "B/D": ("oil", True),    # barrels/day
        "M/M": ("gas", False),   # mcf/month
        "M/D": ("gas", True),    # mcf/day
        "G/M": ("gas", False),   # mcf/month (alternate)
        "G/D": ("gas", True),    # mcf/day (alternate)
        "W/M": ("water", False), # water barrels/month
        "W/D": ("water", True),  # water barrels/day
    }

    def __init__(self, lazy: bool = False) -> None:
        """Initialize the importer.

        Args:
            lazy: If True, stream through file on each lookup instead of
                  loading all data into memory. Slower per-lookup but uses
                  constant memory for large datasets.
        """
        self._forecasts: dict[tuple[str, str], AriesForecastParams] = {}
        self._parse_failures: list[tuple[str, str, str]] = []  # (well_id, product, expression)
        self._lazy = lazy
        self._filepath: Path | None = None
        self._well_count: int = 0

    def load(self, filepath: Path | str) -> int:
        """Load ARIES forecasts from CSV file.

        Supports two formats:
        1. AC_ECONOMIC format (row-based): PROPNUM, KEYWORD, EXPRESSION columns
           where KEYWORD is OIL/GAS/WATER and EXPRESSION contains the decline params
        2. Flat format: PROPNUM with expression columns (OIL_EXPRESSION, etc.)

        In lazy mode, only validates the file and counts wells without
        loading data into memory. Actual parsing happens on get() calls.

        Args:
            filepath: Path to CSV file with ARIES expressions

        Returns:
            Number of forecasts loaded (or counted in lazy mode)

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"ARIES forecast file not found: {filepath}")

        self._filepath = filepath

        if self._lazy:
            # In lazy mode, just count rows and validate format
            return self._count_and_validate(filepath)

        # Eager mode: load all data into memory
        return self._load_all(filepath)

    def _count_and_validate(self, filepath: Path) -> int:
        """Count forecasts without loading into memory (lazy mode)."""
        count = 0
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError("CSV file has no headers")

            fieldnames_upper = [fn.strip().upper() for fn in reader.fieldnames]
            is_ac_economic = "KEYWORD" in fieldnames_upper and "EXPRESSION" in fieldnames_upper

            for row in reader:
                row = {k.strip().upper(): v for k, v in row.items()}
                propnum = self._get_propnum(row)
                if not propnum:
                    continue

                if is_ac_economic:
                    keyword = row.get("KEYWORD", "").strip().upper()
                    expression = row.get("EXPRESSION", "").strip()
                    if keyword in ("OIL", "GAS", "WATER") and expression:
                        count += 1
                else:
                    # Count any non-empty expression-like values
                    for col, value in row.items():
                        if value and " X " in value.upper():
                            count += 1

        self._well_count = count
        return count

    def _load_all(self, filepath: Path) -> int:
        """Load all forecasts into memory (eager mode)."""
        count = 0
        with open(filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError("CSV file has no headers")

            # Normalize column names
            fieldnames_upper = [fn.strip().upper() for fn in reader.fieldnames]

            # Detect format: AC_ECONOMIC has KEYWORD and EXPRESSION columns
            is_ac_economic = "KEYWORD" in fieldnames_upper and "EXPRESSION" in fieldnames_upper

            for row in reader:
                # Normalize row keys
                row = {k.strip().upper(): v for k, v in row.items()}

                propnum = self._get_propnum(row)
                if not propnum:
                    continue

                if is_ac_economic:
                    # AC_ECONOMIC format: parse based on KEYWORD
                    count += self._parse_ac_economic_row(propnum, row)
                else:
                    # Flat format: try all columns for expressions
                    count += self._parse_flat_row(propnum, row)

        return count

    def _parse_ac_economic_row(self, propnum: str, row: dict[str, str]) -> int:
        """Parse AC_ECONOMIC format row.

        Returns number of forecasts parsed (0 or 1).
        """
        keyword = row.get("KEYWORD", "").strip().upper()
        expression = row.get("EXPRESSION", "").strip()

        # Only process OIL, GAS, WATER keywords (skip CUMS, START, continuation rows)
        if keyword not in ("OIL", "GAS", "WATER"):
            return 0

        if not expression:
            return 0

        # Map keyword to product
        product_map = {"OIL": "oil", "GAS": "gas", "WATER": "water"}
        expected_product = product_map[keyword]

        params = self._parse_expression(propnum, expression, product_hint=expected_product)
        if params is None:
            return 0

        # Override product based on KEYWORD (more reliable than unit inference)
        params = AriesForecastParams(
            propnum=params.propnum,
            product=expected_product,
            qi=params.qi,
            di=params.di,
            b=params.b,
            dmin=params.dmin,
            decline_type=params.decline_type,
        )

        # Store with normalized ID for consistent lookup
        normalized_id = normalize_well_id(propnum)
        key = (normalized_id, expected_product)
        self._forecasts[key] = params
        return 1

    def _parse_flat_row(self, propnum: str, row: dict[str, str]) -> int:
        """Parse flat format row (expression columns).

        Returns number of forecasts parsed.
        """
        count = 0
        normalized_id = normalize_well_id(propnum)
        for col, value in row.items():
            if not value or not value.strip():
                continue

            params = self._parse_expression(propnum, value.strip())
            if params:
                key = (normalized_id, params.product)
                self._forecasts[key] = params
                count += 1
        return count

    def _get_propnum(self, row: dict[str, str]) -> str | None:
        """Extract property number from row."""
        for col in ("PROPNUM", "PROP_NUM", "WELL_ID", "API", "ENTITY_ID"):
            if col in row and row[col]:
                return row[col].strip()
        return None

    def _parse_expression(
        self,
        propnum: str,
        expression: str,
        product_hint: str | None = None,
    ) -> AriesForecastParams | None:
        """Parse ARIES expression into forecast parameters.

        Args:
            propnum: Well property number
            expression: ARIES expression string
            product_hint: Optional product type for logging context

        Returns:
            AriesForecastParams if parsed successfully, None otherwise
        """
        if not expression or not expression.strip():
            return None

        # Try main pattern first
        match = self.EXPRESSION_PATTERN.match(expression)
        if match:
            return self._build_params(propnum, match, has_type=True)

        # Try alternative pattern
        match = self.EXPRESSION_PATTERN_ALT.match(expression)
        if match:
            return self._build_params(propnum, match, has_type=False)

        # Log parse failure
        product_str = product_hint or "unknown"
        self._parse_failures.append((propnum, product_str, expression))
        logger.warning(
            f"Failed to parse ARIES expression for {propnum}/{product_str}: '{expression}'"
        )
        return None

    def _build_params(
        self,
        propnum: str,
        match: re.Match,
        has_type: bool,
    ) -> AriesForecastParams | None:
        """Build AriesForecastParams from regex match."""
        try:
            qi_raw = float(match.group("qi"))
            unit = match.group("unit").upper()
            dmin_pct = float(match.group("dmin"))
            b = float(match.group("b"))
            di_pct = float(match.group("di"))

            if has_type:
                decline_type = match.group("type").upper()
            else:
                # Infer type from b-factor
                if b <= 0.1:
                    decline_type = "EXP"
                elif b >= 0.95:
                    decline_type = "HRM"
                else:
                    decline_type = "HYP"

            # Get product and rate type from unit
            if unit not in self.UNIT_MAP:
                return None
            product, is_daily = self.UNIT_MAP[unit]

            # Convert qi to daily rate
            if is_daily:
                qi = qi_raw
            else:
                qi = qi_raw / DAYS_PER_MONTH

            # Convert decline rates from annual % to monthly fraction
            di = (di_pct / 100) / 12
            dmin = (dmin_pct / 100) / 12

            return AriesForecastParams(
                propnum=propnum,
                product=product,
                qi=qi,
                di=di,
                b=b,
                dmin=dmin,
                decline_type=decline_type,
            )
        except (ValueError, AttributeError):
            return None

    def get(
        self,
        propnum: str,
        product: str,
    ) -> AriesForecastParams | None:
        """Get forecast parameters for a well/product.

        The propnum is normalized before lookup to handle API format variations
        (e.g., "42-123-45678" matches "4212345678").

        In lazy mode, streams through the file on each call.

        Args:
            propnum: Well property number or API
            product: Product type (oil, gas, water)

        Returns:
            AriesForecastParams if found, None otherwise
        """
        normalized_id = normalize_well_id(propnum)

        if self._lazy and self._filepath:
            return self._stream_find(normalized_id, product)

        return self._forecasts.get((normalized_id, product))

    def _stream_find(
        self,
        normalized_id: str,
        product: str,
    ) -> AriesForecastParams | None:
        """Stream through file to find specific well/product (lazy mode)."""
        if self._filepath is None:
            return None

        with open(self._filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                return None

            fieldnames_upper = [fn.strip().upper() for fn in reader.fieldnames]
            is_ac_economic = "KEYWORD" in fieldnames_upper and "EXPRESSION" in fieldnames_upper

            for row in reader:
                row = {k.strip().upper(): v for k, v in row.items()}
                propnum = self._get_propnum(row)
                if not propnum:
                    continue

                if normalize_well_id(propnum) != normalized_id:
                    continue

                if is_ac_economic:
                    keyword = row.get("KEYWORD", "").strip().upper()
                    expression = row.get("EXPRESSION", "").strip()

                    if keyword not in ("OIL", "GAS", "WATER"):
                        continue

                    product_map = {"OIL": "oil", "GAS": "gas", "WATER": "water"}
                    if product_map[keyword] != product:
                        continue

                    params = self._parse_expression(propnum, expression, product_hint=product)
                    if params:
                        # Override product based on keyword
                        return AriesForecastParams(
                            propnum=params.propnum,
                            product=product,
                            qi=params.qi,
                            di=params.di,
                            b=params.b,
                            dmin=params.dmin,
                            decline_type=params.decline_type,
                        )
                else:
                    # Flat format - try to find matching product
                    for col, value in row.items():
                        if not value or not value.strip():
                            continue
                        params = self._parse_expression(propnum, value.strip())
                        if params and params.product == product:
                            return params

        return None

    def to_model(self, params: AriesForecastParams) -> HyperbolicModel:
        """Convert ARIES parameters to HyperbolicModel.

        Args:
            params: ARIES forecast parameters

        Returns:
            HyperbolicModel configured with ARIES parameters
        """
        return HyperbolicModel(
            qi=params.qi,
            di=params.di,
            b=params.b,
            dmin=params.dmin,
        )

    def list_wells(self) -> list[str]:
        """List all unique well IDs loaded (normalized form).

        In lazy mode, streams through the file to collect well IDs.
        """
        if self._lazy and self._filepath:
            return self._stream_list_wells()
        return sorted(set(propnum for propnum, _ in self._forecasts.keys()))

    def _stream_list_wells(self) -> list[str]:
        """Stream through file to list wells (lazy mode)."""
        if self._filepath is None:
            return []

        well_ids: set[str] = set()
        with open(self._filepath, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                return []

            for row in reader:
                row = {k.strip().upper(): v for k, v in row.items()}
                propnum = self._get_propnum(row)
                if propnum:
                    well_ids.add(normalize_well_id(propnum))

        return sorted(well_ids)

    def list_products(self, propnum: str) -> list[str]:
        """List products available for a well."""
        normalized_id = normalize_well_id(propnum)
        return sorted(
            product for p, product in self._forecasts.keys() if p == normalized_id
        )

    @property
    def parse_failures(self) -> list[tuple[str, str, str]]:
        """Return list of expressions that failed to parse.

        Returns:
            List of (well_id, product, expression) tuples that couldn't be parsed
        """
        return self._parse_failures.copy()

    def __len__(self) -> int:
        """Return number of loaded forecasts."""
        return len(self._forecasts)

    def __contains__(self, key: tuple[str, str]) -> bool:
        """Check if forecast exists for (propnum, product).

        The propnum is normalized before lookup.
        """
        propnum, product = key
        normalized_id = normalize_well_id(propnum)
        return (normalized_id, product) in self._forecasts
