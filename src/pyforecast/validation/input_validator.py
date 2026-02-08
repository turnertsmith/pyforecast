"""Input validation for production data.

Validates column structure, date formats, and production value ranges.
"""

from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np

from .result import (
    ValidationResult,
    ValidationIssue,
    IssueCategory,
    IssueSeverity,
)

if TYPE_CHECKING:
    from ..data.well import Well


class InputValidator:
    """Validates production data inputs.

    Error codes:
        IV001: Negative production values
        IV002: Values exceed threshold (>50k bbl/mo oil, >500k mcf/mo gas)
        IV003: Date parsing failed
        IV004: Future dates in data
    """

    def __init__(
        self,
        max_oil_rate: float = 50000.0,
        max_gas_rate: float = 500000.0,
        max_water_rate: float = 100000.0,
    ):
        """Initialize input validator.

        Args:
            max_oil_rate: Maximum expected oil rate (bbl/mo)
            max_gas_rate: Maximum expected gas rate (mcf/mo)
            max_water_rate: Maximum expected water rate (bbl/mo)
        """
        self.max_rates = {
            "oil": max_oil_rate,
            "gas": max_gas_rate,
            "water": max_water_rate,
        }

    def validate(self, well: "Well") -> ValidationResult:
        """Validate all aspects of a well's input data.

        Args:
            well: Well object to validate

        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult(well_id=well.well_id)

        # Validate dates
        result = result.merge(self._validate_dates(well))

        # Validate each product
        for product in ["oil", "gas", "water"]:
            result = result.merge(self._validate_production_values(well, product))

        return result

    def validate_product(self, well: "Well", product: str) -> ValidationResult:
        """Validate a specific product's data.

        Args:
            well: Well object to validate
            product: Product to validate (oil, gas, water)

        Returns:
            ValidationResult with any issues found
        """
        result = ValidationResult(well_id=well.well_id, product=product)
        result = result.merge(self._validate_dates(well))
        result = result.merge(self._validate_production_values(well, product))
        return result

    def _validate_dates(self, well: "Well") -> ValidationResult:
        """Validate production dates.

        Checks:
            - Dates are parseable (already done by loader)
            - No future dates

        Args:
            well: Well object to validate

        Returns:
            ValidationResult with date issues
        """
        result = ValidationResult(well_id=well.well_id)
        today = np.datetime64(date.today())

        try:
            dates = well.production.dates
            if len(dates) == 0:
                return result

            # Convert to numpy datetime64 if needed
            dates_arr = np.asarray(dates, dtype='datetime64[D]')
            future_mask = dates_arr > today
            future_indices = np.where(future_mask)[0].tolist()

            if future_indices:
                result.add_issue(ValidationIssue.future_dates(
                    count=len(future_indices),
                    first_date=str(dates_arr[future_indices[0]]),
                    indices=future_indices,
                ))

        except Exception as e:
            result.add_issue(ValidationIssue(
                code="IV003",
                category=IssueCategory.DATA_FORMAT,
                severity=IssueSeverity.ERROR,
                message=f"Date parsing failed: {str(e)}",
                guidance="Check date format in input data; expected YYYY-MM-DD or similar",
                details={"error": str(e)},
            ))

        return result

    def _validate_production_values(
        self,
        well: "Well",
        product: str,
    ) -> ValidationResult:
        """Validate production values for a product.

        Checks:
            - No negative values
            - Values within reasonable range

        Args:
            well: Well object to validate
            product: Product to validate

        Returns:
            ValidationResult with value issues
        """
        result = ValidationResult(well_id=well.well_id, product=product)

        try:
            values = well.production.get_product(product)
        except (AttributeError, KeyError):
            return result  # Product not present, not an error

        if values is None or len(values) == 0:
            return result

        values = np.asarray(values)

        # Check for negative values
        negative_mask = values < 0
        if np.any(negative_mask):
            negative_indices = np.where(negative_mask)[0].tolist()
            negative_values = values[negative_mask].tolist()

            result.add_issue(ValidationIssue.negative_values(
                product=product,
                count=len(negative_indices),
                indices=negative_indices,
                values=negative_values,
            ))

        # Check for values exceeding threshold
        max_rate = self.max_rates.get(product, 50000.0)
        exceeds_mask = values > max_rate

        if np.any(exceeds_mask):
            exceeds_indices = np.where(exceeds_mask)[0].tolist()
            exceeds_values = values[exceeds_mask].tolist()

            result.add_issue(ValidationIssue.exceeds_threshold(
                product=product,
                count=len(exceeds_indices),
                threshold=max_rate,
                indices=exceeds_indices,
                values=exceeds_values,
                max_value=float(np.max(values)),
            ))

        return result
