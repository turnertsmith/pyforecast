"""Validation orchestrator for running all validators in sequence.

Provides a single entry point for running input, data quality, and fitting
validation on wells, with unified result reporting.
"""

from dataclasses import dataclass
from typing import Literal

from ..config import PyForecastConfig
from ..data.well import Well
from .result import ValidationResult, merge_results
from .input_validator import InputValidator
from .data_quality import DataQualityValidator
from .fitting_validator import FittingValidator


@dataclass
class ValidationOrchestrator:
    """Orchestrates all validation stages for wells.

    Runs input validation, data quality checks, and fitting validation
    in sequence, producing unified validation results.

    Attributes:
        config: PyForecast configuration with validation settings
        input_validator: Validator for input data format and ranges
        quality_validator: Validator for data quality issues
        fitting_validator: Validator for fit quality checks
    """

    config: PyForecastConfig
    input_validator: InputValidator = None  # type: ignore
    quality_validator: DataQualityValidator = None  # type: ignore
    fitting_validator: FittingValidator = None  # type: ignore

    def __post_init__(self) -> None:
        """Initialize validators from config."""
        val_config = self.config.validation

        self.input_validator = InputValidator(
            max_oil_rate=val_config.max_oil_rate,
            max_gas_rate=val_config.max_gas_rate,
            max_water_rate=val_config.max_water_rate,
        )

        self.quality_validator = DataQualityValidator(
            gap_threshold_months=val_config.gap_threshold_months,
            outlier_sigma=val_config.outlier_sigma,
            shutin_threshold=val_config.shutin_threshold,
            min_cv=val_config.min_cv,
        )

        self.fitting_validator = FittingValidator(
            min_r_squared=val_config.min_r_squared,
            max_annual_decline=val_config.max_annual_decline,
        )

    @classmethod
    def from_config(cls, config: PyForecastConfig) -> "ValidationOrchestrator":
        """Create orchestrator from PyForecastConfig.

        Args:
            config: PyForecast configuration

        Returns:
            Configured ValidationOrchestrator
        """
        return cls(config=config)

    def validate_input(self, well: Well) -> ValidationResult:
        """Run input validation on a well.

        Checks data format, value ranges, and date validity.

        Args:
            well: Well to validate

        Returns:
            ValidationResult with input validation issues
        """
        return self.input_validator.validate(well)

    def validate_quality(
        self,
        well: Well,
        product: Literal["oil", "gas", "water"],
    ) -> ValidationResult:
        """Run data quality validation on a well for a specific product.

        Checks for gaps, outliers, shut-ins, and data variability.

        Args:
            well: Well to validate
            product: Product to validate

        Returns:
            ValidationResult with data quality issues
        """
        return self.quality_validator.validate(well, product)

    def validate_pre_fit(
        self,
        well: Well,
        product: Literal["oil", "gas", "water"],
    ) -> ValidationResult:
        """Run pre-fit validation on a well.

        Checks data sufficiency, trend direction, and fit prerequisites.

        Args:
            well: Well to validate
            product: Product to validate

        Returns:
            ValidationResult with pre-fit issues
        """
        return self.fitting_validator.validate_pre_fit(well, product)

    def validate_post_fit(
        self,
        well: Well,
        product: Literal["oil", "gas", "water"],
    ) -> ValidationResult:
        """Run post-fit validation on a well.

        Checks fit quality metrics (R², b-factor bounds, decline rate).

        Args:
            well: Well to validate (must have forecast set)
            product: Product to validate

        Returns:
            ValidationResult with post-fit issues
        """
        return self.fitting_validator.validate_post_fit(well, product)

    def validate_well_full(
        self,
        well: Well,
        products: list[Literal["oil", "gas", "water"]] | None = None,
        include_post_fit: bool = False,
    ) -> ValidationResult:
        """Run all validation stages on a well.

        Runs input validation, then data quality and pre-fit validation
        for each product. Optionally includes post-fit validation if
        forecasts are available.

        Args:
            well: Well to validate
            products: Products to validate (default: config output.products)
            include_post_fit: Whether to run post-fit validation

        Returns:
            Merged ValidationResult with all issues
        """
        if products is None:
            products = self.config.output.products

        results: list[ValidationResult] = []

        # Input validation (all products)
        results.append(self.validate_input(well))

        # Per-product validation
        for product in products:
            try:
                # Data quality
                results.append(self.validate_quality(well, product))

                # Pre-fit checks
                results.append(self.validate_pre_fit(well, product))

                # Post-fit checks (if forecast exists and requested)
                if include_post_fit:
                    forecast = well.get_forecast(product)
                    if forecast is not None:
                        results.append(self.validate_post_fit(well, product))

            except Exception:
                # Product may not be available - skip silently
                pass

        # Merge all results
        combined = merge_results(results)
        combined.well_id = well.well_id
        return combined

    def validate_wells(
        self,
        wells: list[Well],
        products: list[Literal["oil", "gas", "water"]] | None = None,
        include_post_fit: bool = False,
    ) -> list[ValidationResult]:
        """Validate multiple wells.

        Args:
            wells: Wells to validate
            products: Products to validate
            include_post_fit: Whether to run post-fit validation

        Returns:
            List of ValidationResults, one per well
        """
        return [
            self.validate_well_full(well, products, include_post_fit)
            for well in wells
        ]

    def get_summary(self, results: list[ValidationResult]) -> dict:
        """Get summary statistics from validation results.

        Args:
            results: List of validation results

        Returns:
            Dictionary with summary statistics
        """
        total_wells = len(results)
        wells_with_errors = sum(1 for r in results if r.has_errors)
        wells_with_warnings = sum(1 for r in results if r.has_warnings)
        total_errors = sum(r.error_count for r in results)
        total_warnings = sum(r.warning_count for r in results)

        # Count by code
        by_code: dict[str, int] = {}
        for result in results:
            for issue in result.issues:
                by_code[issue.code] = by_code.get(issue.code, 0) + 1

        # Count by category
        by_category: dict[str, int] = {}
        for result in results:
            for issue in result.issues:
                cat_name = issue.category.name
                by_category[cat_name] = by_category.get(cat_name, 0) + 1

        return {
            "total_wells": total_wells,
            "wells_with_errors": wells_with_errors,
            "wells_with_warnings": wells_with_warnings,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "by_code": by_code,
            "by_category": by_category,
        }

    def format_report(
        self,
        results: list[ValidationResult],
        include_ok: bool = False,
    ) -> str:
        """Format validation results as a text report.

        Args:
            results: List of validation results
            include_ok: Whether to include wells with no issues

        Returns:
            Formatted text report
        """
        lines = ["PyForecast Validation Report", "=" * 40, ""]

        summary = self.get_summary(results)

        lines.append("Summary:")
        lines.append(f"  Total wells: {summary['total_wells']}")
        lines.append(f"  Wells with errors: {summary['wells_with_errors']}")
        lines.append(f"  Wells with warnings: {summary['wells_with_warnings']}")
        lines.append(f"  Total errors: {summary['total_errors']}")
        lines.append(f"  Total warnings: {summary['total_warnings']}")
        lines.append("")

        if summary["by_category"]:
            lines.append("Issues by category:")
            for cat, count in sorted(summary["by_category"].items()):
                lines.append(f"  {cat}: {count}")
            lines.append("")

        lines.append("Detailed Issues:")
        lines.append("-" * 40)

        for result in results:
            if result.issues or include_ok:
                lines.append(f"\n{result.well_id}:")
                if result.issues:
                    for issue in result.issues:
                        lines.append(f"  [{issue.code}] {issue.severity.name}: {issue.message}")
                        lines.append(f"    Guidance: {issue.guidance}")
                else:
                    lines.append("  No issues")

        return "\n".join(lines)
