"""Data quality validation for production data.

Detects gaps, outliers, shut-ins, and other quality issues.
"""

from typing import TYPE_CHECKING

import numpy as np

from .result import (
    ValidationResult,
    ValidationIssue,
    IssueSeverity,
    IssueCategory,
)

if TYPE_CHECKING:
    from ..data.well import Well


class DataQualityValidator:
    """Validates production data quality.

    Error codes:
        DQ001: Data gaps detected (>N months missing)
        DQ002: Outliers detected (modified z-score or N-sigma)
        DQ003: Shut-in periods detected (rate < threshold)
        DQ004: Low data variability (CV < threshold)
    """

    def __init__(
        self,
        gap_threshold_months: int = 2,
        outlier_sigma: float = 3.0,
        shutin_threshold: float = 1.0,
        min_cv: float = 0.05,
    ):
        """Initialize data quality validator.

        Args:
            gap_threshold_months: Minimum gap size to flag (months)
            outlier_sigma: Number of std devs for outlier detection
            shutin_threshold: Rate below which is considered shut-in
            min_cv: Minimum coefficient of variation (below = too flat)
        """
        self.gap_threshold_months = gap_threshold_months
        self.outlier_sigma = outlier_sigma
        self.shutin_threshold = shutin_threshold
        self.min_cv = min_cv

    def validate(self, well: "Well", product: str) -> ValidationResult:
        """Validate data quality for a well's product.

        Args:
            well: Well object to validate
            product: Product to validate (oil, gas, water)

        Returns:
            ValidationResult with any quality issues found
        """
        result = ValidationResult(well_id=well.well_id, product=product)

        try:
            values = well.production.get_product(product)
        except (AttributeError, KeyError):
            return result  # Product not present

        if values is None or len(values) == 0:
            return result

        values = np.asarray(values, dtype=float)

        # Skip validation if no significant production
        if np.max(values) < 1.0:
            return result

        result = result.merge(self._check_gaps(well, values))
        result = result.merge(self._check_outliers(well, product, values))
        result = result.merge(self._check_shutins(well, product, values))
        result = result.merge(self._check_variability(well, product, values))

        return result

    def _check_gaps(
        self,
        well: "Well",
        values: np.ndarray,
    ) -> ValidationResult:
        """Check for gaps in production data.

        A gap is defined as consecutive months with zero or near-zero
        production surrounded by non-zero production.

        Args:
            well: Well object
            values: Production values array

        Returns:
            ValidationResult with gap issues
        """
        result = ValidationResult(well_id=well.well_id)

        # Identify zero/near-zero values
        zero_mask = values < 0.1

        # Find consecutive runs of zeros
        gaps = []
        in_gap = False
        gap_start = 0

        for i, is_zero in enumerate(zero_mask):
            if is_zero and not in_gap:
                # Start of potential gap
                gap_start = i
                in_gap = True
            elif not is_zero and in_gap:
                # End of gap
                gap_length = i - gap_start
                if gap_length >= self.gap_threshold_months:
                    # Only count as gap if there was production before
                    if gap_start > 0:
                        gaps.append((gap_start, i - 1, gap_length))
                in_gap = False

        if gaps:
            result.add_issue(ValidationIssue.data_gaps(
                gap_count=len(gaps),
                gaps=[(g[0], g[1], g[2]) for g in gaps],
                total_gap_months=sum(g[2] for g in gaps),
            ))

        return result

    def _check_outliers(
        self,
        well: "Well",
        product: str,
        values: np.ndarray,
    ) -> ValidationResult:
        """Check for outliers using modified z-score.

        Uses median absolute deviation (MAD) for robust outlier detection.

        Args:
            well: Well object
            product: Product name
            values: Production values array

        Returns:
            ValidationResult with outlier issues
        """
        result = ValidationResult(well_id=well.well_id, product=product)

        # Filter to non-zero values for statistics
        nonzero = values[values > 0]
        if len(nonzero) < 5:
            return result

        # Modified z-score using MAD
        median = np.median(nonzero)
        mad = np.median(np.abs(nonzero - median))

        if mad < 1e-10:
            # All values essentially the same
            return result

        # Modified z-score threshold (0.6745 adjusts for normal distribution)
        modified_z = 0.6745 * (values - median) / mad
        outlier_mask = np.abs(modified_z) > self.outlier_sigma

        # Only flag outliers in non-zero data
        outlier_mask = outlier_mask & (values > 0)

        if np.any(outlier_mask):
            outlier_indices = np.where(outlier_mask)[0].tolist()
            outlier_values = values[outlier_mask].tolist()

            result.add_issue(ValidationIssue.outliers(
                product=product,
                count=len(outlier_indices),
                indices=outlier_indices,
                values=outlier_values,
                median=float(median),
                mad=float(mad),
                sigma=self.outlier_sigma,
            ))

        return result

    def _check_shutins(
        self,
        well: "Well",
        product: str,
        values: np.ndarray,
    ) -> ValidationResult:
        """Check for shut-in periods.

        Shut-ins are periods where production drops to near-zero
        then resumes at significant levels.

        Args:
            well: Well object
            product: Product name
            values: Production values array

        Returns:
            ValidationResult with shut-in issues
        """
        result = ValidationResult(well_id=well.well_id, product=product)

        # Identify potential shut-in periods (low production)
        shutin_mask = values < self.shutin_threshold

        # Find runs of shut-in followed by resumption
        shutin_periods = []
        in_shutin = False
        shutin_start = 0
        had_production_before = False

        for i, is_shutin in enumerate(shutin_mask):
            if not is_shutin:
                had_production_before = True
                if in_shutin and shutin_start > 0:
                    # End of shut-in with resumption
                    shutin_length = i - shutin_start
                    if shutin_length >= 1:  # At least 1 month
                        shutin_periods.append((shutin_start, i - 1, shutin_length))
                in_shutin = False
            elif is_shutin and not in_shutin and had_production_before:
                # Start of potential shut-in
                shutin_start = i
                in_shutin = True

        if shutin_periods:
            result.add_issue(ValidationIssue(
                code="DQ003",
                category=IssueCategory.DATA_QUALITY,
                severity=IssueSeverity.INFO,
                message=f"Found {len(shutin_periods)} shut-in periods in {product} data",
                guidance="Shut-in periods may trigger regime detection; verify expected behavior",
                details={
                    "shutin_count": len(shutin_periods),
                    "periods": [(p[0], p[1], p[2]) for p in shutin_periods[:5]],
                    "threshold": self.shutin_threshold,
                },
            ))

        return result

    def _check_variability(
        self,
        well: "Well",
        product: str,
        values: np.ndarray,
    ) -> ValidationResult:
        """Check data variability (coefficient of variation).

        Very low CV suggests the data may be synthetic, rounded,
        or otherwise not representative of actual production.

        Args:
            well: Well object
            product: Product name
            values: Production values array

        Returns:
            ValidationResult with variability issues
        """
        result = ValidationResult(well_id=well.well_id, product=product)

        # Filter to non-zero values
        nonzero = values[values > 0]
        if len(nonzero) < 5:
            return result

        mean = np.mean(nonzero)
        if mean < 1e-10:
            return result

        std = np.std(nonzero)
        cv = std / mean

        if cv < self.min_cv:
            result.add_issue(ValidationIssue(
                code="DQ004",
                category=IssueCategory.DATA_QUALITY,
                severity=IssueSeverity.WARNING,
                message=f"Very low variability in {product} data (CV={cv:.4f})",
                guidance="Flat production data may be synthetic or averaged; decline curve may not be appropriate",
                details={
                    "cv": float(cv),
                    "threshold": self.min_cv,
                    "mean": float(mean),
                    "std": float(std),
                },
            ))

        return result
