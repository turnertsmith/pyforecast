"""Validation result types for data quality and fitting checks.

Provides structured validation results with categorized issues,
severity levels, and actionable guidance.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class IssueSeverity(Enum):
    """Severity level of a validation issue."""
    ERROR = auto()    # Cannot proceed - data/fit is unusable
    WARNING = auto()  # Can proceed with caution - review recommended
    INFO = auto()     # Informational - no action required


class IssueCategory(Enum):
    """Category of validation issue for grouping and filtering."""
    DATA_QUALITY = auto()    # Gaps, outliers, shut-ins
    DATA_FORMAT = auto()     # Column, date, value format issues
    FITTING_PREREQ = auto()  # Pre-fit checks failed
    FITTING_RESULT = auto()  # Post-fit quality issues


@dataclass
class ValidationIssue:
    """A single validation issue with context and guidance.

    Attributes:
        code: Unique identifier (e.g., "DQ001", "IV002")
        category: Issue category for grouping
        severity: Issue severity level
        message: User-friendly description of the issue
        guidance: Actionable next step for resolution
        details: Context data (indices, values, thresholds, etc.)
    """
    code: str
    category: IssueCategory
    severity: IssueSeverity
    message: str
    guidance: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """Format issue as string for display."""
        severity_str = self.severity.name
        return f"[{self.code}] {severity_str}: {self.message}"

    # --- Factory methods for common issue patterns ---

    @staticmethod
    def negative_values(product: str, count: int, indices: list, values: list) -> "ValidationIssue":
        """Create IV001: Negative production values issue."""
        return ValidationIssue(
            code="IV001",
            category=IssueCategory.DATA_FORMAT,
            severity=IssueSeverity.ERROR,
            message=f"Found {count} negative {product} values",
            guidance="Production values must be non-negative; check data source for errors",
            details={"negative_count": count, "indices": indices[:10], "values": values[:10], "product": product},
        )

    @staticmethod
    def exceeds_threshold(product: str, count: int, threshold: float, indices: list, values: list, max_value: float) -> "ValidationIssue":
        """Create IV002: Values exceed threshold issue."""
        return ValidationIssue(
            code="IV002",
            category=IssueCategory.DATA_FORMAT,
            severity=IssueSeverity.WARNING,
            message=f"Found {count} {product} values exceeding {threshold:,.0f}",
            guidance="Very high production values may indicate unit conversion issues or data errors",
            details={"exceeds_count": count, "threshold": threshold, "indices": indices[:10], "values": values[:10], "max_value": max_value, "product": product},
        )

    @staticmethod
    def future_dates(count: int, first_date: str, indices: list) -> "ValidationIssue":
        """Create IV004: Future dates issue."""
        return ValidationIssue(
            code="IV004",
            category=IssueCategory.DATA_FORMAT,
            severity=IssueSeverity.WARNING,
            message=f"Found {count} future dates in production data",
            guidance="Verify data dates are correct; future dates may indicate data entry errors",
            details={"future_date_count": count, "first_future_date": first_date, "indices": indices[:10]},
        )

    @staticmethod
    def data_gaps(gap_count: int, gaps: list, total_gap_months: int) -> "ValidationIssue":
        """Create DQ001: Data gaps issue."""
        return ValidationIssue(
            code="DQ001",
            category=IssueCategory.DATA_QUALITY,
            severity=IssueSeverity.WARNING,
            message=f"Found {gap_count} data gaps",
            guidance="Gaps may indicate shut-ins or missing data; consider excluding or interpolating",
            details={"gap_count": gap_count, "gaps": gaps[:5], "total_gap_months": total_gap_months},
        )

    @staticmethod
    def outliers(product: str, count: int, indices: list, values: list, median: float, mad: float, sigma: float) -> "ValidationIssue":
        """Create DQ002: Outliers issue."""
        return ValidationIssue(
            code="DQ002",
            category=IssueCategory.DATA_QUALITY,
            severity=IssueSeverity.WARNING,
            message=f"Found {count} potential outliers in {product} data",
            guidance="Review outlier values for data errors; consider excluding from fit",
            details={"outlier_count": count, "indices": indices[:10], "values": values[:10], "median": median, "mad": mad, "sigma_threshold": sigma},
        )

    @staticmethod
    def poor_fit(product: str, r_squared: float, threshold: float, rmse: float) -> "ValidationIssue":
        """Create FR001: Poor fit quality issue."""
        severity = IssueSeverity.ERROR if r_squared < 0.3 else IssueSeverity.WARNING
        return ValidationIssue(
            code="FR001",
            category=IssueCategory.FITTING_RESULT,
            severity=severity,
            message=f"Poor {product} fit quality: R²={r_squared:.3f}",
            guidance="Low R² suggests poor model fit; consider data quality or alternative model",
            details={"r_squared": r_squared, "threshold": threshold, "rmse": rmse},
        )


@dataclass
class ValidationResult:
    """Collection of validation issues for a well or dataset.

    Attributes:
        well_id: Well identifier (None for file-level validation)
        issues: List of validation issues found
        product: Product being validated (None for multi-product)
    """
    well_id: str | None = None
    issues: list[ValidationIssue] = field(default_factory=list)
    product: str | None = None

    @property
    def has_errors(self) -> bool:
        """Check if any ERROR-severity issues exist."""
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        """Check if any WARNING-severity issues exist."""
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    @property
    def is_valid(self) -> bool:
        """Check if no ERROR-severity issues exist."""
        return not self.has_errors

    @property
    def error_count(self) -> int:
        """Count of ERROR-severity issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        """Count of WARNING-severity issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    @property
    def info_count(self) -> int:
        """Count of INFO-severity issues."""
        return sum(1 for i in self.issues if i.severity == IssueSeverity.INFO)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the result."""
        self.issues.append(issue)

    def by_category(self, category: IssueCategory) -> list[ValidationIssue]:
        """Filter issues by category.

        Args:
            category: Category to filter by

        Returns:
            List of issues in the specified category
        """
        return [i for i in self.issues if i.category == category]

    def by_severity(self, severity: IssueSeverity) -> list[ValidationIssue]:
        """Filter issues by severity.

        Args:
            severity: Severity to filter by

        Returns:
            List of issues with the specified severity
        """
        return [i for i in self.issues if i.severity == severity]

    def errors(self) -> list[ValidationIssue]:
        """Get all ERROR-severity issues."""
        return self.by_severity(IssueSeverity.ERROR)

    def warnings(self) -> list[ValidationIssue]:
        """Get all WARNING-severity issues."""
        return self.by_severity(IssueSeverity.WARNING)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into this one.

        Args:
            other: Another ValidationResult to merge

        Returns:
            New ValidationResult with combined issues
        """
        return ValidationResult(
            well_id=self.well_id or other.well_id,
            issues=self.issues + other.issues,
            product=self.product or other.product,
        )

    def __str__(self) -> str:
        """Format result as summary string."""
        if not self.issues:
            return f"Validation OK for {self.well_id or 'data'}"

        lines = [f"Validation for {self.well_id or 'data'}: "
                 f"{self.error_count} errors, {self.warning_count} warnings"]
        for issue in self.issues:
            lines.append(f"  {issue}")
        return "\n".join(lines)


def merge_results(results: list[ValidationResult]) -> ValidationResult:
    """Merge multiple validation results into one.

    Args:
        results: List of ValidationResults to merge

    Returns:
        Combined ValidationResult with all issues
    """
    if not results:
        return ValidationResult()

    combined = results[0]
    for result in results[1:]:
        combined = combined.merge(result)
    return combined


def summarize_validation(
    results: dict[str, ValidationResult] | list[ValidationResult],
) -> dict:
    """Calculate validation summary statistics from results.

    Consolidates the duplicated summary logic from BatchResult,
    BatchExporter, and ValidationOrchestrator into one place.

    Args:
        results: Dict of well_id -> ValidationResult, or list of results

    Returns:
        Dictionary with summary statistics:
            - wells_with_errors: count of wells with at least one error
            - wells_with_warnings: count of wells with at least one warning
            - total_errors: total error count
            - total_warnings: total warning count
            - by_category: dict of category name -> issue count
            - by_code: dict of issue code -> issue count
    """
    if isinstance(results, dict):
        result_iter = results.values()
    else:
        result_iter = results

    summary: dict[str, Any] = {
        "wells_with_errors": 0,
        "wells_with_warnings": 0,
        "total_errors": 0,
        "total_warnings": 0,
        "by_category": {},
        "by_code": {},
    }

    for result in result_iter:
        if result.has_errors:
            summary["wells_with_errors"] += 1
        if result.has_warnings:
            summary["wells_with_warnings"] += 1
        summary["total_errors"] += result.error_count
        summary["total_warnings"] += result.warning_count

        for issue in result.issues:
            cat_name = issue.category.name
            summary["by_category"][cat_name] = summary["by_category"].get(cat_name, 0) + 1
            summary["by_code"][issue.code] = summary["by_code"].get(issue.code, 0) + 1

    return summary
