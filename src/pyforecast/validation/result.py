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
