"""Data validation and error handling for PyForecast.

Provides comprehensive validation for production data quality,
input format checking, and curve fitting quality assessment.

Usage:
    from pyforecast.validation import (
        ValidationResult,
        ValidationIssue,
        IssueSeverity,
        IssueCategory,
        InputValidator,
        DataQualityValidator,
        FittingValidator,
        ValidationConfig,
    )

    # Validate input data
    input_validator = InputValidator(config)
    result = input_validator.validate(well)

    # Check data quality
    quality_validator = DataQualityValidator(config)
    result = quality_validator.validate(well, product="oil")

    # Validate fitting results
    fitting_validator = FittingValidator(config)
    result = fitting_validator.validate_post_fit(well, product="oil")
"""

from .result import (
    IssueSeverity,
    IssueCategory,
    ValidationIssue,
    ValidationResult,
    merge_results,
)
from .input_validator import InputValidator
from .data_quality import DataQualityValidator
from .fitting_validator import FittingValidator

__all__ = [
    # Result types
    "IssueSeverity",
    "IssueCategory",
    "ValidationIssue",
    "ValidationResult",
    "merge_results",
    # Validators
    "InputValidator",
    "DataQualityValidator",
    "FittingValidator",
]
