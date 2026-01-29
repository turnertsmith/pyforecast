#!/usr/bin/env python3
"""Generate PDF documentation for PyForecast Validation module."""

from fpdf import FPDF


class PyForecastPDF(FPDF):
    """Custom PDF class with headers and footers."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "PyForecast Validation Guide", align="C")
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title: str):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(37, 99, 235)
        self.cell(0, 10, title, ln=True)
        self.set_draw_color(37, 99, 235)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(51, 65, 85)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

    def subsection_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(71, 85, 105)
        self.cell(0, 7, title, ln=True)
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def bullet_point(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.set_x(15)
        self.multi_cell(0, 5, f"  {chr(149)}  {text}")

    def code_block(self, code: str):
        self.set_font("Courier", "", 9)
        self.set_fill_color(30, 41, 59)
        self.set_text_color(226, 232, 240)
        self.multi_cell(0, 4, code, fill=True)
        self.ln(2)

    def issue_box(self, severity: str, description: str):
        colors = {
            "ERROR": (220, 53, 69),
            "WARNING": (245, 158, 11),
            "INFO": (16, 185, 129),
        }
        r, g, b = colors.get(severity, (100, 100, 100))

        self.set_draw_color(r, g, b)
        self.set_fill_color(r, g, b)
        self.rect(10, self.get_y(), 3, 20, "F")

        self.set_x(15)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(r, g, b)
        self.cell(0, 5, severity, ln=True)

        self.set_x(15)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.multi_cell(180, 5, description)
        self.ln(3)

    def table_header(self, headers: list, widths: list):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(248, 250, 252)
        self.set_text_color(30, 41, 59)
        for i, (header, width) in enumerate(zip(headers, widths)):
            self.cell(width, 7, header, border=1, fill=True, align="C")
        self.ln()

    def table_row(self, cells: list, widths: list, severity_col: int = -1):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 41, 59)
        for i, (cell, width) in enumerate(zip(cells, widths)):
            if i == severity_col:
                if "ERROR" in cell:
                    self.set_text_color(220, 53, 69)
                elif "WARNING" in cell or "WARN" in cell:
                    self.set_text_color(245, 158, 11)
                elif "INFO" in cell:
                    self.set_text_color(16, 185, 129)
            self.cell(width, 6, cell, border=1)
            self.set_text_color(30, 41, 59)
        self.ln()


def create_pdf():
    pdf = PyForecastPDF()

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(37, 99, 235)
    pdf.ln(60)
    pdf.cell(0, 15, "PyForecast", align="C", ln=True)

    pdf.set_font("Helvetica", "", 18)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 10, "Data Validation & Error Handling Guide", align="C", ln=True)

    pdf.ln(20)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, "Comprehensive Guide to Production Data Quality Assurance", align="C", ln=True)

    pdf.ln(40)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 8, "January 2026", align="C")

    # Table of Contents
    pdf.add_page()
    pdf.chapter_title("Table of Contents")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(37, 99, 235)
    toc_items = [
        ("1. Executive Summary", 3),
        ("2. Getting Started", 4),
        ("3. Validation Categories", 5),
        ("4. Input Validation (IV Codes)", 6),
        ("5. Data Quality Validation (DQ Codes)", 8),
        ("6. Fitting Prerequisite Validation (FP Codes)", 10),
        ("7. Fitting Result Validation (FR Codes)", 11),
        ("8. Error Code Quick Reference", 13),
        ("9. Programmatic Usage", 14),
        ("10. Configuration Reference", 15),
        ("11. Best Practices", 16),
    ]
    for item, page in toc_items:
        pdf.cell(0, 8, item, ln=True)

    # Executive Summary
    pdf.add_page()
    pdf.chapter_title("1. Executive Summary")
    pdf.body_text(
        "PyForecast includes a comprehensive data validation and error handling system "
        "designed to catch data quality issues before they impact decline curve analysis."
    )

    pdf.body_text("This system provides:")
    pdf.bullet_point("Automatic detection of common production data problems")
    pdf.bullet_point("Structured error codes for easy issue identification")
    pdf.bullet_point("Configurable thresholds to match your operational parameters")
    pdf.bullet_point("Actionable guidance for resolving each issue type")
    pdf.bullet_point("Integrated reporting in batch processing workflows")

    pdf.ln(5)
    pdf.body_text("The validation system operates at three levels:")

    pdf.table_header(["Level", "Description"], [50, 130])
    pdf.table_row(["Input Validation", "Verifies data format and value ranges"], [50, 130])
    pdf.table_row(["Data Quality", "Detects gaps, outliers, and anomalies"], [50, 130])
    pdf.table_row(["Fitting Validation", "Ensures fitting prerequisites and quality"], [50, 130])

    # Getting Started
    pdf.add_page()
    pdf.chapter_title("2. Getting Started")

    pdf.section_title("Basic Usage")
    pdf.body_text("Validation runs automatically during batch processing:")
    pdf.code_block("pyforecast process production_data.csv -o output/")

    pdf.body_text("After processing, you'll see a validation summary showing wells with errors, "
                  "warnings, and issues by category.")

    pdf.section_title("Configuration")
    pdf.body_text("Generate a configuration file with validation settings:")
    pdf.code_block("pyforecast init -o pyforecast.yaml")

    pdf.body_text("Key validation parameters you can customize:")
    pdf.bullet_point("max_oil_rate: Maximum expected oil rate (bbl/mo)")
    pdf.bullet_point("max_gas_rate: Maximum expected gas rate (mcf/mo)")
    pdf.bullet_point("gap_threshold_months: Minimum gap size to flag")
    pdf.bullet_point("outlier_sigma: Standard deviations for outlier detection")
    pdf.bullet_point("min_r_squared: Minimum acceptable R-squared value")

    # Validation Categories
    pdf.add_page()
    pdf.chapter_title("3. Validation Categories")

    pdf.section_title("Overview")
    pdf.body_text("All validation issues are categorized for easy filtering and reporting:")

    pdf.table_header(["Category", "Description", "When Checked"], [45, 80, 55])
    pdf.table_row(["DATA_FORMAT", "Column, date, value format issues", "Before fitting"], [45, 80, 55])
    pdf.table_row(["DATA_QUALITY", "Gaps, outliers, shut-ins", "Before fitting"], [45, 80, 55])
    pdf.table_row(["FITTING_PREREQ", "Pre-fit requirements not met", "Before fitting"], [45, 80, 55])
    pdf.table_row(["FITTING_RESULT", "Post-fit quality concerns", "After fitting"], [45, 80, 55])

    pdf.ln(5)
    pdf.section_title("Severity Levels")
    pdf.body_text("Each issue has a severity level indicating required action:")

    pdf.table_header(["Severity", "Meaning", "Action Required"], [35, 55, 90])
    pdf.table_row(["ERROR", "Cannot proceed safely", "Must resolve before trusting results"], [35, 55, 90], 0)
    pdf.table_row(["WARNING", "Can proceed with caution", "Review recommended"], [35, 55, 90], 0)
    pdf.table_row(["INFO", "Informational only", "No action required"], [35, 55, 90], 0)

    # Input Validation
    pdf.add_page()
    pdf.chapter_title("4. Input Validation (IV Codes)")
    pdf.body_text("Input validation checks that production data is properly formatted and within expected ranges.")

    pdf.subsection_title("IV001: Negative Production Values")
    pdf.issue_box("ERROR", "Production values must be non-negative. Negative values indicate "
                  "data entry errors or unit conversion problems.")
    pdf.body_text("Resolution: Check source data for typos, verify unit conversions, replace "
                  "negative values with zero or interpolated values.")

    pdf.subsection_title("IV002: Values Exceed Threshold")
    pdf.issue_box("WARNING", "Production values exceed configured maximum rates. Very high values "
                  "may indicate unit conversion errors (e.g., daily rates instead of monthly).")
    pdf.body_text("Default thresholds: Oil 50,000 bbl/mo, Gas 500,000 mcf/mo, Water 100,000 bbl/mo")

    pdf.subsection_title("IV003: Date Parsing Failed")
    pdf.issue_box("ERROR", "Production dates could not be parsed. This typically indicates "
                  "an unsupported date format.")
    pdf.body_text("Supported formats: YYYY-MM-DD, MM/DD/YYYY, DD-Mon-YYYY, Excel serial numbers")

    pdf.subsection_title("IV004: Future Dates in Data")
    pdf.issue_box("WARNING", "Production dates are in the future, which may indicate data entry "
                  "errors or placeholder values.")

    # Data Quality
    pdf.add_page()
    pdf.chapter_title("5. Data Quality Validation (DQ Codes)")
    pdf.body_text("Data quality validation detects patterns that may affect fitting accuracy.")

    pdf.subsection_title("DQ001: Data Gaps Detected")
    pdf.issue_box("WARNING", "Consecutive months of zero or near-zero production surrounded by "
                  "non-zero production. Gaps may represent missing data or operational shut-ins.")
    pdf.body_text("Default threshold: 2+ months. Resolution: Determine if gaps are operational "
                  "or data issues; regime detection handles shut-ins automatically.")

    pdf.subsection_title("DQ002: Outliers Detected")
    pdf.issue_box("WARNING", "Values significantly different from typical production pattern. "
                  "Uses Modified Z-Score with Median Absolute Deviation (MAD) for robust detection.")
    pdf.body_text("Default threshold: 3.0 sigma. Resolution: Investigate outliers, correct errors, "
                  "or adjust outlier_sigma sensitivity.")

    pdf.subsection_title("DQ003: Shut-in Periods Detected")
    pdf.issue_box("INFO", "Periods where production drops to near-zero then resumes. These trigger "
                  "regime detection, which fits only the most recent decline period.")
    pdf.body_text("Default threshold: < 1.0 bbl/month. Usually no action required.")

    pdf.subsection_title("DQ004: Low Data Variability")
    pdf.issue_box("WARNING", "Production data has very low variability (near-constant values). "
                  "May indicate synthetic data, smoothed data, or allocation issues.")
    pdf.body_text("Default threshold: CV < 0.05. Resolution: Verify data is metered production.")

    # Fitting Prereq
    pdf.add_page()
    pdf.chapter_title("6. Fitting Prerequisite Validation (FP Codes)")
    pdf.body_text("Pre-fit validation ensures data is suitable for decline curve analysis.")

    pdf.subsection_title("FP001: Insufficient Data Points")
    pdf.issue_box("ERROR", "Not enough data points to perform reliable curve fitting.")
    pdf.body_text("Default minimum: 6 months. Resolution: Obtain more history or skip fitting.")

    pdf.subsection_title("FP002: Increasing Trend")
    pdf.issue_box("WARNING", "Production shows an increasing trend rather than decline. "
                  "Decline curve analysis assumes production is declining.")
    pdf.body_text("Resolution: Verify well is in decline phase, check for recent workover, "
                  "wait for stabilization, or use regime detection.")

    pdf.subsection_title("FP003: Flat Trend")
    pdf.issue_box("WARNING", "Production shows minimal decline, which hyperbolic models may not fit well.")
    pdf.body_text("Resolution: Verify production pattern, consider if decline forecasting is appropriate.")

    # Fitting Result
    pdf.add_page()
    pdf.chapter_title("7. Fitting Result Validation (FR Codes)")
    pdf.body_text("Post-fit validation assesses the quality and reasonableness of fitted parameters.")

    pdf.subsection_title("FR001: Poor Fit Quality")
    pdf.issue_box("WARNING", "The fitted curve does not match the production data well. "
                  "WARNING for R-squared 0.3-0.5, ERROR for R-squared < 0.3.")
    pdf.body_text("Default threshold: R-squared < 0.5. Resolution: Review data quality, check "
                  "regime detection, consider alternative models.")

    pdf.subsection_title("FR003: B-Factor at Lower Bound")
    pdf.issue_box("INFO", "The fitted b-factor is at the configured lower bound (default 0.01), "
                  "suggesting near-exponential decline.")
    pdf.body_text("Usually acceptable - exponential decline is a valid pattern.")

    pdf.subsection_title("FR004: B-Factor at Upper Bound")
    pdf.issue_box("WARNING", "The fitted b-factor is at the configured upper bound (default 1.5). "
                  "Very high b-factors may indicate transient flow or data quality issues.")
    pdf.body_text("Resolution: Review production plot, check for early-time transient behavior.")

    pdf.subsection_title("FR005: Very High Decline Rate")
    pdf.issue_box("WARNING", "The fitted initial decline rate exceeds 100% per year, which is "
                  "unusual and may indicate fitting issues.")
    pdf.body_text("Resolution: Verify data is monthly, check early months quality. High declines "
                  "may be valid for tight oil/gas plays.")

    # Quick Reference
    pdf.add_page()
    pdf.chapter_title("8. Error Code Quick Reference")

    pdf.table_header(["Code", "Severity", "Category", "Description"], [20, 25, 50, 95])
    codes = [
        ("IV001", "ERROR", "DATA_FORMAT", "Negative production values"),
        ("IV002", "WARNING", "DATA_FORMAT", "Values exceed threshold"),
        ("IV003", "ERROR", "DATA_FORMAT", "Date parsing failed"),
        ("IV004", "WARNING", "DATA_FORMAT", "Future dates in data"),
        ("DQ001", "WARNING", "DATA_QUALITY", "Data gaps detected"),
        ("DQ002", "WARNING", "DATA_QUALITY", "Outliers detected"),
        ("DQ003", "INFO", "DATA_QUALITY", "Shut-in periods detected"),
        ("DQ004", "WARNING", "DATA_QUALITY", "Low data variability"),
        ("FP001", "ERROR", "FITTING_PREREQ", "Insufficient data points"),
        ("FP002", "WARNING", "FITTING_PREREQ", "Increasing trend"),
        ("FP003", "WARNING", "FITTING_PREREQ", "Flat trend"),
        ("FR001", "WARN/ERR", "FITTING_RESULT", "Poor fit (R-squared < threshold)"),
        ("FR003", "INFO", "FITTING_RESULT", "B-factor at lower bound"),
        ("FR004", "WARNING", "FITTING_RESULT", "B-factor at upper bound"),
        ("FR005", "WARNING", "FITTING_RESULT", "Very high decline rate"),
    ]
    for row in codes:
        pdf.table_row(list(row), [20, 25, 50, 95], 1)

    # Programmatic Usage
    pdf.add_page()
    pdf.chapter_title("9. Programmatic Usage")

    pdf.section_title("Basic Validation")
    pdf.code_block("""from pyforecast.validation import (
    InputValidator,
    DataQualityValidator,
    FittingValidator,
)

# Create validators
input_validator = InputValidator(max_oil_rate=75000)
quality_validator = DataQualityValidator(outlier_sigma=2.5)
fitting_validator = FittingValidator(min_r_squared=0.6)

# Validate a well
result = input_validator.validate(well)
if result.has_errors:
    for error in result.errors():
        print(f"{error.code}: {error.message}")""")

    pdf.section_title("Working with Results")
    pdf.code_block("""from pyforecast.validation import merge_results, IssueCategory

# Filter by category
quality_issues = result.by_category(IssueCategory.DATA_QUALITY)

# Merge multiple results
combined = merge_results([input_result, quality_result])

# Check status
print(f"Valid: {combined.is_valid}")
print(f"Errors: {combined.error_count}")""")

    # Configuration Reference
    pdf.add_page()
    pdf.chapter_title("10. Configuration Reference")

    pdf.section_title("Complete Validation Configuration")
    pdf.code_block("""validation:
  max_oil_rate: 50000      # bbl/month - IV002
  max_gas_rate: 500000     # mcf/month - IV002
  max_water_rate: 100000   # bbl/month - IV002
  gap_threshold_months: 2  # Months - DQ001
  outlier_sigma: 3.0       # Std deviations - DQ002
  shutin_threshold: 1.0    # Rate threshold - DQ003
  min_cv: 0.05             # Coefficient of variation - DQ004
  min_r_squared: 0.5       # R-squared threshold - FR001
  max_annual_decline: 1.0  # Annual fraction - FR005
  strict_mode: false       # Treat warnings as errors""")

    pdf.ln(5)
    pdf.section_title("Parameter Summary")
    pdf.table_header(["Parameter", "Default", "Description"], [55, 25, 100])
    params = [
        ("max_oil_rate", "50,000", "Max expected oil rate (bbl/mo)"),
        ("max_gas_rate", "500,000", "Max expected gas rate (mcf/mo)"),
        ("gap_threshold_months", "2", "Min consecutive zero months to flag"),
        ("outlier_sigma", "3.0", "Modified Z-score threshold"),
        ("min_cv", "0.05", "Minimum coefficient of variation"),
        ("min_r_squared", "0.5", "Minimum acceptable R-squared"),
        ("max_annual_decline", "1.0", "Max annual decline (1.0 = 100%)"),
        ("strict_mode", "false", "Treat warnings as errors"),
    ]
    for row in params:
        pdf.table_row(list(row), [55, 25, 100])

    # Best Practices
    pdf.add_page()
    pdf.chapter_title("11. Best Practices")

    pdf.section_title("Data Preparation Checklist")
    pdf.body_text("Before running PyForecast:")
    pdf.bullet_point("Verify all dates are in a consistent format")
    pdf.bullet_point("Confirm production values are monthly (not daily/annual)")
    pdf.bullet_point("Check for negative values and correct")
    pdf.bullet_point("Remove forecast/projected data from historical records")
    pdf.bullet_point("Verify units match expected (bbl, mcf)")

    pdf.section_title("Interpreting Results")
    pdf.bullet_point("Address ERRORs first - These prevent reliable analysis")
    pdf.bullet_point("Review WARNINGs - Understand the cause before accepting")
    pdf.bullet_point("Note INFOs - Generally informational, no action needed")

    pdf.section_title("Adjusting Thresholds")
    pdf.table_header(["Scenario", "Adjustment"], [60, 120])
    pdf.table_row(["Prolific wells", "Increase max_oil_rate and max_gas_rate"], [60, 120])
    pdf.table_row(["Tight formations", "Increase max_annual_decline"], [60, 120])
    pdf.table_row(["Noisy data", "Increase outlier_sigma to reduce false positives"], [60, 120])
    pdf.table_row(["Strict QC", "Set strict_mode: true"], [60, 120])

    pdf.ln(5)
    pdf.section_title("Common Workflows")
    pdf.body_text("Initial data load:")
    pdf.code_block("pyforecast process data.csv -o output/ --no-plots\ncat output/validation_report.txt")

    pdf.body_text("Production processing with custom config:")
    pdf.code_block("pyforecast process data.csv -o output/ --config basin_config.yaml")

    # Save
    pdf.output("PyForecast_Validation_Guide.pdf")
    print("PDF generated: PyForecast_Validation_Guide.pdf")


if __name__ == "__main__":
    create_pdf()
