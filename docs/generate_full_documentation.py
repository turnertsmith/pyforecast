#!/usr/bin/env python3
"""Generate comprehensive PyForecast project documentation PDF."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("Installing fpdf2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2", "-q"])
    from fpdf import FPDF


def sanitize(text: str) -> str:
    """Replace Unicode characters with ASCII equivalents."""
    replacements = {
        '\u2014': '-',   # em dash
        '\u2013': '-',   # en dash
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u2026': '...', # ellipsis
        '\u2022': '*',   # bullet
        '\u00d7': 'x',   # multiplication sign
        '\u2264': '<=',  # less than or equal
        '\u2265': '>=',  # greater than or equal
        '\u03c3': 'sigma', # sigma
        '\u03bc': 'mu',    # mu
        '\u2192': '->',    # right arrow
        '\u2190': '<-',    # left arrow
        '\u2713': '+',     # check mark
        '\u2717': 'x',     # x mark
        '\u251c': '|',     # box drawing
        '\u2514': '\\',    # box drawing
        '\u2500': '-',     # horizontal line
        '\u2502': '|',     # vertical line
        '\u00b2': '2',     # superscript 2
        '\u00b0': ' deg ', # degree
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


class DocPDF(FPDF):
    """Custom PDF class for documentation."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "PyForecast Documentation", align="C")
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title: str):
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(37, 99, 235)
        self.cell(0, 12, sanitize(title), ln=True)
        self.set_draw_color(37, 99, 235)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(51, 65, 85)
        self.cell(0, 10, sanitize(title), ln=True)
        self.ln(3)

    def subsection_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(71, 85, 105)
        self.cell(0, 8, sanitize(title), ln=True)
        self.ln(2)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.multi_cell(0, 5, sanitize(text))
        self.ln(3)

    def bullet_point(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.cell(5, 5, "*")
        self.multi_cell(0, 5, sanitize(text))
        self.ln(1)

    def code_block(self, code: str):
        self.set_font("Courier", "", 8)
        self.set_fill_color(30, 41, 59)
        self.set_text_color(226, 232, 240)
        for line in code.split('\n'):
            safe_line = sanitize(line)
            self.cell(0, 4, safe_line[:95], ln=True, fill=True)
        self.ln(3)

    def table_row(self, cols: list, widths: list, header: bool = False):
        self.set_font("Helvetica", "B" if header else "", 9)
        if header:
            self.set_fill_color(241, 245, 249)
            self.set_text_color(30, 41, 59)
        else:
            self.set_text_color(51, 65, 85)
        for i, (col, w) in enumerate(zip(cols, widths)):
            self.cell(w, 6, sanitize(str(col)[:int(w/2)]), border=1, fill=header)
        self.ln()


def create_full_documentation():
    """Create the comprehensive documentation PDF."""
    pdf = DocPDF()

    # === COVER PAGE ===
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(37, 99, 235)
    pdf.ln(40)
    pdf.cell(0, 20, "PyForecast", align="C", ln=True)

    pdf.set_font("Helvetica", "", 18)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 12, "Oil & Gas DCA Auto-Forecasting Tool", align="C", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, "Comprehensive Documentation", align="C", ln=True)

    pdf.ln(40)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 8, "Version 0.1.0", align="C", ln=True)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y')}", align="C", ln=True)

    # === TABLE OF CONTENTS ===
    pdf.add_page()
    pdf.chapter_title("Table of Contents")

    toc = [
        ("1. Project Overview", 3),
        ("2. Installation & Quick Start", 4),
        ("3. Core Concepts", 5),
        ("4. Decline Curve Models", 6),
        ("5. CLI Reference", 7),
        ("6. Data Validation", 9),
        ("7. Refinement & Analysis", 12),
        ("8. Ground Truth Comparison", 14),
        ("9. Configuration Reference", 17),
        ("10. Module Reference", 19),
    ]

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(51, 65, 85)
    for item, page in toc:
        pdf.cell(150, 8, item)
        pdf.cell(0, 8, str(page), align="R", ln=True)

    # === 1. PROJECT OVERVIEW ===
    pdf.add_page()
    pdf.chapter_title("1. Project Overview")

    pdf.body_text(
        "PyForecast is an automated decline curve analysis (DCA) tool for oil and gas "
        "production forecasting. It automatically fits hyperbolic decline curves to "
        "historical production data, detects regime changes (refracs, workovers), and "
        "exports forecasts in ARIES-compatible formats."
    )

    pdf.section_title("Key Features")
    features = [
        "Automatic hyperbolic decline curve fitting with configurable constraints",
        "Regime change detection for refracs and workovers",
        "Recency-weighted fitting to prioritize recent production trends",
        "Batch processing with parallel execution support",
        "ARIES AC_FORECAST and AC_ECONOMIC export formats",
        "Comprehensive data validation and quality checks",
        "Hindcast validation for forecast accuracy measurement",
        "Ground truth comparison against expert ARIES projections",
        "Interactive plots and visualization",
    ]
    for f in features:
        pdf.bullet_point(f)

    pdf.section_title("Supported Products")
    pdf.body_text("PyForecast supports forecasting for three fluid types:")
    pdf.bullet_point("Oil (bbl/month)")
    pdf.bullet_point("Gas (mcf/month)")
    pdf.bullet_point("Water (bbl/month)")

    pdf.section_title("Dependencies")
    deps = [
        ("pandas", "Data manipulation and CSV I/O"),
        ("numpy", "Numerical computations"),
        ("scipy", "Curve fitting optimization"),
        ("plotly", "Interactive visualization"),
        ("typer", "CLI framework"),
        ("pyyaml", "Configuration file support"),
    ]
    for name, desc in deps:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(25, 5, name)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 5, f"- {desc}", ln=True)
    pdf.ln(3)

    # === 2. INSTALLATION & QUICK START ===
    pdf.add_page()
    pdf.chapter_title("2. Installation & Quick Start")

    pdf.section_title("Installation")
    pdf.body_text("Install PyForecast using pip:")
    pdf.code_block("pip install pyforecast")

    pdf.body_text("Or install from source:")
    pdf.code_block("git clone https://github.com/yourorg/pyforecast.git\ncd pyforecast\npip install -e .")

    pdf.section_title("Quick Start")

    pdf.subsection_title("1. Generate a configuration file")
    pdf.code_block("pyforecast init -o pyforecast.yaml")

    pdf.subsection_title("2. Process production data")
    pdf.code_block("pyforecast process production_data.csv -o output/")

    pdf.subsection_title("3. Review outputs")
    pdf.body_text("After processing, the output directory contains:")
    pdf.bullet_point("forecasts.csv - Fitted parameters and EUR estimates")
    pdf.bullet_point("ac_economic.csv - ARIES AC_ECONOMIC export format")
    pdf.bullet_point("validation_report.txt - Data quality issues")
    pdf.bullet_point("plots/ - Individual well decline curve plots")

    pdf.section_title("Input Data Format")
    pdf.body_text(
        "PyForecast accepts CSV files with production data. Required columns include "
        "a well identifier (API or PROPNUM) and date/production columns."
    )
    pdf.code_block("API,Date,Oil,Gas,Water\n42001000010000,2020-01-01,1500,5000,200\n42001000010000,2020-02-01,1400,4800,210")

    # === 3. CORE CONCEPTS ===
    pdf.add_page()
    pdf.chapter_title("3. Core Concepts")

    pdf.section_title("Decline Curve Analysis (DCA)")
    pdf.body_text(
        "Decline curve analysis is a technique for forecasting oil and gas production "
        "by fitting mathematical models to historical production trends. The underlying "
        "assumption is that production will continue to decline in a predictable pattern "
        "based on reservoir physics and well performance."
    )

    pdf.section_title("Key Parameters")

    params = [
        ("qi (Initial Rate)", "Production rate at the start of decline (bbl/day or mcf/day)"),
        ("di (Initial Decline)", "Rate of decline at time zero (fraction per month)"),
        ("b (Decline Exponent)", "Controls curvature: 0=exponential, 0.5=typical, 1=harmonic"),
        ("dmin (Terminal Decline)", "Minimum annual decline rate (typically 6%)"),
    ]
    for name, desc in params:
        pdf.subsection_title(name)
        pdf.body_text(desc)

    pdf.section_title("Regime Detection")
    pdf.body_text(
        "PyForecast automatically detects regime changes (refracs, workovers, artificial lift "
        "installations) that cause production to increase. When detected, fitting starts from "
        "the most recent regime rather than historical peak production."
    )

    pdf.body_text("Detection criteria (configurable):")
    pdf.bullet_point("Production increase > 100% (threshold)")
    pdf.bullet_point("Sustained for 2+ months (sustained_months)")
    pdf.bullet_point("Evaluated over 6-month windows (window)")

    pdf.section_title("Recency Weighting")
    pdf.body_text(
        "By default, PyForecast weights recent data points more heavily than older data "
        "using exponential decay. This helps the fit track current well behavior rather "
        "than being anchored to historical patterns that may no longer apply."
    )
    pdf.body_text("Default half-life: 12 months (configurable via recency_half_life)")

    # === 4. DECLINE CURVE MODELS ===
    pdf.add_page()
    pdf.chapter_title("4. Decline Curve Models")

    pdf.section_title("Hyperbolic Decline")
    pdf.body_text(
        "The general hyperbolic decline equation relates production rate (q) to time (t):"
    )
    pdf.body_text("q(t) = qi / (1 + b * di * t)^(1/b)")

    pdf.body_text("Where:")
    pdf.bullet_point("q(t) = production rate at time t")
    pdf.bullet_point("qi = initial production rate")
    pdf.bullet_point("di = initial decline rate")
    pdf.bullet_point("b = decline exponent (0 to ~1.5)")

    pdf.section_title("Special Cases")

    pdf.subsection_title("Exponential Decline (b = 0)")
    pdf.body_text("q(t) = qi * exp(-di * t)")
    pdf.body_text(
        "Exponential decline represents constant percentage decline. Common in "
        "solution gas drive reservoirs and late-time production."
    )

    pdf.subsection_title("Harmonic Decline (b = 1)")
    pdf.body_text("q(t) = qi / (1 + di * t)")
    pdf.body_text(
        "Harmonic decline represents proportional decline rate. Less common but "
        "may occur in gravity drainage or some water drive reservoirs."
    )

    pdf.section_title("Terminal Decline (dmin)")
    pdf.body_text(
        "Hyperbolic decline predicts ever-decreasing decline rates, which is physically "
        "unrealistic at late times. PyForecast switches to exponential decline when the "
        "instantaneous decline rate reaches dmin (default 6%/year)."
    )

    pdf.section_title("B-Factor Interpretation")
    bfactors = [
        ("0.0 - 0.3", "Near-exponential", "Solution gas drive, depletion"),
        ("0.3 - 0.6", "Typical hyperbolic", "Most unconventional wells"),
        ("0.6 - 1.0", "High hyperbolic", "Multi-phase flow, complex reservoirs"),
        ("1.0 - 1.5", "Super-harmonic", "Transient flow, early time data"),
    ]
    widths = [35, 45, 100]
    pdf.table_row(["B Range", "Description", "Typical Reservoir"], widths, header=True)
    for row in bfactors:
        pdf.table_row(row, widths)

    # === 5. CLI REFERENCE ===
    pdf.add_page()
    pdf.chapter_title("5. CLI Reference")

    pdf.section_title("Main Commands")

    commands = [
        ("pyforecast process", "Process production data and generate forecasts"),
        ("pyforecast init", "Generate a default configuration file"),
        ("pyforecast analyze-fits", "Analyze accumulated fit logs"),
        ("pyforecast suggest-params", "Get parameter suggestions from learning"),
        ("pyforecast calibrate-regime", "Calibrate regime detection thresholds"),
    ]
    for cmd, desc in commands:
        pdf.subsection_title(cmd)
        pdf.body_text(desc)

    pdf.section_title("Process Command Options")

    pdf.code_block("pyforecast process INPUT_FILE [OPTIONS]")

    options = [
        ("-o, --output", "Output directory", "Required"),
        ("-c, --config", "Configuration file", "Optional"),
        ("-p, --product", "Products to forecast (oil,gas,water)", "All"),
        ("--plots/--no-plots", "Generate individual well plots", "True"),
        ("--batch-plot/--no-batch-plot", "Generate batch overlay plot", "True"),
        ("--format", "Export format (ac_forecast, ac_economic, json)", "ac_economic"),
        ("--hindcast", "Enable hindcast validation", "False"),
        ("--log-fits", "Enable fit logging", "False"),
        ("--residuals", "Enable residual analysis", "False"),
        ("--ground-truth", "ARIES file for comparison", "None"),
        ("--gt-months", "Months to compare", "60"),
        ("--gt-plots", "Generate comparison plots", "False"),
        ("--gt-lazy", "Stream ARIES file (low memory)", "False"),
        ("--gt-workers", "Parallel validation workers", "1"),
    ]

    widths = [50, 80, 40]
    pdf.table_row(["Option", "Description", "Default"], widths, header=True)
    for row in options:
        pdf.table_row(row, widths)

    # === 6. DATA VALIDATION ===
    pdf.add_page()
    pdf.chapter_title("6. Data Validation")

    pdf.body_text(
        "PyForecast includes a comprehensive validation system that checks data quality "
        "at multiple stages: input validation, data quality checks, and fit result validation."
    )

    pdf.section_title("Validation Categories")
    categories = [
        ("DATA_FORMAT (IV)", "Column, date, and value format issues"),
        ("DATA_QUALITY (DQ)", "Gaps, outliers, shut-ins, variability"),
        ("FITTING_PREREQ (FP)", "Pre-fit requirements not met"),
        ("FITTING_RESULT (FR)", "Post-fit quality concerns"),
    ]
    widths = [55, 125]
    pdf.table_row(["Category", "Description"], widths, header=True)
    for row in categories:
        pdf.table_row(row, widths)

    pdf.section_title("Severity Levels")
    pdf.bullet_point("ERROR - Cannot proceed safely, must resolve")
    pdf.bullet_point("WARNING - Can proceed with caution, review recommended")
    pdf.bullet_point("INFO - Informational only, no action required")

    pdf.section_title("Input Validation Codes")
    iv_codes = [
        ("IV001", "ERROR", "Negative production values"),
        ("IV002", "WARNING", "Values exceed threshold"),
        ("IV003", "ERROR", "Date parsing failed"),
        ("IV004", "WARNING", "Future dates in data"),
    ]
    widths = [25, 30, 125]
    pdf.table_row(["Code", "Severity", "Description"], widths, header=True)
    for row in iv_codes:
        pdf.table_row(row, widths)

    pdf.section_title("Data Quality Codes")
    dq_codes = [
        ("DQ001", "WARNING", "Data gaps detected (>= threshold months)"),
        ("DQ002", "WARNING", "Outliers detected (> sigma threshold)"),
        ("DQ003", "INFO", "Shut-in periods detected"),
        ("DQ004", "WARNING", "Low data variability (CV < threshold)"),
    ]
    pdf.table_row(["Code", "Severity", "Description"], widths, header=True)
    for row in dq_codes:
        pdf.table_row(row, widths)

    pdf.add_page()
    pdf.section_title("Fitting Prerequisite Codes")
    fp_codes = [
        ("FP001", "ERROR", "Insufficient data points"),
        ("FP002", "WARNING", "Increasing trend (not declining)"),
        ("FP003", "WARNING", "Flat trend (minimal decline)"),
    ]
    pdf.table_row(["Code", "Severity", "Description"], widths, header=True)
    for row in fp_codes:
        pdf.table_row(row, widths)

    pdf.section_title("Fitting Result Codes")
    fr_codes = [
        ("FR001", "WARN/ERR", "Poor fit quality (R-squared < threshold)"),
        ("FR003", "INFO", "B-factor at lower bound"),
        ("FR004", "WARNING", "B-factor at upper bound"),
        ("FR005", "WARNING", "Very high decline rate (>100%/year)"),
    ]
    pdf.table_row(["Code", "Severity", "Description"], widths, header=True)
    for row in fr_codes:
        pdf.table_row(row, widths)

    pdf.section_title("Validation Report")
    pdf.body_text(
        "A validation_report.txt file is generated in the output directory with "
        "detailed information about all validation issues found during processing."
    )

    # === 7. REFINEMENT & ANALYSIS ===
    pdf.add_page()
    pdf.chapter_title("7. Refinement & Analysis")

    pdf.body_text(
        "The refinement module provides tools for measuring, logging, analyzing, and "
        "improving decline curve fit quality. All features are disabled by default."
    )

    pdf.section_title("Hindcast Validation")
    pdf.body_text(
        "Hindcast validation splits historical data into training and holdout periods, "
        "fits on training data, and measures prediction accuracy on the holdout."
    )

    pdf.body_text("Metrics computed:")
    pdf.bullet_point("MAPE - Mean Absolute Percentage Error (good: < 30%)")
    pdf.bullet_point("Correlation - Pearson correlation coefficient (good: > 0.7)")
    pdf.bullet_point("Bias - Systematic over/under prediction (good: < 0.3)")

    pdf.code_block("pyforecast process data.csv --hindcast -o output/")

    pdf.section_title("Fit Logging")
    pdf.body_text(
        "Fit logging persists all fit parameters and metrics to SQLite storage for "
        "analysis and learning across projects."
    )

    pdf.body_text("Default storage: ~/.pyforecast/fit_logs.db")

    pdf.code_block("pyforecast process data.csv --log-fits -o output/\npyforecast analyze-fits ~/.pyforecast/fit_logs.db")

    pdf.section_title("Residual Analysis")
    pdf.body_text(
        "Residual analysis detects systematic fit errors that may not be apparent from "
        "R-squared alone, including autocorrelation and early/late bias patterns."
    )

    pdf.body_text("Diagnostics computed:")
    pdf.bullet_point("Durbin-Watson statistic (ideal: ~2.0)")
    pdf.bullet_point("Early bias (error in first half)")
    pdf.bullet_point("Late bias (error in second half)")
    pdf.bullet_point("Lag-1 autocorrelation")

    pdf.code_block("pyforecast process data.csv --residuals -o output/")

    pdf.section_title("Parameter Learning")
    pdf.body_text(
        "Parameter learning analyzes accumulated fit logs to suggest optimal fitting "
        "parameters for different basins and formations."
    )

    pdf.code_block("pyforecast suggest-params -p oil --basin Permian")

    # === 8. GROUND TRUTH COMPARISON ===
    pdf.add_page()
    pdf.chapter_title("8. Ground Truth Comparison")

    pdf.body_text(
        "Ground truth comparison measures how well pyforecast's auto-fitted decline "
        "curves match expert/approved ARIES projections."
    )

    pdf.section_title("Use Case")
    pdf.body_text(
        "When you have existing ARIES forecasts created by reservoir engineers, you can "
        "compare pyforecast's automated fits against these 'ground truth' projections to:"
    )
    pdf.bullet_point("Validate that auto-fitting produces reasonable results")
    pdf.bullet_point("Identify wells where manual review may be needed")
    pdf.bullet_point("Measure overall fitting accuracy across a portfolio")

    pdf.section_title("CLI Usage")
    pdf.code_block("pyforecast process data.csv --ground-truth aries.csv -o output/\n\n# With comparison plots\npyforecast process data.csv --ground-truth aries.csv --gt-plots -o output/\n\n# Large file support\npyforecast process data.csv --ground-truth large.csv --gt-lazy --gt-workers 4 -o output/")

    pdf.section_title("Metrics Computed")
    metrics = [
        ("MAPE", "Mean Absolute Percentage Error", "< 20%"),
        ("Correlation", "Pearson correlation of curves", "> 0.95"),
        ("Cumulative Diff", "Total production difference", "< 15%"),
        ("B-Factor Diff", "Absolute b-factor difference", "< 0.3"),
    ]
    widths = [45, 80, 45]
    pdf.table_row(["Metric", "Description", "Good Value"], widths, header=True)
    for row in metrics:
        pdf.table_row(row, widths)

    pdf.section_title("Match Quality Grades")
    grades = [
        ("A", "Excellent match - all metrics well within thresholds"),
        ("B", "Good match - minor deviations"),
        ("C", "Fair match - some significant differences"),
        ("D", "Poor match - review recommended"),
        ("X", "Insufficient data - MAPE could not be calculated"),
    ]
    widths = [20, 160]
    pdf.table_row(["Grade", "Description"], widths, header=True)
    for row in grades:
        pdf.table_row(row, widths)

    pdf.add_page()
    pdf.section_title("ARIES Expression Format")
    pdf.body_text("PyForecast parses ARIES AC_ECONOMIC expression format:")
    pdf.code_block('"{Qi} X {unit} {Dmin%} {type} B/{b} {Di%}"\n\nExample: "1000 X B/M 6 EXP B/0.50 8.5"')

    pdf.body_text("Supported units:")
    pdf.bullet_point("B/M - barrels/month (oil)")
    pdf.bullet_point("B/D - barrels/day (oil)")
    pdf.bullet_point("M/M - mcf/month (gas)")
    pdf.bullet_point("M/D - mcf/day (gas)")

    pdf.section_title("Advanced Features")

    pdf.subsection_title("Rate Validation")
    pdf.body_text(
        "Forecast arrays are automatically validated for NaN, infinite, and negative values. "
        "Issues are logged as warnings and values are cleaned before metric calculation."
    )

    pdf.subsection_title("MAPE Edge Case Handling")
    pdf.body_text(
        "When fewer than 3 valid data points exist above the rate threshold, MAPE returns "
        "None and match_grade returns 'X' (insufficient data)."
    )

    pdf.subsection_title("Identifier Mismatch Logging")
    pdf.body_text(
        "The validate_batch() method tracks wells that exist in only one dataset, helping "
        "diagnose ID normalization issues between pyforecast and ARIES data."
    )

    pdf.subsection_title("Comparison Plots")
    pdf.body_text(
        "Generate overlay plots showing ARIES vs pyforecast curves with metrics annotation. "
        "Plots are saved to output/ground_truth_plots/ directory."
    )

    pdf.subsection_title("Lazy Loading")
    pdf.body_text(
        "For large ARIES files (10,000+ wells), use --gt-lazy flag to stream data from disk "
        "instead of loading into memory. Uses constant memory regardless of file size."
    )

    pdf.subsection_title("Parallel Validation")
    pdf.body_text(
        "For large batches, use --gt-workers N to enable parallel validation with "
        "ThreadPoolExecutor. Default is 1 (sequential)."
    )

    # === 9. CONFIGURATION REFERENCE ===
    pdf.add_page()
    pdf.chapter_title("9. Configuration Reference")

    pdf.body_text("PyForecast uses YAML configuration files. Generate a default with:")
    pdf.code_block("pyforecast init -o pyforecast.yaml")

    pdf.section_title("Product Configuration")
    pdf.code_block("oil:\n  b_min: 0.01      # Minimum b-factor\n  b_max: 1.5       # Maximum b-factor\n  dmin: 0.06       # Terminal decline (annual)")

    pdf.section_title("Regime Detection")
    pdf.code_block("regime:\n  threshold: 1.0          # Min increase to trigger (1.0 = 100%)\n  window: 6               # Months of trend data\n  sustained_months: 2     # Months elevation must persist")

    pdf.section_title("Fitting Parameters")
    pdf.code_block("fitting:\n  recency_half_life: 12.0  # Half-life for data weighting\n  min_points: 6            # Minimum months required")

    pdf.section_title("Output Configuration")
    pdf.code_block("output:\n  products: [oil, gas, water]\n  plots: true\n  batch_plot: true\n  format: ac_economic")

    pdf.section_title("Validation Configuration")
    pdf.code_block("validation:\n  max_oil_rate: 50000\n  max_gas_rate: 500000\n  gap_threshold_months: 2\n  outlier_sigma: 3.0\n  min_r_squared: 0.5\n  strict_mode: false")

    pdf.add_page()
    pdf.section_title("Refinement Configuration")
    pdf.code_block("refinement:\n  enable_logging: false\n  log_storage: sqlite\n  enable_hindcast: false\n  hindcast_holdout_months: 6\n  min_training_months: 12\n  enable_residual_analysis: false\n  ground_truth_file: null\n  ground_truth_months: 60\n  ground_truth_lazy: false\n  ground_truth_workers: 1")

    # === 10. MODULE REFERENCE ===
    pdf.add_page()
    pdf.chapter_title("10. Module Reference")

    pdf.section_title("Package Structure")
    pdf.code_block("pyforecast/\n  core/\n    models.py         # Decline curve models\n    fitting.py        # Curve fitting logic\n    regime_detection.py  # Regime change detection\n    selection.py      # Model selection\n  validation/\n    input_validator.py\n    data_quality_validator.py\n    fitting_validator.py\n  refinement/\n    hindcast.py       # Hindcast validation\n    fit_logger.py     # Fit logging\n    residual_analysis.py\n    parameter_learning.py\n    ground_truth.py   # Ground truth comparison\n    plotting.py       # Comparison plots\n    schemas.py        # Data classes\n    storage.py        # SQLite storage\n  import_/\n    aries_forecast.py # ARIES file parser\n  export/\n    ac_forecast.py    # AC_FORECAST export\n    ac_economic.py    # AC_ECONOMIC export\n  batch/\n    processor.py      # Batch processing\n  cli/\n    commands.py       # CLI commands\n  config.py           # Configuration")

    pdf.section_title("Key Classes")

    classes = [
        ("HyperbolicModel", "Decline curve model with qi, di, b, dmin parameters"),
        ("DeclineFitter", "Fits decline curves to production data"),
        ("RegimeDetector", "Detects production regime changes"),
        ("HindcastValidator", "Validates forecasts via backtesting"),
        ("FitLogger", "Logs fit parameters to storage"),
        ("GroundTruthValidator", "Compares against ARIES projections"),
        ("AriesForecastImporter", "Parses ARIES AC_ECONOMIC files"),
        ("InputValidator", "Validates input data format/values"),
        ("DataQualityValidator", "Checks for gaps, outliers, etc."),
    ]
    widths = [55, 125]
    pdf.table_row(["Class", "Description"], widths, header=True)
    for row in classes:
        pdf.table_row(row, widths)

    # Save
    output_path = Path("/Users/turnersmith/pyforecast/drive/PyForecast_Full_Documentation.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    print(f"PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    create_full_documentation()
