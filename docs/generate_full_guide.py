#!/usr/bin/env python3
"""Generate comprehensive PDF documentation for PyForecast."""

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
            self.cell(0, 10, "PyForecast User Guide", align="C")
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title: str):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(37, 99, 235)
        self.cell(0, 10, title)
        self.ln()
        self.set_draw_color(37, 99, 235)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(51, 65, 85)
        self.cell(0, 8, title)
        self.ln(10)

    def subsection_title(self, title: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(71, 85, 105)
        self.cell(0, 7, title)
        self.ln(8)

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

    def numbered_item(self, num: int, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 41, 59)
        self.set_x(15)
        self.multi_cell(0, 5, f"  {num}.  {text}")

    def code_block(self, code: str):
        self.set_font("Courier", "", 9)
        self.set_fill_color(30, 41, 59)
        self.set_text_color(226, 232, 240)
        self.multi_cell(0, 4, code, fill=True)
        self.ln(2)

    def note_box(self, text: str, title: str = "Note"):
        self.set_draw_color(37, 99, 235)
        self.set_fill_color(239, 246, 255)
        y_start = self.get_y()
        self.rect(10, y_start, 190, 20, "DF")
        self.set_xy(15, y_start + 2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(37, 99, 235)
        self.cell(0, 5, title)
        self.set_xy(15, y_start + 8)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 41, 59)
        self.multi_cell(180, 4, text)
        self.set_y(y_start + 22)

    def warning_box(self, text: str):
        self.set_draw_color(245, 158, 11)
        self.set_fill_color(255, 251, 235)
        y_start = self.get_y()
        self.rect(10, y_start, 190, 18, "DF")
        self.set_xy(15, y_start + 2)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(245, 158, 11)
        self.cell(0, 5, "Important")
        self.set_xy(15, y_start + 8)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 41, 59)
        self.multi_cell(180, 4, text)
        self.set_y(y_start + 20)

    def table_header(self, headers: list, widths: list):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(248, 250, 252)
        self.set_text_color(30, 41, 59)
        for header, width in zip(headers, widths):
            self.cell(width, 7, header, border=1, fill=True, align="C")
        self.ln()

    def table_row(self, cells: list, widths: list):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 41, 59)
        for cell, width in zip(cells, widths):
            self.cell(width, 6, str(cell), border=1)
        self.ln()


def create_pdf():
    pdf = PyForecastPDF()

    # ========== COVER PAGE ==========
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 36)
    pdf.set_text_color(37, 99, 235)
    pdf.ln(50)
    pdf.cell(0, 15, "PyForecast", align="C")
    pdf.ln(15)

    pdf.set_font("Helvetica", "", 18)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 10, "Oil & Gas Decline Curve Analysis", align="C")
    pdf.ln(8)
    pdf.cell(0, 10, "User Guide", align="C")

    pdf.ln(30)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 8, "Automated Hyperbolic Decline Curve Fitting", align="C")
    pdf.ln(6)
    pdf.cell(0, 8, "with Regime Detection and Data Validation", align="C")

    pdf.ln(50)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 8, "January 2026", align="C")

    # ========== TABLE OF CONTENTS ==========
    pdf.add_page()
    pdf.chapter_title("Table of Contents")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(37, 99, 235)

    toc = [
        "1. Introduction",
        "2. Quick Start",
        "3. Decline Curve Theory",
        "4. Fitting Parameters",
        "    4.1 B-Factor (Hyperbolic Exponent)",
        "    4.2 Initial Decline Rate (Di)",
        "    4.3 Terminal Decline (Dmin)",
        "5. Regime Detection",
        "6. Configuration",
        "7. Data Validation",
        "8. Output Formats",
        "9. Command Reference",
        "10. Programmatic Usage",
        "Appendix A: Parameter Quick Reference",
    ]
    for item in toc:
        pdf.cell(0, 7, item)
        pdf.ln()

    # ========== 1. INTRODUCTION ==========
    pdf.add_page()
    pdf.chapter_title("1. Introduction")

    pdf.body_text(
        "PyForecast is an automated decline curve analysis (DCA) tool for oil and gas "
        "production forecasting. It fits hyperbolic decline models to historical production "
        "data and generates forecasts suitable for reserves estimation and economic analysis."
    )

    pdf.section_title("Key Features")
    pdf.bullet_point("Hyperbolic decline curve fitting with configurable b-factor bounds")
    pdf.bullet_point("Automatic regime change detection (workovers, refracs, RTPs)")
    pdf.bullet_point("Per-product configuration (oil, gas, water)")
    pdf.bullet_point("Recency-weighted fitting to favor recent production trends")
    pdf.bullet_point("ARIES-compatible forecast export formats")
    pdf.bullet_point("Comprehensive data validation and quality checks")
    pdf.bullet_point("Batch processing with parallel execution")
    pdf.bullet_point("Interactive visualization with semi-log plots")

    pdf.ln(3)
    pdf.section_title("Supported Input Formats")
    pdf.bullet_point("Enverus production exports (CSV/Excel)")
    pdf.bullet_point("ARIES production data format")
    pdf.bullet_point("Auto-detection of file format")

    # ========== 2. QUICK START ==========
    pdf.add_page()
    pdf.chapter_title("2. Quick Start")

    pdf.section_title("Installation")
    pdf.code_block("pip install pyforecast")

    pdf.section_title("Basic Usage")
    pdf.body_text("Process production data with default settings:")
    pdf.code_block("pyforecast process production.csv -o output/")

    pdf.body_text("This will:")
    pdf.numbered_item(1, "Load wells from the input file")
    pdf.numbered_item(2, "Fit decline curves for oil, gas, and water")
    pdf.numbered_item(3, "Export ARIES-format forecasts")
    pdf.numbered_item(4, "Generate interactive plots")
    pdf.numbered_item(5, "Create a validation report")

    pdf.ln(3)
    pdf.section_title("Custom Configuration")
    pdf.body_text("Generate a configuration file to customize fitting parameters:")
    pdf.code_block("pyforecast init -o config.yaml")

    pdf.body_text("Then run with your configuration:")
    pdf.code_block("pyforecast process production.csv -o output/ --config config.yaml")

    pdf.ln(3)
    pdf.section_title("Single Well Analysis")
    pdf.body_text("Plot and analyze a single well:")
    pdf.code_block("pyforecast plot production.csv --well-id \"42-001-12345\" --product oil")

    # ========== 3. DECLINE CURVE THEORY ==========
    pdf.add_page()
    pdf.chapter_title("3. Decline Curve Theory")

    pdf.section_title("The Hyperbolic Decline Equation")
    pdf.body_text(
        "PyForecast uses the Arps hyperbolic decline equation, the industry standard for "
        "production forecasting:"
    )

    pdf.ln(2)
    pdf.set_font("Courier", "B", 11)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 8, "    q(t) = qi / (1 + b * Di * t)^(1/b)", align="C")
    pdf.ln(10)

    pdf.body_text("Where:")
    pdf.bullet_point("q(t) = Production rate at time t")
    pdf.bullet_point("qi = Initial production rate at t=0")
    pdf.bullet_point("Di = Initial decline rate (fraction/month)")
    pdf.bullet_point("b = Hyperbolic exponent (dimensionless)")
    pdf.bullet_point("t = Time (months)")

    pdf.ln(3)
    pdf.section_title("Decline Types")
    pdf.body_text("The b-factor determines the decline behavior:")

    pdf.table_header(["b Value", "Decline Type", "Characteristics"], [30, 45, 115])
    pdf.table_row(["b = 0", "Exponential", "Constant % decline; steepest; most conservative"], [30, 45, 115])
    pdf.table_row(["0 < b < 1", "Hyperbolic", "Declining % decline rate; typical for most wells"], [30, 45, 115])
    pdf.table_row(["b = 1", "Harmonic", "Decline rate proportional to rate; slowest standard"], [30, 45, 115])
    pdf.table_row(["b > 1", "Super-harmonic", "Very slow decline; transient flow; use caution"], [30, 45, 115])

    pdf.ln(5)
    pdf.note_box(
        "Most conventional oil wells have b-factors between 0.3 and 0.8. "
        "Unconventional (tight oil/shale) wells often show b > 1 in early time, "
        "transitioning to lower values as boundary-dominated flow develops.",
        "Industry Guidance"
    )

    # ========== 4. FITTING PARAMETERS ==========
    pdf.add_page()
    pdf.chapter_title("4. Fitting Parameters")

    pdf.body_text(
        "PyForecast allows fine-grained control over decline curve fitting through "
        "per-product configuration. Understanding these parameters is essential for "
        "generating reliable forecasts."
    )

    pdf.section_title("4.1 B-Factor (Hyperbolic Exponent)")
    pdf.body_text(
        "The b-factor controls the curvature of the decline. PyForecast constrains "
        "b within configurable bounds to prevent unrealistic forecasts."
    )

    pdf.subsection_title("Configuration")
    pdf.code_block("""oil:
  b_min: 0.01    # Lower bound (near-exponential)
  b_max: 1.5     # Upper bound (super-harmonic)

gas:
  b_min: 0.01
  b_max: 2.0     # Gas often shows higher b in tight reservoirs""")

    pdf.subsection_title("Recommended Ranges by Play Type")
    pdf.table_header(["Play Type", "Typical b Range", "Recommended b_max"], [60, 50, 60])
    pdf.table_row(["Conventional oil", "0.3 - 0.8", "1.0"], [60, 50, 60])
    pdf.table_row(["Conventional gas", "0.4 - 0.9", "1.2"], [60, 50, 60])
    pdf.table_row(["Tight oil (Bakken, etc.)", "0.8 - 1.5", "1.5 - 2.0"], [60, 50, 60])
    pdf.table_row(["Shale gas (Marcellus, etc.)", "1.0 - 2.0", "2.0 - 2.5"], [60, 50, 60])
    pdf.table_row(["CBM / Coal seam gas", "0.5 - 1.0", "1.2"], [60, 50, 60])

    pdf.ln(3)
    pdf.warning_box(
        "High b-factors (>1.5) can lead to optimistic forecasts. Always validate "
        "fits visually and consider using terminal decline (Dmin) as a constraint."
    )

    pdf.ln(5)
    pdf.subsection_title("What Happens at the Bounds")
    pdf.bullet_point("b at b_min: Near-exponential decline. May indicate mature well or data issues.")
    pdf.bullet_point("b at b_max: May indicate transient flow or fitting to noise. Review carefully.")

    # Page break for next section
    pdf.add_page()
    pdf.section_title("4.2 Initial Decline Rate (Di)")
    pdf.body_text(
        "The initial decline rate represents how fast production is declining at the "
        "start of the forecast period. PyForecast reports Di as a monthly fraction "
        "but you can also think of it annually."
    )

    pdf.subsection_title("Unit Conversion")
    pdf.table_header(["Monthly Di", "Annual Di", "Interpretation"], [40, 40, 100])
    pdf.table_row(["0.01", "12%", "Very slow decline"], [40, 40, 100])
    pdf.table_row(["0.05", "60%", "Moderate decline"], [40, 40, 100])
    pdf.table_row(["0.08", "96%", "Steep decline (typical tight oil)"], [40, 40, 100])
    pdf.table_row(["0.10", "120%", "Very steep (early-time transient)"], [40, 40, 100])

    pdf.ln(3)
    pdf.body_text(
        "PyForecast automatically estimates Di from the data using log-linear regression "
        "as an initial guess, then refines it through optimization."
    )

    pdf.ln(5)
    pdf.section_title("4.3 Terminal Decline (Dmin)")
    pdf.body_text(
        "Terminal decline (Dmin) sets a floor on the decline rate. When the instantaneous "
        "decline drops to Dmin, the model switches to exponential decline at that rate. "
        "This prevents unrealistically long tails in high-b forecasts."
    )

    pdf.subsection_title("Configuration")
    pdf.code_block("""oil:
  dmin: 0.06    # 6% annual terminal decline

gas:
  dmin: 0.05    # 5% annual terminal decline""")

    pdf.subsection_title("Typical Values")
    pdf.table_header(["Fluid", "Typical Dmin", "Rationale"], [40, 40, 100])
    pdf.table_row(["Oil", "5-8%", "Based on field decline statistics"], [40, 40, 100])
    pdf.table_row(["Gas", "4-6%", "Gas wells often decline more slowly late-life"], [40, 40, 100])
    pdf.table_row(["Water", "5-10%", "Varies widely by drive mechanism"], [40, 40, 100])

    # ========== 5. REGIME DETECTION ==========
    pdf.add_page()
    pdf.chapter_title("5. Regime Detection")

    pdf.body_text(
        "Wells often experience production increases due to workovers, refracs, or "
        "return-to-production (RTP) events. PyForecast automatically detects these "
        "regime changes and fits only the most recent decline period."
    )

    pdf.section_title("How It Works")
    pdf.numbered_item(1, "Fits a trend line to recent production history")
    pdf.numbered_item(2, "Extrapolates the trend forward")
    pdf.numbered_item(3, "Detects when production exceeds the trend by a threshold")
    pdf.numbered_item(4, "Confirms the change is sustained (not a spike)")
    pdf.numbered_item(5, "Resets the fitting window to start from the new regime")

    pdf.ln(3)
    pdf.section_title("Configuration")
    pdf.code_block("""regime:
  threshold: 1.0         # 100% increase triggers detection
  window: 6              # Months of data for trend fitting
  sustained_months: 2    # Months elevation must persist""")

    pdf.subsection_title("Parameter Guidance")
    pdf.table_header(["Parameter", "Low Value", "High Value"], [55, 65, 65])
    pdf.table_row(["threshold", "More sensitive (0.5)", "Less sensitive (1.5)"], [55, 65, 65])
    pdf.table_row(["window", "Recent trend only (3)", "Longer history (12)"], [55, 65, 65])
    pdf.table_row(["sustained_months", "Quick trigger (1)", "Confirmed only (3)"], [55, 65, 65])

    pdf.ln(5)
    pdf.note_box(
        "When regime detection triggers, the fit results will show regime_start_idx > 0, "
        "indicating the month index where the new regime began. The forecast is based "
        "only on data from that point forward.",
        "Tip"
    )

    pdf.ln(5)
    pdf.section_title("Recency Weighting")
    pdf.body_text(
        "In addition to regime detection, PyForecast applies exponential decay weighting "
        "to favor recent production data. This helps the model track current trends."
    )

    pdf.code_block("""fitting:
  recency_half_life: 12.0  # Half-life in months""")

    pdf.body_text(
        "With a 12-month half-life, data from 12 months ago has half the weight of "
        "current data. Lower values weight recent data more aggressively."
    )

    # ========== 6. CONFIGURATION ==========
    pdf.add_page()
    pdf.chapter_title("6. Configuration")

    pdf.body_text(
        "PyForecast uses YAML configuration files for reproducible, version-controlled "
        "analysis settings. Generate a template with all options:"
    )
    pdf.code_block("pyforecast init -o pyforecast.yaml")

    pdf.section_title("Complete Configuration Reference")
    pdf.code_block("""# Per-product decline curve parameters
oil:
  b_min: 0.01      # Minimum b-factor
  b_max: 1.5       # Maximum b-factor
  dmin: 0.06       # Terminal decline (annual)

gas:
  b_min: 0.01
  b_max: 1.5
  dmin: 0.06

water:
  b_min: 0.01
  b_max: 1.5
  dmin: 0.06

# Regime change detection
regime:
  threshold: 1.0          # Min increase to trigger (1.0 = 100%)
  window: 6               # Trend fitting window (months)
  sustained_months: 2     # Confirmation period

# General fitting parameters
fitting:
  recency_half_life: 12.0  # Weighting half-life (months)
  min_points: 6            # Minimum data points required

# Output options
output:
  products: [oil, gas, water]  # Products to forecast
  plots: true                   # Generate well plots
  batch_plot: true              # Generate overlay plot
  format: ac_economic           # Export format

# Data validation thresholds
validation:
  max_oil_rate: 50000      # Max rate before warning
  max_gas_rate: 500000
  outlier_sigma: 3.0       # Outlier detection threshold
  min_r_squared: 0.5       # Minimum fit quality""")

    # ========== 7. DATA VALIDATION ==========
    pdf.add_page()
    pdf.chapter_title("7. Data Validation")

    pdf.body_text(
        "PyForecast includes comprehensive data validation to catch quality issues "
        "before they impact your forecasts. Validation runs automatically during "
        "batch processing."
    )

    pdf.section_title("Validation Categories")
    pdf.table_header(["Category", "Checks"], [50, 130])
    pdf.table_row(["DATA_FORMAT", "Negative values, extreme rates, date issues"], [50, 130])
    pdf.table_row(["DATA_QUALITY", "Gaps, outliers, shut-ins, low variability"], [50, 130])
    pdf.table_row(["FITTING_PREREQ", "Sufficient data, declining trend"], [50, 130])
    pdf.table_row(["FITTING_RESULT", "R-squared, b-factor bounds, decline rate"], [50, 130])

    pdf.ln(3)
    pdf.section_title("Key Error Codes")
    pdf.table_header(["Code", "Severity", "Description"], [25, 30, 125])
    pdf.table_row(["IV001", "ERROR", "Negative production values"], [25, 30, 125])
    pdf.table_row(["IV002", "WARNING", "Values exceed configured maximum"], [25, 30, 125])
    pdf.table_row(["DQ001", "WARNING", "Data gaps detected"], [25, 30, 125])
    pdf.table_row(["DQ002", "WARNING", "Outliers detected"], [25, 30, 125])
    pdf.table_row(["FP001", "ERROR", "Insufficient data points"], [25, 30, 125])
    pdf.table_row(["FP002", "WARNING", "Increasing trend (not declining)"], [25, 30, 125])
    pdf.table_row(["FR001", "WARNING", "Poor fit quality (low R-squared)"], [25, 30, 125])
    pdf.table_row(["FR004", "WARNING", "B-factor at upper bound"], [25, 30, 125])

    pdf.ln(3)
    pdf.body_text(
        "After processing, check validation_report.txt in the output directory for "
        "detailed issue descriptions and resolution guidance."
    )

    # ========== 8. OUTPUT FORMATS ==========
    pdf.add_page()
    pdf.chapter_title("8. Output Formats")

    pdf.section_title("ARIES AC Economic Format")
    pdf.body_text(
        "The default export format (ac_economic) is compatible with ARIES economic "
        "evaluation software. It includes monthly forecast volumes for import into "
        "AC_ECONOMIC sections."
    )
    pdf.code_block("pyforecast process data.csv -o output/ --format ac_economic")

    pdf.section_title("ARIES AC Forecast Format")
    pdf.body_text(
        "Alternative format for direct forecast import with decline parameters."
    )
    pdf.code_block("pyforecast process data.csv -o output/ --format ac_forecast")

    pdf.section_title("Output Directory Structure")
    pdf.code_block("""output/
  ac_economic.csv        # Forecast export file
  validation_report.txt  # Data quality issues
  errors.txt             # Processing errors (if any)
  plots/
    WELL001_oil.html     # Interactive well plots
    WELL001_gas.html
    batch_oil.html       # Multi-well overlay plot
    ...""")

    pdf.section_title("Interactive Plots")
    pdf.body_text(
        "PyForecast generates interactive HTML plots using Plotly. Features include:"
    )
    pdf.bullet_point("Semi-log scale (standard for decline analysis)")
    pdf.bullet_point("Historical production with fitted curve overlay")
    pdf.bullet_point("30-year forecast projection")
    pdf.bullet_point("Zoom, pan, and hover for data values")
    pdf.bullet_point("Regime change indicator (if detected)")

    # ========== 9. COMMAND REFERENCE ==========
    pdf.add_page()
    pdf.chapter_title("9. Command Reference")

    pdf.section_title("pyforecast process")
    pdf.body_text("Process production data and generate forecasts.")
    pdf.code_block("""pyforecast process <files> [options]

Arguments:
  files              Input CSV/Excel file(s)

Options:
  -o, --output       Output directory (default: output)
  -c, --config       YAML configuration file
  -p, --product      Products to forecast (oil, gas, water)
  -w, --workers      Parallel workers (default: auto)
  --no-plots         Skip individual well plots
  --no-batch-plot    Skip multi-well overlay plot
  --format           Export format (ac_economic, ac_forecast)""")

    pdf.section_title("pyforecast plot")
    pdf.body_text("Analyze and plot a single well.")
    pdf.code_block("""pyforecast plot <file> [options]

Options:
  --well-id          Well identifier to plot
  -p, --product      Product to plot (default: oil)
  -o, --output       Save plot to file (default: show in browser)
  -c, --config       YAML configuration file""")

    pdf.section_title("pyforecast init")
    pdf.body_text("Generate a default configuration file.")
    pdf.code_block("""pyforecast init [options]

Options:
  -o, --output       Output file path (default: pyforecast.yaml)""")

    pdf.section_title("pyforecast info")
    pdf.body_text("Display information about a production data file.")
    pdf.code_block("""pyforecast info <file>

Shows:
  - Detected format (Enverus, ARIES)
  - Number of wells
  - Date range
  - Available columns""")

    # ========== 10. PROGRAMMATIC USAGE ==========
    pdf.add_page()
    pdf.chapter_title("10. Programmatic Usage")

    pdf.section_title("Loading Data")
    pdf.code_block("""from pyforecast.data.base import load_wells

wells = load_wells("production.csv")
for well in wells:
    print(f"{well.well_id}: {well.production.n_months} months")""")

    pdf.section_title("Fitting a Single Well")
    pdf.code_block("""from pyforecast.core.fitting import DeclineFitter, FittingConfig

# Configure fitting parameters
config = FittingConfig(
    b_min=0.01,
    b_max=1.5,
    dmin_annual=0.06,
    recency_half_life=12.0,
)

# Fit decline curve
fitter = DeclineFitter(config)
result = fitter.fit(
    t=well.production.time_months,
    q=well.production.oil
)

# Access results
print(f"qi: {result.model.qi:.1f} bbl/mo")
print(f"Di: {result.model.di * 12:.1%}/year")
print(f"b:  {result.model.b:.3f}")
print(f"R2: {result.r_squared:.3f}")""")

    pdf.section_title("Using Configuration Files")
    pdf.code_block("""from pyforecast.config import PyForecastConfig
from pyforecast.core.fitting import FittingConfig

# Load configuration
config = PyForecastConfig.from_yaml("config.yaml")

# Get per-product fitting config
oil_config = FittingConfig.from_pyforecast_config(config, "oil")
gas_config = FittingConfig.from_pyforecast_config(config, "gas")""")

    pdf.section_title("Batch Processing")
    pdf.code_block("""from pyforecast.batch.processor import BatchProcessor, BatchConfig
from pyforecast.config import PyForecastConfig

config = PyForecastConfig.from_yaml("config.yaml")

batch_config = BatchConfig(
    products=["oil", "gas"],
    pyforecast_config=config,
    output_dir="output",
)

processor = BatchProcessor(batch_config)
result = processor.run(["data1.csv", "data2.csv"])

print(f"Successful: {result.successful}")
print(f"Failed: {result.failed}")""")

    # ========== APPENDIX A ==========
    pdf.add_page()
    pdf.chapter_title("Appendix A: Parameter Quick Reference")

    pdf.section_title("Fitting Parameters")
    pdf.table_header(["Parameter", "Default", "Description"], [55, 30, 95])
    pdf.table_row(["b_min", "0.01", "Minimum b-factor (near-exponential)"], [55, 30, 95])
    pdf.table_row(["b_max", "1.5", "Maximum b-factor (super-harmonic)"], [55, 30, 95])
    pdf.table_row(["dmin", "0.06", "Terminal decline rate (annual)"], [55, 30, 95])
    pdf.table_row(["recency_half_life", "12.0", "Weighting half-life (months)"], [55, 30, 95])
    pdf.table_row(["min_points", "6", "Minimum data points required"], [55, 30, 95])

    pdf.ln(5)
    pdf.section_title("Regime Detection Parameters")
    pdf.table_header(["Parameter", "Default", "Description"], [55, 30, 95])
    pdf.table_row(["threshold", "1.0", "Min increase to trigger (fraction)"], [55, 30, 95])
    pdf.table_row(["window", "6", "Trend fitting window (months)"], [55, 30, 95])
    pdf.table_row(["sustained_months", "2", "Confirmation period (months)"], [55, 30, 95])

    pdf.ln(5)
    pdf.section_title("Validation Parameters")
    pdf.table_header(["Parameter", "Default", "Description"], [55, 30, 95])
    pdf.table_row(["max_oil_rate", "50,000", "Max oil rate warning (bbl/mo)"], [55, 30, 95])
    pdf.table_row(["max_gas_rate", "500,000", "Max gas rate warning (mcf/mo)"], [55, 30, 95])
    pdf.table_row(["outlier_sigma", "3.0", "Outlier detection threshold"], [55, 30, 95])
    pdf.table_row(["min_r_squared", "0.5", "Minimum fit quality"], [55, 30, 95])
    pdf.table_row(["gap_threshold", "2", "Gap detection (months)"], [55, 30, 95])

    pdf.ln(5)
    pdf.section_title("Typical B-Factor Ranges")
    pdf.table_header(["Formation Type", "Oil b Range", "Gas b Range"], [70, 55, 55])
    pdf.table_row(["Conventional", "0.3 - 0.8", "0.4 - 0.9"], [70, 55, 55])
    pdf.table_row(["Tight / Unconventional", "0.8 - 1.5", "1.0 - 2.0"], [70, 55, 55])
    pdf.table_row(["Shale", "1.0 - 2.0", "1.2 - 2.5"], [70, 55, 55])

    # Save PDF
    pdf.output("PyForecast_User_Guide.pdf")
    print("PDF generated: PyForecast_User_Guide.pdf")


if __name__ == "__main__":
    create_pdf()
