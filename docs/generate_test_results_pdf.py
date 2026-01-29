#!/usr/bin/env python3
"""Generate PDF showing ground truth comparison test results."""

import subprocess
import sys
from datetime import datetime

try:
    from fpdf import FPDF
except ImportError:
    print("Installing fpdf2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2", "-q"])
    from fpdf import FPDF


class TestResultsPDF(FPDF):
    """Custom PDF class for test results."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, "PyForecast Ground Truth Comparison - Test Results", align="C")
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

    def code_block(self, code: str):
        self.set_font("Courier", "", 8)
        self.set_fill_color(30, 41, 59)
        self.set_text_color(226, 232, 240)
        for line in code.split('\n'):
            self.cell(0, 4, line, ln=True, fill=True)
        self.ln(2)

    def test_result(self, name: str, passed: bool):
        self.set_font("Helvetica", "", 9)
        if passed:
            self.set_text_color(16, 185, 129)
            status = "PASSED"
            symbol = "+"
        else:
            self.set_text_color(220, 53, 69)
            status = "FAILED"
            symbol = "x"
        self.cell(10, 5, symbol)
        self.set_text_color(30, 41, 59)
        self.cell(150, 5, name)
        if passed:
            self.set_text_color(16, 185, 129)
        else:
            self.set_text_color(220, 53, 69)
        self.cell(0, 5, status, ln=True)

    def summary_box(self, total: int, passed: int, failed: int, duration: str):
        self.set_fill_color(248, 250, 252)
        self.set_draw_color(200, 200, 200)
        self.rect(10, self.get_y(), 190, 30, "DF")

        y_start = self.get_y() + 5
        self.set_xy(15, y_start)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 41, 59)
        self.cell(40, 6, f"Total: {total}")

        self.set_text_color(16, 185, 129)
        self.cell(40, 6, f"Passed: {passed}")

        self.set_text_color(220, 53, 69)
        self.cell(40, 6, f"Failed: {failed}")

        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"Duration: {duration}")

        self.set_y(self.get_y() + 35)


def run_tests_and_capture():
    """Run pytest and capture output."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest",
         "tests/refinement/test_ground_truth.py",
         "tests/import_/test_aries_forecast.py",
         "-v", "--tb=no"],
        capture_output=True,
        text=True,
        cwd="/Users/turnersmith/pyforecast"
    )
    return result.stdout + result.stderr


def parse_test_output(output: str):
    """Parse pytest output into structured data."""
    tests = []
    for line in output.split('\n'):
        if '::' in line and ('PASSED' in line or 'FAILED' in line):
            # Extract test name and status
            parts = line.split('::')
            if len(parts) >= 2:
                test_class_method = parts[-1].split()[0]
                passed = 'PASSED' in line
                tests.append((test_class_method, passed))

    # Extract summary
    total = passed_count = failed_count = 0
    duration = "N/A"
    for line in output.split('\n'):
        if 'passed' in line and '==' in line:
            # Parse "64 passed in 1.23s"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'passed':
                    total = passed_count = int(parts[i-1])
                elif part == 'failed':
                    failed_count = int(parts[i-1])
            # Find duration
            for part in parts:
                if part.endswith('s') and part[:-1].replace('.', '').isdigit():
                    duration = part

    return tests, total, passed_count, failed_count, duration


def create_pdf():
    """Create the test results PDF."""
    print("Running tests...")
    output = run_tests_and_capture()
    tests, total, passed, failed, duration = parse_test_output(output)

    pdf = TestResultsPDF()

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(37, 99, 235)
    pdf.ln(50)
    pdf.cell(0, 15, "PyForecast", align="C", ln=True)

    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 10, "Ground Truth Comparison", align="C", ln=True)
    pdf.cell(0, 10, "Test Results Report", align="C", ln=True)

    pdf.ln(30)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 8, "Pre-Production Improvements Verification", align="C", ln=True)

    pdf.ln(20)

    # Summary box on cover
    pdf.summary_box(total, passed, failed, duration)

    pdf.ln(20)
    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", align="C")

    # Test Results - Ground Truth
    pdf.add_page()
    pdf.chapter_title("Ground Truth Comparison Tests")

    pdf.section_title("GroundTruthResult Tests")
    gt_tests = [t for t in tests if 'GroundTruth' in t[0] and 'Validator' not in t[0] and 'Summarize' not in t[0] and 'Batch' not in t[0] and 'Rate' not in t[0] and 'Summary' not in t[0]]
    for name, passed in gt_tests[:12]:
        pdf.test_result(name, passed)

    pdf.ln(5)
    pdf.section_title("GroundTruthValidator Tests")
    validator_tests = [t for t in tests if 'Validator' in t[0] and 'Batch' not in t[0]]
    for name, passed in validator_tests:
        pdf.test_result(name, passed)

    pdf.ln(5)
    pdf.section_title("Batch Validation Tests")
    batch_tests = [t for t in tests if 'Batch' in t[0] or 'batch' in t[0]]
    for name, passed in batch_tests:
        pdf.test_result(name, passed)

    # New features tests
    pdf.add_page()
    pdf.chapter_title("New Feature Tests")

    pdf.section_title("MAPE Edge Case Handling")
    pdf.body_text("Tests for handling None MAPE when insufficient data points exist:")
    mape_tests = [t for t in tests if 'Mape' in t[0] or 'mape' in t[0]]
    for name, passed in mape_tests:
        pdf.test_result(name, passed)

    pdf.ln(5)
    pdf.section_title("Rate Validation")
    pdf.body_text("Tests for NaN, infinite, and negative value handling:")
    rate_tests = [t for t in tests if 'Rate' in t[0] or 'rate' in t[0]]
    for name, passed in rate_tests:
        pdf.test_result(name, passed)

    pdf.ln(5)
    pdf.section_title("Summary Functions")
    summary_tests = [t for t in tests if 'Summarize' in t[0] or 'Summary' in t[0]]
    for name, passed in summary_tests:
        pdf.test_result(name, passed)

    # ARIES Importer Tests
    pdf.add_page()
    pdf.chapter_title("ARIES Importer Tests")

    pdf.section_title("Well ID Normalization")
    normalize_tests = [t for t in tests if 'Normalize' in t[0] or 'normalize' in t[0] or 'api' in t[0].lower()]
    for name, passed in normalize_tests:
        pdf.test_result(name, passed)

    pdf.ln(5)
    pdf.section_title("Expression Parsing")
    parse_tests = [t for t in tests if 'parse' in t[0].lower() and 'Failure' not in t[0]]
    for name, passed in parse_tests[:10]:
        pdf.test_result(name, passed)

    pdf.ln(5)
    pdf.section_title("Parse Failure Logging")
    pdf.body_text("Tests for tracking and logging unparseable expressions:")
    failure_tests = [t for t in tests if 'Failure' in t[0] or 'failure' in t[0]]
    for name, passed in failure_tests:
        pdf.test_result(name, passed)

    pdf.ln(5)
    pdf.section_title("Lazy Loading")
    pdf.body_text("Tests for memory-efficient streaming mode:")
    lazy_tests = [t for t in tests if 'Lazy' in t[0] or 'lazy' in t[0] or 'stream' in t[0].lower()]
    for name, passed in lazy_tests:
        pdf.test_result(name, passed)

    # Implementation Summary
    pdf.add_page()
    pdf.chapter_title("Implementation Summary")

    pdf.section_title("Features Implemented")

    features = [
        ("Rate Validation", "Checks for NaN, infinite, and negative values in forecast arrays"),
        ("MAPE Edge Cases", "Returns None when insufficient data, adds mape_valid property"),
        ("ID Mismatch Logging", "Tracks wells in pyforecast-only and ARIES-only"),
        ("Parse Failure Logging", "Logs warning with expression text for unparseable expressions"),
        ("Time-Series Export", "Exports forecast arrays to ground_truth_timeseries.csv"),
        ("Comparison Plots", "Generates overlay plots with --gt-plots flag"),
        ("Lazy Loading", "Streams file with --gt-lazy for large datasets"),
        ("Parallel Validation", "Uses ThreadPoolExecutor with --gt-workers N"),
    ]

    for feature, description in features:
        pdf.subsection_title(feature)
        pdf.body_text(description)
        pdf.ln(1)

    pdf.ln(5)
    pdf.section_title("New CLI Flags")
    pdf.code_block("""--gt-plots      Generate comparison plots for each well
--gt-lazy       Stream ARIES file instead of loading into memory
--gt-workers N  Number of parallel workers for validation""")

    pdf.ln(5)
    pdf.section_title("Files Modified")
    pdf.code_block("""src/pyforecast/
  refinement/
    ground_truth.py    # Rate validation, MAPE handling, batch validation
    schemas.py         # mape_valid property, X grade
    plotting.py        # NEW - comparison plots
    __init__.py        # Export GroundTruthSummary
  import_/
    aries_forecast.py  # Parse failure logging, lazy loading
  cli/commands.py      # New flags, time-series export
  config.py            # gt_lazy, gt_workers options

tests/
  refinement/test_ground_truth.py  # New test cases
  import_/test_aries_forecast.py   # New test cases""")

    # Save
    output_path = "/Users/turnersmith/pyforecast/docs/Ground_Truth_Test_Results.pdf"
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")
    return output_path


if __name__ == "__main__":
    create_pdf()
