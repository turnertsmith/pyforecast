"""Tests for CLI commands."""

import json
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pyforecast.cli.commands import app
from pyforecast.config import PyForecastConfig


FIXTURES_DIR = Path(__file__).parent / "fixtures"
runner = CliRunner()


class TestProcessCommand:
    """Tests for the process command."""

    def test_process_basic(self):
        """Test basic process command with default options."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-o", tmpdir,
                    "--no-plots",
                    "--no-batch-plot",
                ],
            )

            assert result.exit_code == 0
            assert "Processing" in result.stdout
            assert "Successful" in result.stdout

            # Check output files exist
            output_dir = Path(tmpdir)
            assert (output_dir / "ac_economic.csv").exists()

    def test_process_with_product_filter(self):
        """Test process with specific product."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-o", tmpdir,
                    "-p", "oil",
                    "--no-plots",
                    "--no-batch-plot",
                ],
            )

            assert result.exit_code == 0
            assert "oil" in result.stdout.lower()

    def test_process_invalid_product(self):
        """Test process with invalid product name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-o", tmpdir,
                    "-p", "invalid_product",
                ],
            )

            assert result.exit_code == 1
            assert "Invalid product" in result.output

    def test_process_with_config(self):
        """Test process with config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a config file
            config_path = Path(tmpdir) / "config.yaml"
            config = PyForecastConfig()
            config.to_yaml(config_path)

            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-o", tmpdir,
                    "-c", str(config_path),
                    "--no-plots",
                    "--no-batch-plot",
                ],
            )

            assert result.exit_code == 0
            assert "Loading config" in result.stdout

    def test_process_json_format(self):
        """Test process with JSON export format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-o", tmpdir,
                    "--format", "json",
                    "--no-plots",
                    "--no-batch-plot",
                ],
            )

            assert result.exit_code == 0

            # Check JSON file exists and is valid
            json_path = Path(tmpdir) / "forecasts.json"
            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)

            assert "generated" in data
            assert "wells" in data
            assert "config" in data

    def test_process_ac_forecast_format(self):
        """Test process with ac_forecast export format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-o", tmpdir,
                    "--format", "ac_forecast",
                    "--no-plots",
                    "--no-batch-plot",
                ],
            )

            assert result.exit_code == 0
            assert (Path(tmpdir) / "forecasts.csv").exists()

    def test_process_invalid_format(self):
        """Test process with invalid format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-o", tmpdir,
                    "--format", "invalid_format",
                ],
            )

            assert result.exit_code == 1
            assert "Invalid format" in result.output

    def test_process_multiple_files(self):
        """Test process with multiple input files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    str(FIXTURES_DIR / "sample_aries.csv"),
                    "-o", tmpdir,
                    "--no-plots",
                    "--no-batch-plot",
                ],
            )

            assert result.exit_code == 0
            # Should process wells from both files
            assert "Processing 2 file(s)" in result.stdout


class TestValidateCommand:
    """Tests for the validate command."""

    def test_validate_basic(self):
        """Test basic validate command."""
        result = runner.invoke(
            app,
            [
                "validate",
                str(FIXTURES_DIR / "sample_enverus.csv"),
            ],
        )

        # Should complete without crashing
        assert result.exit_code in (0, 1)  # 0 if valid, 1 if errors
        assert "Validation Summary" in result.stdout
        assert "Total wells" in result.stdout

    def test_validate_with_output(self):
        """Test validate with output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.txt"

            result = runner.invoke(
                app,
                [
                    "validate",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-o", str(output_path),
                ],
            )

            assert "Detailed report written to" in result.stdout
            assert output_path.exists()

            # Check report content
            content = output_path.read_text()
            assert "PyForecast Validation Report" in content

    def test_validate_with_config(self):
        """Test validate with config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config
            config_path = Path(tmpdir) / "config.yaml"
            config = PyForecastConfig()
            config.to_yaml(config_path)

            result = runner.invoke(
                app,
                [
                    "validate",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-c", str(config_path),
                ],
            )

            assert "Loading config" in result.stdout

    def test_validate_multiple_files(self):
        """Test validate with multiple input files."""
        result = runner.invoke(
            app,
            [
                "validate",
                str(FIXTURES_DIR / "sample_enverus.csv"),
                str(FIXTURES_DIR / "sample_aries.csv"),
            ],
        )

        assert result.exit_code in (0, 1)
        assert "Validation Summary" in result.stdout

    def test_validate_nonexistent_file(self):
        """Test validate with nonexistent file."""
        result = runner.invoke(
            app,
            [
                "validate",
                "/nonexistent/path/file.csv",
            ],
        )

        assert result.exit_code != 0


class TestPlotCommand:
    """Tests for the plot command."""

    def test_plot_with_output(self):
        """Test plot command with output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "plot.html"

            result = runner.invoke(
                app,
                [
                    "plot",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-p", "oil",
                    "-o", str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()
            assert "Plot saved to" in result.stdout

    def test_plot_with_well_id(self):
        """Test plot with specific well ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "plot.html"

            result = runner.invoke(
                app,
                [
                    "plot",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "--well-id", "WELL001",
                    "-p", "oil",
                    "-o", str(output_path),
                ],
            )

            assert result.exit_code == 0

    def test_plot_invalid_product(self):
        """Test plot with invalid product."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "plot",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-p", "invalid",
                    "-o", str(tmpdir) + "/plot.html",
                ],
            )

            assert result.exit_code == 1
            assert "Invalid product" in result.output

    def test_plot_nonexistent_well(self):
        """Test plot with well ID that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                app,
                [
                    "plot",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "--well-id", "NONEXISTENT_WELL_12345",
                    "-o", str(tmpdir) + "/plot.html",
                ],
            )

            assert result.exit_code == 1
            assert "not found" in result.output or "Available wells" in result.output


class TestInitCommand:
    """Tests for the init command."""

    def test_init_default(self):
        """Test init with default output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory to avoid creating files in repo
            import os
            original_cwd = os.getcwd()
            os.chdir(tmpdir)

            try:
                result = runner.invoke(app, ["init"])

                assert result.exit_code == 0
                assert Path("pyforecast.yaml").exists()
                assert "Config file created" in result.stdout
            finally:
                os.chdir(original_cwd)

    def test_init_custom_path(self):
        """Test init with custom output path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_config.yaml"

            result = runner.invoke(
                app,
                ["init", "-o", str(output_path)],
            )

            assert result.exit_code == 0
            assert output_path.exists()

            # Verify it's valid YAML that can be loaded
            config = PyForecastConfig.from_yaml(output_path)
            assert config is not None

    def test_init_overwrite_confirmation(self):
        """Test init prompts before overwriting existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "config.yaml"

            # Create file first
            output_path.write_text("existing content")

            # Try to overwrite - answer no
            result = runner.invoke(
                app,
                ["init", "-o", str(output_path)],
                input="n\n",
            )

            assert result.exit_code == 0
            # File should still have original content
            assert output_path.read_text() == "existing content"


class TestInfoCommand:
    """Tests for the info command."""

    def test_info_enverus(self):
        """Test info command with Enverus format."""
        result = runner.invoke(
            app,
            ["info", str(FIXTURES_DIR / "sample_enverus.csv")],
        )

        assert result.exit_code == 0
        assert "Inspecting" in result.stdout
        assert "Rows:" in result.stdout
        assert "Columns:" in result.stdout
        assert "Wells found:" in result.stdout

    def test_info_aries(self):
        """Test info command with ARIES format."""
        result = runner.invoke(
            app,
            ["info", str(FIXTURES_DIR / "sample_aries.csv")],
        )

        assert result.exit_code == 0
        assert "Detected format:" in result.stdout

    def test_info_nonexistent_file(self):
        """Test info with nonexistent file."""
        result = runner.invoke(
            app,
            ["info", "/nonexistent/file.csv"],
        )

        assert result.exit_code != 0


class TestConfigValidation:
    """Tests for config validation via CLI."""

    def test_invalid_config_b_min_greater_than_b_max(self):
        """Test that invalid config (b_min > b_max) causes error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad_config.yaml"

            # Write invalid config
            config_path.write_text("""
oil:
  b_min: 2.0
  b_max: 1.0
  dmin: 0.06
""")

            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-c", str(config_path),
                    "-o", tmpdir,
                ],
            )

            assert result.exit_code != 0
            # Should fail due to validation error

    def test_invalid_config_negative_dmin(self):
        """Test that invalid config (dmin <= 0) causes error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad_config.yaml"

            config_path.write_text("""
oil:
  b_min: 0.01
  b_max: 1.5
  dmin: -0.06
""")

            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-c", str(config_path),
                    "-o", tmpdir,
                ],
            )

            assert result.exit_code != 0

    def test_invalid_config_min_points_too_low(self):
        """Test that invalid config (min_points < 3) causes error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "bad_config.yaml"

            config_path.write_text("""
fitting:
  min_points: 2
  recency_half_life: 12.0
""")

            result = runner.invoke(
                app,
                [
                    "process",
                    str(FIXTURES_DIR / "sample_enverus.csv"),
                    "-c", str(config_path),
                    "-o", tmpdir,
                ],
            )

            assert result.exit_code != 0
