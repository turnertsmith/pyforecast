"""Tests for ARIES AC_ECONOMIC forecast parser."""

import csv
import tempfile
from pathlib import Path

import pytest
import numpy as np

from pyforecast.import_.aries_forecast import (
    AriesForecastParams,
    AriesForecastImporter,
    DAYS_PER_MONTH,
)


class TestAriesForecastParams:
    """Tests for AriesForecastParams dataclass."""

    def test_basic_params(self):
        """Test basic parameter storage."""
        params = AriesForecastParams(
            propnum="42-001-00001",
            product="oil",
            qi=100.0,  # daily
            di=0.01,   # monthly
            b=0.5,
            dmin=0.005,
            decline_type="HYP",
        )

        assert params.propnum == "42-001-00001"
        assert params.product == "oil"
        assert params.qi == 100.0
        assert params.di == 0.01
        assert params.b == 0.5
        assert params.decline_type == "HYP"

    def test_monthly_conversion(self):
        """Test qi_monthly property."""
        params = AriesForecastParams(
            propnum="test",
            product="oil",
            qi=100.0,  # 100 bbl/day
            di=0.01,
            b=0.5,
            dmin=0.005,
            decline_type="HYP",
        )

        expected_monthly = 100.0 * DAYS_PER_MONTH
        assert abs(params.qi_monthly - expected_monthly) < 0.1

    def test_annual_decline_conversion(self):
        """Test di_annual and dmin_annual properties."""
        params = AriesForecastParams(
            propnum="test",
            product="oil",
            qi=100.0,
            di=0.01,  # 1%/month
            b=0.5,
            dmin=0.005,  # 0.5%/month
            decline_type="HYP",
        )

        assert abs(params.di_annual - 0.12) < 0.001  # 12%/year
        assert abs(params.dmin_annual - 0.06) < 0.001  # 6%/year


class TestAriesForecastImporter:
    """Tests for AriesForecastImporter class."""

    def test_parse_standard_expression(self):
        """Test parsing standard ARIES expression format."""
        importer = AriesForecastImporter()

        # "1000 X B/M 6 EXP B/0.50 8.5"
        # qi=1000 bbl/month, dmin=6%, type=EXP, b=0.50, di=8.5%
        params = importer._parse_expression(
            "test-well",
            "1000 X B/M 6 EXP B/0.50 8.5"
        )

        assert params is not None
        assert params.propnum == "test-well"
        assert params.product == "oil"
        assert abs(params.qi - 1000 / DAYS_PER_MONTH) < 0.1  # converted to daily
        assert abs(params.di - 0.085 / 12) < 0.001  # 8.5% annual to monthly
        assert params.b == 0.50
        assert abs(params.dmin - 0.06 / 12) < 0.001  # 6% annual to monthly
        assert params.decline_type == "EXP"

    def test_parse_hyperbolic_expression(self):
        """Test parsing hyperbolic decline expression."""
        importer = AriesForecastImporter()

        params = importer._parse_expression(
            "test-well",
            "500 X M/M 6 HYP B/0.75 12"
        )

        assert params is not None
        assert params.product == "gas"
        assert params.b == 0.75
        assert params.decline_type == "HYP"

    def test_parse_harmonic_expression(self):
        """Test parsing harmonic decline expression."""
        importer = AriesForecastImporter()

        params = importer._parse_expression(
            "test-well",
            "200 X B/D 6 HRM B/1.00 10"
        )

        assert params is not None
        assert params.product == "oil"
        assert params.qi == 200  # already daily
        assert params.b == 1.00
        assert params.decline_type == "HRM"

    def test_parse_expression_without_type(self):
        """Test parsing expression without explicit decline type."""
        importer = AriesForecastImporter()

        # Should infer HYP from b=0.5
        params = importer._parse_expression(
            "test-well",
            "1000 X B/M 6 B/0.50 8.5"
        )

        assert params is not None
        assert params.decline_type == "HYP"

        # Should infer EXP from b=0.05
        params = importer._parse_expression(
            "test-well",
            "1000 X B/M 6 B/0.05 8.5"
        )
        assert params.decline_type == "EXP"

        # Should infer HRM from b=1.0
        params = importer._parse_expression(
            "test-well",
            "1000 X B/M 6 B/1.00 8.5"
        )
        assert params.decline_type == "HRM"

    def test_parse_invalid_expression(self):
        """Test that invalid expressions return None."""
        importer = AriesForecastImporter()

        assert importer._parse_expression("test", "invalid") is None
        assert importer._parse_expression("test", "1000 X") is None
        assert importer._parse_expression("test", "") is None

    def test_load_csv_file(self):
        """Test loading forecasts from CSV file."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline=''
        ) as f:
            writer = csv.writer(f)
            writer.writerow(['PROPNUM', 'OIL_EXPRESSION', 'GAS_EXPRESSION'])
            writer.writerow([
                '42-001-00001',
                '1000 X B/M 6 HYP B/0.50 8.5',
                '5000 X M/M 6 EXP B/0.10 12'
            ])
            writer.writerow([
                '42-001-00002',
                '500 X B/D 6 HYP B/0.75 10',
                ''
            ])
            filepath = Path(f.name)

        try:
            importer = AriesForecastImporter()
            count = importer.load(filepath)

            assert count == 3  # 2 oil + 1 gas (one empty)

            # Check first well oil
            params = importer.get('42-001-00001', 'oil')
            assert params is not None
            assert params.b == 0.50

            # Check first well gas
            params = importer.get('42-001-00001', 'gas')
            assert params is not None
            assert params.b == 0.10

            # Check second well oil (daily rate)
            params = importer.get('42-001-00002', 'oil')
            assert params is not None
            assert params.qi == 500  # daily, not converted

            # Check non-existent
            assert importer.get('42-001-00003', 'oil') is None
            assert importer.get('42-001-00001', 'water') is None
        finally:
            filepath.unlink()

    def test_load_nonexistent_file(self):
        """Test that loading non-existent file raises error."""
        importer = AriesForecastImporter()

        with pytest.raises(FileNotFoundError):
            importer.load(Path("/nonexistent/file.csv"))

    def test_to_model(self):
        """Test conversion to HyperbolicModel."""
        importer = AriesForecastImporter()

        params = AriesForecastParams(
            propnum="test",
            product="oil",
            qi=100.0,
            di=0.01,
            b=0.5,
            dmin=0.005,
            decline_type="HYP",
        )

        model = importer.to_model(params)

        assert model.qi == 100.0
        assert model.di == 0.01
        assert model.b == 0.5
        assert model.dmin == 0.005

        # Verify model can generate rates
        rates = model.rate(np.array([0, 1, 2]))
        assert len(rates) == 3
        assert rates[0] == 100.0  # qi at t=0

    def test_list_wells(self):
        """Test listing loaded wells."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline=''
        ) as f:
            writer = csv.writer(f)
            writer.writerow(['PROPNUM', 'EXPRESSION'])
            writer.writerow(['well-a', '1000 X B/M 6 HYP B/0.50 8.5'])
            writer.writerow(['well-b', '500 X B/M 6 HYP B/0.75 10'])
            filepath = Path(f.name)

        try:
            importer = AriesForecastImporter()
            importer.load(filepath)

            wells = importer.list_wells()
            assert 'well-a' in wells
            assert 'well-b' in wells
        finally:
            filepath.unlink()

    def test_list_products(self):
        """Test listing products for a well."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline=''
        ) as f:
            writer = csv.writer(f)
            writer.writerow(['PROPNUM', 'OIL_EXPR', 'GAS_EXPR'])
            writer.writerow([
                'well-a',
                '1000 X B/M 6 HYP B/0.50 8.5',
                '5000 X M/M 6 EXP B/0.10 12'
            ])
            filepath = Path(f.name)

        try:
            importer = AriesForecastImporter()
            importer.load(filepath)

            products = importer.list_products('well-a')
            assert 'oil' in products
            assert 'gas' in products
        finally:
            filepath.unlink()

    def test_len_and_contains(self):
        """Test __len__ and __contains__ methods."""
        importer = AriesForecastImporter()

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False, newline=''
        ) as f:
            writer = csv.writer(f)
            writer.writerow(['PROPNUM', 'EXPRESSION'])
            writer.writerow(['well-a', '1000 X B/M 6 HYP B/0.50 8.5'])
            filepath = Path(f.name)

        try:
            importer.load(filepath)

            assert len(importer) == 1
            assert ('well-a', 'oil') in importer
            assert ('well-b', 'oil') not in importer
        finally:
            filepath.unlink()

    def test_unit_variations(self):
        """Test various ARIES unit specifications."""
        importer = AriesForecastImporter()

        # Test all unit types
        test_cases = [
            ("1000 X B/M 6 HYP B/0.50 8.5", "oil", False),  # barrels/month
            ("100 X B/D 6 HYP B/0.50 8.5", "oil", True),   # barrels/day
            ("5000 X M/M 6 HYP B/0.50 8.5", "gas", False), # mcf/month
            ("500 X M/D 6 HYP B/0.50 8.5", "gas", True),   # mcf/day
            ("5000 X G/M 6 HYP B/0.50 8.5", "gas", False), # gas alternate
        ]

        for expr, expected_product, is_daily in test_cases:
            params = importer._parse_expression("test", expr)
            assert params is not None, f"Failed to parse: {expr}"
            assert params.product == expected_product, f"Wrong product for: {expr}"

    def test_case_insensitivity(self):
        """Test that parser handles case variations."""
        importer = AriesForecastImporter()

        # Lowercase
        params = importer._parse_expression(
            "test",
            "1000 x b/m 6 hyp b/0.50 8.5"
        )
        assert params is not None
        assert params.decline_type == "HYP"

        # Mixed case
        params = importer._parse_expression(
            "test",
            "1000 X b/M 6 Exp B/0.50 8.5"
        )
        assert params is not None
        assert params.decline_type == "EXP"
