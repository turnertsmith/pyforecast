"""Tests for config.py validation and from_dict."""

import pytest
import logging
from pathlib import Path

from pyforecast.config import (
    PyForecastConfig,
    ProductConfig,
    RegimeConfig,
    FittingDefaults,
    OutputConfig,
    ValidationConfig,
    RefinementConfig,
    generate_default_config,
)


class TestProductConfig:
    """Tests for ProductConfig defaults."""

    def test_defaults(self):
        pc = ProductConfig()
        assert pc.b_min == 0.01
        assert pc.b_max == 1.5
        assert pc.dmin == 0.06
        assert pc.recency_half_life is None


class TestPyForecastConfigValidation:
    """Tests for PyForecastConfig.validate()."""

    def test_valid_default_config(self):
        config = PyForecastConfig()
        config.validate()  # Should not raise

    def test_invalid_b_range(self):
        config = PyForecastConfig()
        config.oil.b_min = 2.0
        config.oil.b_max = 1.0
        with pytest.raises(ValueError, match="b_min.*must be less than.*b_max"):
            config.validate()

    def test_invalid_dmin_zero(self):
        config = PyForecastConfig()
        config.gas.dmin = 0.0
        with pytest.raises(ValueError, match="dmin.*must be greater than 0"):
            config.validate()

    def test_invalid_dmin_negative(self):
        config = PyForecastConfig()
        config.water.dmin = -0.01
        with pytest.raises(ValueError, match="dmin.*must be greater than 0"):
            config.validate()

    def test_invalid_min_points(self):
        config = PyForecastConfig()
        config.fitting.min_points = 2
        with pytest.raises(ValueError, match="min_points.*must be at least 3"):
            config.validate()

    def test_invalid_recency_half_life(self):
        config = PyForecastConfig()
        config.fitting.recency_half_life = 0.0
        with pytest.raises(ValueError, match="recency_half_life.*must be greater than 0"):
            config.validate()

    def test_invalid_regime_threshold(self):
        config = PyForecastConfig()
        config.regime.threshold = 0.0
        with pytest.raises(ValueError, match="threshold.*must be greater than 0"):
            config.validate()

    def test_invalid_regime_window(self):
        config = PyForecastConfig()
        config.regime.window = 1
        with pytest.raises(ValueError, match="window.*must be at least 2"):
            config.validate()

    def test_invalid_regime_sustained_months(self):
        config = PyForecastConfig()
        config.regime.sustained_months = 0
        with pytest.raises(ValueError, match="sustained_months.*must be at least 1"):
            config.validate()

    def test_multiple_errors_reported(self):
        config = PyForecastConfig()
        config.oil.b_min = 2.0
        config.oil.b_max = 1.0
        config.fitting.min_points = 1
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        msg = str(exc_info.value)
        assert "b_min" in msg
        assert "min_points" in msg


class TestPyForecastConfigFromDict:
    """Tests for PyForecastConfig.from_dict()."""

    def test_empty_dict(self):
        config = PyForecastConfig.from_dict({})
        assert config.oil.b_min == 0.01  # Defaults

    def test_partial_section(self):
        data = {"oil": {"b_min": 0.1}}
        config = PyForecastConfig.from_dict(data)
        assert config.oil.b_min == 0.1
        assert config.oil.b_max == 1.5  # Default preserved

    def test_multiple_sections(self):
        data = {
            "oil": {"b_min": 0.05, "b_max": 2.0},
            "gas": {"dmin": 0.08},
            "regime": {"threshold": 0.5, "window": 8},
        }
        config = PyForecastConfig.from_dict(data)
        assert config.oil.b_min == 0.05
        assert config.oil.b_max == 2.0
        assert config.gas.dmin == 0.08
        assert config.regime.threshold == 0.5
        assert config.regime.window == 8

    def test_output_products_list(self):
        data = {"output": {"products": ["oil", "gas"]}}
        config = PyForecastConfig.from_dict(data)
        assert config.output.products == ["oil", "gas"]

    def test_fitting_section(self):
        data = {"fitting": {"model_selection": "auto", "min_points": 10}}
        config = PyForecastConfig.from_dict(data)
        assert config.fitting.model_selection == "auto"
        assert config.fitting.min_points == 10

    def test_refinement_section(self):
        data = {"refinement": {"enable_logging": True, "log_storage": "csv"}}
        config = PyForecastConfig.from_dict(data)
        assert config.refinement.enable_logging is True
        assert config.refinement.log_storage == "csv"

    def test_unknown_top_level_keys_warned(self, caplog):
        data = {"bogus_section": {"foo": "bar"}}
        with caplog.at_level(logging.WARNING):
            config = PyForecastConfig.from_dict(data)
        assert "Unknown top-level config section" in caplog.text
        assert "bogus_section" in caplog.text
        # Config should still load with defaults
        assert config.oil.b_min == 0.01

    def test_unknown_section_keys_warned(self, caplog):
        data = {"oil": {"unknown_param": 42, "b_min": 0.1}}
        with caplog.at_level(logging.WARNING):
            config = PyForecastConfig.from_dict(data)
        assert "Unknown key(s)" in caplog.text
        assert "unknown_param" in caplog.text
        assert config.oil.b_min == 0.1

    def test_unknown_keys_filtered(self, caplog):
        data = {"regime": {"threshold": 0.5, "nonexistent_param": True}}
        with caplog.at_level(logging.WARNING):
            config = PyForecastConfig.from_dict(data)
        assert config.regime.threshold == 0.5
        assert "nonexistent_param" in caplog.text


class TestPyForecastConfigGetProduct:
    """Tests for get_product_config."""

    def test_get_oil(self):
        config = PyForecastConfig()
        config.oil.b_min = 0.05
        assert config.get_product_config("oil").b_min == 0.05

    def test_get_gas(self):
        config = PyForecastConfig()
        assert config.get_product_config("gas") is config.gas

    def test_get_water(self):
        config = PyForecastConfig()
        assert config.get_product_config("water") is config.water

    def test_invalid_product(self):
        config = PyForecastConfig()
        with pytest.raises(ValueError, match="Unknown product"):
            config.get_product_config("steam")


class TestPyForecastConfigToDict:
    """Tests for to_dict."""

    def test_roundtrip(self):
        config = PyForecastConfig()
        config.oil.b_min = 0.05
        config.regime.window = 8

        d = config.to_dict()
        assert d["oil"]["b_min"] == 0.05
        assert d["regime"]["window"] == 8

        config2 = PyForecastConfig.from_dict(d)
        assert config2.oil.b_min == 0.05
        assert config2.regime.window == 8


class TestPyForecastConfigYaml:
    """Tests for YAML load/save."""

    def test_from_yaml(self, tmp_path):
        yaml_content = """
oil:
  b_min: 0.05
  b_max: 2.0
regime:
  threshold: 0.8
"""
        filepath = tmp_path / "config.yaml"
        filepath.write_text(yaml_content)

        config = PyForecastConfig.from_yaml(filepath)
        assert config.oil.b_min == 0.05
        assert config.oil.b_max == 2.0
        assert config.regime.threshold == 0.8

    def test_to_yaml(self, tmp_path):
        config = PyForecastConfig()
        config.oil.b_min = 0.05
        filepath = tmp_path / "config.yaml"
        config.to_yaml(filepath)

        config2 = PyForecastConfig.from_yaml(filepath)
        assert config2.oil.b_min == 0.05

    def test_from_yaml_invalid_raises(self, tmp_path):
        yaml_content = """
oil:
  b_min: 2.0
  b_max: 1.0
"""
        filepath = tmp_path / "config.yaml"
        filepath.write_text(yaml_content)

        with pytest.raises(ValueError, match="b_min.*must be less than"):
            PyForecastConfig.from_yaml(filepath)


class TestGenerateDefaultConfig:
    """Tests for generate_default_config."""

    def test_generates_file(self, tmp_path):
        filepath = tmp_path / "pyforecast.yaml"
        result = generate_default_config(filepath)

        assert result == filepath
        assert filepath.exists()
        content = filepath.read_text()
        assert "oil:" in content
        assert "gas:" in content
        assert "regime:" in content
        assert "fitting:" in content

    def test_generated_config_is_valid(self, tmp_path):
        filepath = tmp_path / "pyforecast.yaml"
        generate_default_config(filepath)

        # The generated config should be loadable and valid
        config = PyForecastConfig.from_yaml(filepath)
        config.validate()
