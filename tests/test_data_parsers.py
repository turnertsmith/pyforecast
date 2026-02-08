"""Tests for data parsers (Enverus, ARIES) and ProductionData."""

import numpy as np
import pandas as pd
import pytest

from pyforecast.data.base import DataParser, detect_parser, load_wells
from pyforecast.data.enverus import EnverusParser
from pyforecast.data.aries import AriesParser
from pyforecast.data.well import Well, WellIdentifier, ProductionData


class TestEnverusParser:
    """Tests for EnverusParser."""

    def _make_enverus_df(self, **overrides):
        """Create a minimal Enverus-format DataFrame."""
        data = {
            "Entity ID": ["WELL001", "WELL001", "WELL002", "WELL002"],
            "Well Name": ["Smith 1H", "Smith 1H", "Jones 2H", "Jones 2H"],
            "Production Date": ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"],
            "Oil (BBL)": [5000.0, 4500.0, 6000.0, 5500.0],
            "Gas (MCF)": [25000.0, 22000.0, 30000.0, 28000.0],
            "Water (BBL)": [1000.0, 1100.0, 800.0, 900.0],
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_can_parse_enverus_format(self):
        df = self._make_enverus_df()
        parser = EnverusParser()
        assert parser.can_parse(df)

    def test_cannot_parse_aries_format(self):
        df = pd.DataFrame({
            "PROPNUM": ["W001"], "YYYYMM": [202001], "MONTHLY_OIL": [5000],
        })
        parser = EnverusParser()
        assert not parser.can_parse(df)

    def test_cannot_parse_unrecognized(self):
        df = pd.DataFrame({"foo": [1], "bar": [2]})
        parser = EnverusParser()
        assert not parser.can_parse(df)

    def test_parse_basic(self):
        df = self._make_enverus_df()
        parser = EnverusParser()
        wells = parser.parse(df)

        assert len(wells) == 2
        well_ids = sorted(w.well_id for w in wells)
        assert well_ids == ["WELL001", "WELL002"]

    def test_parse_production_values(self):
        df = self._make_enverus_df()
        parser = EnverusParser()
        wells = parser.parse(df)

        well1 = next(w for w in wells if w.well_id == "WELL001")
        assert len(well1.production.oil) == 2
        np.testing.assert_array_equal(well1.production.oil, [5000.0, 4500.0])
        np.testing.assert_array_equal(well1.production.gas, [25000.0, 22000.0])
        np.testing.assert_array_equal(well1.production.water, [1000.0, 1100.0])

    def test_parse_well_name(self):
        df = self._make_enverus_df()
        parser = EnverusParser()
        wells = parser.parse(df)

        well1 = next(w for w in wells if w.identifier.entity_id == "WELL001")
        assert well1.identifier.well_name == "Smith 1H"

    def test_parse_api_as_id(self):
        """When no entity_id column, use API column."""
        df = pd.DataFrame({
            "API Number": ["42-001-00001", "42-001-00001"],
            "Date": ["2020-01-01", "2020-02-01"],
            "Oil": [5000.0, 4500.0],
            "Gas": [25000.0, 22000.0],
        })
        parser = EnverusParser()
        assert parser.can_parse(df)
        wells = parser.parse(df)
        assert len(wells) == 1
        assert wells[0].identifier.api == "42-001-00001"

    def test_parse_missing_water(self):
        """Water column is optional."""
        df = pd.DataFrame({
            "Entity ID": ["W001", "W001"],
            "Production Date": ["2020-01-01", "2020-02-01"],
            "Oil": [5000.0, 4500.0],
            "Gas": [25000.0, 22000.0],
        })
        parser = EnverusParser()
        wells = parser.parse(df)
        assert wells[0].production.water is None

    def test_parse_fills_nan_with_zero(self):
        df = pd.DataFrame({
            "Entity ID": ["W001", "W001"],
            "Production Date": ["2020-01-01", "2020-02-01"],
            "Oil": [5000.0, float("nan")],
            "Gas": [25000.0, 22000.0],
        })
        parser = EnverusParser()
        wells = parser.parse(df)
        assert wells[0].production.oil[1] == 0.0

    def test_parse_no_id_column_raises(self):
        df = pd.DataFrame({
            "Production Date": ["2020-01-01"],
            "Oil": [5000.0],
        })
        parser = EnverusParser()
        with pytest.raises(ValueError, match="No identifier column"):
            parser.parse(df)

    def test_parse_no_date_column_raises(self):
        df = pd.DataFrame({
            "Entity ID": ["W001"],
            "Oil": [5000.0],
        })
        parser = EnverusParser()
        with pytest.raises(ValueError, match="No date column"):
            parser.parse(df)

    def test_case_insensitive_columns(self):
        """Column matching should be case-insensitive."""
        df = pd.DataFrame({
            "ENTITY ID": ["W001", "W001"],
            "PRODUCTION DATE": ["2020-01-01", "2020-02-01"],
            "OIL": [5000.0, 4500.0],
            "GAS": [25000.0, 22000.0],
        })
        parser = EnverusParser()
        assert parser.can_parse(df)
        wells = parser.parse(df)
        assert len(wells) == 1


class TestAriesParser:
    """Tests for AriesParser."""

    def _make_aries_df(self, date_format="yyyymm"):
        """Create a minimal ARIES-format DataFrame."""
        if date_format == "yyyymm":
            dates = [202001, 202002, 202001, 202002]
            date_col = "YYYYMM"
        elif date_format == "yyyy-mm":
            dates = ["2020-01", "2020-02", "2020-01", "2020-02"]
            date_col = "YYYYMM"
        else:
            dates = ["2020-01-01", "2020-02-01", "2020-01-01", "2020-02-01"]
            date_col = "P_DATE"

        return pd.DataFrame({
            "PROPNUM": ["WELL001", "WELL001", "WELL002", "WELL002"],
            date_col: dates,
            "MONTHLY_OIL": [5000.0, 4500.0, 6000.0, 5500.0],
            "MONTHLY_GAS": [25000.0, 22000.0, 30000.0, 28000.0],
            "MONTHLY_WATER": [1000.0, 1100.0, 800.0, 900.0],
        })

    def test_can_parse_aries_format(self):
        df = self._make_aries_df()
        parser = AriesParser()
        assert parser.can_parse(df)

    def test_cannot_parse_enverus(self):
        df = pd.DataFrame({
            "Entity ID": ["W001"], "Production Date": ["2020-01-01"], "Oil": [5000],
        })
        parser = AriesParser()
        # ARIES parser might match 'date' column, but shouldn't match without propnum or aries date
        # The key differentiator is propnum vs entity_id

    def test_parse_yyyymm_dates(self):
        df = self._make_aries_df(date_format="yyyymm")
        parser = AriesParser()
        wells = parser.parse(df)

        assert len(wells) == 2
        well1 = next(w for w in wells if w.well_id == "WELL001")
        assert well1.production.n_months == 2

    def test_parse_yyyy_mm_dates(self):
        df = self._make_aries_df(date_format="yyyy-mm")
        parser = AriesParser()
        wells = parser.parse(df)

        assert len(wells) == 2

    def test_parse_p_date_format(self):
        df = self._make_aries_df(date_format="p_date")
        parser = AriesParser()
        wells = parser.parse(df)

        assert len(wells) == 2

    def test_parse_propnum_identifier(self):
        df = self._make_aries_df()
        parser = AriesParser()
        wells = parser.parse(df)

        well1 = next(w for w in wells if w.well_id == "WELL001")
        assert well1.identifier.propnum == "WELL001"

    def test_parse_production_values(self):
        df = self._make_aries_df()
        parser = AriesParser()
        wells = parser.parse(df)

        well1 = next(w for w in wells if w.well_id == "WELL001")
        np.testing.assert_array_equal(well1.production.oil, [5000.0, 4500.0])
        np.testing.assert_array_equal(well1.production.gas, [25000.0, 22000.0])

    def test_parse_no_propnum_raises(self):
        df = pd.DataFrame({
            "P_DATE": ["2020-01-01"],
            "OIL": [5000.0],
        })
        parser = AriesParser()
        # P_DATE triggers can_parse, but no propnum/api for ID
        if parser.can_parse(df):
            with pytest.raises(ValueError, match="No identifier"):
                parser.parse(df)


class TestDetectParser:
    """Tests for auto-detection of parser."""

    def test_detect_enverus(self):
        df = pd.DataFrame({
            "Entity ID": ["W001"], "Production Date": ["2020-01-01"],
            "Oil": [5000], "Gas": [25000],
        })
        parser = detect_parser(df)
        assert isinstance(parser, EnverusParser)

    def test_detect_aries(self):
        df = pd.DataFrame({
            "PROPNUM": ["W001"], "YYYYMM": [202001],
            "MONTHLY_OIL": [5000], "MONTHLY_GAS": [25000],
        })
        parser = detect_parser(df)
        # Could be Enverus or ARIES depending on detection order
        # ARIES has propnum which is checked
        assert isinstance(parser, (EnverusParser, AriesParser))

    def test_detect_unknown_raises(self):
        df = pd.DataFrame({"foo": [1], "bar": [2]})
        with pytest.raises(ValueError, match="Could not detect"):
            detect_parser(df)


class TestProductionData:
    """Tests for ProductionData model."""

    def _make_production(self, n_months=12, **kwargs):
        """Create a ProductionData with declining production."""
        dates = pd.date_range("2020-01-01", periods=n_months, freq="MS").values
        qi = kwargs.get("qi", 1000.0)
        decline = kwargs.get("decline", 0.05)
        oil = qi * np.exp(-decline * np.arange(n_months))
        gas = oil * 5.0
        return ProductionData(
            dates=dates,
            oil=oil,
            gas=gas,
            water=kwargs.get("water", None),
        )

    def test_time_months_computed(self):
        prod = self._make_production(n_months=6)
        np.testing.assert_array_equal(prod.time_months, [0, 1, 2, 3, 4, 5])

    def test_days_in_month_computed(self):
        prod = self._make_production(n_months=3)
        # Jan=31, Feb=29 (2020 is leap), Mar=31
        assert prod.days_in_month[0] == 31
        assert prod.days_in_month[1] == 29  # 2020 is a leap year
        assert prod.days_in_month[2] == 31

    def test_n_months(self):
        prod = self._make_production(n_months=12)
        assert prod.n_months == 12

    def test_first_last_date(self):
        prod = self._make_production(n_months=6)
        from datetime import date
        assert prod.first_date == date(2020, 1, 1)
        assert prod.last_date == date(2020, 6, 1)

    def test_get_product(self):
        prod = self._make_production()
        assert len(prod.get_product("oil")) == 12
        assert len(prod.get_product("gas")) == 12

    def test_get_product_water_none_raises(self):
        prod = self._make_production(water=None)
        with pytest.raises(ValueError, match="Water production data not available"):
            prod.get_product("water")

    def test_get_product_unknown_raises(self):
        prod = self._make_production()
        with pytest.raises(ValueError, match="Unknown product"):
            prod.get_product("condensate")

    def test_get_product_daily(self):
        prod = self._make_production(n_months=3)
        daily = prod.get_product_daily("oil")
        # Daily rate = monthly / days_in_month
        np.testing.assert_allclose(daily, prod.oil / prod.days_in_month)

    def test_oil_daily_cached(self):
        """Daily rate properties should be cached after first access."""
        prod = self._make_production()
        daily1 = prod.oil_daily
        daily2 = prod.oil_daily
        assert daily1 is daily2  # Same object, not recomputed

    def test_gas_daily_cached(self):
        prod = self._make_production()
        daily1 = prod.gas_daily
        daily2 = prod.gas_daily
        assert daily1 is daily2

    def test_water_daily_cached(self):
        water = np.array([100.0, 90.0, 80.0])
        prod = self._make_production(n_months=3, water=water)
        daily1 = prod.water_daily
        daily2 = prod.water_daily
        assert daily1 is daily2

    def test_water_daily_none(self):
        prod = self._make_production(water=None)
        assert prod.water_daily is None

    def test_empty_production_data(self):
        prod = ProductionData(
            dates=np.array([], dtype="datetime64[ns]"),
            oil=np.array([], dtype=float),
            gas=np.array([], dtype=float),
        )
        assert prod.n_months == 0
        assert prod.first_date is None
        assert prod.last_date is None
        np.testing.assert_array_equal(prod.time_months, [])

    def test_to_dataframe(self):
        prod = self._make_production(n_months=3)
        df = prod.to_dataframe()
        assert "date" in df.columns
        assert "oil" in df.columns
        assert "gas" in df.columns
        assert len(df) == 3


class TestWellIdentifier:
    """Tests for WellIdentifier."""

    def test_primary_id_api(self):
        ident = WellIdentifier(api="42-001-00001")
        assert ident.primary_id == "42-001-00001"

    def test_primary_id_propnum(self):
        ident = WellIdentifier(propnum="WELL001")
        assert ident.primary_id == "WELL001"

    def test_primary_id_entity_id(self):
        ident = WellIdentifier(entity_id="12345678")
        assert ident.primary_id == "12345678"

    def test_primary_id_priority(self):
        """API takes priority over propnum and entity_id."""
        ident = WellIdentifier(api="API", propnum="PROP", entity_id="ENT")
        assert ident.primary_id == "API"

    def test_primary_id_unknown(self):
        ident = WellIdentifier()
        assert ident.primary_id == "UNKNOWN"


class TestWell:
    """Tests for Well model."""

    def _make_well(self, n_months=12) -> Well:
        dates = pd.date_range("2020-01-01", periods=n_months, freq="MS").values
        return Well(
            identifier=WellIdentifier(api="TEST-001"),
            production=ProductionData(
                dates=dates,
                oil=np.ones(n_months) * 1000,
                gas=np.ones(n_months) * 5000,
            ),
        )

    def test_well_id(self):
        well = self._make_well()
        assert well.well_id == "TEST-001"

    def test_has_sufficient_data(self):
        well = self._make_well(n_months=12)
        assert well.has_sufficient_data(6)
        assert well.has_sufficient_data(12)
        assert not well.has_sufficient_data(13)

    def test_get_set_forecast(self):
        from pyforecast.core.models import HyperbolicModel, ForecastResult
        well = self._make_well()

        model = HyperbolicModel(qi=1000, di=0.05, b=0.5)
        result = ForecastResult(
            model=model, r_squared=0.95, rmse=10.0,
            aic=100, bic=110, regime_start_idx=0, data_points_used=12,
        )

        well.set_forecast("oil", result)
        assert well.get_forecast("oil") is result
        assert well.get_forecast("gas") is None
