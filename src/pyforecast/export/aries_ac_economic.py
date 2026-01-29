"""Export forecasts in ARIES AC_ECONOMIC format."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from ..data.well import Well
from ..core.models import ForecastResult


def _format_decimal(value: float) -> str:
    """Format a number, removing trailing .0 for whole numbers."""
    return f"{value:.2f}".rstrip('0').rstrip('.')


class AriesAcEconomicExporter:
    """Export decline forecasts in ARIES AC_ECONOMIC format.

    AC_ECONOMIC format uses ARIES expression syntax:
    - CUMS: Cumulative production
    - START: Forecast start date
    - OIL/GAS/WATER: Decline expression with b-factor
    - Continuation rows: Terminal exponential decline

    Columns:
    - PROPNUM: Well identifier
    - SECTION: Always 4 for forecasts
    - SEQUENCE: Row ordering (1=CUMS, 2=START, 100+=products)
    - QUALIFIER: Date-based qualifier (e.g., "KA0125")
    - KEYWORD: CUMS, START, OIL, GAS, WATER, or " for continuation
    - EXPRESSION: ARIES decline expression

    Note: Exports use daily rates (B/D = barrels per day, M/D = mcf per day).
    ARIES always uses EXP decline type regardless of b-factor value.
    """

    # Sequence numbers for each product
    SEQUENCE_MAP = {
        "oil": 100,
        "gas": 300,
        "water": 500,
    }

    # Unit codes for ARIES expressions (daily rates)
    UNIT_MAP = {
        "oil": "B/D",    # Barrels per day
        "gas": "M/D",    # Mcf per day
        "water": "B/D",  # Barrels per day
    }

    def __init__(
        self,
        qualifier: str | None = None,
        cumulative_oil: float = 0.0,
        cumulative_gas: float = 0.0,
    ):
        """Initialize exporter.

        Args:
            qualifier: ARIES qualifier string (default: auto-generate from date)
            cumulative_oil: Historical cumulative oil (MBbl) for CUMS row
            cumulative_gas: Historical cumulative gas (MMcf) for CUMS row
        """
        if qualifier is None:
            # Generate qualifier like "KA0125" for Jan 2025
            qualifier = datetime.now().strftime("KA%m%y")
        self.qualifier = qualifier
        self.cumulative_oil = cumulative_oil
        self.cumulative_gas = cumulative_gas

    def _format_expression(
        self,
        result: ForecastResult,
        product: Literal["oil", "gas", "water"],
    ) -> str:
        """Format the ARIES decline expression.

        Format: "{Qn} X {unit} {Dmin%} EXP B/{b} {Di%}"
        Example: "100 X B/D 6 EXP B/0.50 8.5"

        Note: qi in model is daily rate; output directly.
        ARIES always uses EXP regardless of b-factor value.
        """
        model = result.model
        unit = self.UNIT_MAP[product]

        # qi is already daily, round to 1 decimal place
        qn = round(model.qi, 1)

        # Convert rates to annual percentage
        dmin_pct = model.dmin * 12 * 100  # Monthly to annual %
        di_pct = model.di * 12 * 100  # Monthly to annual %

        # Ensure di is slightly higher than dmin (ARIES requirement)
        if di_pct <= dmin_pct:
            di_pct = dmin_pct + 0.1

        qn_str = _format_decimal(qn)
        dmin_str = _format_decimal(dmin_pct)
        di_str = _format_decimal(di_pct)
        b_str = f"{model.b:.2f}"

        return f"{qn_str} X {unit} {dmin_str} EXP B/{b_str} {di_str}"

    def _format_terminal_expression(
        self,
        result: ForecastResult,
        product: Literal["oil", "gas", "water"],
    ) -> str:
        """Format the terminal decline continuation row.

        Format: "X 1 {unit} X YRS EXP {Dmin%}"
        """
        unit = self.UNIT_MAP[product]
        model = result.model
        dmin_pct = model.dmin * 12 * 100
        dmin_str = _format_decimal(dmin_pct)

        return f"X 1 {unit} X YRS EXP {dmin_str}"

    def export_well(
        self,
        well: Well,
        products: list[Literal["oil", "gas", "water"]] | None = None,
    ) -> list[dict]:
        """Export single well to AC_ECONOMIC format rows.

        Args:
            well: Well with forecast results
            products: Products to export (default: oil and gas if available)

        Returns:
            List of row dictionaries
        """
        if products is None:
            products = []
            if well.forecast_oil is not None:
                products.append("oil")
            if well.forecast_gas is not None:
                products.append("gas")
            if well.forecast_water is not None:
                products.append("water")

        rows = []
        propnum = well.identifier.propnum or well.identifier.api or well.identifier.primary_id

        # Calculate cumulative from historical data
        cum_oil = well.production.oil.sum() / 1000 if len(well.production.oil) > 0 else self.cumulative_oil
        cum_gas = well.production.gas.sum() / 1000 if len(well.production.gas) > 0 else self.cumulative_gas

        # CUMS row
        rows.append({
            "PROPNUM": propnum,
            "SECTION": 4,
            "SEQUENCE": 1,
            "QUALIFIER": self.qualifier,
            "KEYWORD": "CUMS",
            "EXPRESSION": f"{cum_oil:.3f} {cum_gas:.3f}",
        })

        # START row
        if well.production.last_date:
            # Start forecast month after last production
            last_date = well.production.last_date
            start_month = last_date.month + 1
            start_year = last_date.year
            if start_month > 12:
                start_month = 1
                start_year += 1
            start_expr = f"{start_month:02d}/{start_year}"
        else:
            start_expr = "01/1900"

        rows.append({
            "PROPNUM": propnum,
            "SECTION": 4,
            "SEQUENCE": 2,
            "QUALIFIER": self.qualifier,
            "KEYWORD": "START",
            "EXPRESSION": start_expr,
        })

        # Product rows
        for product in products:
            result = well.get_forecast(product)
            if result is None:
                continue

            # Skip if qi is effectively zero (qi is daily rate, ~0.3/day = ~10/month)
            if result.model.qi < 0.3:
                continue

            seq = self.SEQUENCE_MAP[product]

            # Main decline row
            rows.append({
                "PROPNUM": propnum,
                "SECTION": 4,
                "SEQUENCE": seq,
                "QUALIFIER": self.qualifier,
                "KEYWORD": product.upper(),
                "EXPRESSION": self._format_expression(result, product),
            })

            # Terminal decline continuation row
            rows.append({
                "PROPNUM": propnum,
                "SECTION": 4,
                "SEQUENCE": seq + 100,
                "QUALIFIER": self.qualifier,
                "KEYWORD": '"',
                "EXPRESSION": self._format_terminal_expression(result, product),
            })

        return rows

    def export_wells(
        self,
        wells: list[Well],
        products: list[Literal["oil", "gas", "water"]] | None = None,
    ) -> pd.DataFrame:
        """Export multiple wells to AC_ECONOMIC format DataFrame.

        Args:
            wells: List of wells with forecast results
            products: Products to export

        Returns:
            DataFrame in AC_ECONOMIC format
        """
        all_rows = []
        for well in wells:
            rows = self.export_well(well, products)
            all_rows.extend(rows)

        if not all_rows:
            return pd.DataFrame(columns=[
                "PROPNUM", "SECTION", "SEQUENCE", "QUALIFIER", "KEYWORD", "EXPRESSION"
            ])

        return pd.DataFrame(all_rows)

    def save(
        self,
        wells: list[Well],
        output_path: Path | str,
        products: list[Literal["oil", "gas", "water"]] | None = None,
    ) -> Path:
        """Export wells and save to CSV file.

        Args:
            wells: List of wells with forecast results
            output_path: Output file path
            products: Products to export

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        df = self.export_wells(wells, products)
        df.to_csv(output_path, index=False)
        return output_path
