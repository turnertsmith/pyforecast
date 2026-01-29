"""Interactive Plotly visualizations for decline curve analysis."""

from pathlib import Path
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..data.well import Well
from ..core.models import ForecastResult


class DeclinePlotter:
    """Create interactive semi-log decline curve plots."""

    # Color palette for multiple wells
    COLORS = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    def __init__(
        self,
        forecast_months: int = 120,
        width: int = 1000,
        height: int = 600,
    ):
        """Initialize plotter.

        Args:
            forecast_months: Number of months to project forecast
            width: Plot width in pixels
            height: Plot height in pixels
        """
        self.forecast_months = forecast_months
        self.width = width
        self.height = height

    def plot_well(
        self,
        well: Well,
        product: Literal["oil", "gas"] = "oil",
        show_regime: bool = True,
    ) -> go.Figure:
        """Create single well decline curve plot.

        Args:
            well: Well with production data and forecast
            product: Product to plot
            show_regime: Whether to mark regime change

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        production = well.production
        result = well.get_forecast(product)

        # Get production data
        t = production.time_months
        q = production.get_product(product)

        # Historical production scatter
        fig.add_trace(go.Scatter(
            x=t,
            y=q,
            mode='markers',
            name='Historical',
            marker=dict(
                size=8,
                color='#1f77b4',
                symbol='circle'
            ),
            hovertemplate=(
                f"<b>{well.well_id}</b><br>"
                "Month: %{x}<br>"
                f"{product.capitalize()}: %{{y:.0f}}<br>"
                "<extra></extra>"
            )
        ))

        if result is not None:
            model = result.model

            # Mark regime change if detected
            if show_regime and result.regime_start_idx > 0:
                regime_t = t[result.regime_start_idx]
                fig.add_vline(
                    x=regime_t,
                    line_dash="dot",
                    line_color="red",
                    annotation_text="Regime Change",
                    annotation_position="top right"
                )

            # Fitted curve over historical period (from regime start)
            t_regime = t[result.regime_start_idx:]
            t_fit = np.linspace(0, t_regime[-1] - t_regime[0], 100)
            q_fit = model.rate(t_fit)

            fig.add_trace(go.Scatter(
                x=t_fit + t_regime[0],
                y=q_fit,
                mode='lines',
                name='Fitted',
                line=dict(color='#ff7f0e', width=2),
                hovertemplate=(
                    f"<b>Fit: {model.decline_type}</b><br>"
                    "Month: %{x:.0f}<br>"
                    f"{product.capitalize()}: %{{y:.0f}}<br>"
                    f"R²: {result.r_squared:.3f}<br>"
                    "<extra></extra>"
                )
            ))

            # Forecast projection (continue from end of historical data)
            t_last = t[-1]
            t_model_end = t_last - t[result.regime_start_idx]  # End time in model coordinates
            t_forecast = np.linspace(t_model_end, t_model_end + self.forecast_months, 200)
            q_forecast = model.rate(t_forecast)

            fig.add_trace(go.Scatter(
                x=t_forecast - t_model_end + t_last,  # Shift to plot coordinates
                y=q_forecast,
                mode='lines',
                name='Forecast',
                line=dict(color='#2ca02c', width=2, dash='dash'),
                hovertemplate=(
                    f"<b>Forecast</b><br>"
                    "Month: %{x:.0f}<br>"
                    f"{product.capitalize()}: %{{y:.0f}}<br>"
                    "<extra></extra>"
                )
            ))

        # Configure layout
        product_label = "Oil (bbl/month)" if product == "oil" else "Gas (mcf/month)"

        fig.update_layout(
            title=dict(
                text=f"Decline Curve Analysis: {well.well_id}",
                font=dict(size=16)
            ),
            xaxis_title="Time (months)",
            yaxis_title=product_label,
            yaxis_type="log",
            width=self.width,
            height=self.height,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            hovermode='closest',
        )

        # Add fit parameters annotation
        if result is not None:
            annotation_text = (
                f"<b>Parameters</b><br>"
                f"qi: {result.model.qi:.0f}<br>"
                f"Di: {result.model.di * 12:.1%}/yr<br>"
                f"b: {result.model.b:.3f}<br>"
                f"R²: {result.r_squared:.3f}"
            )
            fig.add_annotation(
                x=0.02, y=0.02,
                xref='paper', yref='paper',
                text=annotation_text,
                showarrow=False,
                font=dict(size=10),
                align='left',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray',
                borderwidth=1
            )

        return fig

    def plot_multiple_wells(
        self,
        wells: list[Well],
        product: Literal["oil", "gas"] = "oil",
        normalize: bool = False,
    ) -> go.Figure:
        """Create overlay plot of multiple wells.

        Args:
            wells: List of wells with forecasts
            product: Product to plot
            normalize: If True, normalize rates to initial rate

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        for i, well in enumerate(wells):
            color = self.COLORS[i % len(self.COLORS)]
            result = well.get_forecast(product)

            t = well.production.time_months
            q = well.production.get_product(product)

            if normalize and len(q) > 0 and q[0] > 0:
                q = q / q[0]

            # Historical
            fig.add_trace(go.Scatter(
                x=t,
                y=q,
                mode='markers',
                name=f'{well.well_id}',
                marker=dict(size=6, color=color),
                legendgroup=well.well_id,
                hovertemplate=f"<b>{well.well_id}</b><br>Month: %{{x}}<br>Rate: %{{y:.0f}}<extra></extra>"
            ))

            # Forecast
            if result is not None:
                t_last = t[-1] if len(t) > 0 else 0
                t_model_end = t_last - t[result.regime_start_idx] if len(t) > 0 else 0
                t_forecast = np.linspace(t_model_end, t_model_end + self.forecast_months, 100)
                q_forecast = result.model.rate(t_forecast)

                if normalize and result.model.qi > 0:
                    q_forecast = q_forecast / result.model.qi

                fig.add_trace(go.Scatter(
                    x=t_forecast - t_model_end + t_last,
                    y=q_forecast,
                    mode='lines',
                    name=f'{well.well_id} (forecast)',
                    line=dict(color=color, width=1.5, dash='dash'),
                    legendgroup=well.well_id,
                    showlegend=False,
                ))

        # Configure layout
        y_label = "Normalized Rate" if normalize else (
            "Oil (bbl/month)" if product == "oil" else "Gas (mcf/month)"
        )

        fig.update_layout(
            title=f"Multi-Well Decline Comparison ({len(wells)} wells)",
            xaxis_title="Time (months)",
            yaxis_title=y_label,
            yaxis_type="log",
            width=self.width,
            height=self.height,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            hovermode='closest',
        )

        return fig

    def plot_oil_gas(self, well: Well) -> go.Figure:
        """Create dual-axis plot with both oil and gas.

        Args:
            well: Well with production and forecast data

        Returns:
            Plotly Figure with oil and gas subplots
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Oil Production", "Gas Production"),
            vertical_spacing=0.12
        )

        t = well.production.time_months

        for row, product in enumerate(["oil", "gas"], start=1):
            q = well.production.get_product(product)
            result = well.get_forecast(product)
            color = "#1f77b4" if product == "oil" else "#ff7f0e"

            # Historical
            fig.add_trace(
                go.Scatter(
                    x=t, y=q,
                    mode='markers',
                    name=f'{product.capitalize()} Historical',
                    marker=dict(size=6, color=color),
                ),
                row=row, col=1
            )

            # Forecast
            if result is not None:
                t_last = t[-1] if len(t) > 0 else 0
                t_forecast = np.linspace(0, self.forecast_months, 100)
                q_forecast = result.model.rate(t_forecast)

                fig.add_trace(
                    go.Scatter(
                        x=t_forecast + t_last,
                        y=q_forecast,
                        mode='lines',
                        name=f'{product.capitalize()} Forecast',
                        line=dict(color=color, dash='dash'),
                    ),
                    row=row, col=1
                )

            fig.update_yaxes(type="log", row=row, col=1)

        fig.update_layout(
            title=f"Oil & Gas Decline: {well.well_id}",
            height=self.height * 1.2,
            width=self.width,
        )

        return fig

    def save(
        self,
        fig: go.Figure,
        output_path: Path | str,
        format: Literal["html", "png", "svg", "pdf"] = "html"
    ) -> Path:
        """Save figure to file.

        Args:
            fig: Plotly Figure object
            output_path: Output file path
            format: Output format

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)

        if format == "html":
            fig.write_html(output_path)
        else:
            fig.write_image(output_path, format=format)

        return output_path
