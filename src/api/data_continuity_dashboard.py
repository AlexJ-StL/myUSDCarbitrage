"""Data continuity monitoring dashboard for USDC arbitrage application."""

import logging
from datetime import datetime, timedelta

import dash
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

from .gap_detection import GapAnalysisReport, GapDetectionSystem, GapSeverity

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("data_validation.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class DataContinuityDashboard:
    """Dashboard for monitoring data continuity and gap analysis."""

    def __init__(self, connection_string: str):
        """Initialize data continuity dashboard.

        Args:
            connection_string: Database connection string
        """
        self.connection_string = connection_string
        self.gap_detection = GapDetectionSystem(connection_string)
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self) -> None:
        """Set up the dashboard layout."""
        self.app.layout = html.Div(
            [
                html.H1("Data Continuity Monitoring Dashboard"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Exchange:"),
                                dcc.Dropdown(
                                    id="exchange-dropdown",
                                    options=[
                                        {"label": "Coinbase", "value": "coinbase"},
                                        {"label": "Kraken", "value": "kraken"},
                                        {"label": "Binance", "value": "binance"},
                                    ],
                                    value="coinbase",
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "marginRight": "2%",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("Symbol:"),
                                dcc.Dropdown(
                                    id="symbol-dropdown",
                                    options=[
                                        {"label": "USDC/USD", "value": "USDC/USD"},
                                        {"label": "USDC/USDT", "value": "USDC/USDT"},
                                    ],
                                    value="USDC/USD",
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "marginRight": "2%",
                            },
                        ),
                        html.Div(
                            [
                                html.Label("Timeframe:"),
                                dcc.Dropdown(
                                    id="timeframe-dropdown",
                                    options=[
                                        {"label": "1 Hour", "value": "1h"},
                                        {"label": "4 Hours", "value": "4h"},
                                        {"label": "1 Day", "value": "1d"},
                                    ],
                                    value="1h",
                                ),
                            ],
                            style={"width": "30%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Date Range:"),
                        dcc.DatePickerRange(
                            id="date-range",
                            start_date=datetime.now() - timedelta(days=30),
                            end_date=datetime.now(),
                            display_format="YYYY-MM-DD",
                        ),
                    ],
                    style={"marginTop": "20px", "marginBottom": "20px"},
                ),
                html.Button("Run Analysis", id="run-analysis-button", n_clicks=0),
                html.Div(id="analysis-status", style={"marginTop": "10px"}),
                html.Div(
                    [
                        html.H2("Data Completeness Overview"),
                        dcc.Graph(id="completeness-chart"),
                    ],
                    style={"marginTop": "30px"},
                ),
                html.Div(
                    [html.H2("Gap Analysis"), dcc.Graph(id="gap-chart")],
                    style={"marginTop": "30px"},
                ),
                html.Div(
                    [html.H2("Gap Details"), html.Div(id="gap-table")],
                    style={"marginTop": "30px"},
                ),
                html.Div(
                    [
                        html.H2("Data Quality Metrics"),
                        dcc.Graph(id="quality-metrics-chart"),
                    ],
                    style={"marginTop": "30px"},
                ),
            ],
            style={"margin": "20px"},
        )

    def setup_callbacks(self) -> None:
        """Set up dashboard callbacks."""

        @self.app.callback(
            [
                Output("analysis-status", "children"),
                Output("completeness-chart", "figure"),
                Output("gap-chart", "figure"),
                Output("gap-table", "children"),
                Output("quality-metrics-chart", "figure"),
            ],
            [Input("run-analysis-button", "n_clicks")],
            [
                dash.dependencies.State("exchange-dropdown", "value"),
                dash.dependencies.State("symbol-dropdown", "value"),
                dash.dependencies.State("timeframe-dropdown", "value"),
                dash.dependencies.State("date-range", "start_date"),
                dash.dependencies.State("date-range", "end_date"),
            ],
        )
        def update_dashboard(
            n_clicks, exchange, symbol, timeframe, start_date, end_date
        ):
            if n_clicks == 0:
                # Initial state
                return (
                    "Click 'Run Analysis' to start",
                    self._empty_completeness_chart(),
                    self._empty_gap_chart(),
                    self._empty_gap_table(),
                    self._empty_quality_metrics_chart(),
                )

            # Parse dates
            start_date = datetime.strptime(start_date.split("T")[0], "%Y-%m-%d")
            end_date = datetime.strptime(end_date.split("T")[0], "%Y-%m-%d")

            # Run gap analysis
            try:
                report = self.gap_detection.analyze_gaps(
                    exchange, symbol, timeframe, start_date, end_date
                )

                # Create visualizations
                completeness_chart = self._create_completeness_chart(report)
                gap_chart = self._create_gap_chart(report)
                gap_table = self._create_gap_table(report)
                quality_metrics_chart = self._create_quality_metrics_chart(report)

                status = f"Analysis completed: Found {report.total_gaps} gaps, {report.filled_gaps} filled"

                return (
                    status,
                    completeness_chart,
                    gap_chart,
                    gap_table,
                    quality_metrics_chart,
                )

            except Exception as e:
                logger.error(f"Error in dashboard update: {str(e)}")
                return (
                    f"Error: {str(e)}",
                    self._empty_completeness_chart(),
                    self._empty_gap_chart(),
                    self._empty_gap_table(),
                    self._empty_quality_metrics_chart(),
                )

    def _empty_completeness_chart(self) -> go.Figure:
        """Create empty completeness chart."""
        fig = go.Figure()
        fig.update_layout(
            title="Data Completeness",
            xaxis_title="Date",
            yaxis_title="Completeness (%)",
            height=400,
        )
        return fig

    def _empty_gap_chart(self) -> go.Figure:
        """Create empty gap chart."""
        fig = go.Figure()
        fig.update_layout(
            title="Gap Distribution",
            xaxis_title="Date",
            yaxis_title="Gap Duration",
            height=400,
        )
        return fig

    def _empty_gap_table(self) -> html.Div:
        """Create empty gap table."""
        return html.Div("No gap data available")

    def _empty_quality_metrics_chart(self) -> go.Figure:
        """Create empty quality metrics chart."""
        fig = go.Figure()
        fig.update_layout(title="Data Quality Metrics", height=400)
        return fig

    def _create_completeness_chart(self, report: GapAnalysisReport) -> go.Figure:
        """Create data completeness chart.

        Args:
            report: Gap analysis report

        Returns:
            Plotly figure
        """
        fig = go.Figure()

        # Add completeness bar
        fig.add_trace(
            go.Bar(
                x=["Data Completeness"],
                y=[report.data_completeness],
                text=[f"{report.data_completeness:.2f}%"],
                textposition="auto",
                marker_color="royalblue",
            )
        )

        # Add reference line for 100%
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=0.5,
            y0=100,
            y1=100,
            line=dict(color="red", width=2, dash="dash"),
        )

        fig.update_layout(
            title=f"Data Completeness - {report.exchange}/{report.symbol}/{report.timeframe}",
            yaxis=dict(title="Completeness (%)", range=[0, 105]),
            height=400,
        )

        return fig

    def _create_gap_chart(self, report: GapAnalysisReport) -> go.Figure:
        """Create gap distribution chart.

        Args:
            report: Gap analysis report

        Returns:
            Plotly figure
        """
        if not report.gaps:
            return self._empty_gap_chart()

        fig = go.Figure()

        # Prepare data
        start_times = [gap.start_time for gap in report.gaps]
        durations = [
            (gap.duration.total_seconds() / 3600) for gap in report.gaps
        ]  # Convert to hours
        severities = [gap.severity.value for gap in report.gaps]
        filled_status = ["Filled" if gap.filled else "Unfilled" for gap in report.gaps]

        # Create color map for severities
        color_map = {
            "minor": "rgba(173, 216, 230, 0.8)",  # Light blue
            "moderate": "rgba(255, 165, 0, 0.8)",  # Orange
            "critical": "rgba(255, 0, 0, 0.8)",  # Red
        }

        colors = [color_map.get(severity, "gray") for severity in severities]

        # Create hover text
        hover_texts = []
        for i, gap in enumerate(report.gaps):
            text = (
                f"Start: {gap.start_time}<br>"
                f"End: {gap.end_time}<br>"
                f"Duration: {gap.duration}<br>"
                f"Severity: {gap.severity.value}<br>"
                f"Filled: {'Yes' if gap.filled else 'No'}"
            )
            if gap.filled and gap.fill_method:
                text += f"<br>Method: {gap.fill_method}"
            if gap.filled and gap.fill_source:
                text += f"<br>Source: {gap.fill_source}"
            hover_texts.append(text)

        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=start_times,
                y=durations,
                mode="markers",
                marker=dict(size=12, color=colors, line=dict(width=1, color="black")),
                text=hover_texts,
                hoverinfo="text",
                name="Gaps",
            )
        )

        # Add shapes for severity thresholds
        timeframe = report.timeframe
        thresholds = self.gap_detection.severity_thresholds.get(
            timeframe, self.gap_detection.severity_thresholds["1h"]
        )

        minor_threshold = thresholds["minor"].total_seconds() / 3600
        moderate_threshold = thresholds["moderate"].total_seconds() / 3600
        critical_threshold = thresholds["critical"].total_seconds() / 3600

        # Add threshold lines
        fig.add_shape(
            type="line",
            x0=min(start_times),
            x1=max(start_times),
            y0=minor_threshold,
            y1=minor_threshold,
            line=dict(color="lightblue", width=1, dash="dash"),
            name="Minor Threshold",
        )

        fig.add_shape(
            type="line",
            x0=min(start_times),
            x1=max(start_times),
            y0=moderate_threshold,
            y1=moderate_threshold,
            line=dict(color="orange", width=1, dash="dash"),
            name="Moderate Threshold",
        )

        fig.add_shape(
            type="line",
            x0=min(start_times),
            x1=max(start_times),
            y0=critical_threshold,
            y1=critical_threshold,
            line=dict(color="red", width=1, dash="dash"),
            name="Critical Threshold",
        )

        # Update layout
        fig.update_layout(
            title=f"Gap Distribution - {report.exchange}/{report.symbol}/{report.timeframe}",
            xaxis_title="Date",
            yaxis_title="Gap Duration (hours)",
            height=500,
            showlegend=False,
        )

        return fig

    def _create_gap_table(self, report: GapAnalysisReport) -> html.Div:
        """Create gap details table.

        Args:
            report: Gap analysis report

        Returns:
            HTML table
        """
        if not report.gaps:
            return html.Div("No gaps detected")

        # Create table header
        header = html.Tr(
            [
                html.Th("Start Time"),
                html.Th("End Time"),
                html.Th("Duration"),
                html.Th("Severity"),
                html.Th("Filled"),
                html.Th("Fill Method"),
                html.Th("Fill Source"),
                html.Th("Fill Quality"),
            ]
        )

        # Create table rows
        rows = []
        for gap in report.gaps:
            # Set row color based on severity
            if gap.severity == GapSeverity.CRITICAL:
                style = {"backgroundColor": "rgba(255, 0, 0, 0.1)"}
            elif gap.severity == GapSeverity.MODERATE:
                style = {"backgroundColor": "rgba(255, 165, 0, 0.1)"}
            else:
                style = {"backgroundColor": "rgba(173, 216, 230, 0.1)"}

            row = html.Tr(
                [
                    html.Td(gap.start_time.strftime("%Y-%m-%d %H:%M")),
                    html.Td(gap.end_time.strftime("%Y-%m-%d %H:%M")),
                    html.Td(str(gap.duration)),
                    html.Td(gap.severity.value),
                    html.Td("Yes" if gap.filled else "No"),
                    html.Td(gap.fill_method or "-"),
                    html.Td(gap.fill_source or "-"),
                    html.Td(
                        f"{gap.fill_quality:.2f}"
                        if gap.fill_quality is not None
                        else "-"
                    ),
                ],
                style=style,
            )

            rows.append(row)

        # Create table
        table = html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        )

        return table

    def _create_quality_metrics_chart(self, report: GapAnalysisReport) -> go.Figure:
        """Create quality metrics chart.

        Args:
            report: Gap analysis report

        Returns:
            Plotly figure
        """
        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("Gap Severity Distribution", "Gap Fill Methods"),
        )

        # Add severity distribution pie chart
        severity_counts = {
            "minor": sum(1 for gap in report.gaps if gap.severity == GapSeverity.MINOR),
            "moderate": sum(
                1 for gap in report.gaps if gap.severity == GapSeverity.MODERATE
            ),
            "critical": sum(
                1 for gap in report.gaps if gap.severity == GapSeverity.CRITICAL
            ),
        }

        fig.add_trace(
            go.Pie(
                labels=list(severity_counts.keys()),
                values=list(severity_counts.values()),
                marker=dict(
                    colors=[
                        "rgba(173, 216, 230, 0.8)",
                        "rgba(255, 165, 0, 0.8)",
                        "rgba(255, 0, 0, 0.8)",
                    ]
                ),
                textinfo="label+percent",
                hole=0.3,
            ),
            row=1,
            col=1,
        )

        # Add fill methods bar chart
        fill_methods = {}
        for gap in report.gaps:
            if gap.filled and gap.fill_method:
                fill_methods[gap.fill_method] = fill_methods.get(gap.fill_method, 0) + 1

        if fill_methods:
            fig.add_trace(
                go.Bar(
                    x=list(fill_methods.keys()),
                    y=list(fill_methods.values()),
                    marker_color="royalblue",
                    text=list(fill_methods.values()),
                    textposition="auto",
                ),
                row=1,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title=f"Gap Analysis Metrics - {report.exchange}/{report.symbol}/{report.timeframe}",
            height=400,
            showlegend=False,
        )

        return fig

    def run_server(self, debug: bool = False, port: int = 8050) -> None:
        """Run the dashboard server.

        Args:
            debug: Whether to run in debug mode
            port: Port to run the server on
        """
        self.app.run_server(debug=debug, port=port)


def create_dashboard(connection_string: str) -> DataContinuityDashboard:
    """Create and return a data continuity dashboard.

    Args:
        connection_string: Database connection string

    Returns:
        DataContinuityDashboard instance
    """
    return DataContinuityDashboard(connection_string)


if __name__ == "__main__":
    # Example usage
    connection_string = (
        "postgresql://arb_user:strongpassword@localhost:5432/usdc_arbitrage"
    )
    dashboard = create_dashboard(connection_string)
    dashboard.run_server(debug=True)
