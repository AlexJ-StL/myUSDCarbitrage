"""Gap detection and filling system for time series data in USDC arbitrage application."""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from .database import DBConnector

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("data_validation.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class GapSeverity(Enum):
    """Severity levels for time series gaps."""

    MINOR = "minor"  # Small gaps that don't significantly impact analysis
    MODERATE = "moderate"  # Gaps that may affect analysis but can be filled
    CRITICAL = "critical"  # Large gaps that significantly impact data quality


@dataclass
class GapInfo:
    """Information about a detected gap in time series data."""

    start_time: datetime
    end_time: datetime
    duration: timedelta
    severity: GapSeverity
    filled: bool = False
    fill_method: Optional[str] = None
    fill_source: Optional[str] = None
    fill_quality: Optional[float] = None


@dataclass
class GapAnalysisReport:
    """Comprehensive report of gap analysis for a dataset."""

    exchange: str
    symbol: str
    timeframe: str
    analysis_time: datetime
    total_gaps: int
    filled_gaps: int
    unfilled_gaps: int
    total_missing_points: int
    data_completeness: float  # Percentage of complete data
    gaps: List[GapInfo]
    summary: Dict[str, Any]


class DataSourcePriority:
    """Priority configuration for data sources used in gap filling."""

    def __init__(self, primary_source: str, fallback_sources: List[Tuple[str, float]]):
        """Initialize data source priority configuration.

        Args:
            primary_source: The primary data source (exchange name)
            fallback_sources: List of tuples with (source_name, quality_weight)
                where quality_weight is between 0 and 1
        """
        self.primary_source = primary_source
        self.fallback_sources = fallback_sources

    def get_ordered_sources(self) -> List[str]:
        """Get ordered list of data sources by priority."""
        sources = [self.primary_source]
        sources.extend(
            [
                source[0]
                for source in sorted(
                    self.fallback_sources, key=lambda x: x[1], reverse=True
                )
            ]
        )
        return sources


class GapDetectionSystem:
    """System for detecting and filling gaps in time series data."""

    def __init__(self, connection_string: str):
        """Initialize gap detection system.

        Args:
            connection_string: Database connection string
        """
        self.db = DBConnector(connection_string)
        self.engine = create_engine(connection_string)
        self.timeframe_deltas = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "30m": timedelta(minutes=30),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }
        self.severity_thresholds = {
            "1m": {
                "minor": timedelta(minutes=5),
                "moderate": timedelta(minutes=30),
                "critical": timedelta(hours=1),
            },
            "5m": {
                "minor": timedelta(minutes=15),
                "moderate": timedelta(hours=1),
                "critical": timedelta(hours=4),
            },
            "15m": {
                "minor": timedelta(minutes=30),
                "moderate": timedelta(hours=2),
                "critical": timedelta(hours=8),
            },
            "30m": {
                "minor": timedelta(hours=1),
                "moderate": timedelta(hours=4),
                "critical": timedelta(hours=12),
            },
            "1h": {
                "minor": timedelta(hours=2),
                "moderate": timedelta(hours=8),
                "critical": timedelta(days=1),
            },
            "4h": {
                "minor": timedelta(hours=8),
                "moderate": timedelta(days=1),
                "critical": timedelta(days=3),
            },
            "1d": {
                "minor": timedelta(days=2),
                "moderate": timedelta(days=5),
                "critical": timedelta(days=14),
            },
        }

        # Default data source priorities
        self.source_priorities = {
            "coinbase": DataSourcePriority(
                "coinbase", [("kraken", 0.9), ("binance", 0.8)]
            ),
            "kraken": DataSourcePriority(
                "kraken", [("coinbase", 0.9), ("binance", 0.8)]
            ),
            "binance": DataSourcePriority(
                "binance", [("coinbase", 0.9), ("kraken", 0.8)]
            ),
        }

    def detect_gaps(self, df: pd.DataFrame, timeframe: str) -> List[GapInfo]:
        """Detect gaps in time series data.

        Args:
            df: DataFrame with timestamp column
            timeframe: Time interval between data points (e.g., "1h", "1d")

        Returns:
            List of GapInfo objects describing detected gaps
        """
        if len(df) < 2:
            return []

        # Ensure DataFrame is sorted by timestamp
        df_sorted = df.sort_values("timestamp")

        # Calculate time differences between consecutive points
        time_diffs = df_sorted["timestamp"].diff().dropna()

        # Get expected time delta for this timeframe
        expected_delta = self.timeframe_deltas.get(timeframe, timedelta(hours=1))

        # Find gaps (where diff > expected_delta)
        gaps = []
        for i, (idx, diff) in enumerate(time_diffs.items()):
            if diff > expected_delta * 1.1:  # 10% tolerance
                start_time = df_sorted.iloc[i]["timestamp"]
                end_time = df_sorted.iloc[i + 1]["timestamp"]
                duration = end_time - start_time

                # Determine severity based on gap duration
                severity = self._determine_gap_severity(duration, timeframe)

                gap = GapInfo(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    severity=severity,
                )
                gaps.append(gap)

        return gaps

    def _determine_gap_severity(
        self, duration: timedelta, timeframe: str
    ) -> GapSeverity:
        """Determine the severity of a gap based on its duration and timeframe.

        Args:
            duration: Gap duration
            timeframe: Time interval between data points

        Returns:
            GapSeverity enum value
        """
        thresholds = self.severity_thresholds.get(
            timeframe,
            self.severity_thresholds["1h"],  # Default to 1h if timeframe not found
        )

        if duration >= thresholds["critical"]:
            return GapSeverity.CRITICAL
        elif duration >= thresholds["moderate"]:
            return GapSeverity.MODERATE
        else:
            return GapSeverity.MINOR

    def fill_gaps(
        self, exchange: str, symbol: str, timeframe: str, gaps: List[GapInfo]
    ) -> Tuple[pd.DataFrame, List[GapInfo]]:
        """Fill gaps in time series data using multiple sources.

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Time interval between data points
            gaps: List of detected gaps

        Returns:
            Tuple of (filled_data_df, updated_gaps)
        """
        if not gaps:
            return pd.DataFrame(), gaps

        # Get data for the primary exchange
        primary_df = self.db.get_ohlcv_data(exchange, symbol, timeframe)
        if primary_df.empty:
            logger.warning(f"No data found for {exchange}/{symbol}/{timeframe}")
            return pd.DataFrame(), gaps

        # Get ordered list of fallback sources
        fallback_sources = self.source_priorities.get(
            exchange, DataSourcePriority(exchange, [])
        ).get_ordered_sources()

        filled_data = []
        updated_gaps = []

        for gap in gaps:
            # Try to fill the gap using fallback sources
            gap_filled = False

            for source in fallback_sources:
                if source == exchange:
                    continue  # Skip primary source

                try:
                    # Get data from fallback source for the gap period
                    source_df = self.db.get_ohlcv_data(source, symbol, timeframe)

                    if source_df.empty:
                        continue

                    # Filter data for the gap period with a buffer
                    buffer = self.timeframe_deltas.get(timeframe, timedelta(hours=1))
                    gap_data = source_df[
                        (source_df["timestamp"] >= gap.start_time - buffer)
                        & (source_df["timestamp"] <= gap.end_time + buffer)
                    ]

                    if not gap_data.empty:
                        # Check if we have data points within the gap
                        in_gap_data = gap_data[
                            (gap_data["timestamp"] > gap.start_time)
                            & (gap_data["timestamp"] < gap.end_time)
                        ]

                        if not in_gap_data.empty:
                            # We found data to fill the gap
                            filled_data.append(in_gap_data)

                            # Update gap info
                            gap.filled = True
                            gap.fill_method = "alternative_source"
                            gap.fill_source = source
                            gap.fill_quality = 0.9  # High quality as it's real data
                            gap_filled = True
                            break

                except Exception as e:
                    logger.error(f"Error filling gap from {source}: {str(e)}")

            if not gap_filled:
                # If we couldn't fill from alternative sources, use interpolation
                try:
                    # Get data points before and after the gap
                    before_gap = primary_df[
                        primary_df["timestamp"] <= gap.start_time
                    ].iloc[-1:]
                    after_gap = primary_df[
                        primary_df["timestamp"] >= gap.end_time
                    ].iloc[:1]

                    if not before_gap.empty and not after_gap.empty:
                        # Create a DataFrame with just the endpoints
                        endpoints = pd.concat([before_gap, after_gap])

                        # Generate timestamps for missing points
                        expected_delta = self.timeframe_deltas.get(
                            timeframe, timedelta(hours=1)
                        )
                        missing_timestamps = []

                        current_time = gap.start_time + expected_delta
                        while current_time < gap.end_time:
                            missing_timestamps.append(current_time)
                            current_time += expected_delta

                        if missing_timestamps:
                            # Create a DataFrame with the missing timestamps
                            missing_df = pd.DataFrame({"timestamp": missing_timestamps})

                            # Merge with endpoints
                            interp_df = pd.concat([endpoints, missing_df])
                            interp_df = interp_df.sort_values("timestamp")

                            # Interpolate missing values
                            for col in ["open", "high", "low", "close", "volume"]:
                                interp_df[col] = interp_df[col].interpolate(
                                    method="linear"
                                )

                            # Extract only the filled points
                            filled_points = interp_df[
                                interp_df["timestamp"].isin(missing_timestamps)
                            ]
                            filled_data.append(filled_points)

                            # Update gap info
                            gap.filled = True
                            gap.fill_method = "linear_interpolation"
                            gap.fill_source = "interpolation"

                            # Calculate fill quality based on gap size
                            # Longer gaps have lower quality interpolation
                            points_count = len(missing_timestamps)
                            gap.fill_quality = max(0.1, 1.0 - (0.1 * points_count))

                except Exception as e:
                    logger.error(f"Error interpolating gap: {str(e)}")

            updated_gaps.append(gap)

        # Combine all filled data
        if filled_data:
            filled_df = pd.concat(filled_data)
            return filled_df, updated_gaps
        else:
            return pd.DataFrame(), updated_gaps

    def analyze_gaps(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> GapAnalysisReport:
        """Perform comprehensive gap analysis on a dataset.

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Time interval between data points
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis

        Returns:
            GapAnalysisReport object with analysis results
        """
        try:
            # Get data from database
            df = self.db.get_ohlcv_data(exchange, symbol, timeframe)

            if df.empty:
                logger.warning(f"No data found for {exchange}/{symbol}/{timeframe}")
                return GapAnalysisReport(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    analysis_time=datetime.now(),
                    total_gaps=0,
                    filled_gaps=0,
                    unfilled_gaps=0,
                    total_missing_points=0,
                    data_completeness=0.0,
                    gaps=[],
                    summary={"error": "No data found"},
                )

            # Filter by date range if provided
            if start_date:
                df = df[df["timestamp"] >= start_date]
            if end_date:
                df = df[df["timestamp"] <= end_date]

            # Detect gaps
            gaps = self.detect_gaps(df, timeframe)

            if not gaps:
                return GapAnalysisReport(
                    exchange=exchange,
                    symbol=symbol,
                    timeframe=timeframe,
                    analysis_time=datetime.now(),
                    total_gaps=0,
                    filled_gaps=0,
                    unfilled_gaps=0,
                    total_missing_points=0,
                    data_completeness=100.0,
                    gaps=[],
                    summary={"status": "No gaps detected"},
                )

            # Fill gaps
            filled_df, updated_gaps = self.fill_gaps(exchange, symbol, timeframe, gaps)

            # Calculate statistics
            total_gaps = len(gaps)
            filled_gaps = sum(1 for gap in updated_gaps if gap.filled)
            unfilled_gaps = total_gaps - filled_gaps

            # Calculate missing points
            expected_delta = self.timeframe_deltas.get(timeframe, timedelta(hours=1))
            total_missing_points = 0

            for gap in gaps:
                # Calculate how many points should be in this gap
                expected_points = (
                    int((gap.end_time - gap.start_time) / expected_delta) - 1
                )
                total_missing_points += max(0, expected_points)

            # Calculate data completeness
            if start_date and end_date:
                expected_total_points = (
                    int((end_date - start_date) / expected_delta) + 1
                )
            else:
                # Use min and max timestamps in the data
                min_time = df["timestamp"].min()
                max_time = df["timestamp"].max()
                expected_total_points = int((max_time - min_time) / expected_delta) + 1

            actual_points = len(df)
            data_completeness = min(
                100.0, (actual_points / expected_total_points) * 100
            )

            # Create summary
            severity_counts = {
                "minor": sum(1 for gap in gaps if gap.severity == GapSeverity.MINOR),
                "moderate": sum(
                    1 for gap in gaps if gap.severity == GapSeverity.MODERATE
                ),
                "critical": sum(
                    1 for gap in gaps if gap.severity == GapSeverity.CRITICAL
                ),
            }

            fill_method_counts = {}
            for gap in updated_gaps:
                if gap.filled and gap.fill_method:
                    fill_method_counts[gap.fill_method] = (
                        fill_method_counts.get(gap.fill_method, 0) + 1
                    )

            summary = {
                "severity_counts": severity_counts,
                "fill_methods": fill_method_counts,
                "largest_gap": str(
                    max((gap.duration for gap in gaps), default=timedelta(0))
                ),
                "avg_gap_size": str(
                    sum((gap.duration for gap in gaps), timedelta(0))
                    / max(1, len(gaps))
                ),
            }

            return GapAnalysisReport(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                analysis_time=datetime.now(),
                total_gaps=total_gaps,
                filled_gaps=filled_gaps,
                unfilled_gaps=unfilled_gaps,
                total_missing_points=total_missing_points,
                data_completeness=data_completeness,
                gaps=updated_gaps,
                summary=summary,
            )

        except Exception as e:
            logger.error(
                f"Gap analysis failed for {exchange}/{symbol}/{timeframe}: {str(e)}"
            )
            return GapAnalysisReport(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                analysis_time=datetime.now(),
                total_gaps=0,
                filled_gaps=0,
                unfilled_gaps=0,
                total_missing_points=0,
                data_completeness=0.0,
                gaps=[],
                summary={"error": str(e)},
            )

    def save_filled_data(
        self, filled_df: pd.DataFrame, exchange: str, symbol: str, timeframe: str
    ) -> bool:
        """Save filled data to the database.

        Args:
            filled_df: DataFrame with filled data points
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Time interval between data points

        Returns:
            True if successful, False otherwise
        """
        if filled_df.empty:
            return False

        try:
            # Convert DataFrame to list of records for database insertion
            records = []
            for _, row in filled_df.iterrows():
                record = [
                    int(row["timestamp"].timestamp() * 1000),  # Convert to milliseconds
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["volume"]),
                ]
                records.append(record)

            # Use the Database class to insert data
            from .database import Database

            db = Database()
            db.insert_data(exchange, symbol, timeframe, records)
            db.close()

            logger.info(
                f"Saved {len(records)} filled data points for {exchange}/{symbol}/{timeframe}"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving filled data: {str(e)}")
            return False

    def generate_gap_report(
        self, report: GapAnalysisReport, output_format: str = "html"
    ) -> str:
        """Generate a formatted report of gap analysis.

        Args:
            report: GapAnalysisReport object
            output_format: Output format ("html", "json", "text")

        Returns:
            Formatted report as string
        """
        if output_format == "json":
            import json

            # Convert to dictionary
            report_dict = {
                "exchange": report.exchange,
                "symbol": report.symbol,
                "timeframe": report.timeframe,
                "analysis_time": report.analysis_time.isoformat(),
                "total_gaps": report.total_gaps,
                "filled_gaps": report.filled_gaps,
                "unfilled_gaps": report.unfilled_gaps,
                "total_missing_points": report.total_missing_points,
                "data_completeness": report.data_completeness,
                "gaps": [
                    {
                        "start_time": gap.start_time.isoformat(),
                        "end_time": gap.end_time.isoformat(),
                        "duration": str(gap.duration),
                        "severity": gap.severity.value,
                        "filled": gap.filled,
                        "fill_method": gap.fill_method,
                        "fill_source": gap.fill_source,
                        "fill_quality": gap.fill_quality,
                    }
                    for gap in report.gaps
                ],
                "summary": report.summary,
            }

            return json.dumps(report_dict, indent=2)

        elif output_format == "html":
            # Create HTML report
            html = f"""
            <html>
            <head>
                <title>Gap Analysis Report - {report.exchange}/{report.symbol}/{report.timeframe}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .critical {{ color: #d9534f; }}
                    .moderate {{ color: #f0ad4e; }}
                    .minor {{ color: #5bc0de; }}
                    .summary {{ background-color: #eee; padding: 15px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h1>Gap Analysis Report</h1>
                <div class="summary">
                    <p><strong>Exchange:</strong> {report.exchange}</p>
                    <p><strong>Symbol:</strong> {report.symbol}</p>
                    <p><strong>Timeframe:</strong> {report.timeframe}</p>
                    <p><strong>Analysis Time:</strong> {report.analysis_time}</p>
                    <p><strong>Data Completeness:</strong> {report.data_completeness:.2f}%</p>
                    <p><strong>Total Gaps:</strong> {report.total_gaps}</p>
                    <p><strong>Filled Gaps:</strong> {report.filled_gaps}</p>
                    <p><strong>Unfilled Gaps:</strong> {report.unfilled_gaps}</p>
                    <p><strong>Total Missing Points:</strong> {report.total_missing_points}</p>
                </div>
                
                <h2>Gap Summary</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """

            # Add summary items
            for key, value in report.summary.items():
                if isinstance(value, dict):
                    html += f"<tr><td>{key}</td><td>{str(value)}</td></tr>"
                else:
                    html += f"<tr><td>{key}</td><td>{value}</td></tr>"

            html += """
                </table>
                
                <h2>Gap Details</h2>
                <table>
                    <tr>
                        <th>Start Time</th>
                        <th>End Time</th>
                        <th>Duration</th>
                        <th>Severity</th>
                        <th>Filled</th>
                        <th>Fill Method</th>
                        <th>Fill Source</th>
                        <th>Fill Quality</th>
                    </tr>
            """

            # Add gap details
            for gap in report.gaps:
                severity_class = gap.severity.value
                html += f"""
                    <tr>
                        <td>{gap.start_time}</td>
                        <td>{gap.end_time}</td>
                        <td>{gap.duration}</td>
                        <td class="{severity_class}">{gap.severity.value}</td>
                        <td>{"Yes" if gap.filled else "No"}</td>
                        <td>{gap.fill_method or "-"}</td>
                        <td>{gap.fill_source or "-"}</td>
                        <td>{f"{gap.fill_quality:.2f}" if gap.fill_quality is not None else "-"}</td>
                    </tr>
                """

            html += """
                </table>
            </body>
            </html>
            """

            return html

        else:  # text format
            lines = [
                f"Gap Analysis Report - {report.exchange}/{report.symbol}/{report.timeframe}",
                f"Analysis Time: {report.analysis_time}",
                f"Data Completeness: {report.data_completeness:.2f}%",
                f"Total Gaps: {report.total_gaps}",
                f"Filled Gaps: {report.filled_gaps}",
                f"Unfilled Gaps: {report.unfilled_gaps}",
                f"Total Missing Points: {report.total_missing_points}",
                "",
                "Gap Summary:",
            ]

            for key, value in report.summary.items():
                lines.append(f"  {key}: {value}")

            lines.append("")
            lines.append("Gap Details:")

            for i, gap in enumerate(report.gaps):
                lines.extend(
                    [
                        f"  Gap {i+1}:",
                        f"    Start Time: {gap.start_time}",
                        f"    End Time: {gap.end_time}",
                        f"    Duration: {gap.duration}",
                        f"    Severity: {gap.severity.value}",
                        f"    Filled: {'Yes' if gap.filled else 'No'}",
                        f"    Fill Method: {gap.fill_method or '-'}",
                        f"    Fill Source: {gap.fill_source or '-'}",
                        f"    Fill Quality: {f'{gap.fill_quality:.2f}' if gap.fill_quality is not None else '-'}",
                        "",
                    ]
                )

            return "\n".join(lines)

    def setup_data_source_priority(
        self,
        exchange: str,
        primary_source: str,
        fallback_sources: List[Tuple[str, float]],
    ) -> None:
        """Configure data source priority for gap filling.

        Args:
            exchange: Exchange to configure priority for
            primary_source: Primary data source
            fallback_sources: List of tuples with (source_name, quality_weight)
        """
        self.source_priorities[exchange] = DataSourcePriority(
            primary_source, fallback_sources
        )

    def run_gap_detection_and_filling(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        save_filled_data: bool = True,
        generate_report: bool = True,
        report_format: str = "html",
        report_path: Optional[str] = None,
    ) -> Tuple[GapAnalysisReport, Optional[str]]:
        """Run the complete gap detection and filling workflow.

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Time interval between data points
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            save_filled_data: Whether to save filled data to database
            generate_report: Whether to generate a report
            report_format: Report format ("html", "json", "text")
            report_path: Path to save the report to

        Returns:
            Tuple of (GapAnalysisReport, report_content)
        """
        # Run gap analysis
        report = self.analyze_gaps(exchange, symbol, timeframe, start_date, end_date)

        # Save filled data if requested
        if save_filled_data and report.filled_gaps > 0:
            # Get filled data
            filled_df = pd.DataFrame()
            for gap in report.gaps:
                if gap.filled:
                    # Get data for this gap
                    start_time = gap.start_time
                    end_time = gap.end_time

                    if gap.fill_source and gap.fill_source != "interpolation":
                        # Get data from alternative source
                        source_df = self.db.get_ohlcv_data(
                            gap.fill_source, symbol, timeframe
                        )
                        gap_data = source_df[
                            (source_df["timestamp"] > start_time)
                            & (source_df["timestamp"] < end_time)
                        ]
                        filled_df = pd.concat([filled_df, gap_data])
                    else:
                        # For interpolated data, we need to regenerate it
                        # Get data points before and after the gap
                        primary_df = self.db.get_ohlcv_data(exchange, symbol, timeframe)
                        before_gap = primary_df[
                            primary_df["timestamp"] <= start_time
                        ].iloc[-1:]
                        after_gap = primary_df[
                            primary_df["timestamp"] >= end_time
                        ].iloc[:1]

                        if not before_gap.empty and not after_gap.empty:
                            # Create a DataFrame with just the endpoints
                            endpoints = pd.concat([before_gap, after_gap])

                            # Generate timestamps for missing points
                            expected_delta = self.timeframe_deltas.get(
                                timeframe, timedelta(hours=1)
                            )
                            missing_timestamps = []

                            current_time = start_time + expected_delta
                            while current_time < end_time:
                                missing_timestamps.append(current_time)
                                current_time += expected_delta

                            if missing_timestamps:
                                # Create a DataFrame with the missing timestamps
                                missing_df = pd.DataFrame(
                                    {"timestamp": missing_timestamps}
                                )

                                # Merge with endpoints
                                interp_df = pd.concat([endpoints, missing_df])
                                interp_df = interp_df.sort_values("timestamp")

                                # Interpolate missing values
                                for col in ["open", "high", "low", "close", "volume"]:
                                    interp_df[col] = interp_df[col].interpolate(
                                        method="linear"
                                    )

                                # Extract only the filled points
                                filled_points = interp_df[
                                    interp_df["timestamp"].isin(missing_timestamps)
                                ]
                                filled_df = pd.concat([filled_df, filled_points])

            # Save filled data
            if not filled_df.empty:
                self.save_filled_data(filled_df, exchange, symbol, timeframe)

        # Generate report if requested
        report_content = None
        if generate_report:
            report_content = self.generate_gap_report(report, report_format)

            if report_path:
                try:
                    with open(report_path, "w") as f:
                        f.write(report_content)
                    logger.info(f"Gap analysis report saved to {report_path}")
                except Exception as e:
                    logger.error(f"Error saving report: {str(e)}")

        return report, report_content
