"""
On-demand report generation for arbitrage opportunities and strategy performance.

This module provides functionality to generate on-demand reports for arbitrage
opportunities and strategy performance metrics.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy.orm import Session

from src.api.database import DBConnector
from src.api.models import BacktestResult, Strategy
from src.reporting.report_generator import (
    generate_arbitrage_report,
    generate_strategy_performance_report,
)

# Configure logging
logger = logging.getLogger(__name__)


class OnDemandReportGenerator:
    """
    On-demand report generator for arbitrage opportunities and strategy performance.

    This class provides methods to generate HTML reports with detailed analysis
    of arbitrage opportunities and strategy performance metrics on demand.
    """

    def __init__(self, db_session: Session, db_connector: Optional[DBConnector] = None):
        """
        Initialize the on-demand report generator.

        Args:
            db_session: Database session
            db_connector: Optional database connector for raw SQL queries
        """
        self.db_session = db_session
        self.db_connector = db_connector

    def generate_arbitrage_opportunity_report(
        self,
        exchanges: List[str],
        symbol: str = "USDC/USD",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        threshold: float = 0.001,
        output_format: str = "html",
    ) -> Dict[str, Any]:
        """
        Generate an on-demand report for arbitrage opportunities.

        Args:
            exchanges: List of exchanges to compare
            symbol: Trading symbol to analyze
            start_time: Start time for analysis period (defaults to 24 hours ago)
            end_time: End time for analysis period (defaults to now)
            threshold: Minimum percentage difference to be considered an opportunity
            output_format: Output format (html, json, csv)

        Returns:
            Dict[str, Any]: Report content and metadata
        """
        try:
            # Set default times if not provided
            end_time = end_time or datetime.now()
            start_time = start_time or (end_time - timedelta(days=1))

            if start_time >= end_time:
                raise ValueError("Start time must be before end time")

            if len(exchanges) < 2:
                raise ValueError(
                    "At least two exchanges must be provided for comparison"
                )

            # Generate HTML report
            html_content = generate_arbitrage_report(
                db=self.db_session,
                start_time=start_time,
                end_time=end_time,
                exchanges=exchanges,
                symbol=symbol,
                threshold=threshold,
            )

            # Prepare response
            response = {
                "content": html_content,
                "content_type": "text/html",
                "generated_at": datetime.now().isoformat(),
                "report_type": "arbitrage_opportunity",
                "parameters": {
                    "exchanges": exchanges,
                    "symbol": symbol,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "threshold": threshold,
                },
            }

            # Handle different output formats
            if output_format.lower() == "json":
                # For JSON, we'd need to extract data from the HTML or generate JSON directly
                # This is a placeholder for future implementation
                response["content_type"] = "application/json"
                response["content"] = {"message": "JSON format not yet implemented"}

            elif output_format.lower() == "csv":
                # For CSV, we'd need to extract data and convert to CSV
                # This is a placeholder for future implementation
                response["content_type"] = "text/csv"
                response["content"] = "CSV format not yet implemented"

            return response

        except Exception as e:
            logger.exception(f"Error generating arbitrage opportunity report: {e}")
            return {
                "error": str(e),
                "content": f"<html><body><h1>Error Generating Report</h1><p>{str(e)}</p></body></html>",
                "content_type": "text/html",
                "generated_at": datetime.now().isoformat(),
            }

    def generate_strategy_performance_report(
        self,
        strategy_id: Optional[int] = None,
        backtest_id: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_benchmark: bool = False,
        benchmark_symbol: str = "BTC/USD",
        include_sections: Optional[List[str]] = None,
        output_format: str = "html",
    ) -> Dict[str, Any]:
        """
        Generate an on-demand report for strategy performance.

        Args:
            strategy_id: ID of the strategy to analyze
            backtest_id: ID of the specific backtest to analyze
            start_date: Start date for analysis period
            end_date: End date for analysis period
            include_benchmark: Whether to include benchmark comparison
            benchmark_symbol: Symbol to use as benchmark
            include_sections: Specific sections to include in the report
            output_format: Output format (html, json, csv)

        Returns:
            Dict[str, Any]: Report content and metadata
        """
        try:
            # Validate inputs
            if not strategy_id and not backtest_id:
                raise ValueError("Either strategy_id or backtest_id must be provided")

            # Get backtest result
            backtest_result = None

            if backtest_id:
                # Get specific backtest result
                backtest_result = (
                    self.db_session.query(BacktestResult)
                    .filter(BacktestResult.id == backtest_id)
                    .first()
                )

                if not backtest_result:
                    raise ValueError(f"Backtest with ID {backtest_id} not found")

            elif strategy_id:
                # Get latest backtest result for strategy
                query = self.db_session.query(BacktestResult).filter(
                    BacktestResult.strategy_id == strategy_id
                )

                # Filter by date range if provided
                if start_date:
                    query = query.filter(BacktestResult.start_date >= start_date)
                if end_date:
                    query = query.filter(BacktestResult.end_date <= end_date)

                # Get latest backtest
                backtest_result = query.order_by(
                    BacktestResult.created_at.desc()
                ).first()

                if not backtest_result:
                    raise ValueError(
                        f"No backtest results found for strategy ID {strategy_id}"
                    )

            # Extract backtest data
            backtest_data = {
                "strategy_name": self.db_session.query(Strategy.name)
                .filter(Strategy.id == backtest_result.strategy_id)
                .scalar()
                or "Unknown Strategy",
                "start_date": backtest_result.start_date.isoformat(),
                "end_date": backtest_result.end_date.isoformat(),
                "metrics": backtest_result.metrics or {},
                "results": backtest_result.results or {},
            }

            # Get benchmark data if requested
            benchmark_data = None
            if include_benchmark and self.db_connector:
                try:
                    # This is a simplified example - in a real implementation,
                    # you would fetch actual benchmark data from a reliable source
                    benchmark_df = self.db_connector.get_ohlcv_data_range(
                        exchange="coinbase",  # Example exchange
                        symbol=benchmark_symbol,
                        timeframe="1d",  # Daily timeframe
                        start_date=backtest_result.start_date,
                        end_date=backtest_result.end_date,
                    )

                    if not benchmark_df.empty:
                        # Calculate daily returns
                        benchmark_df["return"] = benchmark_df["close"].pct_change()

                        # Create benchmark data structure
                        benchmark_data = {
                            "name": benchmark_symbol,
                            "data": benchmark_df.reset_index().to_dict(
                                orient="records"
                            ),
                        }
                except Exception as e:
                    logger.warning(f"Failed to fetch benchmark data: {e}")
                    benchmark_data = None

            # Generate HTML report
            html_content = generate_strategy_performance_report(
                backtest_result=backtest_data,
                benchmark_data=benchmark_data,
                include_sections=include_sections,
            )

            # Prepare response
            response = {
                "content": html_content,
                "content_type": "text/html",
                "generated_at": datetime.now().isoformat(),
                "report_type": "strategy_performance",
                "parameters": {
                    "strategy_id": strategy_id,
                    "backtest_id": backtest_id,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "include_benchmark": include_benchmark,
                    "benchmark_symbol": benchmark_symbol if include_benchmark else None,
                    "include_sections": include_sections,
                },
            }

            # Handle different output formats
            if output_format.lower() == "json":
                # For JSON, we'd need to extract data from the HTML or generate JSON directly
                # This is a placeholder for future implementation
                response["content_type"] = "application/json"
                response["content"] = {"message": "JSON format not yet implemented"}

            elif output_format.lower() == "csv":
                # For CSV, we'd need to extract data and convert to CSV
                # This is a placeholder for future implementation
                response["content_type"] = "text/csv"
                response["content"] = "CSV format not yet implemented"

            return response

        except Exception as e:
            logger.exception(f"Error generating strategy performance report: {e}")
            return {
                "error": str(e),
                "content": f"<html><body><h1>Error Generating Report</h1><p>{str(e)}</p></body></html>",
                "content_type": "text/html",
                "generated_at": datetime.now().isoformat(),
            }


def get_on_demand_report_generator(
    db_session: Session, db_connector: Optional[DBConnector] = None
) -> OnDemandReportGenerator:
    """
    Factory function to create an on-demand report generator.

    Args:
        db_session: Database session
        db_connector: Optional database connector for raw SQL queries

    Returns:
        OnDemandReportGenerator: On-demand report generator instance
    """
    return OnDemandReportGenerator(db_session, db_connector)
