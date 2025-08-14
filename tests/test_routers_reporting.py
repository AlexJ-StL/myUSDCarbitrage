"""
Integration tests for the reporting API endpoints.

This module contains tests for the reporting API endpoints,
including arbitrage report generation and strategy performance report generation.
"""

import os
import sys
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

# Add src directory to Python path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from reporting.routers.on_demand_reports import router as on_demand_router
from reporting.routers.reporting import router as reporting_router


class TestReportingRouters:
    """Test the reporting API endpoints."""

    @pytest.fixture
    def app(self):
        """Create a FastAPI test application."""
        app = FastAPI()
        app.include_router(reporting_router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client for the FastAPI application."""
        return TestClient(app)

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock(spec=Session)

    @pytest.fixture
    def mock_get_db(self, mock_db):
        """Create a mock get_db dependency."""

        def _get_db():
            return mock_db

        return _get_db

    @patch("src.reporting.routers.reporting.get_db")
    @patch("src.reporting.routers.reporting.generate_arbitrage_report")
    def test_create_arbitrage_report(
        self, mock_generate_report, mock_get_db, app, client
    ):
        """Test the create_arbitrage_report endpoint."""
        # Mock generate_arbitrage_report
        mock_generate_report.return_value = (
            "<html><body>Test Arbitrage Report</body></html>"
        )

        # Mock get_db
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Create test data
        request_data = {
            "exchanges": ["coinbase", "kraken"],
            "symbol": "USDC/USD",
            "threshold": 0.001,
        }

        # Make request
        response = client.post("/reports/arbitrage", json=request_data)

        # Check response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "Test Arbitrage Report" in response.text

        # Check that generate_arbitrage_report was called with correct arguments
        mock_generate_report.assert_called_once()
        args = mock_generate_report.call_args[1]
        assert args["db"] == mock_db
        assert args["exchanges"] == ["coinbase", "kraken"]
        assert args["symbol"] == "USDC/USD"
        assert args["threshold"] == 0.001

    @patch("src.reporting.routers.reporting.get_db")
    @patch("src.reporting.routers.reporting.generate_arbitrage_report")
    def test_create_arbitrage_report_with_dates(
        self, mock_generate_report, mock_get_db, app, client
    ):
        """Test the create_arbitrage_report endpoint with custom dates."""
        # Mock generate_arbitrage_report
        mock_generate_report.return_value = (
            "<html><body>Test Arbitrage Report</body></html>"
        )

        # Mock get_db
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Create test data with dates
        request_data = {
            "exchanges": ["coinbase", "kraken"],
            "symbol": "USDC/USD",
            "start_time": "2023-01-01T00:00:00Z",
            "end_time": "2023-01-02T00:00:00Z",
            "threshold": 0.001,
        }

        # Make request
        response = client.post("/reports/arbitrage", json=request_data)

        # Check response
        assert response.status_code == 200

        # Check that generate_arbitrage_report was called with correct arguments
        mock_generate_report.assert_called_once()
        args = mock_generate_report.call_args[1]
        assert args["start_time"].year == 2023
        assert args["start_time"].month == 1
        assert args["start_time"].day == 1
        assert args["end_time"].year == 2023
        assert args["end_time"].month == 1
        assert args["end_time"].day == 2

    @patch("src.reporting.routers.reporting.get_db")
    @patch("src.reporting.routers.reporting.generate_arbitrage_report")
    def test_create_arbitrage_report_validation_error(
        self, mock_generate_report, mock_get_db, app, client
    ):
        """Test validation error handling in create_arbitrage_report endpoint."""
        # Create invalid test data (missing exchanges)
        request_data = {
            "symbol": "USDC/USD",
            "threshold": 0.001,
        }

        # Make request
        response = client.post("/reports/arbitrage", json=request_data)

        # Check response
        assert response.status_code == 422  # Validation error

        # Check that generate_arbitrage_report was not called
        mock_generate_report.assert_not_called()

    @patch("src.reporting.routers.reporting.get_db")
    @patch("src.reporting.routers.reporting.generate_arbitrage_report")
    def test_create_arbitrage_report_invalid_dates(
        self, mock_generate_report, mock_get_db, app, client
    ):
        """Test invalid date handling in create_arbitrage_report endpoint."""
        # Mock generate_arbitrage_report to raise an exception
        mock_generate_report.side_effect = ValueError(
            "start_time must be before end_time."
        )

        # Mock get_db
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Create test data with invalid dates
        request_data = {
            "exchanges": ["coinbase", "kraken"],
            "symbol": "USDC/USD",
            "start_time": "2023-01-02T00:00:00Z",
            "end_time": "2023-01-01T00:00:00Z",  # End before start
            "threshold": 0.001,
        }

        # Make request
        response = client.post("/reports/arbitrage", json=request_data)

        # Check response
        assert response.status_code == 400
        assert "start_time must be before end_time" in response.json()["detail"]

    @patch("src.reporting.routers.reporting.get_db")
    @patch("src.reporting.routers.reporting.generate_arbitrage_report")
    def test_create_arbitrage_report_server_error(
        self, mock_generate_report, mock_get_db, app, client
    ):
        """Test server error handling in create_arbitrage_report endpoint."""
        # Mock generate_arbitrage_report to raise an exception
        mock_generate_report.side_effect = Exception("Internal server error")

        # Mock get_db
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        # Create test data
        request_data = {
            "exchanges": ["coinbase", "kraken"],
            "symbol": "USDC/USD",
            "threshold": 0.001,
        }

        # Make request
        response = client.post("/reports/arbitrage", json=request_data)

        # Check response
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]


class TestOnDemandReportingRouters:
    """Test the on-demand reporting API endpoints."""

    @pytest.fixture
    def app(self):
        """Create a FastAPI test application."""
        app = FastAPI()
        app.include_router(on_demand_router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create a test client for the FastAPI application."""
        return TestClient(app)

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session."""
        return MagicMock(spec=Session)

    @pytest.fixture
    def mock_db_connector(self):
        """Create a mock database connector."""
        return MagicMock()

    @patch("src.reporting.routers.on_demand_reports.get_db")
    @patch("src.reporting.routers.on_demand_reports.get_db_connector")
    @patch("src.reporting.routers.on_demand_reports.get_on_demand_report_generator")
    def test_create_arbitrage_report(
        self, mock_get_generator, mock_get_db_connector, mock_get_db, app, client
    ):
        """Test the create_arbitrage_report endpoint."""
        # Mock get_on_demand_report_generator
        mock_generator = MagicMock()
        mock_generator.generate_arbitrage_opportunity_report.return_value = {
            "content": "<html><body>Test Arbitrage Report</body></html>",
            "content_type": "text/html",
            "generated_at": datetime.now().isoformat(),
        }
        mock_get_generator.return_value = mock_generator

        # Mock get_db and get_db_connector
        mock_db = MagicMock()
        mock_db_connector = MagicMock()
        mock_get_db.return_value = mock_db
        mock_get_db_connector.return_value = mock_db_connector

        # Create test data
        request_data = {
            "exchanges": ["coinbase", "kraken"],
            "symbol": "USDC/USD",
            "threshold": 0.001,
            "output_format": "html",
        }

        # Make request
        response = client.post("/reports/arbitrage", json=request_data)

        # Check response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "Test Arbitrage Report" in response.text

        # Check that generate_arbitrage_opportunity_report was called with correct arguments
        mock_generator.generate_arbitrage_opportunity_report.assert_called_once_with(
            exchanges=["coinbase", "kraken"],
            symbol="USDC/USD",
            start_time=None,
            end_time=None,
            threshold=0.001,
            output_format="html",
        )

    @patch("src.reporting.routers.on_demand_reports.get_db")
    @patch("src.reporting.routers.on_demand_reports.get_db_connector")
    @patch("src.reporting.routers.on_demand_reports.get_on_demand_report_generator")
    def test_create_arbitrage_report_json_format(
        self, mock_get_generator, mock_get_db_connector, mock_get_db, app, client
    ):
        """Test the create_arbitrage_report endpoint with JSON output format."""
        # Mock get_on_demand_report_generator
        mock_generator = MagicMock()
        mock_generator.generate_arbitrage_opportunity_report.return_value = {
            "content": {"data": "test"},
            "content_type": "application/json",
            "generated_at": datetime.now().isoformat(),
        }
        mock_get_generator.return_value = mock_generator

        # Mock get_db and get_db_connector
        mock_db = MagicMock()
        mock_db_connector = MagicMock()
        mock_get_db.return_value = mock_db
        mock_get_db_connector.return_value = mock_db_connector

        # Create test data
        request_data = {
            "exchanges": ["coinbase", "kraken"],
            "symbol": "USDC/USD",
            "threshold": 0.001,
            "output_format": "json",
        }

        # Make request
        response = client.post("/reports/arbitrage", json=request_data)

        # Check response
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert response.json() == {"data": "test"}

    @patch("src.reporting.routers.on_demand_reports.get_db")
    @patch("src.reporting.routers.on_demand_reports.get_db_connector")
    @patch("src.reporting.routers.on_demand_reports.get_on_demand_report_generator")
    def test_create_arbitrage_report_error(
        self, mock_get_generator, mock_get_db_connector, mock_get_db, app, client
    ):
        """Test error handling in create_arbitrage_report endpoint."""
        # Mock get_on_demand_report_generator
        mock_generator = MagicMock()
        mock_generator.generate_arbitrage_opportunity_report.return_value = {
            "error": "Test error",
            "content": "<html><body>Error</body></html>",
            "content_type": "text/html",
            "generated_at": datetime.now().isoformat(),
        }
        mock_get_generator.return_value = mock_generator

        # Mock get_db and get_db_connector
        mock_db = MagicMock()
        mock_db_connector = MagicMock()
        mock_get_db.return_value = mock_db
        mock_get_db_connector.return_value = mock_db_connector

        # Create test data
        request_data = {
            "exchanges": ["coinbase", "kraken"],
            "symbol": "USDC/USD",
            "threshold": 0.001,
            "output_format": "html",
        }

        # Make request
        response = client.post("/reports/arbitrage", json=request_data)

        # Check response
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

    @patch("src.reporting.routers.on_demand_reports.get_db")
    @patch("src.reporting.routers.on_demand_reports.get_db_connector")
    @patch("src.reporting.routers.on_demand_reports.get_on_demand_report_generator")
    def test_create_strategy_report(
        self, mock_get_generator, mock_get_db_connector, mock_get_db, app, client
    ):
        """Test the create_strategy_report endpoint."""
        # Mock get_on_demand_report_generator
        mock_generator = MagicMock()
        mock_generator.generate_strategy_performance_report.return_value = {
            "content": "<html><body>Test Strategy Report</body></html>",
            "content_type": "text/html",
            "generated_at": datetime.now().isoformat(),
        }
        mock_get_generator.return_value = mock_generator

        # Mock get_db and get_db_connector
        mock_db = MagicMock()
        mock_db_connector = MagicMock()
        mock_get_db.return_value = mock_db
        mock_get_db_connector.return_value = mock_db_connector

        # Create test data
        request_data = {
            "strategy_id": 1,
            "include_benchmark": True,
            "benchmark_symbol": "BTC/USD",
            "include_sections": ["executive_summary", "performance_metrics"],
            "output_format": "html",
        }

        # Make request
        response = client.post("/reports/strategy", json=request_data)

        # Check response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "Test Strategy Report" in response.text

        # Check that generate_strategy_performance_report was called with correct arguments
        mock_generator.generate_strategy_performance_report.assert_called_once_with(
            strategy_id=1,
            backtest_id=None,
            start_date=None,
            end_date=None,
            include_benchmark=True,
            benchmark_symbol="BTC/USD",
            include_sections=["executive_summary", "performance_metrics"],
            output_format="html",
        )

    @patch("src.reporting.routers.on_demand_reports.get_db")
    @patch("src.reporting.routers.on_demand_reports.get_db_connector")
    @patch("src.reporting.routers.on_demand_reports.get_on_demand_report_generator")
    def test_create_strategy_report_validation_error(
        self, mock_get_generator, mock_get_db_connector, mock_get_db, app, client
    ):
        """Test validation error handling in create_strategy_report endpoint."""
        # Create invalid test data (missing both strategy_id and backtest_id)
        request_data = {
            "include_benchmark": True,
            "benchmark_symbol": "BTC/USD",
            "output_format": "html",
        }

        # Make request
        response = client.post("/reports/strategy", json=request_data)

        # Check response
        assert response.status_code == 400
        assert (
            "Either strategy_id or backtest_id must be provided"
            in response.json()["detail"]
        )

    @patch("src.reporting.routers.on_demand_reports.get_db")
    @patch("src.reporting.routers.on_demand_reports.get_db_connector")
    @patch("src.reporting.routers.on_demand_reports.get_on_demand_report_generator")
    def test_get_arbitrage_report(
        self, mock_get_generator, mock_get_db_connector, mock_get_db, app, client
    ):
        """Test the get_arbitrage_report endpoint."""
        # Mock get_on_demand_report_generator
        mock_generator = MagicMock()
        mock_generator.generate_arbitrage_opportunity_report.return_value = {
            "content": "<html><body>Test Arbitrage Report</body></html>",
            "content_type": "text/html",
            "generated_at": datetime.now().isoformat(),
        }
        mock_get_generator.return_value = mock_generator

        # Mock get_db and get_db_connector
        mock_db = MagicMock()
        mock_db_connector = MagicMock()
        mock_get_db.return_value = mock_db
        mock_get_db_connector.return_value = mock_db_connector

        # Make request
        response = client.get(
            "/reports/arbitrage?exchanges=coinbase&exchanges=kraken&symbol=USDC/USD&threshold=0.001"
        )

        # Check response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "Test Arbitrage Report" in response.text

        # Check that generate_arbitrage_opportunity_report was called with correct arguments
        mock_generator.generate_arbitrage_opportunity_report.assert_called_once_with(
            exchanges=["coinbase", "kraken"],
            symbol="USDC/USD",
            start_time=None,
            end_time=None,
            threshold=0.001,
            output_format="html",
        )

    @patch("src.reporting.routers.on_demand_reports.get_db")
    @patch("src.reporting.routers.on_demand_reports.get_db_connector")
    @patch("src.reporting.routers.on_demand_reports.get_on_demand_report_generator")
    def test_get_strategy_report(
        self, mock_get_generator, mock_get_db_connector, mock_get_db, app, client
    ):
        """Test the get_strategy_report endpoint."""
        # Mock get_on_demand_report_generator
        mock_generator = MagicMock()
        mock_generator.generate_strategy_performance_report.return_value = {
            "content": "<html><body>Test Strategy Report</body></html>",
            "content_type": "text/html",
            "generated_at": datetime.now().isoformat(),
        }
        mock_get_generator.return_value = mock_generator

        # Mock get_db and get_db_connector
        mock_db = MagicMock()
        mock_db_connector = MagicMock()
        mock_get_db.return_value = mock_db
        mock_get_db_connector.return_value = mock_db_connector

        # Make request
        response = client.get(
            "/reports/strategy?strategy_id=1&include_benchmark=true&benchmark_symbol=BTC/USD"
        )

        # Check response
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "Test Strategy Report" in response.text

        # Check that generate_strategy_performance_report was called with correct arguments
        mock_generator.generate_strategy_performance_report.assert_called_once_with(
            strategy_id=1,
            backtest_id=None,
            start_date=None,
            end_date=None,
            include_benchmark=True,
            benchmark_symbol="BTC/USD",
            include_sections=None,
            output_format="html",
        )
