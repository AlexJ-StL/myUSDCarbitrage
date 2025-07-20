"""
System Integration Tests for USDC Arbitrage Backtesting Application.

This module contains end-to-end tests that verify the integration of all system components,
testing complete workflows, system recovery, data consistency, and user acceptance scenarios.
"""

import os
import sys
import unittest
import asyncio
import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add src directory to Python path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from api.main import app
from api.database import Base, get_db
from api.security import create_access_token
from api.models import User, Role, UserRole
from api.backtesting import BacktestEngine
from api.backtesting.engine import OrderSide
from api.strategies.strategy_manager import StrategyManager
from monitoring.health_monitoring import HealthMonitor
from monitoring.centralized_logging import initialize_logging, track_error


# Create a test database
TEST_DATABASE_URL = "sqlite:///./test_integration.db"
engine = create_engine(TEST_DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Override the get_db dependency
def override_get_db():
    """Override database session for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def test_client():
    """Create a test client for the FastAPI application."""
    # Create test database tables
    Base.metadata.create_all(bind=engine)

    # Setup test data
    setup_test_data()

    # Create test client
    client = TestClient(app)

    yield client

    # Cleanup
    Base.metadata.drop_all(bind=engine)


def setup_test_data():
    """Set up test data in the database."""
    db = TestingSessionLocal()

    try:
        # Create test user and roles
        admin_role = Role(
            name="admin",
            description="Administrator",
            permissions=json.dumps({"all": True}),
        )
        user_role = Role(
            name="user",
            description="Regular user",
            permissions=json.dumps({"read": True}),
        )

        test_user = User(
            username="testuser",
            email="test@example.com",
            password_hash="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
            is_active=True,
        )

        test_admin = User(
            username="testadmin",
            email="admin@example.com",
            password_hash="$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "password"
            is_active=True,
        )

        db.add_all([admin_role, user_role, test_user, test_admin])
        db.commit()

        # Assign roles
        user_role_assignment = UserRole(user_id=test_user.id, role_id=user_role.id)
        admin_role_assignment = UserRole(user_id=test_admin.id, role_id=admin_role.id)

        db.add_all([user_role_assignment, admin_role_assignment])
        db.commit()

        # Add sample market data
        add_sample_market_data(db)

        # Add sample strategies
        add_sample_strategies(db)

    finally:
        db.close()


def add_sample_market_data(db):
    """Add sample market data to the database."""
    # Create sample data for multiple exchanges
    exchanges = ["coinbase", "kraken", "binance"]
    start_date = datetime(2023, 1, 1, tzinfo=UTC)

    # Execute raw SQL for better performance with bulk inserts
    for exchange in exchanges:
        values = []
        for i in range(24 * 7):  # One week of hourly data
            timestamp = start_date + timedelta(hours=i)
            # Slightly different prices for each exchange to create arbitrage opportunities
            price_offset = {"coinbase": 0.0, "kraken": 0.001, "binance": -0.001}[
                exchange
            ]

            # Create price fluctuations
            base_price = 1.0 + 0.01 * (i % 24) / 24
            price = base_price + price_offset

            values.append(
                f"('{exchange}', 'USDC/USD', '1h', '{timestamp.isoformat()}', "
                f"{price}, {price + 0.005}, {price - 0.005}, {price + 0.001 * (i % 5 - 2)}, "
                f"{100.0 + 10.0 * (i % 10)}, 1.0, TRUE, '{datetime.now(UTC).isoformat()}')"
            )

        # Insert in batches
        batch_size = 100
        for i in range(0, len(values), batch_size):
            batch = values[i : i + batch_size]
            sql = f"""
            INSERT INTO market_data 
            (exchange, symbol, timeframe, timestamp, open, high, low, close, volume, quality_score, is_validated, created_at)
            VALUES {", ".join(batch)}
            """
            db.execute(text(sql))

        db.commit()


def add_sample_strategies(db):
    """Add sample strategies to the database."""
    strategies = [
        {
            "name": "Simple Arbitrage",
            "description": "Basic arbitrage strategy between exchanges",
            "strategy_type": "arbitrage",
            "parameters": json.dumps({
                "threshold": 0.001,
                "exchanges": ["coinbase", "kraken", "binance"],
                "symbols": ["USDC/USD"],
                "position_size": 1000.0,
            }),
            "version": 1,
            "is_active": True,
            "created_by": 1,
        },
        {
            "name": "Mean Reversion",
            "description": "Mean reversion strategy for USDC",
            "strategy_type": "mean_reversion",
            "parameters": json.dumps({
                "window": 24,
                "z_threshold": 2.0,
                "exchanges": ["coinbase"],
                "symbols": ["USDC/USD"],
                "position_size": 500.0,
            }),
            "version": 1,
            "is_active": True,
            "created_by": 1,
        },
    ]

    for strategy in strategies:
        sql = f"""
        INSERT INTO strategies 
        (name, description, strategy_type, parameters, version, is_active, created_by, created_at)
        VALUES 
        ('{strategy["name"]}', '{strategy["description"]}', '{strategy["strategy_type"]}', 
        '{strategy["parameters"]}', {strategy["version"]}, {strategy["is_active"]}, 
        {strategy["created_by"]}, '{datetime.now(UTC).isoformat()}')
        """
        db.execute(text(sql))

    db.commit()


@pytest.fixture
def auth_headers():
    """Create authentication headers for API requests."""
    access_token = create_access_token(
        data={"sub": "testuser", "roles": ["user"]}, expires_delta=timedelta(minutes=30)
    )
    return {"Authorization": f"Bearer {access_token}"}


@pytest.fixture
def admin_auth_headers():
    """Create admin authentication headers for API requests."""
    access_token = create_access_token(
        data={"sub": "testadmin", "roles": ["admin"]},
        expires_delta=timedelta(minutes=30),
    )
    return {"Authorization": f"Bearer {access_token}"}


class TestCompleteBacktestWorkflow:
    """Test the complete backtesting workflow from end to end."""

    def test_end_to_end_backtest_workflow(self, test_client, auth_headers):
        """Test the complete workflow from strategy creation to results analysis."""
        # Step 1: Create a new strategy
        strategy_data = {
            "name": "Test Integration Strategy",
            "description": "Strategy for integration testing",
            "strategy_type": "arbitrage",
            "parameters": {
                "threshold": 0.002,
                "exchanges": ["coinbase", "kraken"],
                "symbols": ["USDC/USD"],
                "position_size": 1000.0,
            },
        }

        response = test_client.post(
            "/api/strategies/", json=strategy_data, headers=auth_headers
        )
        assert response.status_code == 201
        strategy_id = response.json()["id"]

        # Step 2: Verify strategy was created
        response = test_client.get(
            f"/api/strategies/{strategy_id}", headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["name"] == strategy_data["name"]

        # Step 3: Run a backtest
        backtest_data = {
            "strategy_id": strategy_id,
            "start_date": "2023-01-01T00:00:00Z",
            "end_date": "2023-01-03T00:00:00Z",
            "initial_capital": 10000.0,
            "exchanges": ["coinbase", "kraken"],
            "timeframes": ["1h"],
        }

        response = test_client.post(
            "/api/backtest/run", json=backtest_data, headers=auth_headers
        )
        assert response.status_code == 202
        backtest_job_id = response.json()["job_id"]

        # Step 4: Poll for backtest completion
        max_retries = 10
        retry_count = 0
        backtest_id = None

        while retry_count < max_retries:
            response = test_client.get(
                f"/api/backtest/status/{backtest_job_id}", headers=auth_headers
            )
            assert response.status_code == 200
            status = response.json()["status"]

            if status == "completed":
                backtest_id = response.json()["backtest_id"]
                break

            retry_count += 1
            time.sleep(1)  # Wait before retrying

        assert backtest_id is not None, "Backtest did not complete in time"

        # Step 5: Get backtest results
        response = test_client.get(f"/api/results/{backtest_id}", headers=auth_headers)
        assert response.status_code == 200
        results = response.json()

        # Verify results contain expected fields
        assert "metrics" in results
        assert "total_return" in results["metrics"]
        assert "sharpe_ratio" in results["metrics"]
        assert "max_drawdown" in results["metrics"]

        # Step 6: Get visualization data
        response = test_client.get(
            f"/api/visualization/equity-curve/{backtest_id}", headers=auth_headers
        )
        assert response.status_code == 200
        assert "data" in response.json()

        # Step 7: Export results
        response = test_client.get(
            f"/api/data-export/backtest/{backtest_id}?format=json", headers=auth_headers
        )
        assert response.status_code == 200

        # Step 8: Compare with another strategy
        response = test_client.post(
            "/api/results/compare",
            json={
                "backtest_ids": [backtest_id, 1]
            },  # Compare with first sample strategy
            headers=auth_headers,
        )
        assert response.status_code in [
            200,
            404,
        ]  # 404 is acceptable if no other backtests exist

        # Step 9: Update strategy parameters
        update_data = {
            "parameters": {
                "threshold": 0.003,  # Changed threshold
                "exchanges": ["coinbase", "kraken"],
                "symbols": ["USDC/USD"],
                "position_size": 1000.0,
            }
        }

        response = test_client.patch(
            f"/api/strategies/{strategy_id}", json=update_data, headers=auth_headers
        )
        assert response.status_code == 200

        # Verify version was incremented
        response = test_client.get(
            f"/api/strategies/{strategy_id}", headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["version"] == 2

        # Step 10: Run backtest with updated strategy
        backtest_data["strategy_id"] = strategy_id

        response = test_client.post(
            "/api/backtest/run", json=backtest_data, headers=auth_headers
        )
        assert response.status_code == 202


class TestSystemRecoveryAndFailover:
    """Test system recovery and failover scenarios."""

    @patch("api.database.engine")
    def test_database_connection_recovery(self, mock_engine, test_client, auth_headers):
        """Test system recovery after database connection failure."""
        # Simulate database connection failure
        mock_engine.connect.side_effect = Exception("Database connection failed")

        # Attempt to access an endpoint that requires database
        response = test_client.get("/api/strategies/", headers=auth_headers)

        # Should return a 500 error
        assert response.status_code == 500

        # Restore database connection
        mock_engine.connect.side_effect = None

        # Retry the request, should succeed now
        response = test_client.get("/api/strategies/", headers=auth_headers)
        assert response.status_code == 200

    @patch("api.data_downloader.ExchangeDataDownloader.download_data")
    def test_data_source_failover(self, mock_download, test_client, auth_headers):
        """Test failover to alternative data sources when primary source fails."""
        # Simulate primary data source failure
        mock_download.side_effect = [
            Exception("Primary source failed"),  # First call fails
            pd.DataFrame({  # Second call succeeds with backup source
                "timestamp": pd.date_range(start="2023-01-01", periods=24, freq="1H"),
                "open": [1.0] * 24,
                "high": [1.1] * 24,
                "low": [0.9] * 24,
                "close": [1.0] * 24,
                "volume": [100.0] * 24,
            }),
        ]

        # Request data download
        response = test_client.post(
            "/api/data/download",
            json={
                "exchange": "coinbase",
                "symbol": "USDC/USD",
                "timeframe": "1h",
                "start_date": "2023-01-01T00:00:00Z",
                "end_date": "2023-01-02T00:00:00Z",
            },
            headers=auth_headers,
        )

        # Should succeed due to failover
        assert response.status_code == 202

        # Verify that both sources were attempted
        assert mock_download.call_count == 2


class TestDataConsistency:
    """Test data consistency across system components."""

    def test_data_integrity_across_components(self, test_client, auth_headers):
        """Test data consistency between database and API responses."""
        # Get data directly from database
        db = TestingSessionLocal()
        result = db.execute(text("SELECT COUNT(*) FROM market_data")).scalar()
        db_count = result
        db.close()

        # Get data count from API
        response = test_client.get("/api/data/stats", headers=auth_headers)
        assert response.status_code == 200
        api_count = response.json()["total_records"]

        # Verify counts match
        assert db_count == api_count, "Data count mismatch between database and API"

    def test_strategy_versioning_consistency(self, test_client, auth_headers):
        """Test strategy versioning consistency."""
        # Create a new strategy
        strategy_data = {
            "name": "Version Test Strategy",
            "description": "Strategy for testing versioning",
            "strategy_type": "arbitrage",
            "parameters": {
                "threshold": 0.002,
                "exchanges": ["coinbase", "kraken"],
                "symbols": ["USDC/USD"],
                "position_size": 1000.0,
            },
        }

        response = test_client.post(
            "/api/strategies/", json=strategy_data, headers=auth_headers
        )
        assert response.status_code == 201
        strategy_id = response.json()["id"]

        # Update strategy multiple times
        for i in range(3):
            update_data = {
                "parameters": {
                    "threshold": 0.002 + 0.001 * (i + 1),
                    "exchanges": ["coinbase", "kraken"],
                    "symbols": ["USDC/USD"],
                    "position_size": 1000.0,
                }
            }

            response = test_client.patch(
                f"/api/strategies/{strategy_id}", json=update_data, headers=auth_headers
            )
            assert response.status_code == 200

        # Get strategy versions
        response = test_client.get(
            f"/api/strategies/{strategy_id}/versions", headers=auth_headers
        )
        assert response.status_code == 200
        versions = response.json()

        # Should have 4 versions (initial + 3 updates)
        assert len(versions) == 4

        # Verify version numbers are sequential
        version_numbers = [v["version"] for v in versions]
        assert version_numbers == [1, 2, 3, 4]

        # Verify parameters are consistent with updates
        for i, version in enumerate(versions):
            if i == 0:
                assert version["parameters"]["threshold"] == 0.002
            else:
                assert version["parameters"]["threshold"] == 0.002 + 0.001 * i


class TestUserAcceptanceScenarios:
    """Test realistic user acceptance scenarios."""

    def test_realistic_arbitrage_workflow(self, test_client, auth_headers):
        """Test a realistic arbitrage strategy workflow."""
        # Step 1: Check available exchanges
        response = test_client.get("/api/data/exchanges", headers=auth_headers)
        assert response.status_code == 200
        exchanges = response.json()
        assert len(exchanges) >= 2, "Need at least 2 exchanges for arbitrage testing"

        # Step 2: Check data availability
        for exchange in exchanges[:2]:  # Use first two exchanges
            response = test_client.get(
                f"/api/data/availability?exchange={exchange}&symbol=USDC/USD",
                headers=auth_headers,
            )
            assert response.status_code == 200
            assert response.json()["has_data"] is True

        # Step 3: Create arbitrage strategy
        strategy_data = {
            "name": "UAT Arbitrage Strategy",
            "description": "User acceptance test for arbitrage",
            "strategy_type": "arbitrage",
            "parameters": {
                "threshold": 0.001,
                "exchanges": exchanges[:2],
                "symbols": ["USDC/USD"],
                "position_size": 1000.0,
                "max_positions": 5,
                "stop_loss": 0.005,
            },
        }

        response = test_client.post(
            "/api/strategies/", json=strategy_data, headers=auth_headers
        )
        assert response.status_code == 201
        strategy_id = response.json()["id"]

        # Step 4: Run backtest with realistic parameters
        backtest_data = {
            "strategy_id": strategy_id,
            "start_date": "2023-01-01T00:00:00Z",
            "end_date": "2023-01-07T00:00:00Z",  # One week
            "initial_capital": 100000.0,  # Realistic capital
            "exchanges": exchanges[:2],
            "timeframes": ["1h"],
        }

        response = test_client.post(
            "/api/backtest/run", json=backtest_data, headers=auth_headers
        )
        assert response.status_code == 202
        backtest_job_id = response.json()["job_id"]

        # Step 5: Wait for backtest completion
        max_retries = 10
        retry_count = 0
        backtest_id = None

        while retry_count < max_retries:
            response = test_client.get(
                f"/api/backtest/status/{backtest_job_id}", headers=auth_headers
            )
            assert response.status_code == 200
            status = response.json()["status"]

            if status == "completed":
                backtest_id = response.json()["backtest_id"]
                break

            retry_count += 1
            time.sleep(1)  # Wait before retrying

        assert backtest_id is not None, "Backtest did not complete in time"

        # Step 6: Generate report
        response = test_client.post(
            "/api/results/generate-report",
            json={"backtest_id": backtest_id, "format": "html"},
            headers=auth_headers,
        )
        assert response.status_code == 202
        report_job_id = response.json()["job_id"]

        # Step 7: Wait for report generation
        retry_count = 0
        report_url = None

        while retry_count < max_retries:
            response = test_client.get(
                f"/api/results/report-status/{report_job_id}", headers=auth_headers
            )
            assert response.status_code == 200
            status = response.json()["status"]

            if status == "completed":
                report_url = response.json()["report_url"]
                break

            retry_count += 1
            time.sleep(1)  # Wait before retrying

        assert report_url is not None, "Report generation did not complete in time"

        # Step 8: Download report
        response = test_client.get(report_url, headers=auth_headers)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_admin_monitoring_workflow(self, test_client, admin_auth_headers):
        """Test admin monitoring and management workflow."""
        # Step 1: Check system health
        response = test_client.get("/api/monitoring/health", headers=admin_auth_headers)
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

        # Step 2: Check system metrics
        response = test_client.get(
            "/api/monitoring/metrics", headers=admin_auth_headers
        )
        assert response.status_code == 200
        metrics = response.json()
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "database_connections" in metrics

        # Step 3: Check active users
        response = test_client.get(
            "/api/admin/users/active", headers=admin_auth_headers
        )
        assert response.status_code == 200

        # Step 4: Check recent errors
        response = test_client.get(
            "/api/monitoring/errors?limit=10", headers=admin_auth_headers
        )
        assert response.status_code == 200

        # Step 5: Check database status
        response = test_client.get(
            "/api/monitoring/database", headers=admin_auth_headers
        )
        assert response.status_code == 200
        assert response.json()["status"] == "connected"


if __name__ == "__main__":
    pytest.main(["-xvs", "test_system_integration.py"])
