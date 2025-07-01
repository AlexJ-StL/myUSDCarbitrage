import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_backtest_endpoint():
    # Test valid input
    response = client.post("/backtest/", json={"param1": "value1"})
    assert response.status_code == 200
    # Add more assertions


def test_backtest_invalid_input():
    # Test invalid input
    response = client.post("/backtest/", json={"invalid_param": "value"})
    assert response.status_code == 422  # Expect validation error


# Add more test cases for edge scenarios
