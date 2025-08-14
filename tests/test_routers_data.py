import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_data_retrieval_valid():
    # Test valid data retrieval
    response = client.get("/data/")
    assert response.status_code == 200
    # Add assertions on response data


def test_data_retrieval_not_found():
    # Test data not found
    response = client.get("/data/nonexistent")
    assert response.status_code == 404


# Add more edge cases
