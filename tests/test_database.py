"""Tests for database functionality."""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from api.database import get_db


def test_database_connection():
    """Test database connection."""
    db = next(get_db())
    assert db is not None


def test_database_empty_result():
    """Test query with no results."""
    # Example query (replace with actual code)
    # result = db.query(...).all()
    # assert len(result) == 0
    pass


# Add more edge cases
