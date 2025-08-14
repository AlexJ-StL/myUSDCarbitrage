"""
Unit tests for the Jinja filters used in report templates.

This module contains tests for the custom Jinja filters used in the report templates.
"""

import os
import sys
import unittest
from datetime import UTC, datetime

import pytest

# Add src directory to Python path for imports
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from reporting.jinja_filters import (
    format_currency,
    format_date,
    format_datetime,
    format_number,
    format_percentage,
    setup_jinja_filters,
)


class TestJinjaFilters(unittest.TestCase):
    """Test the custom Jinja filters."""

    def test_format_number(self):
        """Test the format_number filter."""
        # Test with default precision
        self.assertEqual(format_number(1234.5678), "1,234.57")

        # Test with custom precision
        self.assertEqual(format_number(1234.5678, 3), "1,234.568")

        # Test with zero precision
        self.assertEqual(format_number(1234.5678, 0), "1,235")

        # Test with negative number
        self.assertEqual(format_number(-1234.5678), "-1,234.57")

        # Test with zero
        self.assertEqual(format_number(0), "0.00")

        # Test with None
        self.assertEqual(format_number(None), "N/A")

        # Test with string
        self.assertEqual(format_number("1234.5678"), "1,234.57")

        # Test with invalid string
        self.assertEqual(format_number("invalid"), "N/A")

    def test_format_currency(self):
        """Test the format_currency filter."""
        # Test with default symbol and precision
        self.assertEqual(format_currency(1234.5678), "$1,234.57")

        # Test with custom symbol
        self.assertEqual(format_currency(1234.5678, "€"), "€1,234.57")

        # Test with custom precision
        self.assertEqual(format_currency(1234.5678, "$", 3), "$1,234.568")

        # Test with negative number
        self.assertEqual(format_currency(-1234.5678), "-$1,234.57")

        # Test with zero
        self.assertEqual(format_currency(0), "$0.00")

        # Test with None
        self.assertEqual(format_currency(None), "N/A")

        # Test with string
        self.assertEqual(format_currency("1234.5678"), "$1,234.57")

        # Test with invalid string
        self.assertEqual(format_currency("invalid"), "N/A")

    def test_format_percentage(self):
        """Test the format_percentage filter."""
        # Test with default precision
        self.assertEqual(format_percentage(0.1234), "12.34%")

        # Test with custom precision
        self.assertEqual(format_percentage(0.1234, 1), "12.3%")

        # Test with zero precision
        self.assertEqual(format_percentage(0.1234, 0), "12%")

        # Test with negative number
        self.assertEqual(format_percentage(-0.1234), "-12.34%")

        # Test with zero
        self.assertEqual(format_percentage(0), "0.00%")

        # Test with None
        self.assertEqual(format_percentage(None), "N/A")

        # Test with string
        self.assertEqual(format_percentage("0.1234"), "12.34%")

        # Test with invalid string
        self.assertEqual(format_percentage("invalid"), "N/A")

        # Test with value already in percentage
        self.assertEqual(format_percentage(12.34, is_percentage=True), "12.34%")

    def test_format_date(self):
        """Test the format_date filter."""
        # Test with datetime object
        dt = datetime(2023, 1, 1, tzinfo=UTC)
        self.assertEqual(format_date(dt), "2023-01-01")

        # Test with custom format
        self.assertEqual(format_date(dt, "%m/%d/%Y"), "01/01/2023")

        # Test with string
        self.assertEqual(format_date("2023-01-01"), "2023-01-01")

        # Test with invalid string
        self.assertEqual(format_date("invalid"), "N/A")

        # Test with None
        self.assertEqual(format_date(None), "N/A")

    def test_format_datetime(self):
        """Test the format_datetime filter."""
        # Test with datetime object
        dt = datetime(2023, 1, 1, 12, 34, 56, tzinfo=UTC)
        self.assertEqual(format_datetime(dt), "2023-01-01 12:34:56")

        # Test with custom format
        self.assertEqual(
            format_datetime(dt, "%m/%d/%Y %I:%M %p"), "01/01/2023 12:34 PM"
        )

        # Test with string
        self.assertEqual(format_datetime("2023-01-01T12:34:56Z"), "2023-01-01 12:34:56")

        # Test with invalid string
        self.assertEqual(format_datetime("invalid"), "N/A")

        # Test with None
        self.assertEqual(format_datetime(None), "N/A")

    def test_setup_jinja_filters(self):
        """Test the setup_jinja_filters function."""
        # Create mock environment
        mock_env = MagicMock()
        mock_env.filters = {}

        # Set up filters
        setup_jinja_filters(mock_env)

        # Check that filters were added
        self.assertIn("format_number", mock_env.filters)
        self.assertIn("format_currency", mock_env.filters)
        self.assertIn("format_percentage", mock_env.filters)
        self.assertIn("format_date", mock_env.filters)
        self.assertIn("format_datetime", mock_env.filters)


# Add MagicMock for the test_setup_jinja_filters test
from unittest.mock import MagicMock


if __name__ == "__main__":
    unittest.main()
