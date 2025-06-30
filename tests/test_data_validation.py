import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from api.data_validation import DataValidator


@pytest.fixture
def sample_data():
    """Creates a sample DataFrame for testing."""
    data = {
        "timestamp": pd.to_datetime(
            [
                "2023-01-01 00:00:00",
                "2023-01-01 01:00:00",
                "2023-01-01 02:00:00",
                "2023-01-01 03:00:00",
            ]
        ),
        "open": [1.0, 1.1, 0.9, 1.2],
        "high": [1.2, 1.2, 1.1, 1.3],
        "low": [0.9, 1.0, 0.8, 1.1],
        "close": [1.1, 1.0, 1.0, 1.25],
        "volume": [100, 150, 80, 200],
    }
    return pd.DataFrame(data)


@pytest.fixture
def validator():
    """Returns a DataValidator instance."""
    connection_string = (
        "postgresql://arb_user:strongpassword@localhost:5432/usdc_arbitrage"
    )
    return DataValidator(connection_string)


def test_check_price_integrity_valid(validator, sample_data):
    """Tests that valid data passes the price integrity check."""
    errors = validator.check_price_integrity(sample_data)
    assert not errors


def test_check_price_integrity_invalid(validator, sample_data):
    """Tests that invalid data fails the price integrity check."""
    invalid_data = sample_data.copy()
    invalid_data.loc[1, "high"] = 0.9  # high < low
    errors = validator.check_price_integrity(invalid_data)
    assert len(errors) == 3
    assert "High < Low" in errors[0]
    assert "Open > High" in errors[1]
    assert "Close > High" in errors[2]


def test_check_time_continuity(validator, sample_data):
    """Tests the time continuity check."""
    # Test with valid data
    gaps = validator.check_time_continuity(sample_data, "1h")
    assert not gaps

    # Test with invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[2, "timestamp"] = invalid_data.loc[2, "timestamp"] + timedelta(
        hours=2
    )
    gaps = validator.check_time_continuity(invalid_data, "1h")
    assert len(gaps) == 1


def test_detect_outliers(validator, sample_data):
    """Tests the outlier detection."""
    # Test with valid data
    outliers = validator.detect_outliers(sample_data)
    assert not outliers

    # Test with invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[1, "close"] = 100
    outliers = validator.detect_outliers(invalid_data)
    assert len(outliers) == 1


def test_detect_volume_anomalies(validator, sample_data):
    """Tests the volume anomaly detection."""
    # Test with valid data
    anomalies = validator.detect_volume_anomalies(sample_data)
    assert not anomalies

    # Test with invalid data
    invalid_data = sample_data.copy()
    invalid_data.loc[1, "volume"] = 10000
    anomalies = validator.detect_volume_anomalies(invalid_data)
    assert len(anomalies) == 1


@pytest.fixture
def changepoint_data():
    """Creates a larger DataFrame with a changepoint for testing."""
    np.random.seed(42)
    data1 = np.random.normal(loc=1.0, scale=0.01, size=100)
    data2 = np.random.normal(loc=1.5, scale=0.01, size=100)
    data = np.concatenate([data1, data2])
    timestamps = pd.to_datetime(
        pd.date_range(start="2023-01-01", periods=200, freq="h")
    )
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": data,
            "high": data + 0.01,
            "low": data - 0.01,
            "close": data,
            "volume": 100,
        }
    )


def test_detect_changepoints(validator, changepoint_data):
    """Tests the changepoint detection."""
    # Test with valid data
    changepoints = validator.detect_changepoints(changepoint_data)
    assert len(changepoints) == 1
