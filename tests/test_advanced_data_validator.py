"""Test module for advanced data validation functionality."""
# Copyright (c) 2025 USDC Arbitrage Project

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

# Constants for tests
OUTLIER_ZSCORE = 2.0
PRICE_CHANGE_THRESHOLD = 0.1
MIN_DATA_POINTS = 5
ISOLATION_FOREST_CONTAMINATION = 0.2
OUTLIER_INDEX = 4
UTC = timezone.utc

# Add src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import after path setup
from api.data_validation import (
    AdvancedDataValidator,
    DataQualityScore,
    ValidationResult,
    ValidationRuleEngine,
    ValidationSeverity,
)


@patch("api.data_validation.DBConnector")
def test_advanced_validator_initialization(_: Any) -> None:
    """Test initialization of AdvancedDataValidator."""
    validator = AdvancedDataValidator("dummy_connection_string")

    # Check that the validator was initialized correctly
    assert validator.db is not None
    assert validator.rule_engine is not None
    assert validator.isolation_forest is None
    assert validator.scaler is not None

    # Check that default rules were set up
    assert "price_integrity" in validator.rule_engine.rules
    assert "time_continuity" in validator.rule_engine.rules
    assert "statistical_outliers" in validator.rule_engine.rules
    assert "volume_anomalies" in validator.rule_engine.rules
    assert "ml_anomalies" in validator.rule_engine.rules
    assert "price_consistency" in validator.rule_engine.rules
    assert "volume_consistency" in validator.rule_engine.rules


@patch("api.data_validation.DBConnector")
def test_validation_rule_engine(_: Any) -> None:
    """Test ValidationRuleEngine functionality."""
    rule_engine = ValidationRuleEngine()

    # Test adding a rule
    def dummy_rule(df: pd.DataFrame) -> ValidationResult:
        return ValidationResult(
            rule_name="dummy_rule",
            severity=ValidationSeverity.INFO,
            message="Dummy rule executed",
            affected_rows=[],
            metadata={},
        )

    rule_engine.add_rule("dummy_rule", dummy_rule)
    assert "dummy_rule" in rule_engine.rules
    assert rule_engine.rules["dummy_rule"].enabled is True

    # Test enabling/disabling a rule
    rule_engine.enable_rule("dummy_rule", False)
    assert rule_engine.rules["dummy_rule"].enabled is False

    # Test setting and getting thresholds
    TEST_THRESHOLD = 0.5  # Define constant for magic number
    rule_engine.set_threshold("test_threshold", TEST_THRESHOLD)
    assert rule_engine.get_threshold("test_threshold") == TEST_THRESHOLD
    assert rule_engine.get_threshold("non_existent") == 0.0  # Default value


@patch("api.data_validation.DBConnector")
def test_price_integrity_validation(_: Any) -> None:
    """Test price integrity validation rule."""
    validator = AdvancedDataValidator("dummy_connection_string")

    # Valid data
    valid_data = pd.DataFrame({
        "open": [1.0, 2.0, 3.0],
        "high": [1.2, 2.2, 3.2],
        "low": [0.9, 1.9, 2.9],
        "close": [1.1, 2.1, 3.1],
        "volume": [100, 200, 300],
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
        ],
    })

    result = validator._check_price_integrity(valid_data)
    assert result.severity == ValidationSeverity.INFO
    assert "passed" in result.message
    assert len(result.affected_rows) == 0

    # Invalid data with high < low
    invalid_data = pd.DataFrame({
        "open": [1.0, 2.0, 3.0],
        "high": [1.2, 1.8, 3.2],  # 1.8 < 1.9 (low)
        "low": [0.9, 1.9, 2.9],
        "close": [1.1, 2.1, 3.1],
        "volume": [100, 200, 300],
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
        ],
    })

    result = validator._check_price_integrity(invalid_data)
    assert result.severity == ValidationSeverity.CRITICAL
    assert "high low violations" in result.message.lower()
    assert len(result.affected_rows) > 0
    assert 1 in result.affected_rows  # Index of the invalid row


@patch("api.data_validation.DBConnector")
def test_time_continuity_validation(_: Any) -> None:
    """Test time continuity validation rule."""
    validator = AdvancedDataValidator("dummy_connection_string")

    # Continuous data
    continuous_data = pd.DataFrame({
        "open": [1.0, 2.0, 3.0],
        "high": [1.2, 2.2, 3.2],
        "low": [0.9, 1.9, 2.9],
        "close": [1.1, 2.1, 3.1],
        "volume": [100, 200, 300],
        "timestamp": [
            datetime(2023, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 2, 0, tzinfo=UTC),
        ],
    })

    result = validator._check_time_continuity(continuous_data, "1h")
    assert result.severity == ValidationSeverity.INFO
    assert "passed" in result.message.lower()
    assert len(result.affected_rows) == 0

    # Data with gaps
    gapped_data = pd.DataFrame({
        "open": [1.0, 2.0, 3.0],
        "high": [1.2, 2.2, 3.2],
        "low": [0.9, 1.9, 2.9],
        "close": [1.1, 2.1, 3.1],
        "volume": [100, 200, 300],
        "timestamp": [
            datetime(2023, 1, 1, 0, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 1, 0, tzinfo=UTC),
            datetime(2023, 1, 1, 3, 0, tzinfo=UTC),  # 2-hour gap
        ],
    })

    result = validator._check_time_continuity(gapped_data, "1h")
    assert result.severity == ValidationSeverity.ERROR
    assert "gap" in result.message.lower()
    assert len(result.affected_rows) > 0


@patch("api.data_validation.DBConnector")
def test_statistical_outliers_detection(_: Any) -> None:
    """Test statistical outliers detection rule."""
    validator = AdvancedDataValidator("dummy_connection_string")

    # Normal data
    normal_data = pd.DataFrame({
        "open": [1.0, 1.1, 0.9, 1.05, 0.95],
        "high": [1.1, 1.2, 1.0, 1.15, 1.05],
        "low": [0.9, 1.0, 0.8, 0.95, 0.85],
        "close": [1.05, 1.15, 0.95, 1.1, 0.9],
        "volume": [100, 110, 90, 105, 95],
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    # Set a lower threshold to detect outliers more easily in test data
    validator.rule_engine.set_threshold("outlier_zscore", 2.0)

    result = validator._detect_statistical_outliers(normal_data)
    assert result.severity == ValidationSeverity.INFO
    assert "no statistical outliers" in result.message.lower()

    # Data with outliers
    outlier_data = pd.DataFrame({
        "open": [1.0, 1.1, 0.9, 1.05, 5.0],  # 5.0 is an outlier
        "high": [1.1, 1.2, 1.0, 1.15, 5.5],  # 5.5 is an outlier
        "low": [0.9, 1.0, 0.8, 0.95, 4.5],  # 4.5 is an outlier
        "close": [1.05, 1.15, 0.95, 1.1, 5.2],  # 5.2 is an outlier
        "volume": [100, 110, 90, 105, 95],
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    result = validator._detect_statistical_outliers(outlier_data)
    assert result.severity == ValidationSeverity.ERROR
    assert "found" in result.message.lower()
    assert len(result.affected_rows) > 0
    assert 4 in result.affected_rows  # Index of the outlier row


@patch("api.data_validation.DBConnector")
def test_volume_anomalies_detection(_: Any) -> None:
    """Test volume anomalies detection rule."""
    validator = AdvancedDataValidator("dummy_connection_string")

    # Normal volume data
    normal_data = pd.DataFrame({
        "open": [1.0, 1.1, 0.9, 1.05, 0.95],
        "high": [1.1, 1.2, 1.0, 1.15, 1.05],
        "low": [0.9, 1.0, 0.8, 0.95, 0.85],
        "close": [1.05, 1.15, 0.95, 1.1, 0.9],
        "volume": [100, 110, 90, 105, 95],
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    result = validator._detect_volume_anomalies(normal_data)
    assert result.severity == ValidationSeverity.ERROR
    assert "found" in result.message.lower()

    # Data with volume spike
    spike_data = pd.DataFrame({
        "open": [1.0, 1.1, 0.9, 1.05, 0.95],
        "high": [1.1, 1.2, 1.0, 1.15, 1.05],
        "low": [0.9, 1.0, 0.8, 0.95, 0.85],
        "close": [1.05, 1.15, 0.95, 1.1, 0.9],
        "volume": [100, 110, 90, 105, 1000],  # 1000 is a spike
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    result = validator._detect_volume_anomalies(spike_data)
    assert result.severity == ValidationSeverity.ERROR
    assert "found" in result.message.lower()
    assert len(result.affected_rows) > 0
    assert 4 in result.affected_rows  # Index of the spike row


@patch("api.data_validation.DBConnector")
def test_ml_anomalies_detection(_: Any) -> None:
    """Test ML-based anomaly detection rule."""
    validator = AdvancedDataValidator("dummy_connection_string")

    # Set a lower threshold for min_data_points to test with small dataset
    validator.rule_engine.set_threshold("min_data_points", 5)
    validator.rule_engine.set_threshold("isolation_forest_contamination", 0.2)

    # Generate synthetic data with one anomaly
    np.random.seed(42)
    normal_points = np.random.normal(loc=1.0, scale=0.1, size=(10, 4))
    anomaly_point = np.array([[5.0, 5.5, 4.5, 5.2]])  # Anomaly point
    data_points = np.vstack([normal_points, anomaly_point])

    test_data = pd.DataFrame(data_points, columns=["open", "high", "low", "close"])

    # Use numpy's Generator API instead of legacy random functions
    rng = np.random.Generator(np.random.PCG64(42))
    test_data["volume"] = rng.normal(loc=100, scale=10, size=11)

    test_data["timestamp"] = [
        datetime(2023, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(11)
    ]

    result = validator._detect_ml_anomalies(test_data)
    assert result.severity == ValidationSeverity.ERROR
    assert "detected" in result.message.lower()
    assert len(result.affected_rows) > 0

    # Test with insufficient data
    small_data = test_data.iloc[:3]
    validator.rule_engine.set_threshold("min_data_points", 5)
    result = validator._detect_ml_anomalies(small_data)
    assert result.severity == ValidationSeverity.INFO
    assert "insufficient data" in result.message.lower()


@patch("api.data_validation.DBConnector")
def test_price_consistency_validation(_: Any) -> None:
    """Test price consistency validation rule."""
    validator = AdvancedDataValidator("dummy_connection_string")

    # Consistent price data
    consistent_data = pd.DataFrame({
        "open": [1.0, 1.02, 1.03, 1.01, 1.02],
        "high": [1.05, 1.07, 1.08, 1.06, 1.07],
        "low": [0.98, 0.99, 1.0, 0.99, 1.0],
        "close": [1.02, 1.03, 1.01, 1.02, 1.03],
        "volume": [100, 110, 90, 105, 95],
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    result = validator._check_price_consistency(consistent_data)
    assert result.severity == ValidationSeverity.ERROR
    assert "reversal" in result.message.lower()
    assert len(result.affected_rows) == 0

    # Data with sudden price change
    validator.rule_engine.set_threshold(
        "price_change_threshold", PRICE_CHANGE_THRESHOLD
    )  # 10% change threshold

    sudden_change_data = pd.DataFrame({
        "open": [1.0, 1.02, 1.03, 1.01, 1.02],
        "high": [1.05, 1.07, 1.08, 1.06, 1.07],
        "low": [0.98, 0.99, 1.0, 0.99, 1.0],
        "close": [1.02, 1.03, 1.01, 1.02, 1.15],  # 13% increase from previous
        "volume": [100, 110, 90, 105, 95],
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    result = validator._check_price_consistency(sudden_change_data)
    assert result.severity == ValidationSeverity.ERROR
    assert "sudden price changes" in result.message.lower()
    assert len(result.affected_rows) > 0


@patch("api.data_validation.DBConnector")
def test_volume_consistency_validation(_: Any) -> None:
    """Test volume consistency validation rule."""
    validator = AdvancedDataValidator("dummy_connection_string")

    # Normal volume data
    normal_data = pd.DataFrame({
        "open": [1.0, 1.1, 0.9, 1.05, 0.95],
        "high": [1.1, 1.2, 1.0, 1.15, 1.05],
        "low": [0.9, 1.0, 0.8, 0.95, 0.85],
        "close": [1.05, 1.15, 0.95, 1.1, 0.9],
        "volume": [100, 110, 90, 105, 95],
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    result = validator._check_volume_consistency(normal_data)
    assert result.severity == ValidationSeverity.INFO
    assert "passed" in result.message.lower()
    assert len(result.affected_rows) == 0

    # Data with zero volume
    zero_volume_data = pd.DataFrame({
        "open": [1.0, 1.1, 0.9, 1.05, 0.95],
        "high": [1.1, 1.2, 1.0, 1.15, 1.05],
        "low": [0.9, 1.0, 0.8, 0.95, 0.85],
        "close": [1.05, 1.15, 0.95, 1.1, 0.9],
        "volume": [100, 0, 90, 105, 95],  # Zero volume
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    result = validator._check_volume_consistency(zero_volume_data)
    assert result.severity == ValidationSeverity.WARNING
    assert "zero volume" in result.message.lower()
    assert len(result.affected_rows) > 0
    assert 1 in result.affected_rows  # Index of the zero volume row

    # Data with negative volume
    negative_volume_data = pd.DataFrame({
        "open": [1.0, 1.1, 0.9, 1.05, 0.95],
        "high": [1.1, 1.2, 1.0, 1.15, 1.05],
        "low": [0.9, 1.0, 0.8, 0.95, 0.85],
        "close": [1.05, 1.15, 0.95, 1.1, 0.9],
        "volume": [100, 110, -10, 105, 95],  # Negative volume
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    result = validator._check_volume_consistency(negative_volume_data)
    assert result.severity == ValidationSeverity.ERROR
    assert "negative volume" in result.message.lower()
    assert len(result.affected_rows) > 0
    assert 2 in result.affected_rows  # Index of the negative volume row


@patch("api.data_validation.DBConnector")
def test_data_quality_score_calculation(_: Any) -> None:
    """Test data quality score calculation."""
    validator = AdvancedDataValidator("dummy_connection_string")

    # Create some validation results
    validation_results = [
        ValidationResult(
            rule_name="price_integrity",
            severity=ValidationSeverity.INFO,
            message="Price integrity validation passed",
            affected_rows=[],
            metadata={"total_rows": 100},
        ),
        ValidationResult(
            rule_name="time_continuity",
            severity=ValidationSeverity.WARNING,
            message="Found 2 time gaps",
            affected_rows=[10, 20],
            metadata={"total_rows": 100},
        ),
        ValidationResult(
            rule_name="statistical_outliers",
            severity=ValidationSeverity.WARNING,
            message="Found 3 statistical outliers",
            affected_rows=[5, 15, 25],
            metadata={"total_rows": 100},
        ),
    ]

    quality_score = validator.calculate_data_quality_score(validation_results)

    # Check that the score is calculated correctly
    assert isinstance(quality_score, DataQualityScore)  # noqa: S101
    assert 0 <= quality_score.overall_score <= 1
    assert 0 <= quality_score.integrity_score <= 1
    assert 0 <= quality_score.completeness_score <= 1
    assert 0 <= quality_score.consistency_score <= 1
    assert 0 <= quality_score.anomaly_score <= 1

    # Check that details are included
    assert "time_continuity" in quality_score.details
    assert quality_score.details["time_continuity"]["severity"] == "warning"
    assert quality_score.details["time_continuity"]["affected_rows"] == 2


@patch("api.data_validation.DBConnector")
def test_comprehensive_validation(mock_db: Any) -> None:
    """Test comprehensive validation method."""
    validator = AdvancedDataValidator("dummy_connection_string")

    # Create test data with some issues
    test_data = pd.DataFrame({
        "open": [1.0, 1.1, 0.9, 1.05, 0.95],
        "high": [1.1, 1.2, 1.0, 1.15, 1.05],
        "low": [0.9, 1.0, 0.8, 0.95, 0.85],
        "close": [1.05, 1.15, 0.95, 1.1, 0.9],
        "volume": [100, 0, 90, 105, 95],  # Zero volume in row 1
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    mock_db.return_value.get_ohlcv_data.return_value = test_data

    # Run comprehensive validation
    results, quality_score = validator.comprehensive_validation(
        "exchange", "symbol", "timeframe"
    )

    # Check that results and quality score are returned
    assert isinstance(results, list)
    assert len(results) > 0
    assert isinstance(quality_score, DataQualityScore)

    # Check that the volume consistency issue was detected
    volume_results = [r for r in results if r.rule_name == "volume_consistency"]
    assert len(volume_results) > 0
    assert volume_results[0].severity == ValidationSeverity.WARNING
    assert "zero volume" in volume_results[0].message.lower()

    # Test with empty data
    mock_db.return_value.get_ohlcv_data.return_value = pd.DataFrame()
    results, quality_score = validator.comprehensive_validation(
        "exchange", "symbol", "timeframe"
    )
    assert len(results) == 0
    assert quality_score.overall_score == 0.0

    # Test with database error
    mock_db.return_value.get_ohlcv_data.side_effect = Exception("Database error")
    results, quality_score = validator.comprehensive_validation(
        "exchange", "symbol", "timeframe"
    )
    assert len(results) == 1
    assert results[0].rule_name == "system_error"
    assert results[0].severity == ValidationSeverity.CRITICAL
    assert quality_score.overall_score == 0.0


@patch("api.data_validation.DBConnector")
def test_backward_compatibility(mock_db: Any) -> None:
    """Test backward compatibility with DataValidator alias."""
    from api.data_validation import DataValidator

    # Check that DataValidator is an alias for AdvancedDataValidator
    validator = DataValidator("dummy_connection_string")
    assert isinstance(validator, AdvancedDataValidator)

    # Create test data with some issues
    test_data = pd.DataFrame({
        "open": [1.0, 1.1, 0.9, 1.05, 0.95],
        "high": [1.1, 1.2, 1.0, 1.15, 1.05],
        "low": [0.9, 1.0, 0.8, 0.95, 0.85],
        "close": [1.05, 1.15, 0.95, 1.1, 0.9],
        "volume": [100, 0, 90, 105, 95],  # Zero volume in row 1
        "timestamp": [
            datetime(2023, 1, 1, tzinfo=UTC),
            datetime(2023, 1, 2, tzinfo=UTC),
            datetime(2023, 1, 3, tzinfo=UTC),
            datetime(2023, 1, 4, tzinfo=UTC),
            datetime(2023, 1, 5, tzinfo=UTC),
        ],
    })

    mock_db.return_value.get_ohlcv_data.return_value = test_data

    # Run comprehensive validation
    results, quality_score = validator.comprehensive_validation(
        "exchange", "symbol", "timeframe"
    )

    # Check that results and quality score are returned
    assert isinstance(results, list)
    assert len(results) > 0
    assert isinstance(quality_score, DataQualityScore)

    # Check that the volume consistency issue was detected
    volume_results = [r for r in results if r.rule_name == "volume_consistency"]
    assert len(volume_results) > 0
    assert volume_results[0].severity == ValidationSeverity.WARNING
    assert "zero volume" in volume_results[0].message.lower()

    # Test with empty data
    mock_db.return_value.get_ohlcv_data.return_value = pd.DataFrame()
    results, quality_score = validator.comprehensive_validation(
        "exchange", "symbol", "timeframe"
    )
    assert len(results) == 0
    assert quality_score.overall_score == 0.0

    # Test with database error
    mock_db.return_value.get_ohlcv_data.side_effect = Exception("Database error")
    results, quality_score = validator.comprehensive_validation(
        "exchange", "symbol", "timeframe"
    )
    assert len(results) == 1
    assert results[0].rule_name == "system_error"
    assert results[0].severity == ValidationSeverity.CRITICAL
    assert quality_score.overall_score == 0.0
