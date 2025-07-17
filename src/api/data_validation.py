"""Advanced data validation framework for USDC arbitrage application."""

import logging
import json
from datetime import timedelta
from typing import List, Dict, Any, Tuple, Callable, Optional, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.cluster import DBSCAN  # type: ignore
import joblib

from .database import DBConnector

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("data_validation.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""

    rule_name: str
    severity: ValidationSeverity
    message: str
    affected_rows: List[int]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary."""
        result = asdict(self)
        result["severity"] = self.severity.value
        return result


@dataclass
class DataQualityScore:
    """Data quality scoring result."""

    overall_score: float
    integrity_score: float
    completeness_score: float
    consistency_score: float
    anomaly_score: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert quality score to dictionary."""
        return asdict(self)


@dataclass
class ValidationRule:
    """Validation rule configuration."""

    name: str
    function: Callable
    enabled: bool = True
    description: str = ""
    category: str = "general"
    severity_threshold: Dict[str, float] = field(default_factory=dict)


class ValidationRuleEngine:
    """Configurable validation rule engine."""

    def __init__(self):
        """Initialize validation rule engine."""
        self.rules: Dict[str, ValidationRule] = {}
        self.thresholds: Dict[str, float] = {
            "outlier_zscore": 3.0,
            "volume_anomaly_zscore": 3.0,
            "price_change_threshold": 0.1,  # 10% price change threshold
            "volume_spike_threshold": 5.0,  # 5x volume spike threshold
            "gap_tolerance": 1.15,  # 15% tolerance for time gaps
            "isolation_forest_contamination": 0.1,  # 10% contamination rate
            "min_data_points": 100,  # Minimum data points for analysis
            "dbscan_eps": 0.5,  # DBSCAN epsilon parameter
            "dbscan_min_samples": 5,  # DBSCAN min samples parameter
            "zero_volume_allowed": False,  # Whether zero volume is allowed
            "negative_price_allowed": False,  # Whether negative prices are allowed
            "max_price_deviation": 0.2,  # Maximum allowed price deviation (20%)
            "min_quality_score": 0.7,  # Minimum acceptable quality score
        }
        self.rule_categories: Set[str] = {"integrity", "completeness", "consistency", "anomaly"}

    def add_rule(
        self, 
        name: str, 
        func: Callable, 
        enabled: bool = True, 
        description: str = "", 
        category: str = "general"
    ) -> None:
        """Add a custom validation rule."""
        self.rules[name] = ValidationRule(
            name=name,
            function=func,
            enabled=enabled,
            description=description,
            category=category
        )
        if category not in self.rule_categories:
            self.rule_categories.add(category)

    def set_threshold(self, name: str, value: float) -> None:
        """Set threshold for validation rules."""
        self.thresholds[name] = value

    def get_threshold(self, name: str) -> float:
        """Get threshold value."""
        return self.thresholds.get(name, 0.0)

    def enable_rule(self, name: str, enabled: bool = True) -> None:
        """Enable or disable a validation rule."""
        if name in self.rules:
            self.rules[name].enabled = enabled

    def get_rules_by_category(self, category: str) -> List[ValidationRule]:
        """Get all rules in a specific category."""
        return [rule for rule in self.rules.values() if rule.category == category]

    def load_config(self, config_path: str) -> None:
        """Load rule engine configuration from JSON file."""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                
            # Load thresholds
            if "thresholds" in config:
                for name, value in config["thresholds"].items():
                    self.thresholds[name] = value
                    
            # Load rule configurations
            if "rules" in config:
                for rule_name, rule_config in config["rules"].items():
                    if rule_name in self.rules:
                        self.rules[rule_name].enabled = rule_config.get("enabled", True)
                        if "severity_threshold" in rule_config:
                            self.rules[rule_name].severity_threshold = rule_config["severity_threshold"]
                            
            logger.info(f"Loaded validation configuration from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load validation configuration: {e}")

    def save_config(self, config_path: str) -> None:
        """Save rule engine configuration to JSON file."""
        try:
            config = {
                "thresholds": self.thresholds,
                "rules": {
                    name: {
                        "enabled": rule.enabled,
                        "category": rule.category,
                        "description": rule.description,
                        "severity_threshold": rule.severity_threshold
                    }
                    for name, rule in self.rules.items()
                }
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Saved validation configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save validation configuration: {e}")


class AdvancedDataValidator:
    """Advanced data validation framework with ML-based anomaly detection."""

    def __init__(self, connection_string: str):
        """Initialize advanced data validator."""
        self.db = DBConnector(connection_string)
        self.rule_engine = ValidationRuleEngine()
        self.isolation_forest: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        self._setup_default_rules()
        
        # Try to load configuration if it exists
        config_path = os.path.join(os.path.dirname(__file__), "..", "config", "validation_config.json")
        if os.path.exists(config_path):
            self.rule_engine.load_config(config_path)

    def calculate_data_quality_score(
        self, validation_results: List[ValidationResult]
    ) -> DataQualityScore:
        """Calculate comprehensive data quality score."""
        # Initialize scores
        integrity_score = 1.0
        completeness_score = 1.0
        consistency_score = 1.0
        anomaly_score = 1.0

        # Weight factors for validation aspects
        weights = {
            "price_integrity": 0.3,
            "time_continuity": 0.2,
            "statistical_outliers": 0.15,
            "volume_anomalies": 0.15,
            "ml_anomalies": 0.1,
            "price_consistency": 0.05,
            "volume_consistency": 0.05,
        }

        # Severity penalties
        severity_penalties = {
            ValidationSeverity.INFO: 0.0,
            ValidationSeverity.WARNING: 0.1,
            ValidationSeverity.ERROR: 0.3,
            ValidationSeverity.CRITICAL: 0.5,
        }

        details = {}

        for result in validation_results:
            rule_name = result.rule_name
            severity = result.severity
            affected_count = len(result.affected_rows)

            # Calculate penalty based on severity and affected rows
            base_penalty = severity_penalties.get(severity, 0.0)
            weight = weights.get(rule_name, 0.05)

            # Adjust penalty based on proportion of affected rows
            if "total_rows" in result.metadata:
                total_rows = result.metadata["total_rows"]
                if total_rows > 0:
                    proportion_affected = affected_count / total_rows
                    adjusted_penalty = base_penalty * (1 + proportion_affected)
                else:
                    adjusted_penalty = base_penalty
            else:
                adjusted_penalty = base_penalty

            # Apply penalties to relevant score categories
            if rule_name in ["price_integrity", "volume_consistency"]:
                integrity_score -= adjusted_penalty * weight
            elif rule_name in ["time_continuity"]:
                completeness_score -= adjusted_penalty * weight
            elif rule_name in ["price_consistency", "volume_consistency"]:
                consistency_score -= adjusted_penalty * weight
            elif rule_name in [
                "statistical_outliers",
                "volume_anomalies",
                "ml_anomalies",
                "clustering_anomalies",
            ]:
                anomaly_score -= adjusted_penalty * weight

            # Store details for each rule
            details[rule_name] = {
                "severity": severity.value,
                "affected_rows": affected_count,
                "penalty_applied": adjusted_penalty * weight,
                "message": result.message,
            }

        # Ensure scores don't go below 0
        integrity_score = max(0.0, integrity_score)
        completeness_score = max(0.0, completeness_score)
        consistency_score = max(0.0, consistency_score)
        anomaly_score = max(0.0, anomaly_score)

        # Calculate overall score as weighted average
        overall_score = (
            integrity_score * 0.4
            + completeness_score * 0.25
            + consistency_score * 0.2
            + anomaly_score * 0.15
        )

        return DataQualityScore(
            overall_score=round(overall_score, 3),
            integrity_score=round(integrity_score, 3),
            completeness_score=round(completeness_score, 3),
            consistency_score=round(consistency_score, 3),
            anomaly_score=round(anomaly_score, 3),
            details=details,
        )

    def comprehensive_validation(
        self, exchange: str, symbol: str, timeframe: str
    ) -> Tuple[List[ValidationResult], DataQualityScore]:
        """Run comprehensive validation and return results with quality score."""
        logger.info(
            "Running comprehensive validation for %s/%s/%s", exchange, symbol, timeframe
        )

        try:
            # Get data from database
            df = self.db.get_ohlcv_data(exchange, symbol, timeframe)

            if df.empty:
                logger.warning(
                    "No data found for %s/%s/%s", exchange, symbol, timeframe
                )
                return [], DataQualityScore(
                    0.0, 0.0, 0.0, 0.0, 0.0, {"error": "No data found"}
                )

            # Add total rows to metadata for scoring
            total_rows = len(df)

            validation_results = []

            # Run all enabled validation rules
            for rule_name, rule in self.rule_engine.rules.items():
                if not rule.enabled:
                    continue

                try:
                    if rule_name == "time_continuity":
                        result = rule.function(df, timeframe)
                    else:
                        result = rule.function(df)

                    # Add total rows to metadata for quality scoring
                    result.metadata["total_rows"] = total_rows
                    validation_results.append(result)

                    logger.info(
                        "Rule %s: %s - %s",
                        rule_name,
                        result.severity.value,
                        result.message,
                    )

                except Exception as e:
                    logger.error("Error running validation rule %s: %s", rule_name, e)
                    error_result = ValidationResult(
                        rule_name=rule_name,
                        severity=ValidationSeverity.ERROR,
                        message=f"Rule execution failed: {str(e)}",
                        affected_rows=[],
                        metadata={"error": str(e), "total_rows": total_rows},
                    )
                    validation_results.append(error_result)

            # Calculate data quality score
            quality_score = self.calculate_data_quality_score(validation_results)

            logger.info(
                "Validation completed for %s/%s/%s - Overall quality score: %.3f",
                exchange,
                symbol,
                timeframe,
                quality_score.overall_score,
            )

            # Flag data for review if quality score is below threshold
            min_quality_score = self.rule_engine.get_threshold("min_quality_score")
            if quality_score.overall_score < min_quality_score:
                logger.warning(
                    "Data quality score %.3f is below threshold %.3f for %s/%s/%s - Flagging for review",
                    quality_score.overall_score,
                    min_quality_score,
                    exchange,
                    symbol,
                    timeframe,
                )
                self._flag_data_for_review(exchange, symbol, timeframe, quality_score, validation_results)

            return validation_results, quality_score

        except Exception as e:
            logger.error(
                "Comprehensive validation failed for %s/%s/%s: %s",
                exchange,
                symbol,
                timeframe,
                e,
            )
            error_result = ValidationResult(
                rule_name="system_error",
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation system error: {str(e)}",
                affected_rows=[],
                metadata={"error": str(e)},
            )
            error_score = DataQualityScore(
                0.0, 0.0, 0.0, 0.0, 0.0, {"system_error": str(e)}
            )
            return [error_result], error_score

    def _flag_data_for_review(
        self, 
        exchange: str, 
        symbol: str, 
        timeframe: str, 
        quality_score: DataQualityScore, 
        validation_results: List[ValidationResult]
    ) -> None:
        """Flag data for manual review."""
        try:
            # Create a review record with validation results
            review_data = {
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "quality_score": quality_score.to_dict(),
                "validation_results": [result.to_dict() for result in validation_results],
                "timestamp": pd.Timestamp.now().isoformat(),
                "status": "pending_review"
            }
            
            # Create directory if it doesn't exist
            review_dir = os.path.join(os.path.dirname(__file__), "..", "data", "review")
            os.makedirs(review_dir, exist_ok=True)
            
            # Save review data to file
            filename = f"{exchange}_{symbol}_{timeframe}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.json"
            filepath = os.path.join(review_dir, filename)
            
            with open(filepath, "w") as f:
                json.dump(review_data, f, indent=2)
                
            logger.info(f"Flagged data for review: {filepath}")
            
            # TODO: Implement notification system for administrators
            # This would typically involve sending an email or other alert
            
        except Exception as e:
            logger.error(f"Failed to flag data for review: {e}")

    def _setup_default_rules(self) -> None:
        """Setup default validation rules."""
        self.rule_engine.add_rule(
            "price_integrity", 
            self._check_price_integrity,
            description="Validates OHLC price relationships and integrity",
            category="integrity"
        )
        self.rule_engine.add_rule(
            "time_continuity", 
            self._check_time_continuity,
            description="Checks for gaps in time series data",
            category="completeness"
        )
        self.rule_engine.add_rule(
            "statistical_outliers", 
            self._detect_statistical_outliers,
            description="Detects statistical outliers using multiple methods",
            category="anomaly"
        )
        self.rule_engine.add_rule(
            "volume_anomalies", 
            self._detect_volume_anomalies,
            description="Detects volume anomalies using statistical methods",
            category="anomaly"
        )
        self.rule_engine.add_rule(
            "ml_anomalies", 
            self._detect_ml_anomalies,
            description="ML-based anomaly detection using Isolation Forest",
            category="anomaly"
        )
        self.rule_engine.add_rule(
            "clustering_anomalies", 
            self._detect_clustering_anomalies,
            description="Anomaly detection using DBSCAN clustering",
            category="anomaly"
        )
        self.rule_engine.add_rule(
            "price_consistency", 
            self._check_price_consistency,
            description="Checks price consistency and sudden changes",
            category="consistency"
        )
        self.rule_engine.add_rule(
            "volume_consistency", 
            self._check_volume_consistency,
            description="Checks volume consistency and unusual patterns",
            category="consistency"
        )
        self.rule_engine.add_rule(
            "cross_exchange_consistency", 
            self._check_cross_exchange_consistency,
            description="Checks price consistency across exchanges",
            category="consistency",
            enabled=False  # Disabled by default as it requires data from multiple exchanges
        )

    def enable_rule(self, rule_name: str, enabled: bool = True) -> None:
        """Enable or disable a specific validation rule."""
        self.rule_engine.enable_rule(rule_name, enabled)

    def set_threshold(self, name: str, value: float) -> None:
        """Set threshold for validation rules."""
        self.rule_engine.set_threshold(name, value)

    def save_configuration(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "validation_config.json")
        self.rule_engine.save_config(config_path)

    def load_configuration(self, config_path: Optional[str] = None) -> None:
        """Load configuration from file."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "..", "config", "validation_config.json")
        self.rule_engine.load_config(config_path)

    def save_ml_model(self, model_path: Optional[str] = None) -> None:
        """Save trained ML model to file."""
        if self.isolation_forest is None:
            logger.warning("No ML model to save")
            return
            
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "isolation_forest.joblib")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        try:
            joblib.dump(self.isolation_forest, model_path)
            logger.info(f"Saved ML model to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save ML model: {e}")

    def load_ml_model(self, model_path: Optional[str] = None) -> None:
        """Load trained ML model from file."""
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "isolation_forest.joblib")
            
        if not os.path.exists(model_path):
            logger.warning(f"ML model file not found: {model_path}")
            return
            
        try:
            self.isolation_forest = joblib.load(model_path)
            logger.info(f"Loaded ML model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")

    def _check_price_integrity(self, df: pd.DataFrame) -> ValidationResult:
        """Comprehensive OHLC price integrity validation."""
        errors = []
        affected_rows = []

        # Basic OHLC relationships
        violations = {
            "high_low": df["high"] < df["low"],
            "open_high": df["open"] > df["high"],
            "open_low": df["open"] < df["low"],
            "close_high": df["close"] > df["high"],
            "close_low": df["close"] < df["low"],
            "negative_prices": (df[["open", "high", "low", "close"]] < 0).any(axis=1),
            "zero_prices": (df[["open", "high", "low", "close"]] == 0).any(axis=1),
        }

        for violation_type, mask in violations.items():
            if mask.any():
                count = mask.sum()
                errors.append(f"{count} {violation_type.replace('_', ' ')} violations")
                affected_rows.extend(df[mask].index.tolist())

        # Check for NaN values
        nan_mask = df[["open", "high", "low", "close"]].isna().any(axis=1)
        if nan_mask.any():
            count = nan_mask.sum()
            errors.append(f"{count} rows with NaN values")
            affected_rows.extend(df[nan_mask].index.tolist())

        # Check for infinity values
        inf_mask = (~np.isfinite(df[["open", "high", "low", "close"]])).any(axis=1)
        if inf_mask.any():
            count = inf_mask.sum()
            errors.append(f"{count} rows with infinity values")
            affected_rows.extend(df[inf_mask].index.tolist())

        # Determine severity based on violation types and counts
        if any(v in ["negative_prices", "high_low"] for v in violations.keys() if violations[v].any()):
            severity = ValidationSeverity.CRITICAL
        elif nan_mask.any() or inf_mask.any():
            severity = ValidationSeverity.CRITICAL
        elif any(violations[v].any() for v in violations.keys()):
            severity = ValidationSeverity.ERROR
        else:
            severity = ValidationSeverity.INFO

        message = "; ".join(errors) if errors else "Price integrity validation passed"

        return ValidationResult(
            rule_name="price_integrity",
            severity=severity,
            message=message,
            affected_rows=list(set(affected_rows)),
            metadata={
                "violation_counts": {k: int(v.sum()) for k, v in violations.items()},
                "nan_count": int(nan_mask.sum()),
                "inf_count": int(inf_mask.sum()),
            },
        )

    def _check_time_continuity(
        self, df: pd.DataFrame, timeframe: str
    ) -> ValidationResult:
        """Check for gaps and irregularities in time series data."""
        if len(df) < 2:
            return ValidationResult(
                rule_name="time_continuity",
                severity=ValidationSeverity.WARNING,
                message="Insufficient data for time continuity check",
                affected_rows=[],
                metadata={},
            )

        df_sorted = df.sort_values("timestamp")
        time_diffs = df_sorted["timestamp"].diff().dropna()

        # Expected time delta based on timeframe
        timeframe_deltas = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "4h": timedelta(hours=4),
            "1d": timedelta(days=1),
        }

        expected_delta = timeframe_deltas.get(timeframe, timedelta(minutes=1))
        tolerance = self.rule_engine.get_threshold("gap_tolerance")
        max_allowed_gap = expected_delta * tolerance

        gaps = time_diffs[time_diffs > max_allowed_gap]
        gap_indices = gaps.index.tolist()
        
        # Check for duplicate timestamps
        duplicate_timestamps = df_sorted["timestamp"].duplicated()
        duplicate_indices = df_sorted[duplicate_timestamps].index.tolist()
        
        # Check for out-of-order timestamps
        is_sorted = df_sorted["timestamp"].equals(df["timestamp"])
        
        # Determine severity based on findings
        if len(gaps) > 0:
            if len(gaps) / len(df) > 0.1:  # More than 10% of data has gaps
                severity = ValidationSeverity.ERROR
            else:
                severity = ValidationSeverity.WARNING
        elif duplicate_timestamps.any():
            severity = ValidationSeverity.ERROR
        elif not is_sorted:
            severity = ValidationSeverity.WARNING
        else:
            severity = ValidationSeverity.INFO
            
        # Build message
        messages = []
        if len(gaps) > 0:
            messages.append(f"Found {len(gaps)} time gaps")
        if duplicate_timestamps.any():
            messages.append(f"Found {duplicate_timestamps.sum()} duplicate timestamps")
        if not is_sorted:
            messages.append("Timestamps are not in chronological order")
            
        message = "; ".join(messages) if messages else "Time continuity validation passed"
        
        # Combine affected rows
        affected_rows = list(set(gap_indices + duplicate_indices))

        return ValidationResult(
            rule_name="time_continuity",
            severity=severity,
            message=message,
            affected_rows=affected_rows,
            metadata={
                "expected_delta": str(expected_delta),
                "max_gaps": len(gaps),
                "largest_gap": str(gaps.max()) if len(gaps) > 0 else None,
                "duplicate_timestamps": int(duplicate_timestamps.sum()),
                "is_chronological": is_sorted,
                "gap_percentage": len(gaps) / len(df) if len(df) > 0 else 0,
            },
        )

    def _detect_statistical_outliers(self, df: pd.DataFrame) -> ValidationResult:
        """Detect statistical outliers using multiple methods."""
        outlier_indices = []
        methods_used = []
        outlier_details = {}

        # Modified Z-score method
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))

                if mad > 0:
                    threshold = self.rule_engine.get_threshold("outlier_zscore")
                    scaled_diff = 0.6745 * (df[col] - median) / mad
                    outliers = df[abs(scaled_diff) > threshold].index.tolist()
                    outlier_indices.extend(outliers)
                    methods_used.append(f"Modified Z-score ({col})")
                    
                    # Store outlier details
                    for idx in outliers:
                        if idx not in outlier_details:
                            outlier_details[idx] = {}
                        outlier_details[idx][f"{col}_zscore"] = float(scaled_diff.iloc[idx])
                        outlier_details[idx][f"{col}_value"] = float(df[col].iloc[idx])
                        outlier_details[idx][f"{col}_median"] = float(median)

        # IQR method for additional validation
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                iqr_outliers = df[
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ].index.tolist()
                outlier_indices.extend(iqr_outliers)
                methods_used.append(f"IQR ({col})")
                
                # Store outlier details
                for idx in iqr_outliers:
                    if idx not in outlier_details:
                        outlier_details[idx] = {}
                    outlier_details[idx][f"{col}_iqr_lower"] = float(lower_bound)
                    outlier_details[idx][f"{col}_iqr_upper"] = float(upper_bound)
                    outlier_details[idx][f"{col}_value"] = float(df[col].iloc[idx])

        # Rolling window analysis for contextual outliers
        if len(df) >= 10:  # Need enough data for meaningful window
            for col in ["close"]:
                # Calculate rolling mean and std
                window_size = min(20, len(df) // 5)  # Adaptive window size
                rolling_mean = df[col].rolling(window=window_size, center=True).mean()
                rolling_std = df[col].rolling(window=window_size, center=True).std()
                
                # Fill NaN values at edges
                rolling_mean = rolling_mean.fillna(method="bfill").fillna(method="ffill")
                rolling_std = rolling_std.fillna(method="bfill").fillna(method="ffill")
                
                # Identify values outside of 3 standard deviations
                threshold = self.rule_engine.get_threshold("outlier_zscore")
                upper_bound = rolling_mean + threshold * rolling_std
                lower_bound = rolling_mean - threshold * rolling_std
                
                contextual_outliers = df[
                    (df[col] > upper_bound) | (df[col] < lower_bound)
                ].index.tolist()
                
                outlier_indices.extend(contextual_outliers)
                methods_used.append(f"Rolling window ({col})")
                
                # Store outlier details
                for idx in contextual_outliers:
                    if idx not in outlier_details:
                        outlier_details[idx] = {}
                    outlier_details[idx][f"{col}_rolling_mean"] = float(rolling_mean.iloc[idx])
                    outlier_details[idx][f"{col}_rolling_std"] = float(rolling_std.iloc[idx])
                    outlier_details[idx][f"{col}_value"] = float(df[col].iloc[idx])

        unique_outliers = list(set(outlier_indices))
        
        # Determine severity based on proportion of outliers
        outlier_proportion = len(unique_outliers) / len(df) if len(df) > 0 else 0
        if outlier_proportion > 0.1:  # More than 10% are outliers
            severity = ValidationSeverity.ERROR
        elif outlier_proportion > 0.05:  # More than 5% are outliers
            severity = ValidationSeverity.WARNING
        elif unique_outliers:
            severity = ValidationSeverity.INFO
        else:
            severity = ValidationSeverity.INFO
            
        message = (
            f"Found {len(unique_outliers)} statistical outliers ({outlier_proportion:.1%} of data)"
            if unique_outliers
            else "No statistical outliers detected"
        )

        return ValidationResult(
            rule_name="statistical_outliers",
            severity=severity,
            message=message,
            affected_rows=unique_outliers,
            metadata={
                "methods_used": methods_used,
                "total_outliers": len(unique_outliers),
                "outlier_proportion": outlier_proportion,
                "outlier_details": outlier_details,
            },
        )

    def _detect_volume_anomalies(self, df: pd.DataFrame) -> ValidationResult:
        """Detect volume anomalies using statistical methods."""
        if "volume" not in df.columns or len(df) == 0:
            return ValidationResult(
                rule_name="volume_anomalies",
                severity=ValidationSeverity.WARNING,
                message="Volume data not available",
                affected_rows=[],
                metadata={},
            )

        # Log-transform volume to handle skewness
        log_volume = np.log(df["volume"] + 1e-6)

        # Detect volume spikes
        volume_median = df["volume"].median()
        spike_threshold = self.rule_engine.get_threshold("volume_spike_threshold")
        volume_spikes = df[
            df["volume"] > volume_median * spike_threshold
        ].index.tolist()

        # Statistical outliers in log-transformed volume
        median = log_volume.median()
        mad = np.median(np.abs(log_volume - median))

        statistical_outliers = []
        if mad > 0:
            threshold = self.rule_engine.get_threshold("volume_anomaly_zscore")
            scaled_diff = 0.6745 * (log_volume - median) / mad
            statistical_outliers = df[abs(scaled_diff) > threshold].index.tolist()
            
        # Detect sudden volume changes
        if len(df) > 1:
            df_sorted = df.sort_values("timestamp")
            volume_changes = df_sorted["volume"].pct_change().abs()
            sudden_changes = volume_changes[volume_changes > 1.0].index.tolist()  # 100% change
        else:
            sudden_changes = []

        # Detect unusual volume-price relationships
        unusual_relationships = []
        if "close" in df.columns and len(df) > 1:
            df_sorted = df.sort_values("timestamp")
            price_changes = df_sorted["close"].pct_change().abs()
            volume_changes = df_sorted["volume"].pct_change()
            
            # Identify cases where price changes significantly but volume doesn't increase
            price_change_threshold = 0.05  # 5% price change
            significant_price_changes = price_changes > price_change_threshold
            
            # Unusual: big price change with low volume
            unusual_mask = significant_price_changes & (volume_changes < 0)
            unusual_relationships = df_sorted[unusual_mask].index.tolist()

        all_anomalies = list(set(volume_spikes + statistical_outliers + sudden_changes + unusual_relationships))
        
        # Determine severity based on proportion of anomalies
        anomaly_proportion = len(all_anomalies) / len(df) if len(df) > 0 else 0
        if anomaly_proportion > 0.1:  # More than 10% are anomalies
            severity = ValidationSeverity.ERROR
        elif anomaly_proportion > 0.05:  # More than 5% are anomalies
            severity = ValidationSeverity.WARNING
        elif all_anomalies:
            severity = ValidationSeverity.INFO
        else:
            severity = ValidationSeverity.INFO
            
        message = (
            f"Found {len(all_anomalies)} volume anomalies ({anomaly_proportion:.1%} of data)"
            if all_anomalies
            else "No volume anomalies detected"
        )

        return ValidationResult(
            rule_name="volume_anomalies",
            severity=severity,
            message=message,
            affected_rows=all_anomalies,
            metadata={
                "volume_spikes": len(volume_spikes),
                "statistical_outliers": len(statistical_outliers),
                "sudden_changes": len(sudden_changes),
                "unusual_relationships": len(unusual_relationships),
                "median_volume": float(volume_median),
                "anomaly_proportion": anomaly_proportion,
            },
        )

    def _detect_ml_anomalies(self, df: pd.DataFrame) -> ValidationResult:
        """ML-based anomaly detection using Isolation Forest."""
        min_points = self.rule_engine.get_threshold("min_data_points")

        if len(df) < min_points:
            return ValidationResult(
                rule_name="ml_anomalies",
                severity=ValidationSeverity.INFO,
                message=f"Insufficient data for ML detection "
                f"(need {min_points}, got {len(df)})",
                affected_rows=[],
                metadata={},
            )

        try:
            # Prepare features for ML model
            features: List[np.ndarray] = []
            feature_names: List[str] = []

            # Price-based features
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                df_features = df[["open", "high", "low", "close"]].copy()

                # Add derived features
                df_features["price_range"] = df_features["high"] - df_features["low"]
                close_open_diff = df_features["close"] - df_features["open"]
                df_features["body_size"] = np.abs(close_open_diff)
                max_vals = df_features[["open", "close"]].max(axis=1)
                min_vals = df_features[["open", "close"]].min(axis=1)
                df_features["upper_shadow"] = df_features["high"] - max_vals
                df_features["lower_shadow"] = min_vals - df_features["low"]

                if "volume" in df.columns:
                    df_features["volume"] = np.log(df["volume"] + 1e-6)
                    df_features["price_volume"] = (
                        df_features["close"] * df_features["volume"]
                    )

                # Add rolling statistics
                for window in [5, 10]:
                    if len(df) > window:
                        close_series = pd.Series(df_features["close"], index=df.index)
                        df_features[f"close_ma_{window}"] = close_series.rolling(
                            window
                        ).mean()
                        if "volume" in df_features.columns:
                            volume_series = pd.Series(
                                df_features["volume"], index=df.index
                            )
                            df_features[f"volume_ma_{window}"] = volume_series.rolling(
                                window
                            ).mean()
                            
                # Add price momentum features
                if len(df) > 1:
                    df_sorted = df.sort_values("timestamp")
                    df_features["price_change"] = df_sorted["close"].pct_change()
                    df_features["price_acceleration"] = df_features["price_change"].diff()
                    
                    # Add volatility features
                    df_features["volatility"] = df_sorted["close"].rolling(5).std() / df_sorted["close"].rolling(5).mean()

                # Remove NaN values
                df_features = df_features.bfill()
                df_features = df_features.ffill()
                features = [df_features.values]
                feature_names = df_features.columns.tolist()

            if not features or len(features[0]) == 0:
                return ValidationResult(
                    rule_name="ml_anomalies",
                    severity=ValidationSeverity.WARNING,
                    message="No suitable features for ML anomaly detection",
                    affected_rows=[],
                    metadata={},
                )

            # Scale features
            features_array = np.vstack(features)
            features_scaled = self.scaler.fit_transform(features_array)

            # Initialize and fit Isolation Forest
            contamination = self.rule_engine.get_threshold(
                "isolation_forest_contamination"
            )
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
            )

            # Predict anomalies
            anomaly_labels = self.isolation_forest.fit_predict(features_scaled)
            anomaly_scores = self.isolation_forest.score_samples(features_scaled)

            # Get anomaly indices
            anomaly_indices = df.index[anomaly_labels == -1].tolist()
            
            # Calculate anomaly proportion
            anomaly_proportion = len(anomaly_indices) / len(df) if len(df) > 0 else 0
            
            # Determine severity based on proportion of anomalies
            if anomaly_proportion > 0.1:  # More than 10% are anomalies
                severity = ValidationSeverity.ERROR
            elif anomaly_proportion > 0.05:  # More than 5% are anomalies
                severity = ValidationSeverity.WARNING
            elif anomaly_indices:
                severity = ValidationSeverity.INFO
            else:
                severity = ValidationSeverity.INFO

            message = (
                f"ML detected {len(anomaly_indices)} anomalies ({anomaly_proportion:.1%} of data)"
                if anomaly_indices
                else "No ML anomalies detected"
            )
            
            # Save the trained model for future use
            self.save_ml_model()

            return ValidationResult(
                rule_name="ml_anomalies",
                severity=severity,
                message=message,
                affected_rows=anomaly_indices,
                metadata={
                    "features_used": feature_names,
                    "contamination_rate": contamination,
                    "anomaly_scores": anomaly_scores.tolist(),
                    "model_type": "IsolationForest",
                    "anomaly_proportion": anomaly_proportion,
                },
            )

        except (ValueError, RuntimeError) as e:
            logger.error("Error in ML anomaly detection: %s", e)
            return ValidationResult(
                rule_name="ml_anomalies",
                severity=ValidationSeverity.ERROR,
                message=f"ML anomaly detection failed: {str(e)}",
                affected_rows=[],
                metadata={"error": str(e)},
            )

    def _detect_clustering_anomalies(self, df: pd.DataFrame) -> ValidationResult:
        """Anomaly detection using DBSCAN clustering."""
        min_points = self.rule_engine.get_threshold("min_data_points")

        if len(df) < min_points:
            return ValidationResult(
                rule_name="clustering_anomalies",
                severity=ValidationSeverity.INFO,
                message=f"Insufficient data for clustering detection "
                f"(need {min_points}, got {len(df)})",
                affected_rows=[],
                metadata={},
            )

        try:
            # Prepare features for clustering
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                # Use price and volume features
                features = df[["open", "high", "low", "close"]].copy()
                
                if "volume" in df.columns:
                    features["volume"] = np.log(df["volume"] + 1e-6)
                
                # Add derived features
                features["price_range"] = features["high"] - features["low"]
                features["body_size"] = np.abs(features["close"] - features["open"])
                
                # Scale features
                features_scaled = self.scaler.fit_transform(features)
                
                # Apply DBSCAN clustering
                eps = self.rule_engine.get_threshold("dbscan_eps")
                min_samples = int(self.rule_engine.get_threshold("dbscan_min_samples"))
                
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = dbscan.fit_predict(features_scaled)
                
                # Points labeled as -1 are considered outliers/anomalies
                anomaly_indices = df.index[clusters == -1].tolist()
                
                # Calculate cluster statistics
                unique_clusters = np.unique(clusters)
                cluster_counts = {int(c): int((clusters == c).sum()) for c in unique_clusters if c != -1}
                
                # Calculate anomaly proportion
                anomaly_proportion = len(anomaly_indices) / len(df) if len(df) > 0 else 0
                
                # Determine severity based on proportion of anomalies
                if anomaly_proportion > 0.1:  # More than 10% are anomalies
                    severity = ValidationSeverity.ERROR
                elif anomaly_proportion > 0.05:  # More than 5% are anomalies
                    severity = ValidationSeverity.WARNING
                elif anomaly_indices:
                    severity = ValidationSeverity.INFO
                else:
                    severity = ValidationSeverity.INFO
                
                message = (
                    f"Clustering detected {len(anomaly_indices)} anomalies ({anomaly_proportion:.1%} of data)"
                    if anomaly_indices
                    else "No clustering anomalies detected"
                )
                
                return ValidationResult(
                    rule_name="clustering_anomalies",
                    severity=severity,
                    message=message,
                    affected_rows=anomaly_indices,
                    metadata={
                        "cluster_counts": cluster_counts,
                        "eps": eps,
                        "min_samples": min_samples,
                        "anomaly_proportion": anomaly_proportion,
                        "model_type": "DBSCAN",
                    },
                )
            else:
                return ValidationResult(
                    rule_name="clustering_anomalies",
                    severity=ValidationSeverity.WARNING,
                    message="Required price columns not available for clustering",
                    affected_rows=[],
                    metadata={},
                )
                
        except Exception as e:
            logger.error("Error in clustering anomaly detection: %s", e)
            return ValidationResult(
                rule_name="clustering_anomalies",
                severity=ValidationSeverity.ERROR,
                message=f"Clustering anomaly detection failed: {str(e)}",
                affected_rows=[],
                metadata={"error": str(e)},
            )

    def _check_price_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Check price consistency and detect sudden price changes."""
        if len(df) < 2:
            return ValidationResult(
                rule_name="price_consistency",
                severity=ValidationSeverity.INFO,
                message="Insufficient data for price consistency check",
                affected_rows=[],
                metadata={},
            )

        df_sorted = df.sort_values("timestamp")
        price_changes = df_sorted["close"].pct_change().abs()

        threshold = self.rule_engine.get_threshold("price_change_threshold")
        sudden_changes = price_changes[price_changes > threshold]
        affected_indices = sudden_changes.index.tolist()
        
        # Calculate additional metrics
        max_change = float(price_changes.max()) if len(price_changes) > 0 else 0
        avg_change = float(price_changes.mean()) if len(price_changes) > 0 else 0
        
        # Check for price reversals (price changes direction frequently)
        if len(df_sorted) > 3:
            price_diff = df_sorted["close"].diff()
            direction_changes = ((price_diff > 0) != (price_diff.shift(1) > 0))
            reversal_count = direction_changes.sum()
            reversal_rate = reversal_count / (len(df_sorted) - 2)  # Exclude first two rows
            
            # High reversal rate might indicate noisy data
            high_reversal = reversal_rate > 0.5  # More than 50% of points change direction
        else:
            reversal_count = 0
            reversal_rate = 0
            high_reversal = False
        
        # Determine severity based on findings
        if len(affected_indices) > len(df) * 0.1 or high_reversal:  # More than 10% have sudden changes
            severity = ValidationSeverity.ERROR
        elif len(affected_indices) > 0:
            severity = ValidationSeverity.WARNING
        else:
            severity = ValidationSeverity.INFO
            
        # Build message
        messages = []
        if len(affected_indices) > 0:
            messages.append(f"Found {len(affected_indices)} sudden price changes")
        if high_reversal:
            messages.append(f"High price reversal rate ({reversal_rate:.1%})")
            
        message = "; ".join(messages) if messages else "Price consistency validation passed"

        return ValidationResult(
            rule_name="price_consistency",
            severity=severity,
            message=message,
            affected_rows=affected_indices,
            metadata={
                "threshold": threshold,
                "max_change": max_change,
                "avg_change": avg_change,
                "reversal_count": int(reversal_count),
                "reversal_rate": float(reversal_rate),
                "high_reversal": high_reversal,
            },
        )

    def _check_volume_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Check volume consistency and detect unusual patterns."""
        if "volume" not in df.columns or len(df) < 2:
            return ValidationResult(
                rule_name="volume_consistency",
                severity=ValidationSeverity.INFO,
                message="Volume data not available or insufficient",
                affected_rows=[],
                metadata={},
            )

        # Check for zero volume
        zero_volume = df[df["volume"] == 0].index.tolist()

        # Check for negative volume
        negative_volume = df[df["volume"] < 0].index.tolist()
        
        # Check for NaN volume
        nan_volume = df[df["volume"].isna()].index.tolist()
        
        # Check for volume trends
        if len(df) >= 10:
            df_sorted = df.sort_values("timestamp")
            
            # Calculate rolling statistics
            rolling_mean = df_sorted["volume"].rolling(window=5).mean()
            rolling_std = df_sorted["volume"].rolling(window=5).std()
            
            # Check for declining volume trend
            volume_trend = np.polyfit(range(len(df_sorted)), df_sorted["volume"].values, 1)[0]
            declining_volume = volume_trend < 0 and abs(volume_trend) > rolling_mean.mean() * 0.01
            
            # Check for increasing volatility in volume
            if len(df_sorted) >= 10:
                early_std = df_sorted["volume"].iloc[:len(df_sorted)//2].std()
                late_std = df_sorted["volume"].iloc[len(df_sorted)//2:].std()
                increasing_volatility = late_std > early_std * 1.5  # 50% increase in volatility
            else:
                increasing_volatility = False
        else:
            declining_volume = False
            increasing_volatility = False
            volume_trend = 0

        issues = []
        affected_rows = []

        if zero_volume:
            zero_allowed = self.rule_engine.get_threshold("zero_volume_allowed")
            if not zero_allowed:
                issues.append(f"{len(zero_volume)} zero volume entries")
                affected_rows.extend(zero_volume)

        if negative_volume:
            issues.append(f"{len(negative_volume)} negative volume entries")
            affected_rows.extend(negative_volume)
            
        if nan_volume:
            issues.append(f"{len(nan_volume)} NaN volume entries")
            affected_rows.extend(nan_volume)
            
        if declining_volume:
            issues.append("Declining volume trend")
            
        if increasing_volatility:
            issues.append("Increasing volume volatility")

        # Determine severity based on findings
        if negative_volume:
            severity = ValidationSeverity.ERROR
        elif nan_volume:
            severity = ValidationSeverity.ERROR
        elif zero_volume and not self.rule_engine.get_threshold("zero_volume_allowed"):
            severity = ValidationSeverity.WARNING
        elif declining_volume and increasing_volatility:
            severity = ValidationSeverity.WARNING
        elif declining_volume or increasing_volatility:
            severity = ValidationSeverity.INFO
        else:
            severity = ValidationSeverity.INFO
            
        message = (
            "; ".join(issues) if issues else "Volume consistency validation passed"
        )

        return ValidationResult(
            rule_name="volume_consistency",
            severity=severity,
            message=message,
            affected_rows=list(set(affected_rows)),
            metadata={
                "zero_volume_count": len(zero_volume),
                "negative_volume_count": len(negative_volume),
                "nan_volume_count": len(nan_volume),
                "volume_trend": float(volume_trend),
                "declining_volume": declining_volume,
                "increasing_volatility": increasing_volatility,
            },
        )

    def _check_cross_exchange_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """Check price consistency across exchanges."""
        # This is a placeholder for cross-exchange validation
        # In a real implementation, this would compare data from multiple exchanges
        # For now, we'll just return a placeholder result
        return ValidationResult(
            rule_name="cross_exchange_consistency",
            severity=ValidationSeverity.INFO,
            message="Cross-exchange consistency check not implemented",
            affected_rows=[],
            metadata={},
        )

    def check_price_integrity(self, df: pd.DataFrame) -> List[str]:
        """Validate OHLC price relationships and detect integrity violations."""
        errors = []
        mask_high_low = df["high"] < df["low"]
        mask_open_high = df["open"] > df["high"]
        mask_open_low = df["open"] < df["low"]
        mask_close_high = df["close"] > df["high"]
        mask_close_low = df["close"] < df["low"]

        if mask_high_low.any():
            errors.append(f"{mask_high_low.sum()} High < Low violations")
        if mask_open_high.any():
            errors.append(f"{mask_open_high.sum()} Open > High violations")
        if mask_open_low.any():
            errors.append(f"{mask_open_low.sum()} Open < Low violations")
        if mask_close_high.any():
            errors.append(f"{mask_close_high.sum()} Close > High violations")
        if mask_close_low.any():
            errors.append(f"{mask_close_low.sum()} Close < Low violations")

        mask_low_negative = df["low"] < 0
        if mask_low_negative.any():
            errors.append(f"{mask_low_negative.sum()} Low < 0 violations")

        return errors

    def check_time_continuity(self, df: pd.DataFrame, timeframe: str) -> List[tuple]:
        """Check for gaps in time series data."""
        df_sorted = df.sort_values("timestamp")
        time_diffs = df_sorted["timestamp"].diff().dropna()

        if timeframe == "1m":
            expected_delta = timedelta(minutes=1)
        elif timeframe == "5m":
            expected_delta = timedelta(minutes=5)
        elif timeframe == "1h":
            expected_delta = timedelta(hours=1)
        elif timeframe == "4h":
            expected_delta = timedelta(hours=4)
        elif timeframe == "1d":
            expected_delta = timedelta(days=1)
        else:
            expected_delta = timedelta(minutes=1)

        max_allowed_gap = expected_delta * 1.15
        gaps = time_diffs[time_diffs > max_allowed_gap]
        gap_details = [
            (str(gap_start), gap_length) for gap_start, gap_length in gaps.items()
        ]

        return gap_details if not gaps.empty else []

    def detect_outliers(self, df: pd.DataFrame, n_sigmas: int = 5) -> List[int]:
        """Identify statistical outliers using modified Z-score."""
        try:
            median = df["close"].median()
            mad = np.median(np.abs(df["close"] - median))

            if mad == 0:
                return []

            scaled_diff = 0.6745 * (df["close"] - median) / mad
            outlier_mask = abs(scaled_diff) > n_sigmas
            return df[outlier_mask].index.tolist()
        except Exception as e:
            logger.error("Error in outlier detection: %s", e)
            return []

    def detect_volume_anomalies(self, df: pd.DataFrame, n_sigmas: int = 5) -> List[int]:
        """Detect volume anomalies using log-transformed data."""
        log_volume = np.log(df["volume"] + 1e-6)
        median = log_volume.median()
        mad = np.median(np.abs(log_volume - median))

        if mad == 0:
            return []

        scaled_diff = 0.6745 * (log_volume - median) / mad
        outlier_mask = abs(scaled_diff) > n_sigmas
        return df[outlier_mask].index.tolist()


# For backward compatibility
DataValidator = AdvancedDataValidator