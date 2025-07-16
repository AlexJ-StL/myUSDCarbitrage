"""Advanced data validation framework for USDC arbitrage application."""

import logging
from datetime import timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

from .database import DBConnector

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


@dataclass
class DataQualityScore:
    """Data quality scoring result."""

    overall_score: float
    integrity_score: float
    completeness_score: float
    consistency_score: float
    anomaly_score: float
    details: Dict[str, Any]


class ValidationRuleEngine:
    """Configurable validation rule engine."""

    def __init__(self):
        """Initialize validation rule engine."""
        self.rules = {}
        self.thresholds = {
            "outlier_zscore": 3.0,
            "volume_anomaly_zscore": 3.0,
            "price_change_threshold": 0.1,  # 10% price change threshold
            "volume_spike_threshold": 5.0,  # 5x volume spike threshold
            "gap_tolerance": 1.15,  # 15% tolerance for time gaps
            "isolation_forest_contamination": 0.1,  # 10% contamination rate
            "min_data_points": 100,  # Minimum data points for analysis
        }

    def add_rule(self, name: str, func: callable, enabled: bool = True) -> None:
        """Add a custom validation rule."""
        self.rules[name] = {"function": func, "enabled": enabled}

    def set_threshold(self, name: str, value: float) -> None:
        """Set threshold for validation rules."""
        self.thresholds[name] = value

    def get_threshold(self, name: str) -> float:
        """Get threshold value."""
        return self.thresholds.get(name, 0.0)

    def enable_rule(self, name: str, enabled: bool = True) -> None:
        """Enable or disable a validation rule."""
        if name in self.rules:
            self.rules[name]["enabled"] = enabled


class AdvancedDataValidator:
    """Advanced data validation framework with ML-based anomaly detection."""

    def __init__(self, connection_string: str):
        """Initialize advanced data validator."""
        self.db = DBConnector(connection_string)
        self.rule_engine = ValidationRuleEngine()
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self._setup_default_rules()

    def calculate_data_quality_score(
        self, validation_results: List[ValidationResult]
    ) -> DataQualityScore:
        """Calculate comprehensive data quality score based on validation results."""
        # Initialize scores
        integrity_score = 1.0
        completeness_score = 1.0
        consistency_score = 1.0
        anomaly_score = 1.0

        # Weight factors for different validation aspects
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
            for rule_name, rule_config in self.rule_engine.rules.items():
                if not rule_config["enabled"]:
                    continue

                try:
                    if rule_name == "time_continuity":
                        result = rule_config["function"](df, timeframe)
                    else:
                        result = rule_config["function"](df)

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

    def _setup_default_rules(self) -> None:
        """Setup default validation rules."""
        self.rule_engine.add_rule("price_integrity", self._check_price_integrity)
        self.rule_engine.add_rule("time_continuity", self._check_time_continuity)
        self.rule_engine.add_rule(
            "statistical_outliers", self._detect_statistical_outliers
        )
        self.rule_engine.add_rule("volume_anomalies", self._detect_volume_anomalies)
        self.rule_engine.add_rule("ml_anomalies", self._detect_ml_anomalies)
        self.rule_engine.add_rule("price_consistency", self._check_price_consistency)
        self.rule_engine.add_rule("volume_consistency", self._check_volume_consistency)

    def enable_rule(self, rule_name: str, enabled: bool = True) -> None:
        """Enable or disable a specific validation rule."""
        self.rule_engine.enable_rule(rule_name, enabled)

    def set_threshold(self, name: str, value: float) -> None:
        """Set threshold for validation rules."""
        self.rule_engine.set_threshold(name, value)

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

        severity = ValidationSeverity.CRITICAL if errors else ValidationSeverity.INFO
        message = "; ".join(errors) if errors else "Price integrity validation passed"

        return ValidationResult(
            rule_name="price_integrity",
            severity=severity,
            message=message,
            affected_rows=list(set(affected_rows)),
            metadata={"violation_counts": {k: v.sum() for k, v in violations.items()}},
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

        severity = (
            ValidationSeverity.WARNING if len(gaps) > 0 else ValidationSeverity.INFO
        )
        message = (
            f"Found {len(gaps)} time gaps"
            if gaps.any()
            else "Time continuity validation passed"
        )

        return ValidationResult(
            rule_name="time_continuity",
            severity=severity,
            message=message,
            affected_rows=gap_indices,
            metadata={
                "expected_delta": str(expected_delta),
                "max_gaps": len(gaps),
                "largest_gap": str(gaps.max()) if len(gaps) > 0 else None,
            },
        )

    def _detect_statistical_outliers(self, df: pd.DataFrame) -> ValidationResult:
        """Detect statistical outliers using multiple methods."""
        outlier_indices = []
        methods_used = []

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

        # IQR method for additional validation
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                iqr_outliers = df[
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ].index.tolist()
                outlier_indices.extend(iqr_outliers)
                methods_used.append(f"IQR ({col})")

        unique_outliers = list(set(outlier_indices))
        severity = (
            ValidationSeverity.WARNING if unique_outliers else ValidationSeverity.INFO
        )
        message = (
            f"Found {len(unique_outliers)} statistical outliers"
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

        all_anomalies = list(set(volume_spikes + statistical_outliers))
        severity = (
            ValidationSeverity.WARNING if all_anomalies else ValidationSeverity.INFO
        )
        message = (
            f"Found {len(all_anomalies)} volume anomalies"
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
                "median_volume": float(volume_median),
            },
        )

    def _detect_ml_anomalies(self, df: pd.DataFrame) -> ValidationResult:
        """ML-based anomaly detection using Isolation Forest."""
        min_points = self.rule_engine.get_threshold("min_data_points")

        if len(df) < min_points:
            return ValidationResult(
                rule_name="ml_anomalies",
                severity=ValidationSeverity.INFO,
                message=f"Insufficient data for ML anomaly detection (need {min_points}, got {len(df)})",
                affected_rows=[],
                metadata={},
            )

        try:
            # Prepare features for ML model
            features = []
            feature_names = []

            # Price-based features
            if all(col in df.columns for col in ["open", "high", "low", "close"]):
                df_features = df[["open", "high", "low", "close"]].copy()

                # Add derived features
                df_features["price_range"] = df_features["high"] - df_features["low"]
                df_features["body_size"] = abs(
                    df_features["close"] - df_features["open"]
                )
                df_features["upper_shadow"] = df_features["high"] - df_features[
                    ["open", "close"]
                ].max(axis=1)
                df_features["lower_shadow"] = (
                    df_features[["open", "close"]].min(axis=1) - df_features["low"]
                )

                if "volume" in df.columns:
                    df_features["volume"] = np.log(df["volume"] + 1e-6)
                    df_features["price_volume"] = (
                        df_features["close"] * df_features["volume"]
                    )

                # Add rolling statistics
                for window in [5, 10]:
                    if len(df) > window:
                        df_features[f"close_ma_{window}"] = (
                            df_features["close"].rolling(window).mean()
                        )
                        df_features[f"volume_ma_{window}"] = (
                            df_features.get("volume", pd.Series(0, index=df.index))
                            .rolling(window)
                            .mean()
                        )

                # Remove NaN values
                df_features = df_features.fillna(method="bfill").fillna(method="ffill")
                features = df_features.values
                feature_names = df_features.columns.tolist()

            if len(features) == 0:
                return ValidationResult(
                    rule_name="ml_anomalies",
                    severity=ValidationSeverity.WARNING,
                    message="No suitable features for ML anomaly detection",
                    affected_rows=[],
                    metadata={},
                )

            # Scale features
            features_scaled = self.scaler.fit_transform(features)

            # Initialize and fit Isolation Forest
            contamination = self.rule_engine.get_threshold(
                "isolation_forest_contamination"
            )
            self.isolation_forest = IsolationForest(
                contamination=contamination, random_state=42, n_estimators=100
            )

            # Predict anomalies
            anomaly_labels = self.isolation_forest.fit_predict(features_scaled)
            anomaly_scores = self.isolation_forest.score_samples(features_scaled)

            # Get anomaly indices
            anomaly_indices = df.index[anomaly_labels == -1].tolist()

            severity = (
                ValidationSeverity.WARNING
                if anomaly_indices
                else ValidationSeverity.INFO
            )
            message = (
                f"ML detected {len(anomaly_indices)} anomalies"
                if anomaly_indices
                else "No ML anomalies detected"
            )

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
                },
            )

        except Exception as e:
            logger.error("Error in ML anomaly detection: %s", e)
            return ValidationResult(
                rule_name="ml_anomalies",
                severity=ValidationSeverity.ERROR,
                message=f"ML anomaly detection failed: {str(e)}",
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

        severity = (
            ValidationSeverity.WARNING
            if len(affected_indices) > 0
            else ValidationSeverity.INFO
        )
        message = (
            f"Found {len(affected_indices)} sudden price changes"
            if affected_indices
            else "Price consistency validation passed"
        )

        return ValidationResult(
            rule_name="price_consistency",
            severity=severity,
            message=message,
            affected_rows=affected_indices,
            metadata={
                "threshold": threshold,
                "max_change": (
                    float(price_changes.max()) if len(price_changes) > 0 else 0
                ),
                "avg_change": (
                    float(price_changes.mean()) if len(price_changes) > 0 else 0
                ),
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

        issues = []
        affected_rows = []

        if zero_volume:
            issues.append(f"{len(zero_volume)} zero volume entries")
            affected_rows.extend(zero_volume)

        if negative_volume:
            issues.append(f"{len(negative_volume)} negative volume entries")
            affected_rows.extend(negative_volume)

        severity = (
            ValidationSeverity.ERROR
            if negative_volume
            else (
                ValidationSeverity.WARNING if zero_volume else ValidationSeverity.INFO
            )
        )
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
            },
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
        anomaly_mask = abs(scaled_diff) > n_sigmas
        return df[anomaly_mask].index.tolist()

    def detect_changepoints(self, df: pd.DataFrame) -> List[int]:
        """Detect structural breaks in price series using CUSUM."""
        if len(df) < 100:
            return []

        values = df["close"].values
        cumulative_sum = np.cumsum(values - np.mean(values))
        cumulative_sum_abs = np.abs(cumulative_sum)
        max_change_idx = np.argmax(cumulative_sum_abs)

        if cumulative_sum_abs[max_change_idx] > 10 * np.std(values):
            return [df.index[max_change_idx]]

        return []

    def validate_data(
        self, exchange: str, symbol: str, timeframe: str
    ) -> Dict[str, Any]:
        """Run all enabled validation rules"""
        logger.info("Validating %s/%s/%s", exchange, symbol, timeframe)
        validation_results = {}
        try:
            df = self.db.get_ohlcv_data(exchange, symbol, timeframe)

            # Check for missing required columns
            required_columns = ["open", "close", "high", "low", "volume", "timestamp"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                validation_results["missing_values"] = missing_columns
                logger.warning("Missing columns found: %s", missing_columns)
                return validation_results

            # Check for null values
            null_mask = df.isnull().any(axis=1)
            if null_mask.any():
                null_count = null_mask.sum()
                logger.warning(
                    "%s missing values found in data for %s/%s/%s",
                    null_count,
                    exchange,
                    symbol,
                    timeframe,
                )
                validation_results["null_values"] = [
                    f"{null_count} null value(s) found"
                ]
            else:
                logger.info(
                    "Data validation passed for %s/%s/%s", exchange, symbol, timeframe
                )
        except Exception as e:
            logger.error("Data retrieval failed: %s", e)
            validation_results["price_errors"] = ["Data retrieval failed"]
        return validation_results

    def validate_dataset(self, exchange, symbol, timeframe):
        """Validate a dataset for a given exchange, symbol, and timeframe"""
        logger.info("Validating dataset for %s/%s/%s", exchange, symbol, timeframe)
        validation_results = {}
        try:
            df = self.db.get_ohlcv_data(exchange, symbol, timeframe)
            # Detect outliers using modified Z-score
            median = df["close"].median()
            mad = np.median(np.abs(df["close"] - median))

            if mad == 0:
                validation_results["outliers"] = []
            else:
                scaled_diff = 0.6745 * (df["close"] - median) / mad
                outlier_mask = abs(scaled_diff) > 3  # Using 3 sigma threshold
                validation_results["outliers"] = df[outlier_mask].index.tolist()
        except Exception as e:
            logger.error("Error in dataset validation: %s", e)
            validation_results["price_errors"] = ["Error in dataset validation"]
        return validation_results


# Backward compatibility alias
DataValidator = AdvancedDataValidator
