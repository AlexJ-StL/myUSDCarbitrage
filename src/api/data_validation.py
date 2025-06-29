import pandas as pd
import numpy as np
from datetime import timedelta
import logging
from .database import DBConnector

# Set up logging
logger = logging.getLogger('data_validation')
logger.setLevel(logging.INFO)
handler = logging.FileHandler('data_validation.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

class DataValidator:
    def __init__(self):
        self.db = DBConnector()
        self.rules = {
            'price_integrity': True,
            'time_continuity': True,
            'outlier_detection': True,
            'volume_anomaly': True,
            'changepoint_detection': False  # More CPU intensive
        }

    def enable_rule(self, rule_name, enabled=True):
        self.rules[rule_name] = enabled

    def check_price_integrity(self, df):
        """Rule 1: Validate OHLC relationships"""
        errors = []
        mask_high_low = df['high'] < df['low']
        mask_open_high = df['open'] > df['high']
        mask_open_low = df['open'] < df['low']
        mask_close_high = df['close'] > df['high']
        mask_close_low = df['close'] < df['low']

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

        return errors

    def check_time_continuity(self, df, timeframe):
        """Rule 2: Check for gaps in time series"""
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dropna()

        # Expected time deltas
        if timeframe == '1m': expected_delta = timedelta(minutes=1)
        elif timeframe == '5m': expected_delta = timedelta(minutes=5)
        elif timeframe == '1h': expected_delta = timedelta(hours=1)
        elif timeframe == '4h': expected_delta = timedelta(hours=4)
        elif timeframe == '1d': expected_delta = timedelta(days=1)
        else: expected_delta = timedelta(minutes=1)  # Default to 1min

        max_allowed_gap = expected_delta * 1.15  # 15% tolerance
        gaps = time_diffs[time_diffs > max_allowed_gap]
        gap_details = [(str(gap_start), gap_length)
                      for gap_start, gap_length in gaps.items()]

        return gap_details if not gaps.empty else []

    def detect_outliers(self, df, n_sigmas=5):
        """Rule 3: Identify statistical outliers using robust statistics"""
        try:
            median = df['close'].median()
            mad = np.median(np.abs(df['close'] - median))

            if mad == 0:  # Handle constant data
                return []

            scaled_diff = 0.6745 * (df['close'] - median) / mad  # consistency constant
            outlier_mask = abs(scaled_diff) > n_sigmas
            return df[outlier_mask].index.tolist()
        except:
            return []

    def detect_volume_anomalies(self, df, n_sigmas=5):
        """Rule 4: Volume spikes detection"""
        log_volume = np.log(df['volume'] + 1e-6)  # Avoid log(0)
        median = log_volume.median()
        mad = np.median(np.abs(log_volume - median))

        if mad == 0:
            return []

        scaled_diff = 0.6745 * (log_volume - median) / mad
        anomaly_mask = abs(scaled_diff) > n_sigmas
        return df[anomaly_mask].index.tolist()

    def detect_changepoints(self, df):
        """Rule 5: Detect structural breaks (CUSUM algorithm)"""
        if len(df) < 100:  # Need sufficient data
            return []

        values = df['close'].values
        cumulative_sum = np.cumsum(values - np.mean(values))
        cumulative_sum_abs = np.abs(cumulative_sum)
        max_change_idx = np.argmax(cumulative_sum_abs)

        if cumulative_sum_abs[max_change_idx] > 10 * np.std(values):
            return [df.index[max_change_idx]]
        return []

    def validate_dataset(self, exchange, symbol, timeframe):
        """Run all enabled validation rules"""
        logger.info(f"Validating {exchange}/{symbol}/{timeframe}")
        try:
            df = self.db.get_ohlcv_data(exchange, symbol, timeframe)
        except Exception as e:
            logger.error(f"Data retrieval failed: {e}")
            return {'status': 'error', 'message': str(e)}

        if df.empty:
            logger.warning("Empty dataset received")
            return {'status': 'warning', 'message': 'Empty dataset'}

        results = {'exchange': exchange, 'symbol': symbol, 'timeframe': timeframe}

        # Apply validation rules
        if self.rules['price_integrity']:
            price_errors = self.check_price_integrity(df)
            results['price_errors'] = price_errors if price_errors else []

        if self.rules['time_continuity']:
            time_gaps = self.check_time_continuity(df, timeframe)
            results['time_gaps'] = time_gaps

        if self.rules['outlier_detection']:
            outliers = self.detect_outliers(df)
            results['outliers'] = outliers

        if self.rules['volume_anomaly']:
            volume_anomalies = self.detect_volume_anomalies(df)
            results['volume_anomalies'] = volume_anomalies

        if self.rules['changepoint_detection']:
            changepoints = self.detect_changepoints(df)
            results['changepoints'] = changepoints

        self.log_validation_results(results)
        return results

    def log_validation_results(self, results):
        """Log validation results systematically"""
        issue_count = 0

        if results.get('price_errors', []):
            issue_count += len(results['price_errors'])
            for err in results['price_errors']:
                logger.warning(f"PRICE: {err}")

        if results.get('time_gaps', []):
            issue_count += len(results['time_gaps'])
            for gap in results['time_gaps']:
                logger.warning(f"TIME GAP: {gap[0]} | Length: {gap[1]}")

        if results.get('outliers', []):
            issue_count += len(results['outliers'])
            logger.warning(f"OUTLIERS: {len(results['outliers'])} detected")

        if results.get('volume_anomalies', []):
            issue_count += len(results['volume_anomalies'])
            logger.warning(f"VOLUME ANOMALIES: {len(results['volume_anomalies'])} detected")

        if results.get('changepoints', []):
            issue_count += len(results['changepoints'])
            logger.warning(f"CHANGEPOINTS: {len(results['changepoints'])} detected")

        logger.info(f"Validation complete: Found {issue_count} issues")