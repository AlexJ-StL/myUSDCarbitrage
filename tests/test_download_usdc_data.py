import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone
import ccxt

from download_usdc_data import fetch_ohlcv, main


class TestDownloadUSDCData(unittest.TestCase):

    @patch("download_usdc_data.ccxt")
    @patch("download_usdc_data.logger")
    def test_fetch_ohlcv_success(self, mock_logger, mock_ccxt):
        mock_exchange = MagicMock()
        mock_exchange.timeframes = {"1h": 3600}
        mock_exchange.fetch_ohlcv.return_value = [
            [1609459200000, 1.1, 1.2, 1.0, 1.15, 1000],
            [1609462800000, 1.15, 1.25, 1.1, 1.2, 1500],
        ]
        mock_ccxt.coinbase.return_value = mock_exchange

        result = fetch_ohlcv(
            mock_exchange, "USDC/USD", "1h", 1609459200000, 1609466400000
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 1609459200000)
        self.assertEqual(result[1][0], 1609462800000)
        mock_logger.info.assert_called_with("Fetching from 2021-01-01 00:00:00+00:00")

    @patch("download_usdc_data.ccxt")
    @patch("download_usdc_data.logger")
    def test_fetch_ohlcv_network_error(self, mock_logger, mock_ccxt):
        mock_exchange = MagicMock()
        mock_exchange.timeframes = {"1h": 3600}
        mock_exchange.fetch_ohlcv.side_effect = ccxt.NetworkError("Network error")
        mock_ccxt.coinbase.return_value = mock_exchange

        result = fetch_ohlcv(
            mock_exchange, "USDC/USD", "1h", 1609459200000, 1609466400000
        )
        self.assertEqual(len(result), 0)
        mock_logger.error.assert_called_with("Network error: Network error")

    @patch("download_usdc_data.ccxt")
    @patch("download_usdc_data.logger")
    def test_fetch_ohlcv_exchange_error(self, mock_logger, mock_ccxt):
        mock_exchange = MagicMock()
        mock_exchange.timeframes = {"1h": 3600}
        mock_exchange.fetch_ohlcv.side_effect = ccxt.ExchangeError("Exchange error")
        mock_ccxt.coinbase.return_value = mock_exchange

        result = fetch_ohlcv(
            mock_exchange, "USDC/USD", "1h", 1609459200000, 1609466400000
        )
        self.assertEqual(len(result), 0)
        mock_logger.error.assert_called_with("Exchange error: Exchange error")

    @patch("download_usdc_data.ccxt")
    @patch("download_usdc_data.logger")
    def test_fetch_ohlcv_unexpected_error(self, mock_logger, mock_ccxt):
        mock_exchange = MagicMock()
        mock_exchange.timeframes = {"1h": 3600}
        mock_exchange.fetch_ohlcv.side_effect = Exception("Unexpected error")
        mock_ccxt.coinbase.return_value = mock_exchange

        result = fetch_ohlcv(
            mock_exchange, "USDC/USD", "1h", 1609459200000, 1609466400000
        )
        self.assertEqual(len(result), 0)
        mock_logger.error.assert_called_with("Unexpected error: Unexpected error")

    @patch("download_usdc_data.fetch_ohlcv")
    @patch("download_usdc_data.DataValidator")
    @patch("download_usdc_data.ccxt")
    @patch("download_usdc_data.logger")
    def test_main_success(
        self, mock_logger, mock_ccxt, mock_validator, mock_fetch_ohlcv
    ):
        mock_exchange = MagicMock()
        mock_exchange.timeframes = {"1h": 3600}
        mock_exchange.symbols = ["USDC/USD"]
        mock_ccxt.coinbase.return_value = mock_exchange
        mock_ccxt.kraken.return_value = mock_exchange
        mock_ccxt.binance.return_value = mock_exchange

        mock_fetch_ohlcv.return_value = [
            [1609459200000, 1.1, 1.2, 1.0, 1.15, 1000],
            [1609462800000, 1.15, 1.25, 1.1, 1.2, 1500],
        ]

        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_data.return_value = {"price_errors": []}
        mock_validator.return_value = mock_validator_instance

        main()

        mock_logger.info.assert_called_with("Inserting 2 records into database...")
        mock_validator_instance.validate_data.assert_called_with(
            "coinbase", "USDC/USD", "1h"
        )

    @patch("download_usdc_data.fetch_ohlcv")
    @patch("download_usdc_data.DataValidator")
    @patch("download_usdc_data.ccxt")
    @patch("download_usdc_data.logger")
    def test_main_critical_error(
        self, mock_logger, mock_ccxt, mock_validator, mock_fetch_ohlcv
    ):
        mock_exchange = MagicMock()
        mock_exchange.timeframes = {"1h": 3600}
        mock_exchange.symbols = ["USDC/USD"]
        mock_ccxt.coinbase.return_value = mock_exchange
        mock_ccxt.kraken.return_value = mock_exchange
        mock_ccxt.binance.return_value = mock_exchange

        mock_fetch_ohlcv.return_value = [
            [1609459200000, 1.1, 1.2, 1.0, 1.15, 1000],
            [1609462800000, 1.15, 1.25, 1.1, 1.2, 1500],
        ]

        mock_validator_instance = MagicMock()
        mock_validator_instance.validate_data.return_value = {"price_errors": ["Error"]}
        mock_validator.return_value = mock_validator_instance

        main()

        mock_logger.error.assert_called_with("Critical data issues detected")



