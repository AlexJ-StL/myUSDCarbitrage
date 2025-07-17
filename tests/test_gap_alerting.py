"""Test module for gap alerting functionality."""

import sys
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from datetime import datetime, timedelta

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from api.gap_alerting import (
    AlertConfig,
    AlertManager,
    GapAlertingSystem,
    GapSeverity,
    GapInfo,
    GapAnalysisReport,
)


@patch("api.gap_alerting.smtplib.SMTP")
def test_alert_manager_initialization(mock_smtp):
    """Test initialization of AlertManager."""
    # Test with explicit parameters
    manager = AlertManager(
        smtp_server="smtp.example.com",
        smtp_port=587,
        smtp_username="user",
        smtp_password="pass",
        sender_email="sender@example.com",
    )

    assert manager.smtp_server == "smtp.example.com"
    assert manager.smtp_port == 587
    assert manager.smtp_username == "user"
    assert manager.smtp_password == "pass"
    assert manager.sender_email == "sender@example.com"
    assert isinstance(manager.last_alert_time, dict)

    # Test with environment variables
    with patch.dict(
        os.environ,
        {
            "SMTP_SERVER": "env.example.com",
            "SMTP_USERNAME": "env_user",
            "SMTP_PASSWORD": "env_pass",
            "SENDER_EMAIL": "env_sender@example.com",
        },
    ):
        manager = AlertManager()
        assert manager.smtp_server == "env.example.com"
        assert manager.smtp_username == "env_user"
        assert manager.smtp_password == "env_pass"
        assert manager.sender_email == "env_sender@example.com"


def test_should_send_alert():
    """Test should_send_alert method."""
    manager = AlertManager()
    config = AlertConfig(
        enabled=True,
        min_severity=GapSeverity.MODERATE,
        alert_cooldown=timedelta(hours=1),
    )

    # Test with severity below threshold
    assert not manager.should_send_alert(
        "coinbase", "USDC/USD", "1h", GapSeverity.MINOR, config
    )

    # Test with severity at threshold
    assert manager.should_send_alert(
        "coinbase", "USDC/USD", "1h", GapSeverity.MODERATE, config
    )

    # Test with severity above threshold
    assert manager.should_send_alert(
        "coinbase", "USDC/USD", "1h", GapSeverity.CRITICAL, config
    )

    # Test with disabled alerts
    config.enabled = False
    assert not manager.should_send_alert(
        "coinbase", "USDC/USD", "1h", GapSeverity.CRITICAL, config
    )

    # Test cooldown
    config.enabled = True
    key = "coinbase_USDC/USD_1h"
    manager.last_alert_time[key] = datetime.now() - timedelta(minutes=30)
    assert not manager.should_send_alert(
        "coinbase", "USDC/USD", "1h", GapSeverity.CRITICAL, config
    )

    # Test after cooldown
    manager.last_alert_time[key] = datetime.now() - timedelta(hours=2)
    assert manager.should_send_alert(
        "coinbase", "USDC/USD", "1h", GapSeverity.CRITICAL, config
    )


@patch("api.gap_alerting.smtplib.SMTP")
def test_send_email_alert(mock_smtp):
    """Test send_email_alert method."""
    manager = AlertManager(
        smtp_server="smtp.example.com", smtp_username="user", smtp_password="pass"
    )

    # Test successful email
    mock_smtp_instance = MagicMock()
    mock_smtp.return_value.__enter__.return_value = mock_smtp_instance

    result = manager.send_email_alert(
        ["recipient@example.com"], "Test Subject", "<p>HTML Body</p>", "Text Body"
    )

    assert result is True
    mock_smtp.assert_called_once_with("smtp.example.com", 587)
    mock_smtp_instance.starttls.assert_called_once()
    mock_smtp_instance.login.assert_called_once_with("user", "pass")
    mock_smtp_instance.sendmail.assert_called_once()

    # Test with no SMTP server
    manager.smtp_server = None
    result = manager.send_email_alert(
        ["recipient@example.com"], "Test Subject", "<p>HTML Body</p>", "Text Body"
    )
    assert result is False

    # Test with no recipients
    manager.smtp_server = "smtp.example.com"
    result = manager.send_email_alert(
        [], "Test Subject", "<p>HTML Body</p>", "Text Body"
    )
    assert result is False

    # Test with SMTP error
    mock_smtp.side_effect = Exception("SMTP Error")
    result = manager.send_email_alert(
        ["recipient@example.com"], "Test Subject", "<p>HTML Body</p>", "Text Body"
    )
    assert result is False


@patch("api.gap_alerting.requests.post")
def test_send_slack_alert(mock_post):
    """Test send_slack_alert method."""
    manager = AlertManager()

    # Test successful alert
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    result = manager.send_slack_alert(
        "https://hooks.slack.com/services/xxx", "Test message"
    )

    assert result is True
    mock_post.assert_called_once()

    # Test with no webhook URL
    result = manager.send_slack_alert("", "Test message")
    assert result is False

    # Test with error response
    mock_response.status_code = 400
    mock_response.text = "Invalid payload"
    result = manager.send_slack_alert(
        "https://hooks.slack.com/services/xxx", "Test message"
    )
    assert result is False

    # Test with exception
    mock_post.side_effect = Exception("Network error")
    result = manager.send_slack_alert(
        "https://hooks.slack.com/services/xxx", "Test message"
    )
    assert result is False


@patch("api.gap_alerting.requests.post")
def test_send_teams_alert(mock_post):
    """Test send_teams_alert method."""
    manager = AlertManager()

    # Test successful alert
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    result = manager.send_teams_alert(
        "https://outlook.office.com/webhook/xxx", "Test Title", "Test message"
    )

    assert result is True
    mock_post.assert_called_once()

    # Test with no webhook URL
    result = manager.send_teams_alert("", "Test Title", "Test message")
    assert result is False

    # Test with error response
    mock_response.status_code = 400
    mock_response.text = "Invalid payload"
    result = manager.send_teams_alert(
        "https://outlook.office.com/webhook/xxx", "Test Title", "Test message"
    )
    assert result is False

    # Test with exception
    mock_post.side_effect = Exception("Network error")
    result = manager.send_teams_alert(
        "https://outlook.office.com/webhook/xxx", "Test Title", "Test message"
    )
    assert result is False


@patch("api.gap_alerting.GapDetectionSystem")
def test_format_alert_message(mock_gap_detection):
    """Test format_alert_message method."""
    manager = AlertManager()

    # Create a mock report
    gaps = [
        GapInfo(
            start_time=datetime(2023, 1, 1, 10, 0),
            end_time=datetime(2023, 1, 1, 12, 0),
            duration=timedelta(hours=2),
            severity=GapSeverity.CRITICAL,
            filled=True,
            fill_method="alternative_source",
            fill_source="binance",
            fill_quality=0.9,
        ),
        GapInfo(
            start_time=datetime(2023, 1, 2, 10, 0),
            end_time=datetime(2023, 1, 2, 11, 0),
            duration=timedelta(hours=1),
            severity=GapSeverity.MODERATE,
            filled=False,
            fill_method=None,
            fill_source=None,
            fill_quality=None,
        ),
    ]

    report = GapAnalysisReport(
        exchange="coinbase",
        symbol="USDC/USD",
        timeframe="1h",
        analysis_time=datetime(2023, 1, 3, 10, 0),
        total_gaps=2,
        filled_gaps=1,
        unfilled_gaps=1,
        total_missing_points=3,
        data_completeness=98.5,
        gaps=gaps,
        summary={"largest_gap": "2:00:00", "avg_gap_size": "1:30:00"},
    )

    # Mock the generate_gap_report method
    mock_gap_detection_instance = MagicMock()
    mock_gap_detection_instance.generate_gap_report.return_value = (
        "<html>Full report</html>"
    )
    mock_gap_detection.return_value = mock_gap_detection_instance

    # Test with include_report=True
    html, text = manager.format_alert_message(report, include_report=True)

    assert "Data Gap Alert - coinbase/USDC/USD/1h" in html
    assert "Data completeness: <strong>98.50%</strong>" in html
    assert "1 critical" in html
    assert "1 moderate" in html
    assert "0 minor" in html
    assert "Total missing points: <strong>3</strong>" in html
    assert "Filled gaps: <strong>1/2</strong>" in html
    assert "<h3>Critical Gaps</h3>" in html
    assert "2023-01-01 10:00:00" in html
    assert "2023-01-01 12:00:00" in html
    assert "<h3>Full Report</h3>" in html

    assert "Data Gap Alert - coinbase/USDC/USD/1h" in text
    assert "Data completeness: 98.50%" in text
    assert "1 critical, 1 moderate, 0 minor" in text
    assert "Total missing points: 3" in text
    assert "Filled gaps: 1/2" in text
    assert "Critical Gaps:" in text
    assert "2023-01-01 10:00:00 to 2023-01-01 12:00:00" in text

    # Test with include_report=False
    html, text = manager.format_alert_message(report, include_report=False)
    assert "<h3>Full Report</h3>" not in html
    assert "Full Report:" not in text


@patch("api.gap_alerting.AlertManager.send_email_alert")
@patch("api.gap_alerting.AlertManager.send_slack_alert")
@patch("api.gap_alerting.AlertManager.send_teams_alert")
def test_send_alert(mock_teams, mock_slack, mock_email):
    """Test send_alert method."""
    manager = AlertManager()

    # Create a mock report with critical gap
    gaps = [
        GapInfo(
            start_time=datetime(2023, 1, 1, 10, 0),
            end_time=datetime(2023, 1, 1, 12, 0),
            duration=timedelta(hours=2),
            severity=GapSeverity.CRITICAL,
            filled=True,
        )
    ]

    report = GapAnalysisReport(
        exchange="coinbase",
        symbol="USDC/USD",
        timeframe="1h",
        analysis_time=datetime(2023, 1, 3, 10, 0),
        total_gaps=1,
        filled_gaps=1,
        unfilled_gaps=0,
        total_missing_points=2,
        data_completeness=99.0,
        gaps=gaps,
        summary={},
    )

    # Set up mock returns
    mock_email.return_value = True
    mock_slack.return_value = True
    mock_teams.return_value = True

    # Test with all channels enabled
    config = AlertConfig(
        enabled=True,
        min_severity=GapSeverity.MODERATE,
        email_recipients=["test@example.com"],
        slack_webhook="https://hooks.slack.com/services/xxx",
        teams_webhook="https://outlook.office.com/webhook/xxx",
    )

    result = manager.send_alert(report, config)
    assert result is True
    mock_email.assert_called_once()
    mock_slack.assert_called_once()
    mock_teams.assert_called_once()

    # Check that last_alert_time was updated
    key = "coinbase_USDC/USD_1h"
    assert key in manager.last_alert_time

    # Test with only email enabled
    mock_email.reset_mock()
    mock_slack.reset_mock()
    mock_teams.reset_mock()
    manager.last_alert_time = {}  # Reset last alert time

    config = AlertConfig(
        enabled=True,
        min_severity=GapSeverity.MODERATE,
        email_recipients=["test@example.com"],
        slack_webhook=None,
        teams_webhook=None,
    )

    result = manager.send_alert(report, config)
    assert result is True
    mock_email.assert_called_once()
    mock_slack.assert_not_called()
    mock_teams.assert_not_called()

    # Test with severity below threshold
    mock_email.reset_mock()
    manager.last_alert_time = {}  # Reset last alert time

    # Create a report with only minor gaps
    gaps = [
        GapInfo(
            start_time=datetime(2023, 1, 1, 10, 0),
            end_time=datetime(2023, 1, 1, 10, 30),
            duration=timedelta(minutes=30),
            severity=GapSeverity.MINOR,
            filled=True,
        )
    ]

    report = GapAnalysisReport(
        exchange="coinbase",
        symbol="USDC/USD",
        timeframe="1h",
        analysis_time=datetime(2023, 1, 3, 10, 0),
        total_gaps=1,
        filled_gaps=1,
        unfilled_gaps=0,
        total_missing_points=1,
        data_completeness=99.5,
        gaps=gaps,
        summary={},
    )

    result = manager.send_alert(report, config)
    assert result is False
    mock_email.assert_not_called()


@patch("api.gap_alerting.GapDetectionSystem")
def test_gap_alerting_system_initialization(mock_gap_detection):
    """Test initialization of GapAlertingSystem."""
    system = GapAlertingSystem("dummy_connection_string")

    assert system.gap_detection is not None
    assert system.alert_manager is not None
    assert isinstance(system.default_config, AlertConfig)
    assert isinstance(system.exchange_configs, dict)

    # Test with environment variables
    with patch.dict(
        os.environ,
        {
            "ALERT_EMAIL_RECIPIENTS": "admin@example.com,alerts@example.com",
            "SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/xxx",
            "TEAMS_WEBHOOK_URL": "https://outlook.office.com/webhook/xxx",
        },
    ):
        system = GapAlertingSystem("dummy_connection_string")
        assert system.default_config.email_recipients == [
            "admin@example.com",
            "alerts@example.com",
        ]
        assert (
            system.default_config.slack_webhook
            == "https://hooks.slack.com/services/xxx"
        )
        assert (
            system.default_config.teams_webhook
            == "https://outlook.office.com/webhook/xxx"
        )


def test_alert_config_management():
    """Test alert configuration management."""
    system = GapAlertingSystem("dummy_connection_string")

    # Test default config
    config = system.get_alert_config("coinbase")
    assert config is system.default_config

    # Test setting exchange-specific config
    custom_config = AlertConfig(
        enabled=True,
        min_severity=GapSeverity.CRITICAL,
        email_recipients=["exchange@example.com"],
    )

    system.set_alert_config("coinbase", custom_config)
    config = system.get_alert_config("coinbase")
    assert config is custom_config
    assert config.min_severity == GapSeverity.CRITICAL
    assert config.email_recipients == ["exchange@example.com"]

    # Test getting config for different exchange
    config = system.get_alert_config("binance")
    assert config is system.default_config


@patch("api.gap_alerting.GapDetectionSystem")
@patch("api.gap_alerting.AlertManager")
def test_analyze_and_alert(mock_alert_manager_class, mock_gap_detection_class):
    """Test analyze_and_alert method."""
    # Set up mocks
    mock_gap_detection = MagicMock()
    mock_gap_detection_class.return_value = mock_gap_detection

    mock_alert_manager = MagicMock()
    mock_alert_manager_class.return_value = mock_alert_manager

    # Create a mock report
    gaps = [
        GapInfo(
            start_time=datetime(2023, 1, 1, 10, 0),
            end_time=datetime(2023, 1, 1, 12, 0),
            duration=timedelta(hours=2),
            severity=GapSeverity.CRITICAL,
            filled=False,
        )
    ]

    report = GapAnalysisReport(
        exchange="coinbase",
        symbol="USDC/USD",
        timeframe="1h",
        analysis_time=datetime(2023, 1, 3, 10, 0),
        total_gaps=1,
        filled_gaps=0,
        unfilled_gaps=1,
        total_missing_points=2,
        data_completeness=99.0,
        gaps=gaps,
        summary={},
    )

    mock_gap_detection.analyze_gaps.return_value = report

    # Mock the gap filling process
    mock_df = pd.DataFrame({"timestamp": [datetime(2023, 1, 1)]})
    mock_gap_detection.db.get_ohlcv_data.return_value = mock_df
    mock_gap_detection.detect_gaps.return_value = gaps

    filled_df = pd.DataFrame({"timestamp": [datetime(2023, 1, 1, 11, 0)]})
    updated_gaps = [
        GapInfo(
            start_time=datetime(2023, 1, 1, 10, 0),
            end_time=datetime(2023, 1, 1, 12, 0),
            duration=timedelta(hours=2),
            severity=GapSeverity.CRITICAL,
            filled=True,
            fill_method="interpolation",
            fill_source="interpolation",
            fill_quality=0.5,
        )
    ]
    mock_gap_detection.fill_gaps.return_value = (filled_df, updated_gaps)

    # Mock the alert sending
    mock_alert_manager.send_alert.return_value = True

    # Create the system and run analyze_and_alert
    system = GapAlertingSystem("dummy_connection_string")
    system.gap_detection = mock_gap_detection
    system.alert_manager = mock_alert_manager

    result_report, alert_sent = system.analyze_and_alert(
        "coinbase",
        "USDC/USD",
        "1h",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 2),
    )

    # Check that gap detection was called
    mock_gap_detection.analyze_gaps.assert_called_once_with(
        "coinbase", "USDC/USD", "1h", datetime(2023, 1, 1), datetime(2023, 1, 2)
    )

    # Check that gaps were filled
    mock_gap_detection.detect_gaps.assert_called_once()
    mock_gap_detection.fill_gaps.assert_called_once()
    mock_gap_detection.save_filled_data.assert_called_once_with(
        filled_df, "coinbase", "USDC/USD", "1h"
    )

    # Check that alert was sent
    mock_alert_manager.send_alert.assert_called_once()
    assert alert_sent is True

    # Check that the report was updated with filled gaps
    assert result_report.filled_gaps == 1
    assert result_report.unfilled_gaps == 0
    assert result_report.gaps == updated_gaps

    # Test with no gaps
    mock_gap_detection.reset_mock()
    mock_alert_manager.reset_mock()

    no_gaps_report = GapAnalysisReport(
        exchange="coinbase",
        symbol="USDC/USD",
        timeframe="1h",
        analysis_time=datetime(2023, 1, 3, 10, 0),
        total_gaps=0,
        filled_gaps=0,
        unfilled_gaps=0,
        total_missing_points=0,
        data_completeness=100.0,
        gaps=[],
        summary={},
    )

    mock_gap_detection.analyze_gaps.return_value = no_gaps_report

    result_report, alert_sent = system.analyze_and_alert(
        "coinbase",
        "USDC/USD",
        "1h",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 2),
    )

    # Check that no alert was sent
    mock_alert_manager.send_alert.assert_not_called()
    assert alert_sent is False


@patch("api.gap_alerting.GapAlertingSystem.analyze_and_alert")
def test_schedule_regular_checks(mock_analyze_and_alert):
    """Test schedule_regular_checks method."""
    # Set up mock
    mock_report = MagicMock()
    mock_analyze_and_alert.return_value = (mock_report, True)

    # Create the system and run schedule_regular_checks
    system = GapAlertingSystem("dummy_connection_string")

    result = system.schedule_regular_checks(
        "coinbase",
        "USDC/USD",
        "1h",
        check_interval=timedelta(hours=1),
        lookback_period=timedelta(days=1),
    )

    # Check that analyze_and_alert was called with correct parameters
    mock_analyze_and_alert.assert_called_once()
    args, kwargs = mock_analyze_and_alert.call_args
    assert args[0] == "coinbase"
    assert args[1] == "USDC/USD"
    assert args[2] == "1h"

    # Check that start_date and end_date were set correctly
    now = datetime.now()
    start_date = kwargs["start_date"]
    end_date = kwargs["end_date"]

    # Allow for small time differences during test execution
    assert abs((now - end_date).total_seconds()) < 5
    assert abs(((now - timedelta(days=1)) - start_date).total_seconds()) < 5

    # Check that the result is the report
    assert result is mock_report


def test_create_gap_alerting_system():
    """Test create_gap_alerting_system function."""
    with patch("api.gap_alerting.GapAlertingSystem") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        from api.gap_alerting import create_gap_alerting_system

        result = create_gap_alerting_system("dummy_connection_string")

        mock_class.assert_called_once_with("dummy_connection_string")
        assert result is mock_instance
