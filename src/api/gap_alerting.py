"""Gap analysis reporting and alerting system for USDC arbitrage application."""

import logging
import smtplib
import json
import os
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import requests

from .gap_detection import GapDetectionSystem, GapAnalysisReport, GapSeverity, GapInfo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler("data_validation.log")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


@dataclass
class AlertConfig:
    """Configuration for gap alerts."""

    enabled: bool = True
    min_severity: GapSeverity = GapSeverity.MODERATE
    email_recipients: List[str] = None
    slack_webhook: Optional[str] = None
    teams_webhook: Optional[str] = None
    alert_cooldown: timedelta = timedelta(hours=6)
    include_report: bool = True
    report_format: str = "html"


class AlertManager:
    """Manager for sending alerts about data gaps."""

    def __init__(
        self,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        smtp_username: Optional[str] = None,
        smtp_password: Optional[str] = None,
        sender_email: Optional[str] = None,
    ):
        """Initialize alert manager.

        Args:
            smtp_server: SMTP server for sending emails
            smtp_port: SMTP port
            smtp_username: SMTP username
            smtp_password: SMTP password
            sender_email: Email address to send alerts from
        """
        self.smtp_server = smtp_server or os.environ.get("SMTP_SERVER")
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username or os.environ.get("SMTP_USERNAME")
        self.smtp_password = smtp_password or os.environ.get("SMTP_PASSWORD")
        self.sender_email = sender_email or os.environ.get(
            "SENDER_EMAIL", "alerts@usdcarbitrage.com"
        )
        self.last_alert_time: Dict[str, datetime] = {}

    def should_send_alert(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        severity: GapSeverity,
        config: AlertConfig,
    ) -> bool:
        """Determine if an alert should be sent based on cooldown and severity.

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Time interval
            severity: Gap severity
            config: Alert configuration

        Returns:
            True if alert should be sent, False otherwise
        """
        if not config.enabled:
            return False

        # Check severity threshold
        severity_levels = {
            GapSeverity.MINOR: 1,
            GapSeverity.MODERATE: 2,
            GapSeverity.CRITICAL: 3,
        }

        if severity_levels[severity] < severity_levels[config.min_severity]:
            return False

        # Check cooldown
        key = f"{exchange}_{symbol}_{timeframe}"
        now = datetime.now()

        if key in self.last_alert_time:
            time_since_last = now - self.last_alert_time[key]
            if time_since_last < config.alert_cooldown:
                logger.info(
                    f"Skipping alert for {key} due to cooldown "
                    f"(last alert: {time_since_last} ago)"
                )
                return False

        return True

    def send_email_alert(
        self, recipients: List[str], subject: str, body_html: str, body_text: str
    ) -> bool:
        """Send email alert.

        Args:
            recipients: List of email recipients
            subject: Email subject
            body_html: HTML email body
            body_text: Plain text email body

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self.smtp_server or not recipients:
            logger.warning(
                "Cannot send email: SMTP server or recipients not configured"
            )
            return False

        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = self.sender_email
            message["To"] = ", ".join(recipients)

            # Attach parts
            part1 = MIMEText(body_text, "plain")
            part2 = MIMEText(body_html, "html")
            message.attach(part1)
            message.attach(part2)

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                if self.smtp_username and self.smtp_password:
                    server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.sender_email, recipients, message.as_string())

            logger.info(f"Email alert sent to {len(recipients)} recipients")
            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False

    def send_slack_alert(
        self, webhook_url: str, message: str, blocks: List[Dict] = None
    ) -> bool:
        """Send Slack alert.

        Args:
            webhook_url: Slack webhook URL
            message: Alert message
            blocks: Optional Slack blocks for rich formatting

        Returns:
            True if alert was sent successfully, False otherwise
        """
        if not webhook_url:
            logger.warning("Cannot send Slack alert: webhook URL not configured")
            return False

        try:
            payload = {"text": message}
            if blocks:
                payload["blocks"] = blocks

            response = requests.post(
                webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                logger.info("Slack alert sent successfully")
                return True
            else:
                logger.error(
                    f"Failed to send Slack alert: {response.status_code} {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False

    def send_teams_alert(self, webhook_url: str, title: str, message: str) -> bool:
        """Send Microsoft Teams alert.

        Args:
            webhook_url: Teams webhook URL
            title: Alert title
            message: Alert message

        Returns:
            True if alert was sent successfully, False otherwise
        """
        if not webhook_url:
            logger.warning("Cannot send Teams alert: webhook URL not configured")
            return False

        try:
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": "0076D7",
                "summary": title,
                "sections": [
                    {
                        "activityTitle": title,
                        "activitySubtitle": datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),
                        "text": message,
                    }
                ],
            }

            response = requests.post(
                webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                logger.info("Teams alert sent successfully")
                return True
            else:
                logger.error(
                    f"Failed to send Teams alert: {response.status_code} {response.text}"
                )
                return False

        except Exception as e:
            logger.error(f"Failed to send Teams alert: {str(e)}")
            return False

    def format_alert_message(
        self,
        report: GapAnalysisReport,
        include_report: bool = True,
        report_format: str = "html",
    ) -> Tuple[str, str]:
        """Format alert message for email and other channels.

        Args:
            report: Gap analysis report
            include_report: Whether to include the full report
            report_format: Report format (html or text)

        Returns:
            Tuple of (html_message, text_message)
        """
        # Count gaps by severity
        severity_counts = {
            "minor": sum(1 for gap in report.gaps if gap.severity == GapSeverity.MINOR),
            "moderate": sum(
                1 for gap in report.gaps if gap.severity == GapSeverity.MODERATE
            ),
            "critical": sum(
                1 for gap in report.gaps if gap.severity == GapSeverity.CRITICAL
            ),
        }

        # Create HTML message
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .critical {{ color: #d9534f; font-weight: bold; }}
                .moderate {{ color: #f0ad4e; font-weight: bold; }}
                .minor {{ color: #5bc0de; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h2>Data Gap Alert - {report.exchange}/{report.symbol}/{report.timeframe}</h2>
            <p>Data completeness: <strong>{report.data_completeness:.2f}%</strong></p>
            <p>
                Detected gaps: 
                <span class="critical">{severity_counts["critical"]} critical</span>, 
                <span class="moderate">{severity_counts["moderate"]} moderate</span>, 
                <span class="minor">{severity_counts["minor"]} minor</span>
            </p>
            <p>Total missing points: <strong>{report.total_missing_points}</strong></p>
            <p>Filled gaps: <strong>{report.filled_gaps}/{report.total_gaps}</strong></p>
        """

        # Create text message
        text = f"""
        Data Gap Alert - {report.exchange}/{report.symbol}/{report.timeframe}
        
        Data completeness: {report.data_completeness:.2f}%
        Detected gaps: {severity_counts["critical"]} critical, {severity_counts["moderate"]} moderate, {severity_counts["minor"]} minor
        Total missing points: {report.total_missing_points}
        Filled gaps: {report.filled_gaps}/{report.total_gaps}
        """

        # Add critical gaps details
        critical_gaps = [
            gap for gap in report.gaps if gap.severity == GapSeverity.CRITICAL
        ]
        if critical_gaps:
            html += "<h3>Critical Gaps</h3><table><tr><th>Start Time</th><th>End Time</th><th>Duration</th><th>Filled</th></tr>"
            text += "\nCritical Gaps:\n"

            for gap in critical_gaps:
                html += f"""
                <tr>
                    <td>{gap.start_time}</td>
                    <td>{gap.end_time}</td>
                    <td>{gap.duration}</td>
                    <td>{"Yes" if gap.filled else "No"}</td>
                </tr>
                """
                text += f"- {gap.start_time} to {gap.end_time} ({gap.duration}) - {'Filled' if gap.filled else 'Not filled'}\n"

            html += "</table>"

        # Include full report if requested
        if include_report:
            from .gap_detection import GapDetectionSystem

            # Create a temporary GapDetectionSystem to generate the report
            gap_detection = GapDetectionSystem("dummy_connection")

            if report_format == "html":
                full_report = gap_detection.generate_gap_report(report, "html")
                html += f"<h3>Full Report</h3>{full_report}"
            else:
                full_report = gap_detection.generate_gap_report(report, "text")
                text += f"\n\nFull Report:\n{full_report}"

        html += "</body></html>"

        return html, text

    def send_alert(self, report: GapAnalysisReport, config: AlertConfig) -> bool:
        """Send alert based on gap analysis report.

        Args:
            report: Gap analysis report
            config: Alert configuration

        Returns:
            True if at least one alert was sent successfully, False otherwise
        """
        # Determine the highest severity gap
        highest_severity = GapSeverity.MINOR
        for gap in report.gaps:
            if gap.severity == GapSeverity.CRITICAL:
                highest_severity = GapSeverity.CRITICAL
                break
            elif (
                gap.severity == GapSeverity.MODERATE
                and highest_severity != GapSeverity.CRITICAL
            ):
                highest_severity = GapSeverity.MODERATE

        # Check if we should send an alert
        if not self.should_send_alert(
            report.exchange, report.symbol, report.timeframe, highest_severity, config
        ):
            return False

        # Format messages
        html_message, text_message = self.format_alert_message(
            report, config.include_report, config.report_format
        )

        # Send alerts through configured channels
        success = False

        # Email
        if config.email_recipients:
            subject = (
                f"Data Gap Alert - {report.exchange}/{report.symbol}/{report.timeframe}"
            )
            email_success = self.send_email_alert(
                config.email_recipients, subject, html_message, text_message
            )
            success = success or email_success

        # Slack
        if config.slack_webhook:
            # Create Slack blocks for better formatting
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Data Gap Alert - {report.exchange}/{report.symbol}/{report.timeframe}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Data Completeness:*\n{report.data_completeness:.2f}%",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Missing Points:*\n{report.total_missing_points}",
                        },
                    ],
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Filled Gaps:*\n{report.filled_gaps}/{report.total_gaps}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Highest Severity:*\n{highest_severity.value}",
                        },
                    ],
                },
            ]

            slack_success = self.send_slack_alert(
                config.slack_webhook, text_message, blocks
            )
            success = success or slack_success

        # Microsoft Teams
        if config.teams_webhook:
            title = (
                f"Data Gap Alert - {report.exchange}/{report.symbol}/{report.timeframe}"
            )
            teams_success = self.send_teams_alert(
                config.teams_webhook, title, text_message
            )
            success = success or teams_success

        # Update last alert time if any alert was sent
        if success:
            key = f"{report.exchange}_{report.symbol}_{report.timeframe}"
            self.last_alert_time[key] = datetime.now()

        return success


class GapAlertingSystem:
    """System for analyzing gaps and sending alerts."""

    def __init__(self, connection_string: str):
        """Initialize gap alerting system.

        Args:
            connection_string: Database connection string
        """
        self.gap_detection = GapDetectionSystem(connection_string)
        self.alert_manager = AlertManager()
        self.default_config = AlertConfig(
            enabled=True,
            min_severity=GapSeverity.MODERATE,
            email_recipients=[],
            slack_webhook=os.environ.get("SLACK_WEBHOOK_URL"),
            teams_webhook=os.environ.get("TEAMS_WEBHOOK_URL"),
            alert_cooldown=timedelta(hours=6),
            include_report=True,
            report_format="html",
        )

        # Load configuration from environment variables
        if os.environ.get("ALERT_EMAIL_RECIPIENTS"):
            self.default_config.email_recipients = os.environ.get(
                "ALERT_EMAIL_RECIPIENTS"
            ).split(",")

        # Exchange-specific configurations
        self.exchange_configs: Dict[str, AlertConfig] = {}

    def set_alert_config(self, exchange: str, config: AlertConfig) -> None:
        """Set alert configuration for a specific exchange.

        Args:
            exchange: Exchange name
            config: Alert configuration
        """
        self.exchange_configs[exchange] = config

    def get_alert_config(self, exchange: str) -> AlertConfig:
        """Get alert configuration for a specific exchange.

        Args:
            exchange: Exchange name

        Returns:
            Alert configuration for the exchange or default config
        """
        return self.exchange_configs.get(exchange, self.default_config)

    def analyze_and_alert(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        save_filled_data: bool = True,
    ) -> Tuple[GapAnalysisReport, bool]:
        """Analyze gaps and send alerts if necessary.

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Time interval
            start_date: Optional start date for analysis
            end_date: Optional end date for analysis
            save_filled_data: Whether to save filled data to database

        Returns:
            Tuple of (gap_analysis_report, alert_sent)
        """
        # Run gap analysis
        report = self.gap_detection.analyze_gaps(
            exchange, symbol, timeframe, start_date, end_date
        )

        # If there are gaps, try to fill them
        if report.total_gaps > 0:
            # Get data from database
            df = self.gap_detection.db.get_ohlcv_data(exchange, symbol, timeframe)

            if not df.empty:
                # Detect gaps
                gaps = self.gap_detection.detect_gaps(df, timeframe)

                # Fill gaps
                filled_df, updated_gaps = self.gap_detection.fill_gaps(
                    exchange, symbol, timeframe, gaps
                )

                # Save filled data if requested
                if save_filled_data and not filled_df.empty:
                    self.gap_detection.save_filled_data(
                        filled_df, exchange, symbol, timeframe
                    )

                    # Update the report with filled gaps
                    report.filled_gaps = sum(1 for gap in updated_gaps if gap.filled)
                    report.unfilled_gaps = report.total_gaps - report.filled_gaps
                    report.gaps = updated_gaps

        # Send alert if necessary
        config = self.get_alert_config(exchange)
        alert_sent = False

        if report.total_gaps > 0:
            alert_sent = self.alert_manager.send_alert(report, config)

        return report, alert_sent

    def schedule_regular_checks(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        check_interval: timedelta = timedelta(hours=1),
        lookback_period: timedelta = timedelta(days=1),
    ) -> None:
        """Schedule regular gap checks.

        This method is intended to be called by a scheduler like Celery.

        Args:
            exchange: Exchange name
            symbol: Trading pair symbol
            timeframe: Time interval
            check_interval: How often to check for gaps
            lookback_period: How far back to look for gaps
        """
        end_date = datetime.now()
        start_date = end_date - lookback_period

        logger.info(
            f"Running scheduled gap check for {exchange}/{symbol}/{timeframe} "
            f"from {start_date} to {end_date}"
        )

        report, alert_sent = self.analyze_and_alert(
            exchange, symbol, timeframe, start_date, end_date
        )

        if alert_sent:
            logger.info(f"Alert sent for {exchange}/{symbol}/{timeframe}")
        else:
            logger.info(f"No alert sent for {exchange}/{symbol}/{timeframe}")

        return report


def create_gap_alerting_system(connection_string: str) -> GapAlertingSystem:
    """Create and return a gap alerting system.

    Args:
        connection_string: Database connection string

    Returns:
        GapAlertingSystem instance
    """
    return GapAlertingSystem(connection_string)


if __name__ == "__main__":
    # Example usage
    connection_string = (
        "postgresql://arb_user:strongpassword@localhost:5432/usdc_arbitrage"
    )
    alerting = create_gap_alerting_system(connection_string)

    # Configure email recipients
    alerting.default_config.email_recipients = ["admin@example.com"]

    # Run analysis and send alerts
    report, alert_sent = alerting.analyze_and_alert(
        "coinbase",
        "USDC/USD",
        "1h",
        start_date=datetime.now() - timedelta(days=1),
        end_date=datetime.now(),
    )

    print(f"Analysis complete. Alert sent: {alert_sent}")
