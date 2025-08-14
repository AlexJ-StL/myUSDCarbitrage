"""Automated reporting system for daily/weekly summaries."""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from jinja2 import Template

import redis
from sqlalchemy.orm import Session

from .business_metrics import BusinessMetricsCollector
from .alerting import AlertingSystem, AlertSeverity

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates automated reports for system and business metrics."""

    def __init__(
        self,
        business_metrics: Optional[BusinessMetricsCollector] = None,
        alerting_system: Optional[AlertingSystem] = None,
    ):
        """Initialize report generator."""
        self.business_metrics = business_metrics or BusinessMetricsCollector()
        self.alerting_system = alerting_system or AlertingSystem()

    async def generate_daily_summary(self, db: Session) -> Dict[str, Any]:
        """Generate daily summary report."""
        try:
            report_date = datetime.now(timezone.utc).date()

            # Collect metrics for the last 24 hours
            backtest_metrics = await self.business_metrics.collect_backtest_metrics(db)
            pipeline_metrics = (
                await self.business_metrics.collect_data_pipeline_metrics(db)
            )
            activity_metrics = (
                await self.business_metrics.collect_user_activity_metrics(db)
            )

            # Get alerts from the last 24 hours
            alerts = await self.alerting_system.get_alerts(hours=24, limit=50)

            # Categorize alerts by severity
            alert_summary = {
                "critical": [
                    a for a in alerts if a["severity"] == AlertSeverity.CRITICAL.value
                ],
                "high": [
                    a for a in alerts if a["severity"] == AlertSeverity.HIGH.value
                ],
                "medium": [
                    a for a in alerts if a["severity"] == AlertSeverity.MEDIUM.value
                ],
                "low": [a for a in alerts if a["severity"] == AlertSeverity.LOW.value],
            }

            # Extract key metrics
            backtest_24h = backtest_metrics.get("periods", {}).get("24h", {})
            pipeline_summary = pipeline_metrics.get("quality_summary", {})
            user_activity = activity_metrics.get("user_activity", {})

            # Calculate trends (compare with previous day if available)
            trends = await self._calculate_daily_trends(db)

            report = {
                "report_type": "daily_summary",
                "report_date": report_date.isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                # Executive Summary
                "executive_summary": {
                    "total_backtests": backtest_24h.get("total_backtests", 0),
                    "success_rate": backtest_24h.get("success_rate", 0),
                    "avg_return": backtest_24h.get("avg_return", 0),
                    "active_users": user_activity.get("active_users_24h", 0),
                    "data_quality": pipeline_summary.get("high_quality_percentage", 0),
                    "critical_alerts": len(alert_summary["critical"]),
                    "high_alerts": len(alert_summary["high"]),
                },
                # Detailed Metrics
                "backtest_performance": {
                    "total_backtests": backtest_24h.get("total_backtests", 0),
                    "successful_backtests": backtest_24h.get("successful_backtests", 0),
                    "failed_backtests": backtest_24h.get("failed_backtests", 0),
                    "success_rate": backtest_24h.get("success_rate", 0),
                    "avg_return": backtest_24h.get("avg_return", 0),
                    "avg_sharpe": backtest_24h.get("avg_sharpe", 0),
                    "avg_drawdown": backtest_24h.get("avg_drawdown", 0),
                    "total_trades": backtest_24h.get("total_trades", 0),
                    "top_strategies": backtest_24h.get("top_strategies", [])[:5],
                },
                # Data Pipeline Health
                "data_pipeline": {
                    "total_records": pipeline_summary.get("total_records", 0),
                    "high_quality_percentage": pipeline_summary.get(
                        "high_quality_percentage", 0
                    ),
                    "data_gaps_count": pipeline_metrics.get("gaps_count", 0),
                    "data_sources_count": len(pipeline_metrics.get("data_sources", [])),
                    "quality_distribution": {
                        "high_quality": pipeline_summary.get("high_quality_count", 0),
                        "medium_quality": pipeline_summary.get(
                            "medium_quality_count", 0
                        ),
                        "low_quality": pipeline_summary.get("low_quality_count", 0),
                    },
                },
                # User Activity
                "user_activity": {
                    "active_users": user_activity.get("active_users_24h", 0),
                    "total_requests": user_activity.get("total_requests_24h", 0),
                    "backtest_requests": user_activity.get("backtest_requests", 0),
                    "strategy_requests": user_activity.get("strategy_requests", 0),
                    "data_requests": user_activity.get("data_requests", 0),
                },
                # Alerts Summary
                "alerts_summary": {
                    "total_alerts": len(alerts),
                    "by_severity": {
                        "critical": len(alert_summary["critical"]),
                        "high": len(alert_summary["high"]),
                        "medium": len(alert_summary["medium"]),
                        "low": len(alert_summary["low"]),
                    },
                    "recent_critical": alert_summary["critical"][:3],
                    "recent_high": alert_summary["high"][:5],
                },
                # Trends
                "trends": trends,
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate daily summary: {e}")
            return {
                "report_type": "daily_summary",
                "report_date": datetime.now(timezone.utc).date().isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def generate_weekly_summary(self, db: Session) -> Dict[str, Any]:
        """Generate weekly summary report."""
        try:
            report_date = datetime.now(timezone.utc).date()
            week_start = report_date - timedelta(days=report_date.weekday())

            # Collect metrics for the last 7 days
            backtest_metrics = await self.business_metrics.collect_backtest_metrics(db)
            pipeline_metrics = (
                await self.business_metrics.collect_data_pipeline_metrics(db)
            )
            activity_metrics = (
                await self.business_metrics.collect_user_activity_metrics(db)
            )

            # Get alerts from the last 7 days
            alerts = await self.alerting_system.get_alerts(
                hours=168, limit=200
            )  # 7 days

            # Extract weekly metrics
            backtest_7d = backtest_metrics.get("periods", {}).get("7d", {})

            # Calculate weekly trends
            weekly_trends = await self._calculate_weekly_trends(db)

            # Strategy performance analysis
            strategy_analysis = await self._analyze_strategy_performance(db, days=7)

            report = {
                "report_type": "weekly_summary",
                "week_start": week_start.isoformat(),
                "week_end": report_date.isoformat(),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                # Executive Summary
                "executive_summary": {
                    "total_backtests": backtest_7d.get("total_backtests", 0),
                    "avg_daily_backtests": backtest_7d.get("total_backtests", 0) / 7,
                    "success_rate": backtest_7d.get("success_rate", 0),
                    "best_strategy_return": max(
                        [
                            s.get("avg_return", 0)
                            for s in backtest_7d.get("top_strategies", [])
                        ],
                        default=0,
                    ),
                    "total_alerts": len(alerts),
                    "critical_issues": len([
                        a for a in alerts if a["severity"] == "critical"
                    ]),
                },
                # Weekly Performance
                "weekly_performance": {
                    "backtests": backtest_7d,
                    "data_quality_avg": pipeline_metrics.get("quality_summary", {}).get(
                        "high_quality_percentage", 0
                    ),
                    "uptime_percentage": 99.5,  # This would come from system monitoring
                },
                # Strategy Analysis
                "strategy_analysis": strategy_analysis,
                # Trends and Insights
                "trends": weekly_trends,
                # Issues and Recommendations
                "issues_and_recommendations": await self._generate_recommendations(
                    alerts, backtest_7d
                ),
                # Alert Summary
                "alerts_summary": {
                    "total": len(alerts),
                    "daily_average": len(alerts) / 7,
                    "by_type": self._categorize_alerts_by_type(alerts),
                    "by_severity": self._categorize_alerts_by_severity(alerts),
                },
            }

            return report

        except Exception as e:
            logger.error(f"Failed to generate weekly summary: {e}")
            return {
                "report_type": "weekly_summary",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def _calculate_daily_trends(self, db: Session) -> Dict[str, Any]:
        """Calculate daily trends by comparing with previous day."""
        try:
            # Get historical business metrics for trend calculation
            backtest_history = await self.business_metrics.get_business_metrics_history(
                "backtests", days=2, limit=10
            )

            if len(backtest_history) < 2:
                return {"note": "Insufficient data for trend calculation"}

            current = backtest_history[0].get("periods", {}).get("24h", {})
            previous = backtest_history[1].get("periods", {}).get("24h", {})

            def calculate_change(current_val, previous_val):
                if previous_val == 0:
                    return 0 if current_val == 0 else 100
                return ((current_val - previous_val) / previous_val) * 100

            return {
                "backtests_change": calculate_change(
                    current.get("total_backtests", 0),
                    previous.get("total_backtests", 0),
                ),
                "success_rate_change": calculate_change(
                    current.get("success_rate", 0), previous.get("success_rate", 0)
                ),
                "avg_return_change": calculate_change(
                    current.get("avg_return", 0), previous.get("avg_return", 0)
                ),
            }

        except Exception as e:
            logger.error(f"Failed to calculate daily trends: {e}")
            return {"error": str(e)}

    async def _calculate_weekly_trends(self, db: Session) -> Dict[str, Any]:
        """Calculate weekly trends by comparing with previous weeks."""
        try:
            # Get historical data for multiple weeks
            backtest_history = await self.business_metrics.get_business_metrics_history(
                "backtests", days=14, limit=20
            )

            if len(backtest_history) < 2:
                return {"note": "Insufficient data for weekly trend calculation"}

            # Simple trend calculation - would be more sophisticated in production
            recent_avg = sum(
                h.get("periods", {}).get("7d", {}).get("total_backtests", 0)
                for h in backtest_history[:7]
            ) / min(7, len(backtest_history))

            older_avg = sum(
                h.get("periods", {}).get("7d", {}).get("total_backtests", 0)
                for h in backtest_history[7:14]
            ) / min(7, len(backtest_history[7:]))

            trend = (
                "increasing"
                if recent_avg > older_avg
                else "decreasing"
                if recent_avg < older_avg
                else "stable"
            )

            return {
                "backtest_volume_trend": trend,
                "recent_avg": recent_avg,
                "older_avg": older_avg,
            }

        except Exception as e:
            logger.error(f"Failed to calculate weekly trends: {e}")
            return {"error": str(e)}

    async def _analyze_strategy_performance(
        self, db: Session, days: int = 7
    ) -> Dict[str, Any]:
        """Analyze strategy performance over the specified period."""
        try:
            from sqlalchemy import text

            # Get strategy performance data
            strategy_data = db.execute(
                text("""
                SELECT 
                    s.name,
                    COUNT(br.id) as backtest_count,
                    AVG(br.total_return) as avg_return,
                    STDDEV(br.total_return) as return_volatility,
                    AVG(br.sharpe_ratio) as avg_sharpe,
                    AVG(br.max_drawdown) as avg_drawdown,
                    MAX(br.total_return) as best_return,
                    MIN(br.total_return) as worst_return
                FROM strategies s
                LEFT JOIN backtest_results br ON s.id = br.strategy_id
                    AND br.created_at >= NOW() - INTERVAL :days DAY
                WHERE s.is_active = true
                GROUP BY s.id, s.name
                HAVING COUNT(br.id) > 0
                ORDER BY avg_return DESC
            """),
                {"days": days},
            ).fetchall()

            strategies = []
            for row in strategy_data:
                strategies.append({
                    "name": row.name,
                    "backtest_count": row.backtest_count,
                    "avg_return": float(row.avg_return) if row.avg_return else 0,
                    "return_volatility": float(row.return_volatility)
                    if row.return_volatility
                    else 0,
                    "avg_sharpe": float(row.avg_sharpe) if row.avg_sharpe else 0,
                    "avg_drawdown": float(row.avg_drawdown) if row.avg_drawdown else 0,
                    "best_return": float(row.best_return) if row.best_return else 0,
                    "worst_return": float(row.worst_return) if row.worst_return else 0,
                })

            return {
                "total_active_strategies": len(strategies),
                "top_performers": strategies[:5],
                "bottom_performers": strategies[-3:] if len(strategies) > 3 else [],
                "avg_strategy_return": sum(s["avg_return"] for s in strategies)
                / len(strategies)
                if strategies
                else 0,
            }

        except Exception as e:
            logger.error(f"Failed to analyze strategy performance: {e}")
            return {"error": str(e)}

    async def _generate_recommendations(
        self, alerts: List[Dict], backtest_metrics: Dict
    ) -> List[str]:
        """Generate recommendations based on alerts and metrics."""
        recommendations = []

        try:
            # Check for high alert volume
            critical_alerts = [a for a in alerts if a["severity"] == "critical"]
            if len(critical_alerts) > 5:
                recommendations.append(
                    f"High number of critical alerts ({len(critical_alerts)}) detected. "
                    "Review system stability and consider scaling resources."
                )

            # Check backtest success rate
            success_rate = backtest_metrics.get("success_rate", 0)
            if success_rate < 80:
                recommendations.append(
                    f"Backtest success rate is {success_rate:.1f}%, below optimal threshold. "
                    "Review strategy parameters and data quality."
                )

            # Check strategy performance
            avg_return = backtest_metrics.get("avg_return", 0)
            if avg_return < 0:
                recommendations.append(
                    "Average strategy returns are negative. "
                    "Consider reviewing strategy logic and market conditions."
                )

            # Data quality recommendations
            data_alerts = [a for a in alerts if a.get("alert_type") == "data_pipeline"]
            if len(data_alerts) > 10:
                recommendations.append(
                    "Multiple data pipeline issues detected. "
                    "Review data sources and validation rules."
                )

            if not recommendations:
                recommendations.append(
                    "System is performing well. Continue monitoring key metrics."
                )

        except Exception as e:
            logger.error(f"Failed to generate recommendations: {e}")
            recommendations.append(
                "Unable to generate recommendations due to analysis error."
            )

        return recommendations

    def _categorize_alerts_by_type(self, alerts: List[Dict]) -> Dict[str, int]:
        """Categorize alerts by type."""
        categories = {}
        for alert in alerts:
            alert_type = alert.get("alert_type", "unknown")
            categories[alert_type] = categories.get(alert_type, 0) + 1
        return categories

    def _categorize_alerts_by_severity(self, alerts: List[Dict]) -> Dict[str, int]:
        """Categorize alerts by severity."""
        categories = {}
        for alert in alerts:
            severity = alert.get("severity", "unknown")
            categories[severity] = categories.get(severity, 0) + 1
        return categories

    def format_report_as_text(self, report: Dict[str, Any]) -> str:
        """Format report as plain text for email."""
        if report["report_type"] == "daily_summary":
            return self._format_daily_report_text(report)
        elif report["report_type"] == "weekly_summary":
            return self._format_weekly_report_text(report)
        else:
            return f"Unknown report type: {report['report_type']}"

    def _format_daily_report_text(self, report: Dict[str, Any]) -> str:
        """Format daily report as text."""
        template = Template("""
USDC Arbitrage System - Daily Summary Report
Date: {{ report_date }}
Generated: {{ generated_at }}

=== EXECUTIVE SUMMARY ===
• Total Backtests: {{ executive_summary.total_backtests }}
• Success Rate: {{ "%.1f"|format(executive_summary.success_rate) }}%
• Average Return: {{ "%.2f"|format(executive_summary.avg_return) }}%
• Active Users: {{ executive_summary.active_users }}
• Data Quality: {{ "%.1f"|format(executive_summary.data_quality) }}%
• Critical Alerts: {{ executive_summary.critical_alerts }}

=== BACKTEST PERFORMANCE ===
• Successful: {{ backtest_performance.successful_backtests }}/{{ backtest_performance.total_backtests }}
• Average Sharpe Ratio: {{ "%.2f"|format(backtest_performance.avg_sharpe) }}
• Average Drawdown: {{ "%.2f"|format(backtest_performance.avg_drawdown) }}%
• Total Trades: {{ backtest_performance.total_trades }}

Top Strategies:
{% for strategy in backtest_performance.top_strategies[:3] %}
• {{ strategy.name }}: {{ "%.2f"|format(strategy.avg_return) }}% return
{% endfor %}

=== DATA PIPELINE ===
• Total Records: {{ data_pipeline.total_records }}
• High Quality: {{ "%.1f"|format(data_pipeline.high_quality_percentage) }}%
• Data Gaps: {{ data_pipeline.data_gaps_count }}

=== ALERTS ===
• Total: {{ alerts_summary.total_alerts }}
• Critical: {{ alerts_summary.by_severity.critical }}
• High: {{ alerts_summary.by_severity.high }}

{% if alerts_summary.recent_critical %}
Recent Critical Alerts:
{% for alert in alerts_summary.recent_critical %}
• {{ alert.title }} ({{ alert.timestamp }})
{% endfor %}
{% endif %}

=== TRENDS ===
{% if trends.backtests_change is defined %}
• Backtests: {{ "%.1f"|format(trends.backtests_change) }}% change from yesterday
• Success Rate: {{ "%.1f"|format(trends.success_rate_change) }}% change
{% endif %}
        """)

        return template.render(**report)

    def _format_weekly_report_text(self, report: Dict[str, Any]) -> str:
        """Format weekly report as text."""
        template = Template("""
USDC Arbitrage System - Weekly Summary Report
Week: {{ week_start }} to {{ week_end }}
Generated: {{ generated_at }}

=== EXECUTIVE SUMMARY ===
• Total Backtests: {{ executive_summary.total_backtests }}
• Daily Average: {{ "%.1f"|format(executive_summary.avg_daily_backtests) }}
• Success Rate: {{ "%.1f"|format(weekly_performance.backtests.success_rate) }}%
• Best Strategy Return: {{ "%.2f"|format(executive_summary.best_strategy_return) }}%
• Total Alerts: {{ executive_summary.total_alerts }}

=== STRATEGY ANALYSIS ===
• Active Strategies: {{ strategy_analysis.total_active_strategies }}
• Average Return: {{ "%.2f"|format(strategy_analysis.avg_strategy_return) }}%

Top Performers:
{% for strategy in strategy_analysis.top_performers[:3] %}
• {{ strategy.name }}: {{ "%.2f"|format(strategy.avg_return) }}% ({{ strategy.backtest_count }} tests)
{% endfor %}

=== RECOMMENDATIONS ===
{% for rec in issues_and_recommendations %}
• {{ rec }}
{% endfor %}

=== ALERT SUMMARY ===
• Total: {{ alerts_summary.total }}
• Daily Average: {{ "%.1f"|format(alerts_summary.daily_average) }}
• By Severity: Critical({{ alerts_summary.by_severity.critical }}), High({{ alerts_summary.by_severity.high }})
        """)

        return template.render(**report)


# Global report generator instance
report_generator = ReportGenerator()


async def get_report_generator() -> ReportGenerator:
    """Dependency to get report generator instance."""
    return report_generator
