"""Business metrics collection and monitoring dashboard."""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import redis
from sqlalchemy import text, func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class BusinessMetricsCollector:
    """Collects and monitors business-specific metrics."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize business metrics collector."""
        self.redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
        self.metrics_retention_days = 30

    @property
    def redis_client(self) -> redis.Redis:
        """Get Redis client with lazy initialization."""
        if self._redis_client is None:
            self._redis_client = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis_client

    async def collect_backtest_metrics(self, db: Session) -> Dict[str, Any]:
        """Collect backtesting performance metrics."""
        try:
            # Get metrics for different time periods
            metrics = {}

            for period_name, hours in [
                ("1h", 1),
                ("24h", 24),
                ("7d", 168),
                ("30d", 720),
            ]:
                period_start = datetime.now(timezone.utc) - timedelta(hours=hours)

                # Backtest execution metrics
                backtest_stats = db.execute(
                    text("""
                    SELECT 
                        COUNT(*) as total_backtests,
                        COUNT(CASE WHEN total_return IS NOT NULL THEN 1 END) as successful_backtests,
                        COUNT(CASE WHEN total_return IS NULL THEN 1 END) as failed_backtests,
                        AVG(CASE WHEN total_return IS NOT NULL THEN total_return END) as avg_return,
                        AVG(CASE WHEN sharpe_ratio IS NOT NULL THEN sharpe_ratio END) as avg_sharpe,
                        AVG(CASE WHEN max_drawdown IS NOT NULL THEN max_drawdown END) as avg_drawdown,
                        SUM(trade_count) as total_trades
                    FROM backtest_results 
                    WHERE created_at >= :start_time
                """),
                    {"start_time": period_start},
                ).fetchone()

                # Strategy performance distribution
                strategy_performance = db.execute(
                    text("""
                    SELECT 
                        s.name as strategy_name,
                        COUNT(br.id) as backtest_count,
                        AVG(br.total_return) as avg_return,
                        AVG(br.sharpe_ratio) as avg_sharpe,
                        MAX(br.total_return) as best_return,
                        MIN(br.total_return) as worst_return
                    FROM strategies s
                    LEFT JOIN backtest_results br ON s.id = br.strategy_id 
                        AND br.created_at >= :start_time
                    GROUP BY s.id, s.name
                    ORDER BY avg_return DESC NULLS LAST
                """),
                    {"start_time": period_start},
                ).fetchall()

                # Top performing strategies
                top_strategies = [
                    {
                        "name": row.strategy_name,
                        "backtest_count": row.backtest_count,
                        "avg_return": float(row.avg_return) if row.avg_return else 0,
                        "avg_sharpe": float(row.avg_sharpe) if row.avg_sharpe else 0,
                        "best_return": float(row.best_return) if row.best_return else 0,
                        "worst_return": float(row.worst_return)
                        if row.worst_return
                        else 0,
                    }
                    for row in strategy_performance[:10]
                ]

                metrics[period_name] = {
                    "total_backtests": backtest_stats.total_backtests or 0,
                    "successful_backtests": backtest_stats.successful_backtests or 0,
                    "failed_backtests": backtest_stats.failed_backtests or 0,
                    "success_rate": (
                        (
                            backtest_stats.successful_backtests
                            / backtest_stats.total_backtests
                            * 100
                        )
                        if backtest_stats.total_backtests > 0
                        else 0
                    ),
                    "avg_return": float(backtest_stats.avg_return)
                    if backtest_stats.avg_return
                    else 0,
                    "avg_sharpe": float(backtest_stats.avg_sharpe)
                    if backtest_stats.avg_sharpe
                    else 0,
                    "avg_drawdown": float(backtest_stats.avg_drawdown)
                    if backtest_stats.avg_drawdown
                    else 0,
                    "total_trades": backtest_stats.total_trades or 0,
                    "top_strategies": top_strategies,
                }

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "periods": metrics,
            }

        except Exception as e:
            logger.error(f"Failed to collect backtest metrics: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def collect_data_pipeline_metrics(self, db: Session) -> Dict[str, Any]:
        """Collect data pipeline performance metrics."""
        try:
            # Data ingestion metrics
            data_stats = db.execute(
                text("""
                SELECT 
                    exchange,
                    symbol,
                    timeframe,
                    COUNT(*) as record_count,
                    MAX(timestamp) as latest_data,
                    MIN(timestamp) as earliest_data,
                    AVG(quality_score) as avg_quality_score,
                    COUNT(CASE WHEN quality_score < 0.8 THEN 1 END) as low_quality_count
                FROM market_data 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
                GROUP BY exchange, symbol, timeframe
                ORDER BY record_count DESC
            """)
            ).fetchall()

            # Data quality summary
            quality_summary = db.execute(
                text("""
                SELECT 
                    AVG(quality_score) as overall_avg_quality,
                    COUNT(CASE WHEN quality_score >= 0.9 THEN 1 END) as high_quality_count,
                    COUNT(CASE WHEN quality_score >= 0.8 AND quality_score < 0.9 THEN 1 END) as medium_quality_count,
                    COUNT(CASE WHEN quality_score < 0.8 THEN 1 END) as low_quality_count,
                    COUNT(*) as total_records
                FROM market_data 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)
            ).fetchone()

            # Data gaps analysis
            gaps_analysis = db.execute(
                text("""
                SELECT 
                    exchange,
                    symbol,
                    timeframe,
                    EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/3600 as hours_since_last_update
                FROM market_data 
                GROUP BY exchange, symbol, timeframe
                HAVING EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))/3600 > 1
                ORDER BY hours_since_last_update DESC
            """)
            ).fetchall()

            data_sources = [
                {
                    "exchange": row.exchange,
                    "symbol": row.symbol,
                    "timeframe": row.timeframe,
                    "record_count": row.record_count,
                    "latest_data": str(row.latest_data),
                    "earliest_data": str(row.earliest_data),
                    "avg_quality_score": float(row.avg_quality_score)
                    if row.avg_quality_score
                    else 0,
                    "low_quality_count": row.low_quality_count,
                }
                for row in data_stats
            ]

            data_gaps = [
                {
                    "exchange": row.exchange,
                    "symbol": row.symbol,
                    "timeframe": row.timeframe,
                    "hours_since_last_update": float(row.hours_since_last_update),
                }
                for row in gaps_analysis
            ]

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_sources": data_sources,
                "quality_summary": {
                    "overall_avg_quality": float(quality_summary.overall_avg_quality)
                    if quality_summary.overall_avg_quality
                    else 0,
                    "high_quality_count": quality_summary.high_quality_count or 0,
                    "medium_quality_count": quality_summary.medium_quality_count or 0,
                    "low_quality_count": quality_summary.low_quality_count or 0,
                    "total_records": quality_summary.total_records or 0,
                    "high_quality_percentage": (
                        (
                            quality_summary.high_quality_count
                            / quality_summary.total_records
                            * 100
                        )
                        if quality_summary.total_records > 0
                        else 0
                    ),
                },
                "data_gaps": data_gaps,
                "gaps_count": len(data_gaps),
            }

        except Exception as e:
            logger.error(f"Failed to collect data pipeline metrics: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def collect_user_activity_metrics(self, db: Session) -> Dict[str, Any]:
        """Collect user activity and API usage metrics."""
        try:
            # User activity metrics (if audit logs exist)
            try:
                user_activity = db.execute(
                    text("""
                    SELECT 
                        COUNT(DISTINCT user_id) as active_users_24h,
                        COUNT(*) as total_requests_24h,
                        COUNT(CASE WHEN action LIKE '%backtest%' THEN 1 END) as backtest_requests,
                        COUNT(CASE WHEN action LIKE '%strategy%' THEN 1 END) as strategy_requests,
                        COUNT(CASE WHEN action LIKE '%data%' THEN 1 END) as data_requests
                    FROM audit_logs 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                """)
                ).fetchone()

                # Top active users
                top_users = db.execute(
                    text("""
                    SELECT 
                        user_id,
                        COUNT(*) as request_count,
                        COUNT(DISTINCT action) as unique_actions
                    FROM audit_logs 
                    WHERE created_at >= NOW() - INTERVAL '24 hours'
                    GROUP BY user_id
                    ORDER BY request_count DESC
                    LIMIT 10
                """)
                ).fetchall()

                user_metrics = {
                    "active_users_24h": user_activity.active_users_24h or 0,
                    "total_requests_24h": user_activity.total_requests_24h or 0,
                    "backtest_requests": user_activity.backtest_requests or 0,
                    "strategy_requests": user_activity.strategy_requests or 0,
                    "data_requests": user_activity.data_requests or 0,
                    "top_users": [
                        {
                            "user_id": row.user_id,
                            "request_count": row.request_count,
                            "unique_actions": row.unique_actions,
                        }
                        for row in top_users
                    ],
                }

            except Exception:
                # Audit logs table might not exist yet
                user_metrics = {
                    "active_users_24h": 0,
                    "total_requests_24h": 0,
                    "backtest_requests": 0,
                    "strategy_requests": 0,
                    "data_requests": 0,
                    "top_users": [],
                }

            # Strategy usage metrics
            strategy_usage = db.execute(
                text("""
                SELECT 
                    s.name as strategy_name,
                    COUNT(br.id) as usage_count,
                    s.created_at as created_date,
                    s.is_active
                FROM strategies s
                LEFT JOIN backtest_results br ON s.id = br.strategy_id 
                    AND br.created_at >= NOW() - INTERVAL '7 days'
                GROUP BY s.id, s.name, s.created_at, s.is_active
                ORDER BY usage_count DESC
            """)
            ).fetchall()

            strategy_metrics = [
                {
                    "name": row.strategy_name,
                    "usage_count": row.usage_count,
                    "created_date": str(row.created_date),
                    "is_active": row.is_active,
                }
                for row in strategy_usage
            ]

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_activity": user_metrics,
                "strategy_usage": strategy_metrics,
            }

        except Exception as e:
            logger.error(f"Failed to collect user activity metrics: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def store_business_metrics(self, metrics: Dict[str, Any], metric_type: str):
        """Store business metrics in Redis."""
        try:
            import json
            import time

            key = f"business_metrics:{metric_type}:{int(time.time())}"

            # Store metrics with TTL
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.setex(
                    key,
                    self.metrics_retention_days * 24 * 3600,  # TTL in seconds
                    json.dumps(metrics, default=str),
                ),
            )

            # Add to time-series index
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zadd(
                    f"business_metrics_index:{metric_type}", {key: time.time()}
                ),
            )

            # Clean up old entries
            cutoff_time = time.time() - (self.metrics_retention_days * 24 * 3600)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zremrangebyscore(
                    f"business_metrics_index:{metric_type}", 0, cutoff_time
                ),
            )

        except Exception as e:
            logger.error(f"Failed to store business metrics: {e}")

    async def get_business_metrics_history(
        self, metric_type: str, days: int = 7, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve historical business metrics."""
        try:
            import json
            import time

            start_time = time.time() - (days * 24 * 3600)
            end_time = time.time()

            # Get metric keys from index
            keys = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.redis_client.zrangebyscore(
                    f"business_metrics_index:{metric_type}",
                    start_time,
                    end_time,
                    start=0,
                    num=limit,
                    desc=True,
                ),
            )

            if not keys:
                return []

            # Get metric data
            metrics_data = []
            for key in keys:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                if data:
                    try:
                        metrics_data.append(json.loads(data))
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to decode business metrics for key: {key}"
                        )

            return sorted(metrics_data, key=lambda x: x.get("timestamp", ""))

        except Exception as e:
            logger.error(f"Failed to retrieve business metrics history: {e}")
            return []

    async def collect_and_store_all_business_metrics(self, db: Session):
        """Collect and store all business metrics."""
        try:
            # Collect all business metrics concurrently
            backtest_metrics, pipeline_metrics, activity_metrics = await asyncio.gather(
                self.collect_backtest_metrics(db),
                self.collect_data_pipeline_metrics(db),
                self.collect_user_activity_metrics(db),
                return_exceptions=True,
            )

            # Store metrics
            if not isinstance(backtest_metrics, Exception):
                await self.store_business_metrics(backtest_metrics, "backtests")

            if not isinstance(pipeline_metrics, Exception):
                await self.store_business_metrics(pipeline_metrics, "data_pipeline")

            if not isinstance(activity_metrics, Exception):
                await self.store_business_metrics(activity_metrics, "user_activity")

            logger.info("Successfully collected and stored all business metrics")

        except Exception as e:
            logger.error(f"Failed to collect and store business metrics: {e}")


# Global business metrics collector instance
business_metrics_collector = BusinessMetricsCollector()


async def get_business_metrics_collector() -> BusinessMetricsCollector:
    """Dependency to get business metrics collector instance."""
    return business_metrics_collector
