"""Comprehensive audit logging system for security events."""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from fastapi import Request
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import Session

from .database import Base


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    TOKEN_REVOKED = "token_revoked"

    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_DENIED = "permission_denied"

    # API Key events
    API_KEY_CREATED = "api_key_created"
    API_KEY_USED = "api_key_used"
    API_KEY_REVOKED = "api_key_revoked"
    API_KEY_EXPIRED = "api_key_expired"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RATE_LIMIT_WARNING = "rate_limit_warning"

    # Data access events
    SENSITIVE_DATA_ACCESS = "sensitive_data_access"
    DATA_EXPORT = "data_export"
    BULK_DATA_ACCESS = "bulk_data_access"

    # Administrative events
    USER_CREATED = "user_created"
    USER_MODIFIED = "user_modified"
    USER_DELETED = "user_deleted"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"

    # System events
    SYSTEM_ERROR = "system_error"
    SECURITY_VIOLATION = "security_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

    # Strategy and backtest events
    STRATEGY_CREATED = "strategy_created"
    STRATEGY_MODIFIED = "strategy_modified"
    STRATEGY_DELETED = "strategy_deleted"
    BACKTEST_STARTED = "backtest_started"
    BACKTEST_COMPLETED = "backtest_completed"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLog(Base):
    """SQLAlchemy model for audit logs."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), default=AuditSeverity.LOW, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    username = Column(String(100), nullable=True, index=True)
    api_key_id = Column(Integer, nullable=True, index=True)
    client_ip = Column(String(45), nullable=True, index=True)
    user_agent = Column(String(500), nullable=True)
    endpoint = Column(String(200), nullable=True, index=True)
    method = Column(String(10), nullable=True)
    status_code = Column(Integer, nullable=True, index=True)
    message = Column(Text, nullable=False)
    details = Column(Text, nullable=True)  # JSON string for additional details
    session_id = Column(String(100), nullable=True, index=True)
    request_id = Column(String(100), nullable=True, index=True)


class AuditLogEntry(BaseModel):
    """Pydantic model for audit log entries."""

    event_type: AuditEventType
    severity: AuditSeverity = AuditSeverity.LOW
    message: str
    user_id: Optional[int] = None
    username: Optional[str] = None
    api_key_id: Optional[int] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status_code: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None


class AuditLogger:
    """Service for audit logging."""

    def __init__(self, db: Session):
        self.db = db

        # Set up structured logging
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)

        # Create handler if not exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - AUDIT - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def log_event(self, entry: AuditLogEntry):
        """Log an audit event to both database and log file."""
        try:
            # Create database record
            audit_log = AuditLog(
                event_type=entry.event_type.value,
                severity=entry.severity.value,
                user_id=entry.user_id,
                username=entry.username,
                api_key_id=entry.api_key_id,
                client_ip=entry.client_ip,
                user_agent=entry.user_agent,
                endpoint=entry.endpoint,
                method=entry.method,
                status_code=entry.status_code,
                message=entry.message,
                details=json.dumps(entry.details) if entry.details else None,
                session_id=entry.session_id,
                request_id=entry.request_id,
            )

            self.db.add(audit_log)
            self.db.commit()

            # Log to file/console
            log_data = {
                "event_type": entry.event_type.value,
                "severity": entry.severity.value,
                "message": entry.message,
                "user_id": entry.user_id,
                "username": entry.username,
                "client_ip": entry.client_ip,
                "endpoint": entry.endpoint,
                "details": entry.details,
            }

            log_message = (
                f"{entry.event_type.value}: {entry.message} | {json.dumps(log_data)}"
            )

            if entry.severity == AuditSeverity.CRITICAL:
                self.logger.critical(log_message)
            elif entry.severity == AuditSeverity.HIGH:
                self.logger.error(log_message)
            elif entry.severity == AuditSeverity.MEDIUM:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)

        except Exception as e:
            # Fallback logging if database fails
            self.logger.error(f"Failed to log audit event: {e}")
            self.logger.info(f"Original audit event: {entry.model_dump()}")

    def log_authentication_event(
        self,
        event_type: AuditEventType,
        username: str,
        client_ip: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        user_agent: Optional[str] = None,
    ):
        """Log authentication-related events."""
        severity = AuditSeverity.LOW if success else AuditSeverity.MEDIUM
        message = f"Authentication {event_type.value} for user {username}"

        if not success:
            message += " (FAILED)"
            severity = AuditSeverity.HIGH

        entry = AuditLogEntry(
            event_type=event_type,
            severity=severity,
            message=message,
            username=username,
            client_ip=client_ip,
            user_agent=user_agent,
            details=details,
        )

        self.log_event(entry)

    def log_authorization_event(
        self,
        event_type: AuditEventType,
        user_id: int,
        username: str,
        endpoint: str,
        method: str,
        granted: bool = True,
        required_permission: Optional[str] = None,
        client_ip: Optional[str] = None,
    ):
        """Log authorization-related events."""
        severity = AuditSeverity.LOW if granted else AuditSeverity.HIGH
        message = f"Access {'granted' if granted else 'denied'} for {username} to {method} {endpoint}"

        details = {}
        if required_permission:
            details["required_permission"] = required_permission

        entry = AuditLogEntry(
            event_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            username=username,
            endpoint=endpoint,
            method=method,
            client_ip=client_ip,
            details=details if details else None,
        )

        self.log_event(entry)

    def log_api_key_event(
        self,
        event_type: AuditEventType,
        api_key_id: int,
        api_key_name: str,
        client_ip: Optional[str] = None,
        endpoint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log API key-related events."""
        severity = AuditSeverity.MEDIUM
        if event_type in [
            AuditEventType.API_KEY_CREATED,
            AuditEventType.API_KEY_REVOKED,
        ]:
            severity = AuditSeverity.HIGH

        message = f"API key {event_type.value}: {api_key_name}"

        entry = AuditLogEntry(
            event_type=event_type,
            severity=severity,
            message=message,
            api_key_id=api_key_id,
            client_ip=client_ip,
            endpoint=endpoint,
            details=details,
        )

        self.log_event(entry)

    def log_rate_limit_event(
        self,
        client_ip: str,
        endpoint: str,
        limit: int,
        current_count: int,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        api_key_id: Optional[int] = None,
    ):
        """Log rate limiting events."""
        severity = AuditSeverity.MEDIUM
        if current_count > limit * 1.5:  # 50% over limit
            severity = AuditSeverity.HIGH

        message = f"Rate limit exceeded for {client_ip} on {endpoint}: {current_count}/{limit}"

        entry = AuditLogEntry(
            event_type=AuditEventType.RATE_LIMIT_EXCEEDED,
            severity=severity,
            message=message,
            user_id=user_id,
            username=username,
            api_key_id=api_key_id,
            client_ip=client_ip,
            endpoint=endpoint,
            details={
                "limit": limit,
                "current_count": current_count,
                "overage_percentage": ((current_count - limit) / limit) * 100,
            },
        )

        self.log_event(entry)

    def log_security_violation(
        self,
        violation_type: str,
        message: str,
        client_ip: str,
        endpoint: Optional[str] = None,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log security violations."""
        entry = AuditLogEntry(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            message=f"Security violation ({violation_type}): {message}",
            user_id=user_id,
            username=username,
            client_ip=client_ip,
            endpoint=endpoint,
            details=details,
        )

        self.log_event(entry)

    def log_data_access(
        self,
        user_id: int,
        username: str,
        endpoint: str,
        method: str,
        data_type: str,
        record_count: Optional[int] = None,
        client_ip: Optional[str] = None,
    ):
        """Log sensitive data access."""
        event_type = AuditEventType.SENSITIVE_DATA_ACCESS
        severity = AuditSeverity.LOW

        if record_count and record_count > 1000:
            event_type = AuditEventType.BULK_DATA_ACCESS
            severity = AuditSeverity.MEDIUM

        message = f"Data access: {username} accessed {data_type}"
        if record_count:
            message += f" ({record_count} records)"

        entry = AuditLogEntry(
            event_type=event_type,
            severity=severity,
            message=message,
            user_id=user_id,
            username=username,
            endpoint=endpoint,
            method=method,
            client_ip=client_ip,
            details={
                "data_type": data_type,
                "record_count": record_count,
            }
            if record_count
            else {"data_type": data_type},
        )

        self.log_event(entry)


def extract_request_info(request: Request) -> Dict[str, Any]:
    """Extract relevant information from request for audit logging."""
    client_ip = request.client.host
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        client_ip = forwarded_for.split(",")[0].strip()

    return {
        "client_ip": client_ip,
        "user_agent": request.headers.get("User-Agent"),
        "endpoint": str(request.url.path),
        "method": request.method,
        "session_id": request.headers.get("X-Session-ID"),
        "request_id": request.headers.get("X-Request-ID"),
    }


# Middleware for automatic audit logging
class AuditLoggingMiddleware:
    """Middleware for automatic audit logging of requests."""

    def __init__(self, app, db_session_factory):
        self.app = app
        self.db_session_factory = db_session_factory

    async def __call__(self, scope, receive, send):
        """Process request with audit logging."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from starlette.requests import Request

        start_time = datetime.now(timezone.utc)
        request = Request(scope, receive)

        # Extract request information
        request_info = extract_request_info(request)

        # Create a response capture mechanism
        response_info = {"status_code": 200}

        async def send_with_capture(message):
            if message["type"] == "http.response.start":
                response_info["status_code"] = message["status"]
            await send(message)

        # Process request
        await self.app(scope, receive, send_with_capture)

        # Log request if it's a sensitive endpoint or failed
        if self._should_log_request_by_path(
            request.url.path, response_info["status_code"]
        ):
            try:
                db = next(self.db_session_factory())
                audit_logger = AuditLogger(db)

                # Get user info if available (these would be set by auth middleware)
                user_id = (
                    getattr(request.state, "user_id", None)
                    if hasattr(request, "state")
                    else None
                )
                username = (
                    getattr(request.state, "username", None)
                    if hasattr(request, "state")
                    else None
                )
                api_key_id = (
                    getattr(request.state, "api_key_id", None)
                    if hasattr(request, "state")
                    else None
                )

                # Determine event type and severity
                event_type, severity = self._determine_event_type_and_severity_by_path(
                    request.url.path, request.method, response_info["status_code"]
                )

                message = f"{request.method} {request.url.path} - Status: {response_info['status_code']}"

                entry = AuditLogEntry(
                    event_type=event_type,
                    severity=severity,
                    message=message,
                    user_id=user_id,
                    username=username,
                    api_key_id=api_key_id,
                    client_ip=request_info["client_ip"],
                    user_agent=request_info["user_agent"],
                    endpoint=request_info["endpoint"],
                    method=request_info["method"],
                    status_code=response_info["status_code"],
                    session_id=request_info["session_id"],
                    request_id=request_info["request_id"],
                    details={
                        "processing_time_ms": (
                            datetime.now(timezone.utc) - start_time
                        ).total_seconds()
                        * 1000,
                    },
                )

                audit_logger.log_event(entry)

            except Exception as e:
                # Don't let audit logging break the request
                print(f"Audit logging error: {e}")

    def _should_log_request_by_path(self, path: str, status_code: int) -> bool:
        """Determine if request should be logged based on path and status."""
        # Always log failed requests
        if status_code >= 400:
            return True

        # Log sensitive endpoints
        sensitive_paths = [
            "/auth/",
            "/admin/",
            "/api-keys/",
            "/strategies/",
            "/backtest/",
        ]

        return any(
            path.startswith(sensitive_path) for sensitive_path in sensitive_paths
        )

    def _determine_event_type_and_severity_by_path(
        self, path: str, method: str, status_code: int
    ) -> tuple[AuditEventType, AuditSeverity]:
        """Determine event type and severity based on path, method, and status."""
        if status_code >= 500:
            return AuditEventType.SYSTEM_ERROR, AuditSeverity.HIGH
        elif status_code == 403:
            return AuditEventType.ACCESS_DENIED, AuditSeverity.HIGH
        elif status_code == 401:
            return AuditEventType.ACCESS_DENIED, AuditSeverity.MEDIUM
        elif status_code == 429:
            return AuditEventType.RATE_LIMIT_EXCEEDED, AuditSeverity.MEDIUM
        elif path.startswith("/auth/"):
            return AuditEventType.LOGIN_SUCCESS, AuditSeverity.LOW
        elif path.startswith("/admin/"):
            return AuditEventType.ACCESS_GRANTED, AuditSeverity.MEDIUM
        else:
            return AuditEventType.ACCESS_GRANTED, AuditSeverity.LOW

    def _should_log_request(self, request: Request, response) -> bool:
        """Determine if request should be logged."""
        # Always log failed requests
        if response.status_code >= 400:
            return True

        # Log sensitive endpoints
        sensitive_paths = [
            "/auth/",
            "/admin/",
            "/api-keys/",
            "/strategies/",
            "/backtest/",
        ]

        return any(request.url.path.startswith(path) for path in sensitive_paths)

    def _determine_event_type_and_severity(
        self, request: Request, response
    ) -> tuple[AuditEventType, AuditSeverity]:
        """Determine event type and severity based on request/response."""
        if response.status_code >= 500:
            return AuditEventType.SYSTEM_ERROR, AuditSeverity.HIGH
        elif response.status_code == 403:
            return AuditEventType.ACCESS_DENIED, AuditSeverity.HIGH
        elif response.status_code == 401:
            return AuditEventType.ACCESS_DENIED, AuditSeverity.MEDIUM
        elif response.status_code == 429:
            return AuditEventType.RATE_LIMIT_EXCEEDED, AuditSeverity.MEDIUM
        elif request.url.path.startswith("/auth/"):
            return AuditEventType.LOGIN_SUCCESS, AuditSeverity.LOW
        elif request.url.path.startswith("/admin/"):
            return AuditEventType.ACCESS_GRANTED, AuditSeverity.MEDIUM
        else:
            return AuditEventType.ACCESS_GRANTED, AuditSeverity.LOW
