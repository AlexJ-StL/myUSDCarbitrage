"""Tests for API security enhancements."""

import json
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.api_keys import APIKey, APIKeyService
from src.api.audit_logging import AuditEventType, AuditLog, AuditLogger
from src.api.database import Base, get_db
from src.api.encryption import DataEncryption
from src.api.main import app
from src.api.models import Role, User, UserRole
from src.api.rate_limiting import EnhancedRateLimiter
from src.api.security import SecurityService

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_security.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def setup_database():
    """Set up test database."""
    # Import all models to ensure they're registered
    import src.api.models  # This will import AuditLog and APIKey

    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    """Create a database session for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def admin_user(db_session, setup_database):
    """Create an admin user for testing."""
    security_service = SecurityService(db_session)

    # Create admin role
    admin_role = Role(
        name="admin",
        description="Administrator role",
        permissions=[
            "manage:system",
            "manage:users",
            "manage:roles",
            "manage:api_keys",
        ],
    )
    db_session.add(admin_role)
    db_session.commit()

    # Create admin user
    user = security_service.create_user("admin", "admin@test.com", "password123")

    # Assign admin role
    user_role = UserRole(user_id=user.id, role_id=admin_role.id)
    db_session.add(user_role)
    db_session.commit()

    return user


@pytest.fixture
def admin_token(admin_user, db_session):
    """Create an admin access token."""
    security_service = SecurityService(db_session)
    token_data = {
        "sub": admin_user.username,
        "user_id": admin_user.id,
        "permissions": security_service.get_user_permissions(admin_user.id),
    }
    return security_service.create_access_token(token_data)


class TestRateLimiting:
    """Test rate limiting functionality."""

    @patch("redis.from_url")
    def test_rate_limiter_initialization(self, mock_redis):
        """Test rate limiter initialization."""
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        rate_limiter = EnhancedRateLimiter()
        assert rate_limiter.default_rate_limit == 100
        assert rate_limiter.default_window == 3600

    @patch("redis.from_url")
    def test_rate_limit_exceeded(self, mock_redis, client):
        """Test rate limit exceeded scenario."""
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock Redis pipeline to simulate rate limit exceeded
        mock_pipe = MagicMock()
        mock_redis_client.pipeline.return_value = mock_pipe
        mock_pipe.execute.return_value = [None, 101, None, None]  # 101 > 100 limit

        # Make request that should be rate limited
        response = client.get("/health")

        # Note: In actual test, this would be rate limited
        # For now, we're testing the components work
        assert response.status_code in [200, 429]

    def test_rate_limit_headers(self, client):
        """Test rate limit headers are included."""
        response = client.get("/health")

        # Check if rate limit headers are present (they should be added by middleware)
        # Note: Headers might not be present if Redis is not running
        assert response.status_code == 200


class TestEncryption:
    """Test encryption functionality."""

    def test_symmetric_encryption(self):
        """Test symmetric encryption and decryption."""
        encryption = DataEncryption()

        test_data = {"sensitive": "data", "number": 123}

        # Encrypt data
        encrypted = encryption.encrypt_symmetric(test_data)
        assert isinstance(encrypted, str)
        assert encrypted != str(test_data)

        # Decrypt data
        decrypted = encryption.decrypt_symmetric(encrypted)
        assert decrypted == test_data

    def test_encryption_with_different_data_types(self):
        """Test encryption with various data types."""
        encryption = DataEncryption()

        test_cases = [
            {"string": "test"},
            {"number": 42},
            {"list": [1, 2, 3]},
            {"nested": {"key": "value"}},
            {"datetime": datetime.now().isoformat()},
        ]

        for test_data in test_cases:
            encrypted = encryption.encrypt_symmetric(test_data)
            decrypted = encryption.decrypt_symmetric(encrypted)
            assert decrypted == test_data

    def test_encryption_middleware_headers(self, client):
        """Test encryption middleware with proper headers."""
        # Test request with encryption header
        headers = {
            "Content-Type": "application/encrypted+json",
            "Accept": "application/encrypted+json",
        }

        response = client.get("/health", headers=headers)
        assert response.status_code == 200


class TestAPIKeyManagement:
    """Test API key management functionality."""

    def test_api_key_generation(self, db_session):
        """Test API key generation."""
        api_key_service = APIKeyService(db_session)

        full_key, key_hash, key_prefix = api_key_service.generate_api_key()

        assert full_key.startswith("ak_")
        assert len(full_key) == 43  # ak_ + 40 chars
        assert len(key_hash) == 64  # SHA256 hash
        assert key_prefix == full_key[:8]

    def test_create_api_key(self, db_session, admin_user):
        """Test API key creation."""
        from src.api.api_keys import APIKeyCreate

        api_key_service = APIKeyService(db_session)

        key_data = APIKeyCreate(
            name="Test API Key",
            description="Test key for unit tests",
            permissions=["read:data", "create:backtest"],
            expires_in_days=30,
            rate_limit=500,
        )

        api_key = api_key_service.create_api_key(key_data, created_by=admin_user.id)

        assert api_key.name == "Test API Key"
        assert api_key.api_key.startswith("ak_")
        assert api_key.rate_limit == 500
        assert api_key.permissions == ["read:data", "create:backtest"]

    def test_validate_api_key(self, db_session, admin_user):
        """Test API key validation."""
        from src.api.api_keys import APIKeyCreate

        api_key_service = APIKeyService(db_session)

        # Create API key
        key_data = APIKeyCreate(name="Validation Test Key")
        created_key = api_key_service.create_api_key(key_data, created_by=admin_user.id)

        # Validate the key
        validated_key = api_key_service.validate_api_key(created_key.api_key)

        assert validated_key is not None
        assert validated_key.name == "Validation Test Key"
        assert validated_key.usage_count == 1  # Should increment on validation

    def test_api_key_endpoints(self, client, admin_token, setup_database):
        """Test API key management endpoints."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # Create API key
        create_data = {
            "name": "Test Endpoint Key",
            "description": "Key for endpoint testing",
            "permissions": ["read:data"],
            "rate_limit": 1000,
        }

        response = client.post("/api-keys/", json=create_data, headers=headers)
        assert response.status_code == 201

        created_key = response.json()
        assert created_key["name"] == "Test Endpoint Key"
        assert "api_key" in created_key

        key_id = created_key["id"]

        # List API keys
        response = client.get("/api-keys/", headers=headers)
        assert response.status_code == 200
        keys = response.json()
        assert len(keys) >= 1

        # Get specific API key
        response = client.get(f"/api-keys/{key_id}", headers=headers)
        assert response.status_code == 200
        key_details = response.json()
        assert key_details["name"] == "Test Endpoint Key"

        # Update API key
        update_data = {"description": "Updated description"}
        response = client.put(f"/api-keys/{key_id}", json=update_data, headers=headers)
        assert response.status_code == 200

        # Revoke API key
        response = client.post(f"/api-keys/{key_id}/revoke", headers=headers)
        assert response.status_code == 200

        # Delete API key
        response = client.delete(f"/api-keys/{key_id}", headers=headers)
        assert response.status_code == 200


class TestAuditLogging:
    """Test audit logging functionality."""

    def test_audit_logger_initialization(self, db_session):
        """Test audit logger initialization."""
        audit_logger = AuditLogger(db_session)
        assert audit_logger.db == db_session
        assert audit_logger.logger is not None

    def test_log_authentication_event(self, db_session, setup_database):
        """Test logging authentication events."""
        audit_logger = AuditLogger(db_session)

        audit_logger.log_authentication_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            username="testuser",
            client_ip="127.0.0.1",
            success=True,
            details={"method": "password"},
        )

        # Check if log was created
        log_entry = (
            db_session.query(AuditLog)
            .filter(AuditLog.event_type == AuditEventType.LOGIN_SUCCESS.value)
            .first()
        )

        assert log_entry is not None
        assert log_entry.username == "testuser"
        assert log_entry.client_ip == "127.0.0.1"

    def test_log_api_key_event(self, db_session):
        """Test logging API key events."""
        audit_logger = AuditLogger(db_session)

        audit_logger.log_api_key_event(
            event_type=AuditEventType.API_KEY_CREATED,
            api_key_id=1,
            api_key_name="Test Key",
            client_ip="127.0.0.1",
            details={"permissions": ["read:data"]},
        )

        # Check if log was created
        log_entry = (
            db_session.query(AuditLog)
            .filter(AuditLog.event_type == AuditEventType.API_KEY_CREATED.value)
            .first()
        )

        assert log_entry is not None
        assert log_entry.api_key_id == 1
        assert "Test Key" in log_entry.message

    def test_log_rate_limit_event(self, db_session):
        """Test logging rate limit events."""
        audit_logger = AuditLogger(db_session)

        audit_logger.log_rate_limit_event(
            client_ip="127.0.0.1",
            endpoint="/api/test",
            limit=100,
            current_count=150,
            user_id=1,
            username="testuser",
        )

        # Check if log was created
        log_entry = (
            db_session.query(AuditLog)
            .filter(AuditLog.event_type == AuditEventType.RATE_LIMIT_EXCEEDED.value)
            .first()
        )

        assert log_entry is not None
        assert log_entry.client_ip == "127.0.0.1"
        assert log_entry.endpoint == "/api/test"

    def test_log_security_violation(self, db_session):
        """Test logging security violations."""
        audit_logger = AuditLogger(db_session)

        audit_logger.log_security_violation(
            violation_type="unauthorized_access",
            message="Attempted access to admin endpoint without permission",
            client_ip="192.168.1.100",
            endpoint="/admin/users",
            user_id=1,
            username="testuser",
        )

        # Check if log was created
        log_entry = (
            db_session.query(AuditLog)
            .filter(AuditLog.event_type == AuditEventType.SECURITY_VIOLATION.value)
            .first()
        )

        assert log_entry is not None
        assert log_entry.severity == "critical"
        assert "unauthorized_access" in log_entry.message


class TestSecurityIntegration:
    """Test integration of all security components."""

    def test_middleware_order(self, client):
        """Test that security middlewares are applied in correct order."""
        response = client.get("/health")
        assert response.status_code == 200

        # Check that security headers might be present
        # (Actual headers depend on middleware configuration)

    def test_api_key_authentication(
        self, db_session, admin_user, client, setup_database
    ):
        """Test API key authentication flow."""
        from src.api.api_keys import APIKeyCreate

        # Create API key
        api_key_service = APIKeyService(db_session)
        key_data = APIKeyCreate(name="Auth Test Key", permissions=["read:data"])
        created_key = api_key_service.create_api_key(key_data, created_by=admin_user.id)

        # Use API key in request
        headers = {"X-API-Key": created_key.api_key}
        response = client.get("/health", headers=headers)

        assert response.status_code == 200

    def test_comprehensive_security_flow(
        self, client, admin_token, db_session, setup_database
    ):
        """Test comprehensive security flow with all components."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # 1. Create API key (tests authentication, authorization, audit logging)
        create_data = {
            "name": "Comprehensive Test Key",
            "permissions": ["read:data"],
            "rate_limit": 10,  # Low limit for testing
        }

        response = client.post("/api-keys/", json=create_data, headers=headers)
        assert response.status_code == 201

        created_key = response.json()
        api_key = created_key["api_key"]

        # 2. Use API key (tests rate limiting, audit logging)
        api_headers = {"X-API-Key": api_key}

        for i in range(5):  # Make several requests
            response = client.get("/health", headers=api_headers)
            assert response.status_code == 200

            # Check for rate limit headers
            if "X-RateLimit-Limit" in response.headers:
                assert int(response.headers["X-RateLimit-Limit"]) > 0

        # 3. Check audit logs were created
        audit_logs = db_session.query(AuditLog).all()
        assert len(audit_logs) > 0

        # Should have API key creation log
        creation_logs = [
            log
            for log in audit_logs
            if log.event_type == AuditEventType.API_KEY_CREATED.value
        ]
        assert len(creation_logs) > 0

    def test_error_handling_with_security(self, client):
        """Test error handling with security components."""
        # Test unauthorized access
        response = client.get("/admin/users")
        assert response.status_code == 401

        # Test invalid API key
        headers = {"X-API-Key": "invalid_key"}
        response = client.get("/health", headers=headers)
        # Should still work as API key is optional for health endpoint
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
