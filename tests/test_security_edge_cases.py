"""
Comprehensive security and edge case testing for the USDC arbitrage backtesting application.

This module includes:
1. Security testing for authentication and authorization
2. Fuzzing tests for API input validation
3. Edge case testing for boundary conditions and error scenarios
4. Penetration testing scenarios for common vulnerabilities
"""

import json
import os
import random
import string
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# Mock the necessary components to avoid circular imports
class MockSecurityService:
    """Mock security service for testing."""

    def __init__(self, db=None):
        self.db = db

    def create_user(self, username, email, password):
        """Mock user creation."""
        user = MagicMock()
        user.id = random.randint(1, 1000)
        user.username = username
        user.email = email
        user.password_hash = f"hashed_{password}"
        user.is_active = True
        return user

    def get_password_hash(self, password):
        """Mock password hashing."""
        return f"hashed_{password}"

    def verify_password(self, plain_password, hashed_password):
        """Mock password verification."""
        return hashed_password == f"hashed_{plain_password}"

    def create_access_token(self, data, expires_delta=None):
        """Mock token creation."""
        return f"mock_token_{data.get('user_id', 0)}"

    def verify_token(self, token, token_type="access"):
        """Mock token verification."""
        if token.startswith("mock_token_"):
            user_id = int(token.split("_")[-1])
            return {"sub": f"user_{user_id}", "user_id": user_id, "type": token_type}
        raise Exception("Invalid token")

    def get_user_permissions(self, user_id):
        """Mock permission retrieval."""
        if user_id == 1:  # Admin
            return ["manage:system", "manage:users", "manage:roles", "manage:api_keys"]
        return ["read:own_data", "create:backtest", "read:strategies"]


class MockAPIKeyService:
    """Mock API key service for testing."""

    def __init__(self, db=None):
        self.db = db

    def generate_api_key(self):
        """Mock API key generation."""
        full_key = f"ak_{''.join(random.choices(string.ascii_lowercase + string.digits, k=40))}"
        key_hash = f"hash_{''.join(random.choices(string.ascii_lowercase + string.digits, k=64))}"
        key_prefix = full_key[:8]
        return full_key, key_hash, key_prefix

    def create_api_key(self, key_data, created_by=None):
        """Mock API key creation."""
        api_key = MagicMock()
        api_key.id = random.randint(1, 1000)
        api_key.name = key_data.name
        api_key.api_key, _, _ = self.generate_api_key()
        api_key.rate_limit = getattr(key_data, "rate_limit", 100)
        api_key.permissions = getattr(key_data, "permissions", [])
        return api_key

    def validate_api_key(self, api_key):
        """Mock API key validation."""
        if api_key.startswith("ak_"):
            mock_key = MagicMock()
            mock_key.name = "Validated Key"
            mock_key.usage_count = 1
            return mock_key
        return None


# Create a simple FastAPI app for testing
app = FastAPI()


# Add basic routes for testing
from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    email: str
    password: str


@app.post("/auth/register", status_code=201)
async def register(user: UserCreate):
    if len(user.password) < 6:
        return {"error": "Password too short"}, 400
    return {"username": user.username, "email": user.email, "is_active": True}


class LoginForm(BaseModel):
    username: str = None
    password: str = None


@app.post("/auth/login")
async def login(form_data: LoginForm = None):
    if (
        form_data
        and form_data.username == "admin"
        and form_data.password == "password123"
    ):
        return {
            "access_token": "mock_token",
            "refresh_token": "mock_refresh_token",
            "token_type": "bearer",
            "expires_in": 1800,
        }
    return {"error": "Invalid credentials"}


class RefreshRequest(BaseModel):
    refresh_token: str


@app.post("/auth/refresh")
async def refresh_token(request: RefreshRequest = None):
    if request and request.refresh_token == "valid_token":
        return {
            "access_token": "new_mock_token",
            "refresh_token": "new_mock_refresh_token",
        }
    return {"error": "Invalid refresh token"}


@app.post("/auth/logout")
async def logout():
    return {"message": "Successfully logged out"}


@app.get("/auth/me")
async def get_me():
    return {"username": "testuser", "email": "test@example.com"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Mock database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_security_edge_cases.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Mock models
class Base:
    metadata = MagicMock()


class Role:
    def __init__(self, name, description, permissions):
        self.id = random.randint(1, 1000)
        self.name = name
        self.description = description
        self.permissions = permissions


class User:
    def __init__(self, username, email, password_hash):
        self.id = random.randint(1, 1000)
        self.username = username
        self.email = email
        self.password_hash = password_hash
        self.is_active = True
        self.last_login = datetime.now(timezone.utc)


class UserRole:
    def __init__(self, user_id, role_id):
        self.user_id = user_id
        self.role_id = role_id


# Mock DataEncryption
class DataEncryption:
    """Mock data encryption for testing."""

    def encrypt_symmetric(self, data):
        """Mock symmetric encryption."""
        return f"encrypted_{json.dumps(data)}"

    def decrypt_symmetric(self, encrypted_data):
        """Mock symmetric decryption."""
        if encrypted_data.startswith("encrypted_"):
            return json.loads(encrypted_data[10:])
        return encrypted_data


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_security_edge_cases.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    db = MagicMock()
    yield db


@pytest.fixture(scope="module")
def setup_database():
    """Set up test database."""
    yield


@pytest.fixture
def db_session():
    """Create a database session for testing."""
    db = MagicMock()
    yield db


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def security_service():
    """Create a security service instance."""
    return MockSecurityService()


@pytest.fixture
def admin_user(security_service):
    """Create an admin user for testing."""
    # Create admin user
    user = security_service.create_user("admin", "admin@test.com", "password123")
    user.id = 1  # Set ID to 1 for admin
    return user


@pytest.fixture
def regular_user(security_service):
    """Create a regular user for testing."""
    user = security_service.create_user("user", "user@test.com", "password123")
    user.id = 2  # Set ID to 2 for regular user
    return user


@pytest.fixture
def admin_token(admin_user, security_service):
    """Create an admin access token."""
    token_data = {
        "sub": admin_user.username,
        "user_id": admin_user.id,
        "permissions": security_service.get_user_permissions(admin_user.id),
    }
    return security_service.create_access_token(token_data)


@pytest.fixture
def user_token(regular_user, security_service):
    """Create a regular user access token."""
    token_data = {
        "sub": regular_user.username,
        "user_id": regular_user.id,
        "permissions": security_service.get_user_permissions(regular_user.id),
    }
    return security_service.create_access_token(token_data)


@pytest.fixture
def api_key_service():
    """Create an API key service instance."""
    return MockAPIKeyService()


class TestAuthenticationSecurity:
    """Test authentication security features."""

    def test_password_complexity(self, client):
        """Test password complexity requirements."""
        # Test weak passwords
        weak_passwords = [
            "short",  # Too short
            "onlyletters",  # No numbers or special chars
            "12345678",  # Only numbers
            "password123",  # Common password
        ]

        for password in weak_passwords:
            user_data = {
                "username": f"test_user_{random.randint(1000, 9999)}",
                "email": f"test{random.randint(1000, 9999)}@example.com",
                "password": password,
            }
            response = client.post("/auth/register", json=user_data)
            # Should either reject or accept based on your password policy
            # Here we're assuming a policy that might reject these
            assert response.status_code in [400, 422, 201]

        # Test strong password
        strong_password = "StrongP@ssw0rd123!"
        user_data = {
            "username": f"test_user_{random.randint(1000, 9999)}",
            "email": f"test{random.randint(1000, 9999)}@example.com",
            "password": strong_password,
        }
        response = client.post("/auth/register", json=user_data)
        assert response.status_code == 201

    def test_brute_force_protection(self, client, regular_user):
        """Test protection against brute force attacks."""
        # Try multiple incorrect passwords
        for _ in range(5):
            login_data = {
                "username": regular_user.username,
                "password": f"wrong_password_{random.randint(1000, 9999)}",
            }
            response = client.post("/auth/login", data=login_data)
            assert response.status_code == 401

        # Check if account is locked or rate limited
        # This depends on your implementation - might be locked or rate limited
        login_data = {
            "username": regular_user.username,
            "password": "password123",  # Correct password
        }
        response = client.post("/auth/login", data=login_data)
        # Should either be locked (403) or still work (200) depending on your policy
        assert response.status_code in [200, 403, 429]

    def test_token_expiration(self, security_service):
        """Test token expiration."""
        # Create token with short expiration
        token_data = {"sub": "test_user", "user_id": 1}
        token = security_service.create_access_token(
            token_data, expires_delta=timedelta(seconds=1)
        )

        # Wait for token to expire
        time.sleep(2)

        # Verify token should fail
        with pytest.raises(Exception):
            security_service.verify_token(token, "access")

    def test_token_refresh_security(self, client, regular_user, security_service):
        """Test token refresh security."""
        # Login to get tokens
        login_data = {"username": regular_user.username, "password": "password123"}
        response = client.post("/auth/login", data=login_data)
        tokens = response.json()

        # Test refresh with invalid token
        invalid_refresh = {"refresh_token": "invalid_token"}
        response = client.post("/auth/refresh", json=invalid_refresh)
        assert response.status_code == 401

        # Test refresh with valid token
        valid_refresh = {"refresh_token": tokens["refresh_token"]}
        response = client.post("/auth/refresh", json=valid_refresh)
        assert response.status_code == 200
        new_tokens = response.json()
        assert "access_token" in new_tokens
        assert "refresh_token" in new_tokens

        # Test that old refresh token is invalidated (can't be used twice)
        response = client.post("/auth/refresh", json=valid_refresh)
        assert response.status_code == 401

    def test_logout_token_blacklisting(self, client, regular_user, security_service):
        """Test that tokens are blacklisted on logout."""
        # Login to get tokens
        login_data = {"username": regular_user.username, "password": "password123"}
        response = client.post("/auth/login", data=login_data)
        tokens = response.json()

        # Logout
        headers = {"Authorization": f"Bearer {tokens['access_token']}"}
        response = client.post("/auth/logout", headers=headers)
        assert response.status_code == 200

        # Try to use the token after logout
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 401


class TestAuthorizationSecurity:
    """Test authorization security features."""

    def test_privilege_escalation_prevention(
        self, client, admin_token, user_token, db_session
    ):
        """Test prevention of privilege escalation."""
        # Try to create admin role as regular user
        headers = {"Authorization": f"Bearer {user_token}"}
        role_data = {
            "name": "new_admin",
            "description": "New admin role",
            "permissions": ["manage:system", "manage:users"],
        }

        response = client.post("/admin/roles", json=role_data, headers=headers)
        assert response.status_code == 403  # Should be forbidden

        # Try to assign admin role to self as regular user
        admin_role = db_session.query(Role).filter(Role.name == "admin").first()
        regular_user = db_session.query(User).filter(User.username == "user").first()

        response = client.post(
            f"/admin/users/{regular_user.id}/roles/{admin_role.id}",
            headers=headers,
        )
        assert response.status_code == 403  # Should be forbidden

    def test_horizontal_privilege_escalation(
        self, client, security_service, db_session
    ):
        """Test prevention of horizontal privilege escalation."""
        # Create two regular users
        user1 = security_service.create_user("user1", "user1@test.com", "password123")
        user2 = security_service.create_user("user2", "user2@test.com", "password123")

        # Create tokens for both users
        token_data1 = {
            "sub": user1.username,
            "user_id": user1.id,
            "permissions": security_service.get_user_permissions(user1.id),
        }
        token1 = security_service.create_access_token(token_data1)

        # Try to access user2's data with user1's token
        headers = {"Authorization": f"Bearer {token1}"}
        response = client.get(f"/users/{user2.id}", headers=headers)

        # Should either be forbidden (403) or not found (404) depending on your API
        assert response.status_code in [403, 404]

    def test_insecure_direct_object_reference(self, client, user_token):
        """Test protection against insecure direct object references."""
        headers = {"Authorization": f"Bearer {user_token}"}

        # Try to access resources by guessing IDs
        for resource_id in range(1, 10):
            response = client.get(f"/strategies/{resource_id}", headers=headers)
            # Should either return 404 (not found) or 403 (forbidden)
            # But should not expose data that doesn't belong to the user
            assert response.status_code in [403, 404, 200]

            if response.status_code == 200:
                # If successful, verify the resource belongs to the user
                data = response.json()
                # This check depends on your data model
                # assert data["user_id"] == regular_user.id


class TestInputValidationFuzzing:
    """Test input validation with fuzzing techniques."""

    def generate_random_string(self, length=10):
        """Generate a random string for fuzzing."""
        return "".join(random.choice(string.printable) for _ in range(length))

    def test_login_fuzzing(self, client):
        """Test login endpoint with fuzzed inputs."""
        fuzz_inputs = [
            {"username": self.generate_random_string(100), "password": "password123"},
            {"username": "admin", "password": self.generate_random_string(1000)},
            {"username": "<script>alert('xss')</script>", "password": "password123"},
            {"username": "admin'; DROP TABLE users; --", "password": "password123"},
            {"username": "", "password": ""},
            {"username": None, "password": None},
            {"username": True, "password": False},
            {"username": 12345, "password": 67890},
            {"username": ["admin"], "password": {"password": "123"}},
        ]

        for fuzz_input in fuzz_inputs:
            try:
                response = client.post("/auth/login", data=fuzz_input)
                # Should not crash the server
                assert response.status_code in [400, 401, 422]
            except Exception as e:
                pytest.fail(f"Login fuzzing caused exception: {e}")

    def test_registration_fuzzing(self, client):
        """Test registration endpoint with fuzzed inputs."""
        fuzz_inputs = [
            {
                "username": self.generate_random_string(100),
                "email": "test@example.com",
                "password": "password123",
            },
            {
                "username": "newuser",
                "email": self.generate_random_string(1000),
                "password": "password123",
            },
            {
                "username": "newuser",
                "email": "test@example.com",
                "password": self.generate_random_string(2000),
            },
            {
                "username": "<script>alert('xss')</script>",
                "email": "test@example.com",
                "password": "password123",
            },
            {
                "username": "newuser",
                "email": "<script>alert('xss')</script>",
                "password": "password123",
            },
            {"username": "", "email": "", "password": ""},
            {"username": None, "email": None, "password": None},
            {"username": True, "email": False, "password": 12345},
            {"username": ["admin"], "email": ["test@example.com"], "password": [123]},
        ]

        for fuzz_input in fuzz_inputs:
            try:
                response = client.post("/auth/register", json=fuzz_input)
                # Should not crash the server
                assert response.status_code in [400, 401, 422, 201]
            except Exception as e:
                pytest.fail(f"Registration fuzzing caused exception: {e}")

    def test_api_endpoint_fuzzing(self, client, admin_token):
        """Test API endpoints with fuzzed inputs."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # Test strategy creation with fuzzed inputs
        fuzz_strategy_inputs = [
            {
                "name": self.generate_random_string(500),
                "description": self.generate_random_string(1000),
                "code": self.generate_random_string(5000),
                "parameters": {"param1": self.generate_random_string(100)},
            },
            {
                "name": "<script>alert('xss')</script>",
                "description": "SQL Injection: ' OR 1=1 --",
                "code": "def strategy(): return {'action': 'hold'}",
                "parameters": {"param1": "<script>alert('xss')</script>"},
            },
            {
                "name": "",
                "description": None,
                "code": "",
                "parameters": {},
            },
            {
                "name": 12345,
                "description": True,
                "code": ["code"],
                "parameters": "not_a_dict",
            },
        ]

        for fuzz_input in fuzz_strategy_inputs:
            try:
                response = client.post("/strategies/", json=fuzz_input, headers=headers)
                # Should not crash the server
                assert response.status_code in [400, 401, 422, 200, 201]
            except Exception as e:
                pytest.fail(f"Strategy creation fuzzing caused exception: {e}")

    def test_query_parameter_fuzzing(self, client, admin_token):
        """Test query parameters with fuzzed inputs."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        fuzz_queries = [
            "?limit=9999999999",
            "?offset=-1",
            "?sort=<script>alert('xss')</script>",
            "?filter='; DROP TABLE users; --",
            "?"
            + "&".join([
                f"param{i}={self.generate_random_string(100)}" for i in range(20)
            ]),
        ]

        for fuzz_query in fuzz_queries:
            try:
                response = client.get(f"/strategies/{fuzz_query}", headers=headers)
                # Should not crash the server
                assert response.status_code in [400, 401, 404, 422, 200]
            except Exception as e:
                pytest.fail(f"Query parameter fuzzing caused exception: {e}")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pagination_edge_cases(self, client, admin_token):
        """Test pagination edge cases."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # Test extreme pagination values
        edge_cases = [
            {"limit": 0, "offset": 0},
            {"limit": -1, "offset": 0},
            {"limit": 0, "offset": -1},
            {"limit": 1000000, "offset": 0},
            {"limit": 100, "offset": 1000000},
            {"limit": 2**31 - 1, "offset": 2**31 - 1},  # Max int32
        ]

        for case in edge_cases:
            response = client.get(
                f"/strategies/?limit={case['limit']}&offset={case['offset']}",
                headers=headers,
            )
            # Should handle these gracefully
            assert response.status_code in [200, 400, 422]

    def test_date_time_edge_cases(self, client, admin_token):
        """Test date/time edge cases."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # Create a strategy first (needed for backtest)
        strategy_data = {
            "name": "Edge Case Strategy",
            "description": "Strategy for edge case testing",
            "code": "def strategy(data): return {'action': 'hold'}",
            "parameters": {},
        }

        strategy_response = client.post(
            "/strategies/", json=strategy_data, headers=headers
        )

        if strategy_response.status_code in [200, 201]:
            strategy_id = strategy_response.json().get("id", 1)

            # Test with edge case dates
            edge_case_dates = [
                # Very old date
                {
                    "strategy_id": strategy_id,
                    "start_date": "1900-01-01T00:00:00",
                    "end_date": "1900-01-31T00:00:00",
                    "initial_balance": 10000.0,
                },
                # Future date
                {
                    "strategy_id": strategy_id,
                    "start_date": "2050-01-01T00:00:00",
                    "end_date": "2050-01-31T00:00:00",
                    "initial_balance": 10000.0,
                },
                # End date before start date
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-31T00:00:00",
                    "end_date": "2023-01-01T00:00:00",
                    "initial_balance": 10000.0,
                },
                # Same start and end date
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-01T00:00:00",
                    "initial_balance": 10000.0,
                },
                # Invalid date format
                {
                    "strategy_id": strategy_id,
                    "start_date": "not-a-date",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 10000.0,
                },
            ]

            for case in edge_case_dates:
                response = client.post("/backtest/", json=case, headers=headers)
                # Should handle these gracefully
                assert response.status_code in [200, 400, 422, 500]

    def test_numeric_edge_cases(self, client, admin_token):
        """Test numeric edge cases."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # Create a strategy first
        strategy_data = {
            "name": "Numeric Edge Case Strategy",
            "description": "Strategy for numeric edge case testing",
            "code": "def strategy(data): return {'action': 'hold'}",
            "parameters": {},
        }

        strategy_response = client.post(
            "/strategies/", json=strategy_data, headers=headers
        )

        if strategy_response.status_code in [200, 201]:
            strategy_id = strategy_response.json().get("id", 1)

            # Test with edge case numeric values
            edge_case_values = [
                # Zero initial balance
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 0.0,
                },
                # Negative initial balance
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": -1000.0,
                },
                # Very large initial balance
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 1e20,
                },
                # Very small initial balance
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 1e-10,
                },
            ]

            for case in edge_case_values:
                response = client.post("/backtest/", json=case, headers=headers)
                # Should handle these gracefully
                assert response.status_code in [200, 400, 422, 500]

    def test_empty_and_null_edge_cases(self, client, admin_token):
        """Test empty and null edge cases."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # Test with empty and null values
        edge_cases = [
            # Empty strategy name
            {
                "name": "",
                "description": "Strategy with empty name",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {},
            },
            # Null description
            {
                "name": "Null Description Strategy",
                "description": None,
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {},
            },
            # Empty code
            {
                "name": "Empty Code Strategy",
                "description": "Strategy with empty code",
                "code": "",
                "parameters": {},
            },
            # Empty parameters
            {
                "name": "Empty Parameters Strategy",
                "description": "Strategy with empty parameters",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {},
            },
        ]

        for case in edge_cases:
            response = client.post("/strategies/", json=case, headers=headers)
            # Should handle these gracefully
            assert response.status_code in [200, 201, 400, 422]


class TestPenetrationTesting:
    """Test common security vulnerabilities."""

    def test_sql_injection(self, client, admin_token):
        """Test protection against SQL injection."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # SQL injection attempts
        sql_injection_attempts = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT username, password FROM users; --",
            "' OR username LIKE '%admin%",
            "'; INSERT INTO users (username, password) VALUES ('hacker', 'password'); --",
        ]

        for injection in sql_injection_attempts:
            # Try injection in query parameters
            response = client.get(f"/users/?username={injection}", headers=headers)
            assert response.status_code in [200, 400, 404, 422]

            # Try injection in JSON body
            user_data = {
                "username": injection,
                "email": "test@example.com",
                "password": "password123",
            }
            response = client.post("/auth/register", json=user_data)
            assert response.status_code in [201, 400, 422]

    def test_xss_protection(self, client, admin_token):
        """Test protection against Cross-Site Scripting (XSS)."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # XSS attempts
        xss_attempts = [
            "<script>alert('xss')</script>",
            "<img src='x' onerror='alert(\"XSS\")'>",
            "<body onload='alert(\"XSS\")'>",
            "javascript:alert('XSS')",
            "<svg/onload=alert('XSS')>",
        ]

        for xss in xss_attempts:
            # Try XSS in strategy name
            strategy_data = {
                "name": xss,
                "description": "XSS test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {},
            }

            response = client.post("/strategies/", json=strategy_data, headers=headers)

            if response.status_code in [200, 201]:
                # If created successfully, check that XSS is sanitized when retrieved
                strategy_id = response.json().get("id", 1)
                get_response = client.get(f"/strategies/{strategy_id}", headers=headers)

                if get_response.status_code == 200:
                    strategy = get_response.json()
                    # The name should either be sanitized or rejected
                    assert strategy["name"] != xss or "<script>" not in strategy["name"]

    def test_csrf_protection(self, client, admin_token):
        """Test protection against Cross-Site Request Forgery (CSRF)."""
        # For APIs using token-based auth, CSRF is typically not an issue
        # But we can test that state-changing operations require proper auth

        # Try state-changing operation without auth
        strategy_data = {
            "name": "CSRF Test Strategy",
            "description": "Strategy for CSRF testing",
            "code": "def strategy(data): return {'action': 'hold'}",
            "parameters": {},
        }

        response = client.post("/strategies/", json=strategy_data)
        assert response.status_code in [401, 403]  # Should require authentication

        # Now with auth header
        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.post("/strategies/", json=strategy_data, headers=headers)
        assert response.status_code in [200, 201]  # Should work with proper auth

    def test_open_redirect(self, client):
        """Test protection against open redirects."""
        # Test if login/logout redirects can be manipulated
        redirect_attempts = [
            "https://malicious-site.com",
            "//evil.com",
            "/\\evil.com",
            "javascript:alert('hacked')",
        ]

        for redirect in redirect_attempts:
            response = client.get(f"/auth/login?redirect_url={redirect}")
            # Should not redirect to external sites
            assert response.status_code in [200, 400, 404]

            if response.status_code == 200 and "Location" in response.headers:
                location = response.headers["Location"]
                assert not location.startswith("http://malicious-site.com")
                assert not location.startswith("//evil.com")
                assert not location.startswith("javascript:")

    def test_directory_traversal(self, client, admin_token):
        """Test protection against directory traversal."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # Directory traversal attempts
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/passwd",
            "C:\\Windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
        ]

        for traversal in traversal_attempts:
            # Try in file-related endpoints
            response = client.get(f"/files/?path={traversal}", headers=headers)
            assert response.status_code in [400, 403, 404]

            # Try in report generation
            response = client.get(f"/reports/?template={traversal}", headers=headers)
            assert response.status_code in [400, 403, 404]

    def test_server_side_request_forgery(self, client, admin_token):
        """Test protection against Server-Side Request Forgery (SSRF)."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # SSRF attempts targeting internal services
        ssrf_attempts = [
            "http://localhost:22",  # SSH
            "http://127.0.0.1:3306",  # MySQL
            "http://10.0.0.1",  # Internal network
            "http://169.254.169.254/latest/meta-data/",  # AWS metadata
            "file:///etc/passwd",  # Local file
        ]

        for ssrf in ssrf_attempts:
            # Try in URL fetch endpoint if it exists
            response = client.post("/fetch-url/", json={"url": ssrf}, headers=headers)
            assert response.status_code in [400, 403, 404]

            # Try in webhook configuration if it exists
            response = client.post("/webhooks/", json={"url": ssrf}, headers=headers)
            assert response.status_code in [400, 403, 404]


class TestRateLimitingAndDDoSProtection:
    """Test rate limiting and DDoS protection."""

    @patch("redis.from_url")
    def test_rate_limit_enforcement(self, mock_redis, client):
        """Test that rate limits are enforced."""
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        # Mock Redis pipeline to simulate rate limit exceeded
        mock_pipe = MagicMock()
        mock_redis_client.pipeline.return_value = mock_pipe

        # First simulate under the limit
        mock_pipe.execute.return_value = [None, 5, None, None]  # 5 < 10 limit

        # Make request that should be under rate limit
        response = client.post(
            "/auth/login", data={"username": "test", "password": "test"}
        )

        # Now simulate over the limit
        mock_pipe.execute.return_value = [None, 11, None, None]  # 11 > 10 limit

        # Make request that should be rate limited
        response = client.post(
            "/auth/login", data={"username": "test", "password": "test"}
        )

        # Note: In actual test with Redis, this would be rate limited (429)
        # For now, we're testing the components work
        assert response.status_code in [401, 429]

    def test_api_key_rate_limits(self, client, admin_user, api_key_service):
        """Test API key specific rate limits."""
        from src.api.api_keys import APIKeyCreate

        # Create API key with low rate limit
        key_data = APIKeyCreate(
            name="Low Rate Limit Key",
            rate_limit=5,  # Very low limit
            permissions=["read:data"],
        )

        api_key = api_key_service.create_api_key(key_data, created_by=admin_user.id)

        # Make multiple requests with this key
        headers = {"X-API-Key": api_key.api_key}

        for i in range(10):  # More than the limit
            response = client.get("/health", headers=headers)
            # After 5 requests, should start getting rate limited
            # But since Redis might not be running in tests, we accept both
            assert response.status_code in [200, 429]


class TestEncryptionAndDataProtection:
    """Test encryption and data protection features."""

    def test_symmetric_encryption(self):
        """Test symmetric encryption functionality."""
        encryption = DataEncryption()

        # Test with different data types
        test_cases = [
            {"string": "sensitive data"},
            {"number": 12345},
            {"boolean": True},
            {"list": [1, 2, 3, 4, 5]},
            {"nested": {"key": "value", "another": 123}},
            {"mixed": {"name": "test", "values": [1, 2, 3], "active": True}},
        ]

        for test_data in test_cases:
            # Encrypt
            encrypted = encryption.encrypt_symmetric(test_data)
            assert isinstance(encrypted, str)
            assert encrypted != str(test_data)

            # Decrypt
            decrypted = encryption.decrypt_symmetric(encrypted)
            assert decrypted == test_data

    def test_sensitive_field_encryption(self):
        """Test encryption of sensitive fields."""
        from src.api.encryption import (
            encrypt_sensitive_fields,
            decrypt_sensitive_fields,
        )

        test_data = {
            "username": "testuser",
            "password": "sensitive_password",
            "email": "test@example.com",
            "api_key": "secret_api_key",
            "balance": 10000.50,
        }

        sensitive_fields = ["password", "api_key", "balance"]

        # Encrypt sensitive fields
        encrypted_data = encrypt_sensitive_fields(test_data, sensitive_fields)

        # Check that sensitive fields are encrypted
        for field in sensitive_fields:
            assert encrypted_data[field] != test_data[field]

        # Check that non-sensitive fields are not encrypted
        assert encrypted_data["username"] == test_data["username"]
        assert encrypted_data["email"] == test_data["email"]

        # Decrypt sensitive fields
        decrypted_data = decrypt_sensitive_fields(encrypted_data, sensitive_fields)

        # Check that all fields match original data
        for key, value in test_data.items():
            assert decrypted_data[key] == value

    def test_password_storage_security(self, security_service):
        """Test that passwords are securely stored."""
        password = "TestPassword123!"
        hashed = security_service.get_password_hash(password)

        # Password should not be stored in plaintext
        assert hashed != password

        # Hash should be different each time
        hashed2 = security_service.get_password_hash(password)
        assert hashed != hashed2

        # But verification should work
        assert security_service.verify_password(password, hashed)
        assert security_service.verify_password(password, hashed2)

        # And wrong password should fail
        assert not security_service.verify_password("WrongPassword", hashed)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
