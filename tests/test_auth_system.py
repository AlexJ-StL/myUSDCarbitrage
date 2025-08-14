"""Test JWT authentication system functionality."""

import pytest
from datetime import datetime, timezone, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.main import app
from src.api.database import get_db, Base
from src.api.models import User, Role, UserRole
from src.api.security import SecurityService

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_auth.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def test_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def security_service():
    db = TestingSessionLocal()
    return SecurityService(db)


def test_user_registration(client, test_db):
    """Test user registration endpoint."""
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
    }

    response = client.post("/auth/register", json=user_data)
    assert response.status_code == 201

    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"
    assert data["is_active"] is True
    assert "id" in data


def test_user_login(client, test_db):
    """Test user login endpoint."""
    # First register a user
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
    }
    client.post("/auth/register", json=user_data)

    # Then login
    login_data = {"username": "testuser", "password": "testpassword123"}

    response = client.post("/auth/login", data=login_data)
    assert response.status_code == 200

    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["expires_in"] == 1800


def test_token_refresh(client, test_db):
    """Test token refresh endpoint."""
    # Register and login user
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
    }
    client.post("/auth/register", json=user_data)

    login_response = client.post(
        "/auth/login", data={"username": "testuser", "password": "testpassword123"}
    )
    tokens = login_response.json()

    # Refresh token
    refresh_data = {"refresh_token": tokens["refresh_token"]}
    response = client.post("/auth/refresh", json=refresh_data)

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data


def test_get_current_user(client, test_db):
    """Test getting current user information."""
    # Register and login user
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
    }
    client.post("/auth/register", json=user_data)

    login_response = client.post(
        "/auth/login", data={"username": "testuser", "password": "testpassword123"}
    )
    tokens = login_response.json()

    # Get current user
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    response = client.get("/auth/me", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
    assert data["email"] == "test@example.com"


def test_security_service_password_hashing(security_service):
    """Test password hashing and verification."""
    password = "testpassword123"
    hashed = security_service.get_password_hash(password)

    assert hashed != password
    assert security_service.verify_password(password, hashed)
    assert not security_service.verify_password("wrongpassword", hashed)


def test_security_service_token_creation(security_service):
    """Test JWT token creation and verification."""
    data = {"sub": "testuser", "user_id": 1}
    token = security_service.create_access_token(data)

    assert token is not None
    assert isinstance(token, str)

    # Verify token
    payload = security_service.verify_token(token, "access")
    assert payload["sub"] == "testuser"
    assert payload["user_id"] == 1
    assert payload["type"] == "access"


def test_security_service_user_creation(security_service, test_db):
    """Test user creation through security service."""
    user = security_service.create_user(
        username="testuser", email="test@example.com", password="testpassword123"
    )

    assert user.username == "testuser"
    assert user.email == "test@example.com"
    assert user.is_active is True
    assert security_service.verify_password("testpassword123", user.password_hash)


def test_security_service_user_authentication(security_service, test_db):
    """Test user authentication."""
    # Create user
    security_service.create_user(
        username="testuser", email="test@example.com", password="testpassword123"
    )

    # Test successful authentication
    user = security_service.authenticate_user("testuser", "testpassword123")
    assert user is not None
    assert user.username == "testuser"

    # Test failed authentication
    user = security_service.authenticate_user("testuser", "wrongpassword")
    assert user is None


def test_unauthorized_access(client, test_db):
    """Test that protected endpoints require authentication."""
    response = client.get("/auth/me")
    assert response.status_code == 401


def test_invalid_token(client, test_db):
    """Test access with invalid token."""
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 401


def test_duplicate_user_registration(client, test_db):
    """Test that duplicate usernames are rejected."""
    user_data = {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
    }

    # First registration should succeed
    response1 = client.post("/auth/register", json=user_data)
    assert response1.status_code == 201

    # Second registration with same username should fail
    response2 = client.post("/auth/register", json=user_data)
    assert response2.status_code == 400


def test_user_permissions(security_service, test_db):
    """Test user permissions system."""
    # Create user
    user = security_service.create_user(
        username="testuser", email="test@example.com", password="testpassword123"
    )

    # Get user permissions (should have default user permissions)
    permissions = security_service.get_user_permissions(user.id)
    assert isinstance(permissions, list)
    assert "read:own_data" in permissions


if __name__ == "__main__":
    pytest.main([__file__])
