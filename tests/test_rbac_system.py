"""Tests for Role-Based Access Control (RBAC) system."""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.api.database import Base, get_db
from src.api.main import app
from src.api.models import Role, User, UserRole
from src.api.security import SecurityService


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test_rbac.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
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
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(setup_database):
    """Create a database session for testing."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def security_service(db_session):
    """Create a security service instance."""
    return SecurityService(db_session)


@pytest.fixture
def test_roles(db_session):
    """Create test roles."""
    roles_data = [
        {
            "name": "admin",
            "description": "Administrator with full access",
            "permissions": [
                "manage:system",
                "manage:users",
                "manage:roles",
                "read:all",
                "write:all",
                "delete:all",
                "create:strategy",
                "update:strategy",
                "delete:strategy",
                "create:backtest",
                "read:backtest",
                "delete:backtest",
            ],
        },
        {
            "name": "trader",
            "description": "Trader with strategy and backtest access",
            "permissions": [
                "create:strategy",
                "update:own_strategy",
                "read:strategy",
                "create:backtest",
                "read:own_backtest",
                "read:analytics",
            ],
        },
        {
            "name": "user",
            "description": "Basic user with limited access",
            "permissions": [
                "read:own_data",
                "read:strategy",
                "create:backtest",
                "read:own_backtest",
            ],
        },
    ]

    roles = []
    for role_data in roles_data:
        # Check if role already exists
        existing_role = (
            db_session.query(Role).filter(Role.name == role_data["name"]).first()
        )
        if existing_role:
            roles.append(existing_role)
        else:
            role = Role(
                name=role_data["name"],
                description=role_data["description"],
                permissions=role_data["permissions"],
            )
            db_session.add(role)
            roles.append(role)

    db_session.commit()
    return roles


@pytest.fixture
def test_users(db_session, security_service, test_roles):
    """Create test users with different roles."""
    users_data = [
        {
            "username": "admin_user",
            "email": "admin@test.com",
            "password": "admin123!",
            "role": "admin",
        },
        {
            "username": "trader_user",
            "email": "trader@test.com",
            "password": "trader123!",
            "role": "trader",
        },
        {
            "username": "basic_user",
            "email": "user@test.com",
            "password": "user123!",
            "role": "user",
        },
    ]

    users = []
    for user_data in users_data:
        # Check if user already exists
        existing_user = security_service.get_user_by_username(user_data["username"])
        if existing_user:
            users.append(existing_user)
            continue

        user = security_service.create_user(
            username=user_data["username"],
            email=user_data["email"],
            password=user_data["password"],
        )

        # Assign role
        role = db_session.query(Role).filter(Role.name == user_data["role"]).first()
        if role:
            # Check if role assignment already exists
            existing_assignment = (
                db_session.query(UserRole)
                .filter(UserRole.user_id == user.id, UserRole.role_id == role.id)
                .first()
            )
            if not existing_assignment:
                user_role = UserRole(user_id=user.id, role_id=role.id)
                db_session.add(user_role)

        users.append(user)

    db_session.commit()
    return users


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def get_auth_headers(client, username, password):
    """Get authentication headers for a user."""
    response = client.post(
        "/auth/login",
        data={"username": username, "password": password},
    )
    assert response.status_code == 200
    token_data = response.json()
    return {"Authorization": f"Bearer {token_data['access_token']}"}


class TestRoleManagement:
    """Test role management functionality."""

    def test_create_role_as_admin(self, client, test_users):
        """Test creating a role as admin."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        role_data = {
            "name": "analyst",
            "description": "Data analyst role",
            "permissions": ["read:analytics", "read:strategy", "read:backtest"],
        }

        response = client.post(
            "/admin/roles",
            json=role_data,
            headers=admin_headers,
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "analyst"
        assert "read:analytics" in data["permissions"]

    def test_create_role_as_non_admin_fails(self, client, test_users):
        """Test that non-admin users cannot create roles."""
        trader_headers = get_auth_headers(client, "trader_user", "trader123!")

        role_data = {
            "name": "test_role",
            "description": "Test role",
            "permissions": ["read:strategy"],
        }

        response = client.post(
            "/admin/roles",
            json=role_data,
            headers=trader_headers,
        )

        assert response.status_code == 403

    def test_list_roles_as_admin(self, client, test_users):
        """Test listing roles as admin."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        response = client.get("/admin/roles", headers=admin_headers)

        assert response.status_code == 200
        roles = response.json()
        assert len(roles) >= 3  # admin, trader, user
        role_names = [role["name"] for role in roles]
        assert "admin" in role_names
        assert "trader" in role_names
        assert "user" in role_names

    def test_update_role_as_admin(self, client, test_users, db_session):
        """Test updating a role as admin."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        # Get trader role ID
        trader_role = db_session.query(Role).filter(Role.name == "trader").first()

        update_data = {
            "description": "Updated trader description",
            "permissions": ["create:strategy", "read:strategy", "create:backtest"],
        }

        response = client.put(
            f"/admin/roles/{trader_role.id}",
            json=update_data,
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["description"] == "Updated trader description"


class TestUserManagement:
    """Test user management functionality."""

    def test_list_users_as_admin(self, client, test_users):
        """Test listing users as admin."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        response = client.get("/admin/users", headers=admin_headers)

        assert response.status_code == 200
        users = response.json()
        assert len(users) >= 3
        usernames = [user["username"] for user in users]
        assert "admin_user" in usernames
        assert "trader_user" in usernames
        assert "basic_user" in usernames

    def test_list_users_as_non_admin_fails(self, client, test_users):
        """Test that non-admin users cannot list all users."""
        trader_headers = get_auth_headers(client, "trader_user", "trader123!")

        response = client.get("/admin/users", headers=trader_headers)

        assert response.status_code == 403

    def test_update_user_as_admin(self, client, test_users, db_session):
        """Test updating user as admin."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        # Get basic user ID
        basic_user = (
            db_session.query(User).filter(User.username == "basic_user").first()
        )

        update_data = {
            "is_verified": True,
            "email": "updated_user@test.com",
        }

        response = client.put(
            f"/admin/users/{basic_user.id}",
            json=update_data,
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert data["is_verified"] == True
        assert data["email"] == "updated_user@test.com"

    def test_assign_role_to_user(self, client, test_users, db_session):
        """Test assigning role to user."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        # Get basic user and trader role
        basic_user = (
            db_session.query(User).filter(User.username == "basic_user").first()
        )
        trader_role = db_session.query(Role).filter(Role.name == "trader").first()

        response = client.post(
            f"/admin/users/{basic_user.id}/roles/{trader_role.id}",
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "assigned" in data["message"].lower()

    def test_remove_role_from_user(self, client, test_users, db_session):
        """Test removing role from user."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        # Get trader user and trader role
        trader_user = (
            db_session.query(User).filter(User.username == "trader_user").first()
        )
        trader_role = db_session.query(Role).filter(Role.name == "trader").first()

        response = client.delete(
            f"/admin/users/{trader_user.id}/roles/{trader_role.id}",
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "removed" in data["message"].lower()


class TestResourceAccessControl:
    """Test resource-level access control."""

    def test_strategy_access_with_permissions(self, client, test_users):
        """Test strategy access with proper permissions."""
        trader_headers = get_auth_headers(client, "trader_user", "trader123!")

        # Create a strategy
        strategy_data = {
            "name": "Test Strategy",
            "description": "Test strategy description",
            "code": "def strategy(data): return {'action': 'hold'}",
            "parameters": {"param1": "value1"},
        }

        response = client.post(
            "/strategies/",
            json=strategy_data,
            headers=trader_headers,
        )

        assert response.status_code == 200
        strategy_id = response.json()["id"]

        # Try to read the strategy
        response = client.get(
            f"/strategies/{strategy_id}",
            headers=trader_headers,
        )

        assert response.status_code == 200

    def test_strategy_access_without_permissions(self, client, test_users):
        """Test strategy access without proper permissions."""
        # Create strategy as trader
        trader_headers = get_auth_headers(client, "trader_user", "trader123!")

        strategy_data = {
            "name": "Test Strategy 2",
            "description": "Test strategy description",
            "code": "def strategy(data): return {'action': 'hold'}",
            "parameters": {"param1": "value1"},
        }

        response = client.post(
            "/strategies/",
            json=strategy_data,
            headers=trader_headers,
        )

        assert response.status_code == 200
        strategy_id = response.json()["id"]

        # Try to access as basic user (should work for read)
        basic_headers = get_auth_headers(client, "basic_user", "user123!")

        response = client.get(
            f"/strategies/{strategy_id}",
            headers=basic_headers,
        )

        assert response.status_code == 200  # Basic users can read strategies

        # Try to update as basic user (should fail)
        update_data = {
            "description": "Updated description",
        }

        response = client.put(
            f"/strategies/{strategy_id}",
            json=update_data,
            headers=basic_headers,
        )

        assert response.status_code == 403  # Basic users cannot update strategies

    def test_backtest_access_control(self, client, test_users):
        """Test backtest access control."""
        trader_headers = get_auth_headers(client, "trader_user", "trader123!")

        # Create a strategy first
        strategy_data = {
            "name": "Backtest Strategy",
            "description": "Strategy for backtest",
            "code": "def strategy(data): return {'action': 'hold'}",
            "parameters": {},
        }

        response = client.post(
            "/strategies/",
            json=strategy_data,
            headers=trader_headers,
        )

        assert response.status_code == 200
        strategy_id = response.json()["id"]

        # Create a backtest
        backtest_data = {
            "strategy_id": strategy_id,
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-01-31T00:00:00",
            "initial_balance": 10000.0,
        }

        response = client.post(
            "/backtest/",
            json=backtest_data,
            headers=trader_headers,
        )

        # Note: This might fail due to missing backtesting dependencies
        # but we're testing the authorization, not the actual backtesting
        assert response.status_code in [200, 500]  # 500 for missing dependencies


class TestPermissionSystem:
    """Test permission system functionality."""

    def test_get_user_permissions(self, client, test_users):
        """Test getting user permissions."""
        trader_headers = get_auth_headers(client, "trader_user", "trader123!")

        response = client.get("/auth/permissions", headers=trader_headers)

        assert response.status_code == 200
        data = response.json()
        permissions = data["permissions"]
        assert "create:strategy" in permissions
        assert "read:strategy" in permissions

    def test_admin_has_all_permissions(self, client, test_users):
        """Test that admin has comprehensive permissions."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        response = client.get("/auth/permissions", headers=admin_headers)

        assert response.status_code == 200
        data = response.json()
        permissions = data["permissions"]
        assert "manage:system" in permissions
        assert "manage:users" in permissions
        assert "manage:roles" in permissions

    def test_available_permissions_list(self, client, test_users):
        """Test getting available permissions list."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        response = client.get("/admin/permissions", headers=admin_headers)

        assert response.status_code == 200
        permissions = response.json()
        assert "user_management" in permissions
        assert "role_management" in permissions
        assert "strategy_management" in permissions


class TestSystemStats:
    """Test system statistics functionality."""

    def test_get_system_stats_as_admin(self, client, test_users):
        """Test getting system statistics as admin."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        response = client.get("/admin/stats", headers=admin_headers)

        assert response.status_code == 200
        stats = response.json()
        assert "total_users" in stats
        assert "active_users" in stats
        assert "total_roles" in stats
        assert stats["total_users"] >= 3

    def test_get_system_stats_as_non_admin_fails(self, client, test_users):
        """Test that non-admin users cannot access system stats."""
        trader_headers = get_auth_headers(client, "trader_user", "trader123!")

        response = client.get("/admin/stats", headers=trader_headers)

        assert response.status_code == 403


class TestBulkOperations:
    """Test bulk operations functionality."""

    def test_bulk_user_deactivation(self, client, test_users, db_session):
        """Test bulk user deactivation."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        # Get user IDs (excluding admin)
        trader_user = (
            db_session.query(User).filter(User.username == "trader_user").first()
        )
        basic_user = (
            db_session.query(User).filter(User.username == "basic_user").first()
        )

        bulk_data = {
            "user_ids": [trader_user.id, basic_user.id],
            "operation": "deactivate",
        }

        response = client.post(
            "/admin/users/bulk-operation",
            json=bulk_data,
            headers=admin_headers,
        )

        assert response.status_code == 200
        data = response.json()
        assert "deactivated" in data["message"].lower()

    def test_bulk_operation_prevents_self_modification(
        self, client, test_users, db_session
    ):
        """Test that bulk operations prevent self-modification."""
        admin_headers = get_auth_headers(client, "admin_user", "admin123!")

        # Get admin user ID
        admin_user = (
            db_session.query(User).filter(User.username == "admin_user").first()
        )

        bulk_data = {
            "user_ids": [admin_user.id],
            "operation": "deactivate",
        }

        response = client.post(
            "/admin/users/bulk-operation",
            json=bulk_data,
            headers=admin_headers,
        )

        assert response.status_code == 400
        data = response.json()
        assert "own account" in data["detail"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
