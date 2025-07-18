"""Initialize authentication system with default users and roles."""

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from .database import SessionLocal, engine
from .models import Base, Role, User, UserRole
from .security import SecurityService

load_dotenv()


def create_default_roles(db: Session):
    """Create default roles if they don't exist."""
    roles_data = [
        {
            "name": "admin",
            "description": "Administrator with full system access",
            "permissions": [
                "read:all",
                "write:all",
                "delete:all",
                "manage:users",
                "manage:roles",
                "manage:system",
                "create:strategy",
                "update:strategy",
                "delete:strategy",
                "create:backtest",
                "read:backtest",
                "delete:backtest",
                "read:analytics",
                "manage:api_keys",
            ],
        },
        {
            "name": "trader",
            "description": "Trader with strategy and backtest access",
            "permissions": [
                "read:own_data",
                "create:strategy",
                "update:own_strategy",
                "read:strategy",
                "create:backtest",
                "read:own_backtest",
                "read:analytics",
                "read:market_data",
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
                "read:market_data",
            ],
        },
        {
            "name": "analyst",
            "description": "Analyst with read access to all data",
            "permissions": [
                "read:all",
                "read:analytics",
                "read:strategy",
                "read:backtest",
                "read:market_data",
                "create:backtest",
            ],
        },
    ]

    for role_data in roles_data:
        existing_role = db.query(Role).filter(Role.name == role_data["name"]).first()
        if not existing_role:
            role = Role(
                name=role_data["name"],
                description=role_data["description"],
                permissions=role_data["permissions"],
            )
            db.add(role)
            print(f"Created role: {role_data['name']}")
        else:
            # Update permissions if role exists
            existing_role.permissions = role_data["permissions"]
            existing_role.description = role_data["description"]
            print(f"Updated role: {role_data['name']}")

    db.commit()


def create_default_admin(db: Session):
    """Create default admin user if it doesn't exist."""
    admin_username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
    admin_email = os.getenv("DEFAULT_ADMIN_EMAIL", "admin@example.com")
    admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin123!")

    security_service = SecurityService(db)

    # Check if admin user exists
    existing_admin = security_service.get_user_by_username(admin_username)
    if not existing_admin:
        try:
            # Create admin user
            admin_user = User(
                username=admin_username,
                email=admin_email,
                password_hash=security_service.get_password_hash(admin_password),
                is_active=True,
                is_verified=True,
            )
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)

            # Assign admin role
            admin_role = db.query(Role).filter(Role.name == "admin").first()
            if admin_role:
                user_role = UserRole(user_id=admin_user.id, role_id=admin_role.id)
                db.add(user_role)
                db.commit()

            print(f"Created admin user: {admin_username}")
            print(f"Admin password: {admin_password}")
            print("Please change the default password after first login!")

        except Exception as e:
            print(f"Error creating admin user: {e}")
            db.rollback()
    else:
        print(f"Admin user '{admin_username}' already exists")


def cleanup_expired_tokens(db: Session):
    """Clean up expired tokens from blacklist and refresh token tables."""
    from .models import RefreshToken, TokenBlacklist

    current_time = datetime.now(timezone.utc)

    # Clean up expired blacklisted tokens
    expired_blacklist = (
        db.query(TokenBlacklist).filter(TokenBlacklist.expires_at < current_time).all()
    )

    for token in expired_blacklist:
        db.delete(token)

    # Clean up expired refresh tokens
    expired_refresh = (
        db.query(RefreshToken).filter(RefreshToken.expires_at < current_time).all()
    )

    for token in expired_refresh:
        db.delete(token)

    db.commit()
    print(f"Cleaned up {len(expired_blacklist)} expired blacklisted tokens")
    print(f"Cleaned up {len(expired_refresh)} expired refresh tokens")


def initialize_auth_system():
    """Initialize the authentication system."""
    print("Initializing authentication system...")

    # Create database session
    db = SessionLocal()

    try:
        # Import all models to ensure they're registered
        from .models import User, Role, UserRole, RefreshToken, TokenBlacklist

        # Drop and recreate auth tables if they exist (for clean setup)
        print("Dropping existing auth tables if they exist...")
        try:
            TokenBlacklist.__table__.drop(bind=engine, checkfirst=True)
            RefreshToken.__table__.drop(bind=engine, checkfirst=True)
            UserRole.__table__.drop(bind=engine, checkfirst=True)
            Role.__table__.drop(bind=engine, checkfirst=True)
            User.__table__.drop(bind=engine, checkfirst=True)
        except Exception as e:
            print(f"Note: Some tables may not have existed: {e}")

        # Create tables in the correct order
        print("Creating auth tables...")
        User.__table__.create(bind=engine, checkfirst=True)
        Role.__table__.create(bind=engine, checkfirst=True)
        UserRole.__table__.create(bind=engine, checkfirst=True)
        RefreshToken.__table__.create(bind=engine, checkfirst=True)
        TokenBlacklist.__table__.create(bind=engine, checkfirst=True)

        print("Database tables created successfully!")

        # Create default roles
        create_default_roles(db)

        # Create default admin user
        create_default_admin(db)

        # Clean up expired tokens
        cleanup_expired_tokens(db)

        print("Authentication system initialized successfully!")

    except Exception as e:
        print(f"Error initializing authentication system: {e}")
        import traceback

        traceback.print_exc()
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    initialize_auth_system()
