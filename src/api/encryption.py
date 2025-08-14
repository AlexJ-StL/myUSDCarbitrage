"""Request/response encryption utilities for sensitive data."""

import base64
import json
import os
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi import HTTPException, Request, Response, status
from pydantic import BaseModel


class EncryptionConfig:
    """Configuration for encryption settings."""

    def __init__(self):
        # Get encryption key from environment or generate one
        self.encryption_key = os.getenv("ENCRYPTION_KEY")
        if not self.encryption_key:
            # Generate a new key if none exists
            self.encryption_key = Fernet.generate_key().decode()
            print(
                "Warning: Generated new encryption key. Set ENCRYPTION_KEY environment variable for production."
            )

        # Initialize Fernet cipher
        if isinstance(self.encryption_key, str):
            self.encryption_key = self.encryption_key.encode()

        self.fernet = Fernet(self.encryption_key)

        # RSA key pair for asymmetric encryption (optional)
        self.private_key = None
        self.public_key = None
        self._generate_rsa_keys()

    def _generate_rsa_keys(self):
        """Generate RSA key pair for asymmetric encryption."""
        try:
            # Try to load existing keys
            private_key_path = os.getenv("RSA_PRIVATE_KEY_PATH", "private_key.pem")
            public_key_path = os.getenv("RSA_PUBLIC_KEY_PATH", "public_key.pem")

            if os.path.exists(private_key_path) and os.path.exists(public_key_path):
                with open(private_key_path, "rb") as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(), password=None
                    )
                with open(public_key_path, "rb") as f:
                    self.public_key = serialization.load_pem_public_key(f.read())
            else:
                # Generate new keys
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537, key_size=2048
                )
                self.public_key = self.private_key.public_key()

                # Save keys if in development
                if os.getenv("ENVIRONMENT") == "development":
                    self._save_rsa_keys(private_key_path, public_key_path)

        except Exception as e:
            print(f"Warning: Could not initialize RSA keys: {e}")

    def _save_rsa_keys(self, private_path: str, public_path: str):
        """Save RSA keys to files."""
        try:
            # Save private key
            private_pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            with open(private_path, "wb") as f:
                f.write(private_pem)

            # Save public key
            public_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            with open(public_path, "wb") as f:
                f.write(public_pem)

        except Exception as e:
            print(f"Warning: Could not save RSA keys: {e}")


# Global encryption config
encryption_config = EncryptionConfig()


class EncryptedData(BaseModel):
    """Model for encrypted data transmission."""

    encrypted_data: str
    encryption_method: str = "fernet"
    timestamp: Optional[str] = None


class DataEncryption:
    """Service for encrypting and decrypting sensitive data."""

    def __init__(self, config: EncryptionConfig = None):
        self.config = config or encryption_config

    def encrypt_symmetric(self, data: Any) -> str:
        """Encrypt data using symmetric encryption (Fernet)."""
        try:
            # Convert data to JSON string
            json_data = json.dumps(data, default=str)

            # Encrypt the data
            encrypted_data = self.config.fernet.encrypt(json_data.encode())

            # Return base64 encoded string
            return base64.b64encode(encrypted_data).decode()

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Encryption failed: {str(e)}",
            )

    def decrypt_symmetric(self, encrypted_data: str) -> Any:
        """Decrypt data using symmetric encryption (Fernet)."""
        try:
            # Decode base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode())

            # Decrypt the data
            decrypted_data = self.config.fernet.decrypt(encrypted_bytes)

            # Parse JSON
            return json.loads(decrypted_data.decode())

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Decryption failed: {str(e)}",
            )

    def encrypt_asymmetric(self, data: Any) -> str:
        """Encrypt data using asymmetric encryption (RSA)."""
        if not self.config.public_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RSA public key not available",
            )

        try:
            # Convert data to JSON string
            json_data = json.dumps(data, default=str)
            data_bytes = json_data.encode()

            # RSA can only encrypt small amounts of data
            # For larger data, we'll use hybrid encryption
            if len(data_bytes) > 190:  # RSA 2048 can encrypt ~245 bytes, leaving margin
                # Generate a random symmetric key
                symmetric_key = Fernet.generate_key()
                fernet = Fernet(symmetric_key)

                # Encrypt data with symmetric key
                encrypted_data = fernet.encrypt(data_bytes)

                # Encrypt symmetric key with RSA
                encrypted_key = self.config.public_key.encrypt(
                    symmetric_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )

                # Combine encrypted key and data
                combined = (
                    base64.b64encode(encrypted_key).decode()
                    + ":"
                    + base64.b64encode(encrypted_data).decode()
                )
                return combined
            else:
                # Direct RSA encryption for small data
                encrypted_data = self.config.public_key.encrypt(
                    data_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )
                return base64.b64encode(encrypted_data).decode()

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Asymmetric encryption failed: {str(e)}",
            )

    def decrypt_asymmetric(self, encrypted_data: str) -> Any:
        """Decrypt data using asymmetric encryption (RSA)."""
        if not self.config.private_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="RSA private key not available",
            )

        try:
            if ":" in encrypted_data:
                # Hybrid encryption - split key and data
                encrypted_key_b64, encrypted_data_b64 = encrypted_data.split(":", 1)

                # Decrypt symmetric key
                encrypted_key = base64.b64decode(encrypted_key_b64.encode())
                symmetric_key = self.config.private_key.decrypt(
                    encrypted_key,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )

                # Decrypt data with symmetric key
                fernet = Fernet(symmetric_key)
                encrypted_data_bytes = base64.b64decode(encrypted_data_b64.encode())
                decrypted_data = fernet.decrypt(encrypted_data_bytes)
            else:
                # Direct RSA decryption
                encrypted_bytes = base64.b64decode(encrypted_data.encode())
                decrypted_data = self.config.private_key.decrypt(
                    encrypted_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                )

            # Parse JSON
            return json.loads(decrypted_data.decode())

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Asymmetric decryption failed: {str(e)}",
            )


# Global encryption service
data_encryption = DataEncryption()


def encrypt_sensitive_fields(
    data: Dict[str, Any], sensitive_fields: list
) -> Dict[str, Any]:
    """Encrypt specific fields in a dictionary."""
    encrypted_data = data.copy()

    for field in sensitive_fields:
        if field in encrypted_data:
            encrypted_data[field] = data_encryption.encrypt_symmetric(
                encrypted_data[field]
            )

    return encrypted_data


def decrypt_sensitive_fields(
    data: Dict[str, Any], sensitive_fields: list
) -> Dict[str, Any]:
    """Decrypt specific fields in a dictionary."""
    decrypted_data = data.copy()

    for field in sensitive_fields:
        if field in decrypted_data:
            try:
                decrypted_data[field] = data_encryption.decrypt_symmetric(
                    decrypted_data[field]
                )
            except:
                # If decryption fails, assume data is not encrypted
                pass

    return decrypted_data


class EncryptionMiddleware:
    """Middleware for automatic request/response encryption."""

    def __init__(self, app, sensitive_endpoints: list = None):
        self.app = app
        self.sensitive_endpoints = sensitive_endpoints or [
            "/auth/login",
            "/auth/register",
            "/admin/users",
            "/api-keys",
        ]
        self.encryption_service = DataEncryption()

    async def __call__(self, scope, receive, send):
        """Process request with encryption/decryption."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from starlette.requests import Request

        request = Request(scope, receive)

        # Check if this endpoint requires encryption
        requires_encryption = any(
            request.url.path.startswith(endpoint)
            for endpoint in self.sensitive_endpoints
        )

        if not requires_encryption:
            await self.app(scope, receive, send)
            return

        # For encrypted endpoints, we'll pass through for now
        # Full encryption implementation would require more complex request/response handling
        await self.app(scope, receive, send)


# Utility functions for common encryption patterns
def encrypt_password_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Encrypt password-related data."""
    return encrypt_sensitive_fields(data, ["password", "new_password", "old_password"])


def encrypt_api_key_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Encrypt API key data."""
    return encrypt_sensitive_fields(data, ["api_key", "secret_key", "private_key"])


def encrypt_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Encrypt financial data."""
    return encrypt_sensitive_fields(
        data,
        [
            "balance",
            "profit",
            "loss",
            "pnl",
            "equity",
            "account_balance",
            "trading_balance",
        ],
    )


# Decorator for encrypting endpoint responses
def encrypt_response(sensitive_fields: list = None):
    """Decorator to encrypt response data."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            if sensitive_fields and isinstance(result, dict):
                return encrypt_sensitive_fields(result, sensitive_fields)

            return result

        return wrapper

    return decorator
