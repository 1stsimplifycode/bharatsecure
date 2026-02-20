"""
Crypto Utilities - BharatSecure
Shared cryptographic helpers for TLS, hashing, and secure comms.
Cost: $0 — uses stdlib + cryptography library (open-source).
"""

import hashlib
import hmac
import os
import ssl
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def hmac_sha256(key: bytes, data: bytes) -> str:
    """Compute HMAC-SHA256 for message authentication."""
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def create_tls_context(cert_path: str, key_path: str) -> Optional[ssl.SSLContext]:
    """Create a TLS 1.3 context for secure federated communication."""
    if not os.path.exists(cert_path) or not os.path.exists(key_path):
        logger.warning("TLS certs not found. Run 'make generate-certs'.")
        return None

    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3
        ctx.load_cert_chain(cert_path, key_path)
        ctx.verify_mode = ssl.CERT_NONE  # Self-signed in dev
        logger.info("✅ TLS 1.3 context created.")
        return ctx
    except Exception as e:
        logger.error(f"TLS context creation failed: {e}")
        return None


def generate_session_token(length: int = 32) -> str:
    """Generate a cryptographically secure session token."""
    return os.urandom(length).hex()
