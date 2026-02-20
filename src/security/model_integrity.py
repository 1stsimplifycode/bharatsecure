"""
ModelIntegrityChecker - Security Module
SHA-256 runtime verification of the AI model file.
Prevents model tampering attacks.

If hash mismatch detected:
  → System disables execution
  → Security alert triggered
"""

import hashlib
import os
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelIntegrityChecker:
    """
    Computes and verifies SHA-256 hashes of model files at runtime.
    Implements cryptographic integrity checking at zero infrastructure cost.
    """

    def __init__(self, model_path: str):
        self.model_path = model_path

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the model file."""
        sha256 = hashlib.sha256()
        with open(self.model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def compute_and_save_hash(self, hash_path: str) -> str:
        """Compute hash and save to file."""
        h = self.compute_hash()
        with open(hash_path, "w") as f:
            f.write(h)
        logger.info(f"✅ Model hash saved to {hash_path}: {h[:16]}...")
        return h

    def verify(self, hash_path: str) -> bool:
        """
        Verify model file against stored hash.

        Returns:
            True if model is unmodified
            False if tampering detected
        """
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found: {self.model_path}")
            return False

        if not os.path.exists(hash_path):
            logger.error(f"Hash file not found: {hash_path}")
            return False

        with open(hash_path, "r") as f:
            expected_hash = f.read().strip()

        actual_hash = self.compute_hash()

        if actual_hash == expected_hash:
            logger.debug(f"Model integrity OK: {actual_hash[:16]}...")
            return True
        else:
            logger.critical(
                f"⛔ MODEL INTEGRITY VIOLATION!\n"
                f"   Expected: {expected_hash[:32]}...\n"
                f"   Actual:   {actual_hash[:32]}...\n"
                f"   Model path: {self.model_path}"
            )
            return False
