"""
FeatureExtractor - Feature Engineering Module
Converts 21 hand landmarks into a normalised 63-dimensional feature vector.
"""

import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FeatureExtractor:
    """
    Transforms raw MediaPipe landmarks into ML-ready feature vectors.

    Process:
    1. Flatten (21, 3) → 63-dim vector
    2. Normalise relative to wrist landmark (index 0)
    3. Scale to unit range

    This ensures translation-invariant representation — position in frame
    does not affect gesture classification.
    """

    def __init__(self, config: dict):
        self.config = config
        self.num_landmarks = config.get("num_landmarks", 21)
        self.dimensions = config.get("dimensions", 3)
        self.normalize = config.get("normalize", True)
        self.feature_dim = self.num_landmarks * self.dimensions  # 63

        logger.info(f"✅ FeatureExtractor: {self.feature_dim}-dim vectors, normalize={self.normalize}")

    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from landmarks.

        Args:
            landmarks: numpy array of shape (21, 3)

        Returns:
            feature_vector: numpy array of shape (63,) normalised
        """
        if landmarks is None or landmarks.shape != (self.num_landmarks, self.dimensions):
            return np.zeros(self.feature_dim, dtype=np.float32)

        features = landmarks.copy()

        if self.normalize:
            # Translate: make wrist (landmark 0) the origin
            wrist = features[0].copy()
            features -= wrist

            # Scale: normalise by max absolute value for scale invariance
            max_val = np.max(np.abs(features))
            if max_val > 1e-6:
                features /= max_val

        # Flatten to 63-dim
        feature_vector = features.flatten().astype(np.float32)
        return feature_vector

    def get_feature_dim(self) -> int:
        return self.feature_dim
