"""
AnomalyDetector - Security Module
Detects adversarial gloves and injected hand patterns using Z-score analysis.

Z = (x − μ) / σ

Landmarks outside acceptable deviation range are flagged as anomalies.
"""

import numpy as np
from collections import deque
from typing import Tuple, List
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AnomalyDetector:
    """
    Statistical anomaly detection on feature vectors.

    Maintains a rolling baseline of "normal" landmark distributions
    and flags vectors that deviate significantly (Z-score > threshold).

    Detects:
    - Adversarial glove patterns
    - Injected/synthetic hand data
    - Unusual landmark configurations
    """

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.z_threshold = config.get("z_score_threshold", 3.0)
        self.history_size = config.get("history_size", 100)
        self.min_samples = 20  # Min samples before detection activates

        # Rolling statistics
        self._history: deque = deque(maxlen=self.history_size)

        logger.info(
            f"✅ AnomalyDetector: z_threshold={self.z_threshold} "
            f"history={self.history_size} enabled={self.enabled}"
        )

    def check(self, feature_vector: np.ndarray) -> Tuple[bool, List[float]]:
        """
        Check if feature vector is anomalous.

        Args:
            feature_vector: 63-dim feature vector

        Returns:
            is_anomaly: True if anomaly detected
            z_scores: Z-scores for each feature dimension
        """
        if not self.enabled:
            return False, []

        # Add to history for baseline building
        self._history.append(feature_vector.copy())

        # Not enough baseline data yet
        if len(self._history) < self.min_samples:
            return False, []

        # Compute rolling mean and std
        history_arr = np.array(self._history[:-1])  # Exclude current sample
        mu = np.mean(history_arr, axis=0)
        sigma = np.std(history_arr, axis=0)

        # Avoid division by zero
        sigma = np.where(sigma < 1e-8, 1e-8, sigma)

        # Z-score for current vector
        z_scores = np.abs((feature_vector - mu) / sigma)
        z_scores_list = z_scores.tolist()

        # Flag as anomaly if any dimension exceeds threshold significantly
        # Use 90th percentile of z-scores to avoid single-feature false positives
        p90_z = float(np.percentile(z_scores, 90))
        is_anomaly = p90_z > self.z_threshold

        return is_anomaly, z_scores_list

    def reset(self):
        """Clear history (e.g., on user change)."""
        self._history.clear()

    def get_baseline_stats(self) -> dict:
        """Return current baseline statistics for dashboard display."""
        if len(self._history) < self.min_samples:
            return {"status": "building_baseline", "samples": len(self._history)}

        history_arr = np.array(self._history)
        return {
            "status": "active",
            "samples": len(self._history),
            "mean_range": [float(np.min(np.mean(history_arr, axis=0))),
                           float(np.max(np.mean(history_arr, axis=0)))],
            "std_range": [float(np.min(np.std(history_arr, axis=0))),
                          float(np.max(np.std(history_arr, axis=0)))],
        }
