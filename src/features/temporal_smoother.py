"""
TemporalSmoother - Feature Engineering Module
Applies sliding-window averaging to reduce per-frame landmark jitter.
"""

import numpy as np
from collections import deque
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TemporalSmoother:
    """
    Smooths feature vectors over a sliding window of frames.
    Reduces gesture instability caused by minor hand tremors.

    Mathematical operation:
        smoothed[t] = mean(feature_vectors[t-w : t])
    where w = smoothing_window size
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)
        logger.info(f"✅ TemporalSmoother: window={window_size} frames")

    def smooth(self, feature_vector: np.ndarray) -> np.ndarray:
        """
        Add vector to buffer and return smoothed average.

        Args:
            feature_vector: Current frame's feature vector (63,)

        Returns:
            smoothed: Averaged feature vector (63,)
        """
        self.buffer.append(feature_vector)
        return np.mean(np.stack(self.buffer), axis=0).astype(np.float32)

    def reset(self):
        self.buffer.clear()

    def is_full(self) -> bool:
        return len(self.buffer) == self.window_size
