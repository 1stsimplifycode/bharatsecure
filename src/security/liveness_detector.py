"""
LivenessDetector - Security Module
Detects replay attacks by measuring temporal motion between frames.

Mathematical representation:
    Δ = ||F_t − F_{t-1}||

If Δ < threshold → potential static image / video replay attack
"""

import numpy as np
from collections import deque
from typing import Tuple, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LivenessDetector:
    """
    Multi-layer liveness detection to block:
    - Static image replays (Δ ≈ 0)
    - Slow video replays (Δ < threshold over multiple frames)

    Uses pixel-level frame differencing on greyscale images.
    """

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.motion_threshold = config.get("motion_threshold", 0.01)
        self.window = config.get("window", 10)

        self._prev_frame_gray: Optional[np.ndarray] = None
        self._motion_history: deque = deque(maxlen=self.window)
        self._frame_count = 0

        logger.info(
            f"✅ LivenessDetector: threshold={self.motion_threshold} "
            f"window={self.window} enabled={self.enabled}"
        )

    def check(self, rgb_frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check if the frame shows real-time liveness.

        Args:
            rgb_frame: Current RGB frame

        Returns:
            is_live: True if liveness detected
            delta: Motion delta value
        """
        if not self.enabled:
            return True, 1.0

        self._frame_count += 1

        # Convert to greyscale and normalise
        import cv2
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        if self._prev_frame_gray is None:
            self._prev_frame_gray = gray
            return True, 1.0  # First frame — pass

        # Compute normalised frame difference (Frobenius norm)
        diff = gray - self._prev_frame_gray
        delta = float(np.sqrt(np.mean(diff ** 2)))  # RMSE

        self._motion_history.append(delta)
        self._prev_frame_gray = gray

        # Need at least 3 frames to make a decision
        if len(self._motion_history) < 3:
            return True, delta

        mean_motion = float(np.mean(self._motion_history))

        # Liveness check: reject if motion is consistently near-zero
        is_live = mean_motion >= self.motion_threshold

        return is_live, delta

    def reset(self):
        """Reset detector state (e.g., between sessions)."""
        self._prev_frame_gray = None
        self._motion_history.clear()
        self._frame_count = 0
