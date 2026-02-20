"""
ZeroTrustGate - Security Module
Multi-layer command validation. Command executes ONLY if ALL gates pass.

Gates:
1. Confidence > 0.90
2. Gesture stable across N consecutive frames
3. No anomaly detected (passed in from AnomalyDetector)
4. Rate limit satisfied (max 2 commands/second)
"""

import time
from collections import deque
from typing import Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ZeroTrustGate:
    """
    Implements zero-trust command validation for gesture execution.

    Philosophy: Never trust any single signal. Every command must pass
    all security gates independently before execution.
    """

    def __init__(self, config: dict):
        self.confidence_threshold = config.get("confidence_threshold", 0.90)
        self.stability_frames = config.get("stability_frames", 5)
        self.max_commands_per_second = config["rate_limiting"]["max_commands_per_second"]
        self.cooldown_ms = config["rate_limiting"]["cooldown_ms"]

        # Stability tracking
        self._gesture_history: deque = deque(maxlen=self.stability_frames)

        # Rate limiting
        self._last_command_time = 0.0
        self._command_timestamps: deque = deque(maxlen=10)

        logger.info(
            f"✅ ZeroTrustGate: conf>={self.confidence_threshold} "
            f"stability={self.stability_frames}f "
            f"ratelimit={self.max_commands_per_second}/s"
        )

    def evaluate(self, gesture_id: int, confidence: float, is_anomaly: bool) -> Tuple[bool, str]:
        """
        Run all zero-trust gates.

        Args:
            gesture_id: Predicted gesture class ID
            confidence: Model confidence (0–1)
            is_anomaly: Whether anomaly was flagged

        Returns:
            approved: True if command is approved for execution
            reason: Denial reason string (for logging/display)
        """

        # Gate 1: Anomaly check (highest priority)
        if is_anomaly:
            return False, "ANOMALY"

        # Gate 2: Confidence threshold
        if confidence < self.confidence_threshold:
            return False, f"LOW_CONF({confidence:.2f})"

        # Gate 3: Gesture stability (must be consistent across N frames)
        self._gesture_history.append(gesture_id)
        if len(self._gesture_history) < self.stability_frames:
            return False, "WARMING_UP"

        if not self._is_stable():
            return False, "UNSTABLE"

        # Gate 4: Rate limiting
        now = time.time()
        elapsed_since_last = now - self._last_command_time
        if elapsed_since_last < (self.cooldown_ms / 1000.0):
            return False, "RATE_LIMITED"

        # Remove old timestamps
        self._command_timestamps = deque(
            [t for t in self._command_timestamps if now - t <= 1.0],
            maxlen=10
        )
        if len(self._command_timestamps) >= self.max_commands_per_second:
            return False, "RATE_LIMITED"

        # ✅ All gates passed
        self._last_command_time = now
        self._command_timestamps.append(now)
        self._gesture_history.clear()  # Reset stability buffer after execution

        return True, "OK"

    def _is_stable(self) -> bool:
        """Check if all gestures in history buffer are the same class."""
        if len(self._gesture_history) < self.stability_frames:
            return False
        first = self._gesture_history[0]
        return all(g == first for g in self._gesture_history)

    def reset(self):
        self._gesture_history.clear()
        self._command_timestamps.clear()
        self._last_command_time = 0.0
