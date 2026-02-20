"""
LandmarkDetector - Vision Module
Extracts 21 hand landmarks using Google MediaPipe (free, on-device).
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LandmarkDetector:
    """
    Detects hand landmarks using MediaPipe Hands.
    Returns 21 landmarks (x, y, z) = 63-dimensional vector.

    MediaPipe is fully free and runs on-device — zero cloud cost.
    Optimised with model_complexity=0 for Jetson Nano performance.
    """

    def __init__(self, config: dict):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.get("max_num_hands", 1),
            min_detection_confidence=config.get("min_detection_confidence", 0.75),
            min_tracking_confidence=config.get("min_tracking_confidence", 0.60),
            model_complexity=config.get("model_complexity", 0),  # 0=Lite (fastest)
        )

        logger.info(
            f"✅ MediaPipe Hands ready. "
            f"complexity={config.get('model_complexity', 0)} "
            f"(0=Lite for Jetson Nano)"
        )

    def detect(self, rgb_frame: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        Detect hand landmarks in RGB frame.

        Args:
            rgb_frame: RGB image as numpy array

        Returns:
            landmarks: (21, 3) numpy array of normalised [x, y, z] or None
            annotated_frame: Frame with landmark overlay drawn
        """
        annotated = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return None, annotated

        # Take the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks on frame
        self.mp_drawing.draw_landmarks(
            annotated,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style(),
        )

        # Extract (x, y, z) for all 21 landmarks
        landmarks = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )  # shape: (21, 3)

        return landmarks, annotated

    def close(self):
        self.hands.close()
