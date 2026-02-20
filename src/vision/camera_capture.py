"""
CameraCapture - Vision Module
Supports Python (OpenCV) and C++ backend for Jetson Nano optimisation.
"""

import cv2
import subprocess
import numpy as np
from typing import Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class CameraCapture:
    """
    Real-time frame capture for NVIDIA Jetson Nano.
    Supports standard USB cameras and CSI cameras via GStreamer pipeline.
    """

    def __init__(self, config: dict, backend: str = "python"):
        self.config = config
        self.backend = backend
        self.cap = None
        self.width = config.get("width", 640)
        self.height = config.get("height", 480)
        self.fps = config.get("fps", 30)
        self.device_id = config.get("device_id", 0)
        self._prev_frame: Optional[np.ndarray] = None

    def _get_gstreamer_pipeline(self) -> str:
        """
        GStreamer pipeline for Jetson Nano CSI camera.
        Uses nvarguscamerasrc for hardware-accelerated capture.
        Zero licensing cost.
        """
        return (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink"
        )

    def open(self):
        """Open camera with appropriate backend."""
        # Try GStreamer CSI pipeline first (Jetson Nano CSI camera)
        gst_pipeline = self.config.get("pipeline")
        if gst_pipeline:
            logger.info("Attempting Jetson CSI camera via GStreamer...")
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                logger.info("✅ Jetson CSI camera opened via GStreamer.")
                return

        # Fallback: standard USB camera
        logger.info(f"Opening USB camera (device {self.device_id})...")
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Minimize buffer for real-time performance on Jetson
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"✅ Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single RGB frame."""
        if self.cap is None or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("Failed to read frame from camera.")
            return None

        # Convert BGR → RGB for MediaPipe
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_previous_frame(self) -> Optional[np.ndarray]:
        return self._prev_frame

    def release(self):
        if self.cap:
            self.cap.release()
            logger.info("Camera released.")
