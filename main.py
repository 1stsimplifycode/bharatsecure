#!/usr/bin/env python3
"""
BharatSecure Touchless HCI - Main Entry Point
Zero-Trust, Privacy-Preserving Gesture-Based Media Control
Platform: NVIDIA Jetson Nano
Cost: $0 (fully open-source, edge-only)
"""

import argparse
import sys
import os
import signal
import threading
import yaml
import time

from src.utils.logger import setup_logger
from src.vision.camera_capture import CameraCapture
from src.vision.landmark_detector import LandmarkDetector
from src.features.feature_extractor import FeatureExtractor
from src.features.temporal_smoother import TemporalSmoother
from src.ai.gesture_classifier import GestureClassifier
from src.security.liveness_detector import LivenessDetector
from src.security.anomaly_detector import AnomalyDetector
from src.security.zero_trust import ZeroTrustGate
from src.security.model_integrity import ModelIntegrityChecker
from src.media_control.media_controller import MediaController
from src.utils.logger import SecurityEventLogger

logger = setup_logger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(
        description="BharatSecure Touchless HCI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config/system_config.yaml",
                        help="Path to system config YAML")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    parser.add_argument("--camera-backend", choices=["python", "cpp"],
                        default="python", help="Camera capture backend")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Disable security dashboard")
    return parser.parse_args()


class BharatSecureSystem:
    """
    Main orchestrator for the BharatSecure Touchless HCI pipeline.

    Pipeline:
    Camera → Landmark Detection → Feature Engineering → Security Validation
           → AI Inference → Zero-Trust Gate → Secure Command Execution
    """

    def __init__(self, config: dict, debug: bool = False, camera_backend: str = "python"):
        self.config = config
        self.debug = debug
        self.camera_backend = camera_backend
        self.running = False

        logger.info("=" * 60)
        logger.info("  BharatSecure Touchless HCI - Initialising")
        logger.info(f"  Platform: {config['system']['platform'].upper()}")
        logger.info(f"  Version : {config['system']['version']}")
        logger.info("=" * 60)

        # 1. Model integrity check BEFORE loading anything else
        self._verify_model_integrity()

        # 2. Initialise pipeline components
        self.camera = CameraCapture(config["camera"], backend=camera_backend)
        self.landmark_detector = LandmarkDetector(config["mediapipe"])
        self.feature_extractor = FeatureExtractor(config["feature_engineering"])
        self.temporal_smoother = TemporalSmoother(config["feature_engineering"]["smoothing_window"])
        self.classifier = GestureClassifier(config["ai"])

        # 3. Security layers
        self.liveness_detector = LivenessDetector(config["security"]["liveness"])
        self.anomaly_detector = AnomalyDetector(config["security"]["anomaly_detection"])
        self.zero_trust_gate = ZeroTrustGate(config["security"])

        # 4. Media controller & event logger
        self.media_controller = MediaController(config["gestures"])
        self.event_logger = SecurityEventLogger(config["dashboard"]["log_db_path"])

        # 5. Frame stats
        self.frame_count = 0
        self.fps_timer = time.time()
        self.current_fps = 0.0

        logger.info("✅ BharatSecure system initialised successfully.")

    def _verify_model_integrity(self):
        """SHA-256 integrity check before inference begins."""
        model_path = self.config["ai"]["model_path"]
        hash_path = self.config["ai"]["model_hash_path"]

        if not os.path.exists(model_path):
            logger.warning(f"Model not found at {model_path}. Skipping integrity check.")
            logger.warning("Run 'make train' to generate a model first.")
            return

        checker = ModelIntegrityChecker(model_path)
        if os.path.exists(hash_path):
            valid = checker.verify(hash_path)
            if not valid:
                logger.critical("⛔ MODEL INTEGRITY CHECK FAILED! Possible tampering detected.")
                logger.critical("⛔ System execution DISABLED. Contact administrator.")
                sys.exit(1)
            logger.info("✅ Model integrity verified (SHA-256 OK).")
        else:
            logger.warning("No hash file found. Computing and saving model hash...")
            checker.compute_and_save_hash(hash_path)

    def run(self):
        """Main inference loop."""
        self.running = True
        logger.info("🚀 Starting gesture recognition loop...")

        try:
            self.camera.open()

            while self.running:
                frame = self.camera.read_frame()
                if frame is None:
                    continue

                self.frame_count += 1
                self._update_fps()

                # ── Step 1: Liveness Detection ──────────────────────────────
                is_live, liveness_delta = self.liveness_detector.check(frame)
                if not is_live:
                    self.event_logger.log("LIVENESS_FAIL",
                                          f"Potential replay detected. Δ={liveness_delta:.4f}")
                    if self.debug:
                        logger.debug(f"[LIVENESS] FAIL Δ={liveness_delta:.4f}")
                    self._show_frame(frame, "BLOCKED: Liveness", None, None)
                    continue

                # ── Step 2: Landmark Detection ───────────────────────────────
                landmarks, annotated_frame = self.landmark_detector.detect(frame)
                if landmarks is None:
                    self._show_frame(annotated_frame, "No Hand", None, None)
                    continue

                # ── Step 3: Feature Extraction ───────────────────────────────
                feature_vector = self.feature_extractor.extract(landmarks)
                smoothed_vector = self.temporal_smoother.smooth(feature_vector)

                # ── Step 4: Anomaly Detection ────────────────────────────────
                is_anomaly, z_scores = self.anomaly_detector.check(smoothed_vector)
                if is_anomaly:
                    self.event_logger.log("ANOMALY_DETECTED",
                                          f"Z-score anomaly in landmarks. max_z={max(z_scores):.2f}")
                    if self.debug:
                        logger.debug(f"[ANOMALY] Flagged. max_z={max(z_scores):.2f}")
                    self._show_frame(annotated_frame, "ANOMALY", None, None)
                    continue

                # ── Step 5: AI Inference ─────────────────────────────────────
                gesture_id, confidence, all_probs = self.classifier.predict(smoothed_vector)
                gesture_name = self.config["gestures"]["classes"].get(gesture_id, "unknown")

                if self.debug:
                    logger.debug(f"[AI] Gesture={gesture_name} Conf={confidence:.3f} FPS={self.current_fps:.1f}")

                # ── Step 6: Zero-Trust Gate ──────────────────────────────────
                approved, reason = self.zero_trust_gate.evaluate(
                    gesture_id, confidence, is_anomaly=False
                )

                if approved:
                    # ── Step 7: Secure Command Execution ────────────────────
                    self.media_controller.execute(gesture_name)
                    self.event_logger.log("COMMAND_EXECUTED",
                                          f"Gesture={gesture_name} Conf={confidence:.3f}")
                    self._show_frame(annotated_frame, f"✅ {gesture_name.upper()}", confidence, all_probs)
                else:
                    if self.debug:
                        logger.debug(f"[ZERO_TRUST] DENIED: {reason}")
                    self._show_frame(annotated_frame, f"🔒 {reason}", confidence, all_probs)

        except KeyboardInterrupt:
            logger.info("👋 Shutdown requested by user.")
        finally:
            self.shutdown()

    def _update_fps(self):
        elapsed = time.time() - self.fps_timer
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_timer = time.time()

    def _show_frame(self, frame, status: str, confidence, probs):
        """Render debug overlay on frame if display available."""
        try:
            import cv2
            if frame is None:
                return

            overlay = frame.copy()
            h, w = overlay.shape[:2]

            # Status bar
            color = (0, 200, 0) if "✅" in status else (0, 0, 220)
            cv2.rectangle(overlay, (0, 0), (w, 40), color, -1)
            cv2.putText(overlay, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # FPS
            cv2.putText(overlay, f"FPS: {self.current_fps:.1f}", (w - 120, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Confidence bar
            if confidence is not None:
                bar_w = int(w * confidence)
                cv2.rectangle(overlay, (0, h - 10), (bar_w, h), (0, 255, 100), -1)
                cv2.putText(overlay, f"Conf: {confidence:.2f}", (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("BharatSecure Touchless HCI", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False
        except Exception:
            pass  # Headless mode (no display) - continue silently

    def shutdown(self):
        logger.info("🔒 Shutting down BharatSecure system...")
        self.running = False
        self.camera.release()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("✅ Shutdown complete.")


def start_dashboard(config: dict):
    """Launch Flask dashboard in background thread."""
    from src.dashboard.app import create_app
    app = create_app(config)
    app.run(
        host=config["dashboard"]["host"],
        port=config["dashboard"]["port"],
        debug=False,
        use_reloader=False,
    )


def main():
    args = parse_args()
    config = load_config(args.config)

    if args.debug:
        config["system"]["debug"] = True

    os.makedirs("logs", exist_ok=True)

    # Launch dashboard thread
    if config["dashboard"]["enabled"] and not args.no_dashboard:
        dash_thread = threading.Thread(target=start_dashboard, args=(config,), daemon=True)
        dash_thread.start()
        logger.info(f"📊 Dashboard: http://localhost:{config['dashboard']['port']}")

    # Start main system
    system = BharatSecureSystem(
        config=config,
        debug=args.debug,
        camera_backend=args.camera_backend,
    )

    # Graceful shutdown on SIGTERM
    def handle_signal(sig, frame):
        system.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    system.run()


if __name__ == "__main__":
    main()
