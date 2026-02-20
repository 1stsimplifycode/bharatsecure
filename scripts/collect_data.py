#!/usr/bin/env python3
"""
Data Collection Tool - BharatSecure Touchless HCI
Collect labelled gesture feature vectors for model training.

Saves: data/gestures/<gesture_name>/<timestamp>.npy
"""

import cv2
import os
import time
import argparse
import yaml
import numpy as np
from datetime import datetime
from src.vision.camera_capture import CameraCapture
from src.vision.landmark_detector import LandmarkDetector
from src.features.feature_extractor import FeatureExtractor
from src.ai.models.mlp_model import GESTURE_CLASSES
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

GESTURE_NAMES = list(GESTURE_CLASSES.values())
SAMPLES_PER_GESTURE = 200
DATA_DIR = "data/gestures"


def collect_gesture_data(config: dict, gesture_name: str, num_samples: int):
    """Collect feature vectors for a single gesture class."""
    save_dir = os.path.join(DATA_DIR, gesture_name)
    os.makedirs(save_dir, exist_ok=True)

    camera = CameraCapture(config["camera"])
    detector = LandmarkDetector(config["mediapipe"])
    extractor = FeatureExtractor(config["feature_engineering"])

    camera.open()
    collected = 0

    print(f"\n📷 Ready to collect '{gesture_name.upper()}'")
    print(f"   Perform the gesture when you see GREEN border.")
    print(f"   Need: {num_samples} samples | Press 'q' to quit\n")

    # Countdown
    for i in range(3, 0, -1):
        print(f"   Starting in {i}...", end="\r")
        time.sleep(1)
    print("   GO! Show your gesture.              ")

    while collected < num_samples:
        frame = camera.read_frame()
        if frame is None:
            continue

        landmarks, annotated = detector.detect(frame)

        # Display
        bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        color = (0, 200, 0) if landmarks is not None else (0, 0, 200)
        cv2.rectangle(bgr, (0, 0), (w, h), color, 4)
        cv2.putText(bgr, f"Gesture: {gesture_name.upper()}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(bgr, f"Collected: {collected}/{num_samples}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        if landmarks is not None:
            fv = extractor.extract(landmarks)
            fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.npy"
            np.save(os.path.join(save_dir, fname), fv)
            collected += 1

            if collected % 20 == 0:
                logger.info(f"  {gesture_name}: {collected}/{num_samples} samples")

        cv2.imshow("BharatSecure Data Collection", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    logger.info(f"✅ Collected {collected} samples for '{gesture_name}'")
    return collected


def main():
    parser = argparse.ArgumentParser(description="BharatSecure Data Collection")
    parser.add_argument("--config", default="config/system_config.yaml")
    parser.add_argument("--gesture", choices=GESTURE_NAMES, default=None,
                        help="Collect specific gesture only")
    parser.add_argument("--samples", type=int, default=SAMPLES_PER_GESTURE,
                        help="Samples per gesture")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    gestures_to_collect = [args.gesture] if args.gesture else GESTURE_NAMES
    total = 0

    print("\n" + "=" * 55)
    print("   BharatSecure — Gesture Data Collection")
    print("=" * 55)
    print(f"   Gestures: {', '.join(gestures_to_collect)}")
    print(f"   Samples per gesture: {args.samples}")
    print("=" * 55 + "\n")

    for gesture in gestures_to_collect:
        n = collect_gesture_data(config, gesture, args.samples)
        total += n
        print(f"\n   Press ENTER to continue to next gesture, or Ctrl+C to stop...")
        try:
            input()
        except KeyboardInterrupt:
            break

    print(f"\n✅ Data collection complete. Total: {total} samples.")
    print(f"   Now run: make train")


if __name__ == "__main__":
    main()
