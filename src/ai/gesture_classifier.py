"""
GestureClassifier - AI Classification Engine
Runs inference using TFLite (Jetson optimised) or sklearn fallback.
Model: Lightweight MLP, <5MB, <150ms latency.
"""

import os
import pickle
import numpy as np
from typing import Tuple, Optional
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GestureClassifier:
    """
    Dual-backend gesture classifier:
    1. TFLite (primary) - Jetson Nano optimised, uses NNAPI/GPU delegate
    2. scikit-learn (fallback) - for development/testing

    Returns (gesture_id, confidence, all_probabilities)
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_path = config.get("model_path", "src/ai/models/gesture_model.tflite")
        self.sklearn_path = self.model_path.replace(".tflite", ".pkl")
        self.num_classes = config.get("num_gesture_classes", 5)
        self.inference_threads = config.get("inference_threads", 2)

        self.tflite_interpreter = None
        self.sklearn_model = None
        self.backend = None

        self._load_model()

    def _load_model(self):
        """Try TFLite first, then sklearn pickle fallback."""
        # Try TFLite
        if os.path.exists(self.model_path):
            try:
                self._load_tflite()
                return
            except Exception as e:
                logger.warning(f"TFLite load failed: {e}. Trying sklearn fallback.")

        # Try sklearn pickle
        if os.path.exists(self.sklearn_path):
            try:
                self._load_sklearn()
                return
            except Exception as e:
                logger.warning(f"sklearn load failed: {e}")

        logger.warning(
            "⚠️  No trained model found. Run 'make train' to generate one. "
            "Using random predictions (demo mode)."
        )
        self.backend = "demo"

    def _load_tflite(self):
        """Load TFLite model with Jetson Nano optimisations."""
        try:
            import tflite_runtime.interpreter as tflite
            logger.info("Using tflite_runtime (Jetson Nano optimised)")
        except ImportError:
            import tensorflow.lite as tflite
            logger.info("Using tensorflow.lite")

        self.tflite_interpreter = tflite.Interpreter(
            model_path=self.model_path,
            num_threads=self.inference_threads,
        )

        # Try NNAPI delegate (Jetson Nano hardware acceleration)
        try:
            nnapi_delegate = tflite.experimental.load_delegate("libnnapi_util.so")
            self.tflite_interpreter = tflite.Interpreter(
                model_path=self.model_path,
                experimental_delegates=[nnapi_delegate],
                num_threads=self.inference_threads,
            )
            logger.info("✅ NNAPI delegate enabled (Jetson hardware acceleration)")
        except Exception:
            logger.info("NNAPI delegate not available — using CPU (still fast)")

        self.tflite_interpreter.allocate_tensors()
        self.input_details = self.tflite_interpreter.get_input_details()
        self.output_details = self.tflite_interpreter.get_output_details()
        self.backend = "tflite"
        logger.info(f"✅ TFLite model loaded: {self.model_path}")

    def _load_sklearn(self):
        """Load scikit-learn MLP model."""
        with open(self.sklearn_path, "rb") as f:
            data = pickle.load(f)
        self.sklearn_model = data["model"]
        self.backend = "sklearn"
        logger.info(f"✅ sklearn model loaded: {self.sklearn_path}")

    def predict(self, feature_vector: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Run gesture classification.

        Args:
            feature_vector: 63-dim normalised feature vector

        Returns:
            gesture_id: int class index (0–4)
            confidence: float softmax probability of predicted class
            all_probs: numpy array of all class probabilities
        """
        if self.backend == "tflite":
            return self._predict_tflite(feature_vector)
        elif self.backend == "sklearn":
            return self._predict_sklearn(feature_vector)
        else:
            return self._predict_demo()

    def _predict_tflite(self, feature_vector: np.ndarray) -> Tuple[int, float, np.ndarray]:
        input_data = feature_vector.reshape(1, -1).astype(np.float32)
        self.tflite_interpreter.set_tensor(
            self.input_details[0]["index"], input_data
        )
        self.tflite_interpreter.invoke()
        output = self.tflite_interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0]

        # Softmax if raw logits
        probs = self._softmax(output)
        gesture_id = int(np.argmax(probs))
        confidence = float(probs[gesture_id])
        return gesture_id, confidence, probs

    def _predict_sklearn(self, feature_vector: np.ndarray) -> Tuple[int, float, np.ndarray]:
        x = feature_vector.reshape(1, -1)
        probs = self.sklearn_model.predict_proba(x)[0]
        gesture_id = int(np.argmax(probs))
        confidence = float(probs[gesture_id])
        return gesture_id, confidence, probs

    def _predict_demo(self) -> Tuple[int, float, np.ndarray]:
        """Demo mode: random predictions for testing without a trained model."""
        probs = np.random.dirichlet(np.ones(self.num_classes))
        gesture_id = int(np.argmax(probs))
        return gesture_id, float(probs[gesture_id]), probs

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
