"""
GestureModelTrainer - AI Module
Trains the lightweight MLP classifier and exports to TFLite + sklearn pickle.
"""

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.ai.models.mlp_model import build_mlp_sklearn, GESTURE_CLASSES
from src.security.model_integrity import ModelIntegrityChecker
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelTrainer:
    """
    Trains gesture classifier from collected feature vectors.
    Uses sklearn MLP + optional TFLite export.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_path = config["ai"]["model_path"]
        self.sklearn_path = self.model_path.replace(".tflite", ".pkl")
        self.data_dir = "data/gestures"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def load_dataset(self):
        """Load collected gesture feature vectors from disk."""
        X, y = [], []

        for gesture_id, gesture_name in GESTURE_CLASSES.items():
            gesture_dir = os.path.join(self.data_dir, gesture_name)
            if not os.path.exists(gesture_dir):
                logger.warning(f"No data for gesture: {gesture_name}. Run 'make collect-data' first.")
                continue

            for fname in os.listdir(gesture_dir):
                if fname.endswith(".npy"):
                    vec = np.load(os.path.join(gesture_dir, fname))
                    X.append(vec)
                    y.append(gesture_id)

        if len(X) == 0:
            logger.error("No training data found. Run 'make collect-data' first.")
            return None, None

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        logger.info(f"Loaded {len(X)} samples across {len(set(y))} classes.")
        return X, y

    def train(self):
        """Train and save the gesture classifier."""
        X, y = self.load_dataset()
        if X is None:
            return False

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Training: {len(X_train)} samples | Test: {len(X_test)} samples")

        model = build_mlp_sklearn()
        logger.info("Training MLP classifier...")
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean() * 100
        logger.info(f"Test Accuracy: {acc:.2f}%")
        logger.info("\n" + classification_report(
            y_test, y_pred,
            target_names=list(GESTURE_CLASSES.values())
        ))

        # Save sklearn model
        with open(self.sklearn_path, "wb") as f:
            pickle.dump({"model": model, "classes": GESTURE_CLASSES}, f)
        logger.info(f"✅ sklearn model saved: {self.sklearn_path}")

        # Export to TFLite
        self._export_tflite(model, X_train)

        # Compute and save SHA-256 hash
        hash_path = self.config["ai"]["model_hash_path"]
        checker = ModelIntegrityChecker(self.sklearn_path)
        checker.compute_and_save_hash(hash_path)

        return True

    def _export_tflite(self, sklearn_model, X_train: np.ndarray):
        """
        Convert sklearn MLP weights to TFLite model for Jetson inference.
        Uses a reconstruction approach via TensorFlow (one-time export).
        """
        try:
            import tensorflow as tf

            logger.info("Exporting model to TFLite...")

            # Reconstruct as Keras model using sklearn MLP weights
            weights = sklearn_model.coefs_
            biases = sklearn_model.intercepts_

            inputs = tf.keras.Input(shape=(63,))
            x = inputs
            for i, (w, b) in enumerate(zip(weights, biases)):
                x = tf.keras.layers.Dense(
                    w.shape[1],
                    activation="relu" if i < len(weights) - 1 else "softmax",
                    kernel_initializer=tf.constant_initializer(w),
                    bias_initializer=tf.constant_initializer(b),
                )(x)

            keras_model = tf.keras.Model(inputs, x)

            # Convert to TFLite with optimisations
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # 8-bit quantisation
            tflite_model = converter.convert()

            with open(self.model_path, "wb") as f:
                f.write(tflite_model)

            size_kb = os.path.getsize(self.model_path) / 1024
            logger.info(f"✅ TFLite model saved: {self.model_path} ({size_kb:.1f} KB)")

        except ImportError:
            logger.warning("TensorFlow not available. Skipping TFLite export. sklearn model will be used.")
        except Exception as e:
            logger.error(f"TFLite export failed: {e}. sklearn model will be used.")
