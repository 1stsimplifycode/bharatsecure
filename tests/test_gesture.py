"""
Tests - Gesture Pipeline
Unit tests for vision, feature engineering, and AI classification.
"""

import pytest
import numpy as np


class TestFeatureExtractor:

    def setup_method(self):
        from src.features.feature_extractor import FeatureExtractor
        self.extractor = FeatureExtractor({
            "num_landmarks": 21,
            "dimensions": 3,
            "normalize": True,
        })

    def test_output_shape(self):
        landmarks = np.random.rand(21, 3).astype(np.float32)
        fv = self.extractor.extract(landmarks)
        assert fv.shape == (63,)

    def test_normalised_range(self):
        """Normalised features should be in range [-1, 1]."""
        landmarks = np.random.rand(21, 3).astype(np.float32) * 10
        fv = self.extractor.extract(landmarks)
        assert np.all(fv >= -1.0 - 1e-5)
        assert np.all(fv <= 1.0 + 1e-5)

    def test_wrist_at_origin(self):
        """After normalisation, wrist (first landmark) should be at origin."""
        landmarks = np.random.rand(21, 3).astype(np.float32)
        fv = self.extractor.extract(landmarks)
        wrist_xyz = fv[:3]
        assert np.allclose(wrist_xyz, 0.0, atol=1e-5)

    def test_translation_invariant(self):
        """Offset hand position should yield same features."""
        landmarks = np.random.rand(21, 3).astype(np.float32)
        offset = np.array([5.0, 3.0, 1.0])
        landmarks_offset = landmarks + offset
        fv1 = self.extractor.extract(landmarks)
        fv2 = self.extractor.extract(landmarks_offset)
        np.testing.assert_allclose(fv1, fv2, atol=1e-4)

    def test_invalid_landmarks(self):
        """None/wrong-shape input returns zero vector."""
        fv = self.extractor.extract(None)
        assert fv.shape == (63,)
        assert np.all(fv == 0.0)


class TestTemporalSmoother:

    def setup_method(self):
        from src.features.temporal_smoother import TemporalSmoother
        self.smoother = TemporalSmoother(window_size=5)

    def test_single_frame_passthrough(self):
        """First frame should pass through unchanged."""
        vec = np.ones(63, dtype=np.float32)
        smoothed = self.smoother.smooth(vec)
        assert smoothed.shape == (63,)
        np.testing.assert_allclose(smoothed, vec)

    def test_averaging(self):
        """Smoother should average correctly."""
        for _ in range(5):
            self.smoother.smooth(np.zeros(63, dtype=np.float32))

        self.smoother.reset()
        for _ in range(4):
            self.smoother.smooth(np.zeros(63, dtype=np.float32))
        result = self.smoother.smooth(np.ones(63, dtype=np.float32))
        # 4 zeros + 1 one = mean of 0.2
        np.testing.assert_allclose(result, np.full(63, 0.2), atol=1e-5)

    def test_reset(self):
        self.smoother.smooth(np.ones(63, dtype=np.float32))
        self.smoother.reset()
        assert len(self.smoother.buffer) == 0


class TestGestureClassifier:

    def setup_method(self):
        from src.ai.gesture_classifier import GestureClassifier
        # Use demo mode (no model required)
        config = {
            "model_path": "/tmp/nonexistent_model.tflite",
            "model_hash_path": "/tmp/hash.sha256",
            "num_gesture_classes": 5,
            "inference_threads": 1,
        }
        self.classifier = GestureClassifier(config)

    def test_predict_returns_valid_types(self):
        vec = np.random.rand(63).astype(np.float32)
        gesture_id, confidence, probs = self.classifier.predict(vec)
        assert isinstance(gesture_id, int)
        assert isinstance(confidence, float)
        assert probs.shape == (5,)

    def test_confidence_in_range(self):
        vec = np.random.rand(63).astype(np.float32)
        _, confidence, probs = self.classifier.predict(vec)
        assert 0.0 <= confidence <= 1.0
        assert np.all(probs >= 0.0)

    def test_probabilities_sum_to_one(self):
        vec = np.random.rand(63).astype(np.float32)
        _, _, probs = self.classifier.predict(vec)
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_gesture_id_valid_class(self):
        for _ in range(20):
            vec = np.random.rand(63).astype(np.float32)
            gesture_id, _, _ = self.classifier.predict(vec)
            assert 0 <= gesture_id < 5


class TestDifferentialPrivacy:

    def setup_method(self):
        from src.federated.differential_privacy import DifferentialPrivacyNoise
        config = {
            "enabled": True,
            "epsilon": 1.0,
            "delta": 1e-5,
            "noise_multiplier": 1.1,
            "max_grad_norm": 1.0,
        }
        self.dp = DifferentialPrivacyNoise(config)

    def test_noise_added(self):
        """Noised weights should differ from originals."""
        weights = [np.ones((128, 63), dtype=np.float32)]
        noised = self.dp.add_noise(weights)
        assert not np.allclose(noised[0], weights[0])

    def test_shape_preserved(self):
        """Output shape must match input."""
        weights = [np.random.rand(128, 64).astype(np.float32),
                   np.random.rand(64,).astype(np.float32)]
        noised = self.dp.add_noise(weights)
        for w, n in zip(weights, noised):
            assert w.shape == n.shape

    def test_gradient_clipping(self):
        """Large gradients should be clipped to max_grad_norm."""
        big_grad = [np.ones(100, dtype=np.float32) * 100.0]
        clipped = self.dp.clip_gradients(big_grad)
        norm = np.linalg.norm(clipped[0])
        assert norm <= self.dp.max_grad_norm + 1e-5

    def test_disabled_returns_original(self):
        from src.federated.differential_privacy import DifferentialPrivacyNoise
        cfg = {"enabled": False, "epsilon": 1.0, "delta": 1e-5,
               "noise_multiplier": 1.1, "max_grad_norm": 1.0}
        dp = DifferentialPrivacyNoise(cfg)
        w = [np.ones(64, dtype=np.float32)]
        result = dp.add_noise(w)
        np.testing.assert_array_equal(result[0], w[0])
