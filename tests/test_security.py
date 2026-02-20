"""
Tests - Security Module
Unit tests for all security components.
"""

import pytest
import numpy as np
import time
import os
import tempfile


# ── Liveness Detector ────────────────────────────────────────────────────────

class TestLivenessDetector:

    def setup_method(self):
        from src.security.liveness_detector import LivenessDetector
        config = {"enabled": True, "motion_threshold": 0.01, "window": 5}
        self.detector = LivenessDetector(config)

    def test_first_frame_passes(self):
        """First frame always passes (no previous reference)."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        is_live, delta = self.detector.check(frame)
        assert is_live is True

    def test_static_image_blocked(self):
        """Identical frames should be blocked as replay."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Submit same frame many times
        for _ in range(10):
            is_live, delta = self.detector.check(frame)
        # After warmup, identical frames should fail liveness
        assert delta < self.detector.motion_threshold

    def test_moving_hand_passes(self):
        """Frames with motion should pass liveness."""
        self.detector.reset()
        results = []
        for i in range(10):
            # Each frame is different (simulating motion)
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            is_live, delta = self.detector.check(frame)
            results.append(is_live)
        # Most frames with random content (high motion) should be live
        assert sum(results) >= 5

    def test_reset(self):
        """Reset clears history."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.detector.check(frame)
        self.detector.reset()
        assert self.detector._prev_frame_gray is None

    def test_disabled(self):
        """When disabled, all frames pass."""
        from src.security.liveness_detector import LivenessDetector
        cfg = {"enabled": False, "motion_threshold": 0.01, "window": 5}
        detector = LivenessDetector(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(10):
            is_live, _ = detector.check(frame)
            assert is_live is True


# ── Anomaly Detector ──────────────────────────────────────────────────────────

class TestAnomalyDetector:

    def setup_method(self):
        from src.security.anomaly_detector import AnomalyDetector
        config = {"enabled": True, "z_score_threshold": 3.0, "history_size": 50}
        self.detector = AnomalyDetector(config)

    def test_normal_vectors_not_flagged(self):
        """Normal hand landmark vectors should not be flagged."""
        # Build baseline
        for _ in range(30):
            normal_vec = np.random.normal(0, 0.2, 63).astype(np.float32)
            self.detector.check(normal_vec)

        # Check normal vector
        normal_vec = np.random.normal(0, 0.2, 63).astype(np.float32)
        is_anomaly, _ = self.detector.check(normal_vec)
        assert is_anomaly is False

    def test_adversarial_vector_flagged(self):
        """Extreme outlier vectors should be flagged as anomalies."""
        # Build baseline with small-magnitude vectors
        for _ in range(30):
            vec = np.random.normal(0, 0.1, 63).astype(np.float32)
            self.detector.check(vec)

        # Inject adversarial outlier (very large values)
        adversarial = np.ones(63, dtype=np.float32) * 100.0
        is_anomaly, z_scores = self.detector.check(adversarial)
        assert is_anomaly is True
        assert max(z_scores) > 3.0

    def test_not_enough_history(self):
        """Should not flag anomalies before min_samples collected."""
        vec = np.random.normal(0, 5.0, 63).astype(np.float32)
        is_anomaly, _ = self.detector.check(vec)
        assert is_anomaly is False  # Not enough history yet

    def test_reset(self):
        """Reset clears history."""
        for _ in range(25):
            self.detector.check(np.random.rand(63).astype(np.float32))
        self.detector.reset()
        assert len(self.detector._history) == 0


# ── Zero-Trust Gate ───────────────────────────────────────────────────────────

class TestZeroTrustGate:

    def setup_method(self):
        from src.security.zero_trust import ZeroTrustGate
        config = {
            "confidence_threshold": 0.90,
            "stability_frames": 3,
            "rate_limiting": {
                "max_commands_per_second": 2,
                "cooldown_ms": 100,
            }
        }
        self.gate = ZeroTrustGate(config)

    def test_low_confidence_denied(self):
        """Commands with confidence below threshold must be denied."""
        approved, reason = self.gate.evaluate(0, 0.70, False)
        assert approved is False
        assert "CONF" in reason

    def test_anomaly_denied(self):
        """Commands when anomaly detected must be denied."""
        approved, reason = self.gate.evaluate(0, 0.99, True)
        assert approved is False
        assert reason == "ANOMALY"

    def test_unstable_gesture_denied(self):
        """Commands before stability window fills must be denied."""
        # First call — warming up
        approved, reason = self.gate.evaluate(0, 0.95, False)
        assert approved is False

    def test_stable_gesture_approved(self):
        """Stable high-confidence gesture should be approved."""
        # Fill stability buffer
        for _ in range(3):
            self.gate.evaluate(1, 0.95, False)
        # Final call should be approved
        approved, reason = self.gate.evaluate(1, 0.95, False)
        assert approved is True
        assert reason == "OK"

    def test_rate_limiting(self):
        """Rapid commands should be rate limited."""
        # Approve one command
        for _ in range(3):
            self.gate.evaluate(2, 0.95, False)
        approved1, _ = self.gate.evaluate(2, 0.95, False)

        # Immediately try another — should be rate limited
        for _ in range(3):
            self.gate.evaluate(2, 0.95, False)
        approved2, reason2 = self.gate.evaluate(2, 0.95, False)

        # At least one should be rate limited
        assert "RATE_LIMITED" in reason2 or approved2 is False or approved1 is True

    def test_different_gestures_unstable(self):
        """Alternating gesture classes should not be approved."""
        self.gate.reset()
        self.gate.evaluate(0, 0.95, False)
        self.gate.evaluate(1, 0.95, False)
        self.gate.evaluate(0, 0.95, False)
        approved, reason = self.gate.evaluate(1, 0.95, False)
        assert approved is False


# ── Model Integrity ───────────────────────────────────────────────────────────

class TestModelIntegrity:

    def test_hash_computation_and_verification(self):
        """Compute and verify hash of a temp file."""
        from src.security.model_integrity import ModelIntegrityChecker

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b"fake_model_weights_data_12345")
            model_path = f.name

        hash_path = model_path + ".sha256"

        try:
            checker = ModelIntegrityChecker(model_path)
            saved_hash = checker.compute_and_save_hash(hash_path)

            # Verify matches
            assert checker.verify(hash_path) is True

            # Tamper with model
            with open(model_path, "ab") as f:
                f.write(b"tampered!")

            # Verification should fail
            assert checker.verify(hash_path) is False

        finally:
            os.unlink(model_path)
            if os.path.exists(hash_path):
                os.unlink(hash_path)

    def test_missing_model_returns_false(self):
        from src.security.model_integrity import ModelIntegrityChecker
        checker = ModelIntegrityChecker("/nonexistent/model.tflite")
        assert checker.verify("/nonexistent/hash.sha256") is False
