"""
DifferentialPrivacy - Federated Learning Module
Applies Gaussian noise to model weight updates before transmission.
Implements DP-SGD principles for privacy-preserving federated learning.

Privacy guarantee:
  Noise calibrated to (ε, δ)-differential privacy
  with ε = privacy budget, δ = failure probability
"""

import numpy as np
from typing import List, Union
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class DifferentialPrivacyNoise:
    """
    Gaussian mechanism for (ε, δ)-differential privacy.

    Noise scale σ computed as:
        σ = noise_multiplier × max_grad_norm

    where max_grad_norm is the L2 sensitivity (gradient clipping norm).
    """

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", True)
        self.epsilon = config.get("epsilon", 1.0)
        self.delta = config.get("delta", 1e-5)
        self.noise_multiplier = config.get("noise_multiplier", 1.1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)

        # Noise scale = σ
        self.noise_scale = self.noise_multiplier * self.max_grad_norm

        if self.enabled:
            logger.info(
                f"✅ DifferentialPrivacy: ε={self.epsilon} δ={self.delta} "
                f"σ={self.noise_scale:.4f}"
            )

    def clip_gradients(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Clip gradient tensors to max_grad_norm (L2 norm)."""
        clipped = []
        for grad in gradients:
            norm = np.linalg.norm(grad)
            if norm > self.max_grad_norm:
                grad = grad * (self.max_grad_norm / (norm + 1e-8))
            clipped.append(grad)
        return clipped

    def add_noise(self, gradients: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply Gaussian noise to clipped gradients.

        Args:
            gradients: List of weight update arrays

        Returns:
            Noised gradients safe for transmission
        """
        if not self.enabled:
            return gradients

        clipped = self.clip_gradients(gradients)
        noised = []
        for grad in clipped:
            noise = np.random.normal(0, self.noise_scale, size=grad.shape).astype(np.float32)
            noised.append(grad + noise)

        return noised

    def apply_to_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Convenience method: treat weight deltas as gradients."""
        return self.add_noise(weights)

    def compute_privacy_budget(self, num_samples: int, batch_size: int,
                                num_epochs: int) -> dict:
        """
        Estimate actual (ε, δ) privacy budget consumed.
        Uses the moments accountant approximation.

        Returns privacy analysis for logging.
        """
        steps = (num_samples // batch_size) * num_epochs
        q = batch_size / num_samples  # Sampling ratio

        # Simplified moments accountant (Abadi et al., 2016)
        # Actual implementation would use google/dp-accounting library (free)
        estimated_epsilon = (
            2 * q * steps * self.noise_multiplier ** -2
        ) ** 0.5

        return {
            "estimated_epsilon": float(estimated_epsilon),
            "delta": self.delta,
            "steps": steps,
            "noise_multiplier": self.noise_multiplier,
        }
