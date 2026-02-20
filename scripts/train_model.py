#!/usr/bin/env python3
"""
Model Training Script - BharatSecure Touchless HCI
Trains the lightweight MLP gesture classifier.

Usage:
    python scripts/train_model.py
    make train
"""

import yaml
import argparse
from src.ai.model_trainer import ModelTrainer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train BharatSecure gesture classifier")
    parser.add_argument("--config", default="config/system_config.yaml")
    parser.add_argument("--data-dir", default="data/gestures")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 55)
    logger.info("  BharatSecure — Gesture Model Training")
    logger.info("=" * 55)

    trainer = ModelTrainer(config)
    if args.data_dir:
        trainer.data_dir = args.data_dir

    success = trainer.train()

    if success:
        logger.info("=" * 55)
        logger.info("  ✅ Training complete! Run 'make run' to start.")
        logger.info("=" * 55)
    else:
        logger.error("Training failed. Run 'make collect-data' first.")


if __name__ == "__main__":
    main()
