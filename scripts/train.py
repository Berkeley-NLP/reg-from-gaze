#!/usr/bin/env python3
"""
Main training entry point for GazeRL.

Usage:
    # Run canonical preset
    python scripts/train.py --config configs/presets/gaze_bfh.yaml

    # Quick environment smoke test (runs 2 episodes on synthetic data)
    python scripts/train.py --smoke-test

    # Custom CLI overrides
    python scripts/train.py --config configs/presets/gaze_bfh.yaml --lr 5e-6 --batch-size 4
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import yaml

# Ensure project root, reg, and gaze_estimation are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
for p in [str(REPO_ROOT / "gaze_estimation"), str(REPO_ROOT / "reg"), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

from configs.base import ExperimentConfig
from configs.validation import preflight_check
from data.datasets import ReferringExpressionDataset, SyntheticReferringExpressionDataset
from reg.models import get_speaker, get_listener
from reg.rl.rewards import get_reward_function
from reg.training.trainer import GazeRLTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gazerl.train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GazeRL referring expression speaker model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/presets/gaze_bfh.yaml",
        help="Path to YAML experiment preset configuration.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2-episode verification test on synthetic data without downloading models/datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, model paths, and dataset presence without loading models.",
    )
    parser.add_argument("--speaker", type=str, default=None, help="Override speaker architecture.")
    parser.add_argument("--listener", type=str, default=None, help="Override listener type.")
    parser.add_argument("--listener-path", type=str, default=None, help="Override listener model path or HF repo ID.")
    parser.add_argument("--reward", type=str, default=None, help="Override reward type.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override rollout batch size.")
    parser.add_argument("--episodes", type=int, default=None, help="Override total training episodes.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging.")
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name (default: reg-from-gaze).")
    parser.add_argument("--wandb-entity", type=str, default=None, help="WandB username or organization entity.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="WandB run name.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            cfg_data = yaml.safe_load(f)
        config = ExperimentConfig.from_dict(cfg_data)
        logger.info(f"Loaded preset configuration from {config_path}")
    else:
        logger.warning(f"Config file {config_path} not found; using default ExperimentConfig.")
        config = ExperimentConfig()

    # Apply CLI overrides
    if args.speaker:
        config.speaker.architecture = args.speaker
    if args.listener:
        config.listener.listener_type = args.listener
    if args.listener_path:
        config.listener.model_name_or_path = args.listener_path
    if args.reward:
        config.reward.reward_type = args.reward
    if args.lr:
        config.training.learning_rate = args.lr
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.episodes:
        config.training.num_episodes = args.episodes
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.wandb:
        config.training.wandb = True
    if args.wandb_project:
        config.training.wandb_project = args.wandb_project
    if args.wandb_entity:
        config.training.wandb_entity = args.wandb_entity
    if args.wandb_run_name:
        config.training.wandb_run_name = args.wandb_run_name
    if args.seed is not None:
        config.training.seed = args.seed

    if args.smoke_test:
        logger.info("Executing instant smoke test mode...")
        from tests.test_train_step_smoke import DifferentiableMockSpeaker, MockListener

        speaker = DifferentiableMockSpeaker()
        listener = MockListener()
        reward_fn = get_reward_function(config.reward)
        dataset = SyntheticReferringExpressionDataset(num_samples=4)
        config.training.num_episodes = 2
        config.training.batch_size = 2
        config.training.gradient_accumulation_steps = 1
        config.training.warmup_steps = 0
        config.training.wandb = bool(args.wandb)

        trainer = GazeRLTrainer(
            speaker=speaker,
            listener=listener,
            reward_fn=reward_fn,
            train_dataset=dataset,
            config=config,
        )
        trainer.train()
        logger.info("Smoke test passed successfully!")
        return

    # Run pre-flight check before loading any heavy models into GPU memory
    preflight_check(config)
    if args.dry_run:
        logger.info("Pre-flight configuration validation PASSED. (Dry-run mode: exiting without loading models).")
        return

    # Real Training Mode
    logger.info(f"Initializing speaker: {config.speaker.architecture} ({config.speaker.model_name_or_path})")
    speaker = get_speaker(config.speaker)

    logger.info(f"Initializing listener: {config.listener.listener_type} ({config.listener.model_name_or_path})")
    listener = get_listener(config.listener)

    logger.info(f"Initializing reward function: {config.reward.reward_type}")
    reward_fn = get_reward_function(config.reward)

    # Reference speaker for KL divergence if kl_coef > 0
    ref_speaker = None
    if config.training.kl_coef > 0:
        logger.info("Initializing reference policy speaker for KL divergence regularization...")
        ref_speaker = get_speaker(config.speaker)

    # Datasets
    logger.info(f"Loading training dataset: {config.dataset.dataset_name} ({config.dataset.train_split})")
    train_dataset = ReferringExpressionDataset(
        dataset_name=config.dataset.dataset_name,
        split=config.dataset.train_split,
        splits_dir=config.dataset.splits_dir,
        max_samples=config.dataset.max_samples,
    )

    val_dataset = None
    try:
        val_dataset = ReferringExpressionDataset(
            dataset_name=config.dataset.dataset_name,
            split=config.dataset.val_split,
            splits_dir=config.dataset.splits_dir,
            max_samples=config.training.validation_batch_size,
        )
    except Exception as e:
        logger.warning(f"Could not load validation dataset: {e}; training will proceed without validation.")

    trainer = GazeRLTrainer(
        speaker=speaker,
        listener=listener,
        reward_fn=reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        ref_speaker=ref_speaker,
    )

    logger.info(f"Starting training run: {config.name}")
    trainer.train()


if __name__ == "__main__":
    main()
