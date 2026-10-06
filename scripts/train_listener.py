#!/usr/bin/env python3
"""
Unified Training Script for Molmo-REC-Gaze.
Supports YAML configuration files and CLI overrides.

Usage:
    # Train using canonical config:
    python scripts/train_listener.py --config configs/listener/canonical_molmo_gaze.yaml

    # Override parameters via CLI:
    python scripts/train_listener.py \
        --config configs/listener/canonical_molmo_gaze.yaml \
        --output_dir checkpoints/my_run \
        --num_train_epochs 4 \
        --learning_rate 1e-5
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from transformers import logging as hf_logging

# Ensure package is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gaze_estimation.models.loader import load_molmo_model_and_processor
from gaze_estimation.data.dataset import TokenLevelGazeDataset, WordLevelGazeDataset
from gaze_estimation.data.collators import GazeDataCollator
from gaze_estimation.training.trainer import create_training_arguments, create_trainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Molmo-REC-Gaze listener model")
    parser.add_argument("--config", type=str, default="configs/listener/canonical_molmo_gaze.yaml", help="Path to YAML config")
    parser.add_argument("--train_json", type=str, default=None)
    parser.add_argument("--val_json", type=str, default=None)
    parser.add_argument("--image_base", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--level", type=str, choices=["token", "word"], default=None)
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging.")
    parser.add_argument("--wandb_project", type=str, default="gaze-estimation", help="WandB project name (default: gaze-estimation).")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB username or organization entity.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name.")
    return parser.parse_args()


def main():
    hf_logging.set_verbosity_warning()
    args = parse_args()

    # Load configuration from YAML
    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)

    # Extract or override config values
    model_name = cfg.get("model", {}).get("name_or_path", "allenai/Molmo-7B-D-0924")
    freeze_vision = cfg.get("model", {}).get("freeze_vision_backbone", True)

    train_json = args.train_json or cfg.get("data", {}).get("train_json", "data/refcocogaze_train_delay.json")
    val_json = args.val_json or cfg.get("data", {}).get("val_json", "data/refcocogaze_val_delay.json")
    image_base = args.image_base or cfg.get("data", {}).get("image_base", "../../../../scratch/current/teaywright/images")
    max_length = args.max_length or cfg.get("data", {}).get("max_length", 256)
    level = args.level or cfg.get("data", {}).get("level", "token")

    output_dir = args.output_dir or cfg.get("training", {}).get("output_dir", "checkpoints/gaze_predictor_run")
    epochs = args.num_train_epochs or cfg.get("training", {}).get("num_train_epochs", 4)
    lr = args.learning_rate or cfg.get("training", {}).get("learning_rate", 1e-5)
    batch_size = args.per_device_train_batch_size or cfg.get("training", {}).get("per_device_train_batch_size", 1)
    grad_accum = args.gradient_accumulation_steps or cfg.get("training", {}).get("gradient_accumulation_steps", 8)
    eval_steps = cfg.get("training", {}).get("eval_steps", 100)
    optim = cfg.get("training", {}).get("optim", "paged_adamw_8bit")
    warmup_ratio = cfg.get("training", {}).get("warmup_ratio", 0.03)
    weight_decay = cfg.get("training", {}).get("weight_decay", 0.01)
    patience = cfg.get("training", {}).get("early_stopping_patience", 5)

    print("=" * 70)
    print("Molmo-REC-Gaze Training Initializing")
    print(f"  Model: {model_name}")
    print(f"  Level: {level} (subword token alignment)" if level == "token" else f"  Level: {level}")
    print(f"  Train JSON: {train_json}")
    print(f"  Val JSON: {val_json}")
    print(f"  Output dir: {output_dir}")
    print(f"  LR: {lr}, Epochs: {epochs}, Effective Batch Size: {batch_size * grad_accum}")
    print("=" * 70)

    # Load Model and Processor
    model, processor = load_molmo_model_and_processor(
        model_name_or_path=model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        freeze_vision_backbone=freeze_vision,
    )

    # Build Datasets
    if level == "token":
        train_ds = TokenLevelGazeDataset(train_json, split="train", image_base=image_base, tokenizer=processor.tokenizer)
        val_ds = TokenLevelGazeDataset(val_json, split="val", image_base=image_base, tokenizer=processor.tokenizer)
    else:
        train_ds = WordLevelGazeDataset(train_json, split="train", image_base=image_base)
        val_ds = WordLevelGazeDataset(val_json, split="val", image_base=image_base)

    collator = GazeDataCollator(processor, max_length=max_length)

    training_args = create_training_arguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        eval_steps=eval_steps,
        bf16=True,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        optim=optim,
    )

    trainer = create_trainer(
        model=model,
        processor=processor,
        train_dataset=train_ds,
        val_dataset=val_ds,
        collator=collator,
        training_args=training_args,
        early_stopping_patience=patience,
    )

    if args.wandb:
        try:
            import wandb
            wandb_proj = args.wandb_project or cfg.get("training", {}).get("wandb_project", "gaze-estimation")
            wandb_ent = args.wandb_entity or cfg.get("training", {}).get("wandb_entity", os.environ.get("WANDB_ENTITY"))
            wandb_name = args.wandb_run_name or cfg.get("training", {}).get("wandb_run_name", None)
            init_kw = {"project": wandb_proj, "name": wandb_name, "config": cfg}
            if wandb_ent:
                init_kw["entity"] = wandb_ent
            wandb.init(**init_kw)
            print(f"  ✓ Initialized WandB: project='{wandb_proj}' (entity: {wandb_ent})")
        except Exception as e:
            print(f"  [WARNING] Failed to initialize WandB: {e}")

    trainer.train()

    # Save final model & processor
    print(f"\n[INFO] Saving final trained model to {output_dir}...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print(f"  ✓ Model and processor saved to {output_dir}")


if __name__ == "__main__":
    main()
