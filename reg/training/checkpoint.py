"""
Checkpoint saving and loading utilities with LoRA adapter support.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch

from models.base import BaseSpeaker

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: Union[str, Path],
    speaker: BaseSpeaker,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    opt_step: int = 0,
    episode_count: int = 0,
    best_success_rate: float = 0.0,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
) -> str:
    """
    Save training checkpoint: LoRA adapter weights, optimizer state, and training metadata.

    Args:
        checkpoint_dir: Directory path to write checkpoint files.
        speaker: Speaker model instance.
        optimizer: Optimizer instance.
        scheduler: LR scheduler instance.
        opt_step: Completed optimization steps count.
        episode_count: Completed episodes count.
        best_success_rate: Best validation success rate achieved.
        metrics: Dictionary of current metrics.
        config: Experiment configuration object or dictionary.

    Returns:
        String path of the saved checkpoint directory.
    """
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # 1. Save model weights / LoRA adapter
    try:
        # If speaker has PeftModel inside, save adapter
        inner = getattr(speaker, "model", speaker)
        if hasattr(inner, "save_pretrained"):
            inner.save_pretrained(str(ckpt_path))
            logger.info(f"Saved LoRA adapter to {ckpt_path}")
        else:
            torch.save(speaker.state_dict(), ckpt_path / "model_state_dict.pt")
            logger.info(f"Saved model state dict to {ckpt_path / 'model_state_dict.pt'}")
    except Exception as e:
        logger.warning(f"Error saving adapter with save_pretrained: {e}; falling back to state_dict")
        torch.save(speaker.state_dict(), ckpt_path / "model_state_dict.pt")

    # 2. Save training state
    trainer_state: Dict[str, Any] = {
        "opt_step": opt_step,
        "episode_count": episode_count,
        "best_success_rate": best_success_rate,
        "metrics": metrics or {},
    }

    if optimizer is not None:
        trainer_state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        trainer_state["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(trainer_state, ckpt_path / "trainer_state.pt")

    # 3. Save config as JSON for easy human inspection
    if config is not None:
        config_dict = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        with open(ckpt_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    logger.info(f"Successfully saved checkpoint at episode {episode_count}, opt_step {opt_step} -> {ckpt_path}")
    return str(ckpt_path)


def load_checkpoint(
    checkpoint_dir: Union[str, Path],
    speaker: BaseSpeaker,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Load a saved checkpoint and restore speaker weights, optimizer, and scheduler.

    Args:
        checkpoint_dir: Directory containing saved checkpoint.
        speaker: Speaker model to restore weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.

    Returns:
        Dictionary containing restored training state (opt_step, episode_count, metrics, etc.).
    """
    ckpt_str = str(checkpoint_dir)
    ckpt_path = Path(checkpoint_dir)
    inner = getattr(speaker, "model", speaker)

    # 1. Check if checkpoint_dir is a Hugging Face Hub repository identifier
    if not ckpt_path.exists() and "/" in ckpt_str and not ckpt_str.startswith(("/", "./", "../")):
        try:
            from peft import PeftModel
            if hasattr(inner, "load_adapter"):
                inner.load_adapter(ckpt_str, adapter_name="default")
            elif hasattr(speaker, "model"):
                speaker.model = PeftModel.from_pretrained(inner, ckpt_str)
            logger.info("Successfully loaded LoRA adapter from Hugging Face Hub: %s", ckpt_str)
            return {"opt_step": 0, "episode_count": 0, "best_success_rate": 0.0, "metrics": {}}
        except Exception as e:
            logger.error("Failed to load adapter from Hugging Face Hub '%s': %s", ckpt_str, e)
            raise

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    # 2. Restore model weights / adapter
    adapter_file = ckpt_path / "adapter_model.safetensors"
    adapter_bin = ckpt_path / "adapter_model.bin"
    state_file = ckpt_path / "model_state_dict.pt"

    if adapter_file.exists() or adapter_bin.exists():
        if hasattr(inner, "load_adapter"):
            inner.load_adapter(str(ckpt_path), adapter_name="default")
            logger.info(f"Loaded LoRA adapter from {ckpt_path}")
        elif hasattr(inner, "from_pretrained"):
            logger.info(f"Loading adapter weights from {ckpt_path}")
    elif state_file.exists():
        weights = torch.load(state_file, map_location="cpu", weights_only=True)
        speaker.load_state_dict(weights, strict=False)
        logger.info(f"Loaded state dict from {state_file}")

    # 2. Restore trainer state
    trainer_state_file = ckpt_path / "trainer_state.pt"
    state: Dict[str, Any] = {
        "opt_step": 0,
        "episode_count": 0,
        "best_success_rate": 0.0,
        "metrics": {},
    }

    if trainer_state_file.exists():
        loaded_state = torch.load(trainer_state_file, map_location="cpu", weights_only=False)
        state.update(loaded_state)

        if optimizer is not None and "optimizer_state_dict" in loaded_state:
            try:
                optimizer.load_state_dict(loaded_state["optimizer_state_dict"])
                logger.info("Restored optimizer state.")
            except Exception as e:
                logger.warning(f"Failed to restore optimizer state: {e}")

        if scheduler is not None and "scheduler_state_dict" in loaded_state:
            try:
                scheduler.load_state_dict(loaded_state["scheduler_state_dict"])
                logger.info("Restored scheduler state.")
            except Exception as e:
                logger.warning(f"Failed to restore scheduler state: {e}")

    logger.info(f"Checkpoint loaded: episode={state['episode_count']}, opt_step={state['opt_step']}")
    return state
