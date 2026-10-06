"""
Optimization setup: AdamW optimizer, learning rate schedules, and dropout control.
"""

import logging
import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

logger = logging.getLogger(__name__)


def disable_dropout(model: nn.Module) -> int:
    """
    Disable all dropout layers in the model by setting p=0.0.
    Ensures deterministic forward passes during RL training.
    """
    if model is None:
        return 0

    dropout_types = (
        nn.Dropout,
        getattr(nn, "Dropout1d", nn.Dropout),
        getattr(nn, "Dropout2d", nn.Dropout),
        getattr(nn, "Dropout3d", nn.Dropout),
        getattr(nn, "AlphaDropout", nn.Dropout),
    )

    changed = 0
    for m in model.modules():
        if isinstance(m, dropout_types) and hasattr(m, "p"):
            if float(m.p) != 0.0:
                m.p = 0.0
                changed += 1
        elif "dropout" in m.__class__.__name__.lower() and hasattr(m, "p"):
            try:
                if float(getattr(m, "p", 0.0)) != 0.0:
                    m.p = 0.0
                    changed += 1
            except Exception:
                pass
    return changed


def setup_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> AdamW:
    """
    Configure AdamW optimizer for all trainable parameters in the model.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning("No parameters with requires_grad=True found; using all model parameters.")
        trainable_params = list(model.parameters())

    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Setting up AdamW optimizer with {num_trainable:,} trainable parameters (lr={learning_rate}).")

    return AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay, eps=eps)


def setup_scheduler(
    optimizer: AdamW,
    schedule_type: str = "constant",
    warmup_steps: int = 16,
    max_steps: int = 8000,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Configure learning rate schedule (constant with warmup, linear, or cosine).
    """
    if schedule_type == "cosine":
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
            cosine_factor = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        return LambdaLR(optimizer, lr_lambda)

    elif schedule_type == "linear":
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(max_steps - current_step) / float(max(1, max_steps - warmup_steps)))

        return LambdaLR(optimizer, lr_lambda)

    else:
        # Constant schedule with linear warmup
        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda)
