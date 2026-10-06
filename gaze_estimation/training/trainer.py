"""
Unified Trainer configuration for Molmo-REC-Gaze.
"""

from typing import Dict, Any, Optional
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
import torch

from gaze_estimation.training.callbacks import (
    CombinedEvalCallback,
    DTWLoggerCallback,
    BestCheckpointCallback,
)


def create_training_arguments(
    output_dir: str = "checkpoints/gaze_predictor_run",
    num_train_epochs: int = 4,
    learning_rate: float = 1e-5,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    eval_steps: int = 100,
    bf16: bool = True,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    optim: str = "paged_adamw_8bit",
) -> TrainingArguments:
    """
    Construct Hugging Face TrainingArguments matching verified paper configuration.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        bf16=bf16,
        optim=optim,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",  # Managed via BestCheckpointCallback to save disk space
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )


def create_trainer(
    model: Any,
    processor: Any,
    train_dataset: Any,
    val_dataset: Any,
    collator: Any,
    training_args: TrainingArguments,
    early_stopping_patience: int = 5,
) -> Trainer:
    """
    Instantiate Hugging Face Trainer with canonical callbacks.
    """
    callbacks = [
        BestCheckpointCallback(output_dir=training_args.output_dir, processor=processor),
        DTWLoggerCallback(output_dir=training_args.output_dir),
        EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
    ]

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
