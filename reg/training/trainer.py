"""
GazeRL Trainer: Streamlined, modular RL training engine for referring expression generation.
"""

import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import torch

from configs.base import ExperimentConfig, TrainingConfig
from models.base import BaseSpeaker, BaseListener
from rl.rewards.base import BaseRewardFunction
from rl.loss import calculate_policy_loss
from rl.rollouts import RolloutSample, RolloutBatch, collect_single_rollout, collate_rollouts
from training.optimization import setup_optimizer, setup_scheduler, disable_dropout
from training.checkpoint import save_checkpoint, load_checkpoint
from training.logging import MetricLogger
from training.validator import Validator

logger = logging.getLogger(__name__)


class GazeRLTrainer:
    """
    Streamlined Reinforcement Learning Trainer for GazeRL.

    Coordinates:
      - Multi-modal speaker rollouts (sampling & logprob extraction)
      - Listener comprehension (gaze path & target prediction)
      - Reward assignment & credit decay
      - Policy gradient optimization (REINFORCE + advantage standardization + KL divergence)
      - Periodic validation & checkpointing
    """

    def __init__(
        self,
        speaker: BaseSpeaker,
        listener: BaseListener,
        reward_fn: BaseRewardFunction,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        config: Optional[ExperimentConfig] = None,
        ref_speaker: Optional[BaseSpeaker] = None,
    ):
        self.speaker = speaker
        self.listener = listener
        self.reward_fn = reward_fn
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or ExperimentConfig()
        self.train_cfg: TrainingConfig = self.config.training
        self.ref_speaker = ref_speaker

        # Apply LoRA if enabled
        if getattr(self.config, "lora", None) and self.config.lora.enabled:
            logger.info("Applying LoRA adapter to speaker model...")
            self.speaker.apply_lora(self.config.lora)

        # Disable dropout for deterministic policy sampling
        speaker_module = getattr(self.speaker, "model", None) or self.speaker
        disable_dropout(speaker_module)

        # Set up optimizer & scheduler
        self.optimizer = setup_optimizer(
            model=speaker_module,
            learning_rate=self.train_cfg.learning_rate,
            weight_decay=self.train_cfg.weight_decay,
        )

        total_optimization_steps = max(1, self.train_cfg.num_episodes // (self.train_cfg.batch_size * self.train_cfg.gradient_accumulation_steps))
        self.scheduler = setup_scheduler(
            optimizer=self.optimizer,
            schedule_type=self.train_cfg.lr_schedule,
            warmup_steps=self.train_cfg.warmup_steps,
            max_steps=total_optimization_steps,
        )

        # Metric logging
        self.logger = MetricLogger(
            output_dir=self.train_cfg.output_dir,
            use_wandb=self.train_cfg.wandb,
            wandb_project=self.train_cfg.wandb_project,
            wandb_entity=getattr(self.train_cfg, "wandb_entity", None),
            wandb_run_name=self.train_cfg.wandb_run_name or self.config.name,
            config=self.config,
        )

        # In-training validator
        self.validator = Validator(
            speaker=self.speaker,
            listener=self.listener,
            reward_fn=self.reward_fn,
            val_dataset=self.val_dataset,
            max_val_samples=self.train_cfg.validation_batch_size,
        ) if self.val_dataset is not None else None

        # Tracking state
        self.episode_count = 0
        self.opt_step = 0
        self.best_success_rate = 0.0

        # Set seeds
        torch.manual_seed(self.train_cfg.seed)
        np.random.seed(self.train_cfg.seed)
        random.seed(self.train_cfg.seed)

    def train_step(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute one RL batch rollout and optimization step.

        Args:
            batch_items: List of dataset dictionaries containing 'image', 'bbox', etc.

        Returns:
            Dictionary of step metrics (loss, reward, success_rate, kl, etc.).
        """
        prompt = getattr(self.config.speaker, "prompt", "Briefly describe the object in the red box.")

        # 1. Rollout collection across batch items
        rollouts: List[RolloutSample] = []
        for idx, item in enumerate(batch_items):
            if hasattr(item, "image"):
                img = item.image
                bbox = getattr(item, "bbox_minmax", getattr(item, "bbox", None))
                speaker_img = getattr(item, "speaker_image", None)
                listener_img = getattr(item, "listener_image", None)
                sample_prompt = getattr(item, "prompt", prompt)
                sample_id = getattr(item, "sample_id", f"ep_{self.episode_count + idx}")
            else:
                img = item.get("image")
                bbox = item.get("bbox", item.get("bbox_minmax"))
                speaker_img = item.get("speaker_image")
                listener_img = item.get("listener_image")
                sample_prompt = item.get("prompt", prompt)
                sample_id = str(item.get("question_id", item.get("sample_id", f"ep_{self.episode_count + idx}")))

            rollout = collect_single_rollout(
                speaker=self.speaker,
                listener=self.listener,
                reward_fn=self.reward_fn,
                image=img,
                bbox_orig=bbox,
                prompt=sample_prompt,
                sample_id=sample_id,
                speaker_image=speaker_img,
                listener_image=listener_img,
                ref_speaker=self.ref_speaker if self.train_cfg.kl_coef > 0 else None,
            )
            rollouts.append(rollout)

        batch = collate_rollouts(rollouts)
        self.episode_count += len(batch_items)

        # 2. Compute policy gradient loss
        total_loss, pol_loss, ent_loss, kl_loss, num_kl, metrics = calculate_policy_loss(
            log_probs=batch.all_log_probs(),
            rewards=batch.all_rewards(),
            kl_divergence_coef=self.train_cfg.kl_coef,
            entropy_bonus_coef=self.train_cfg.entropy_coef,
            entropies=batch.all_entropies(),
            ref_log_probs=batch.all_ref_log_probs(),
        )

        # 3. Backward pass with gradient accumulation
        loss_to_backward = total_loss / self.train_cfg.gradient_accumulation_steps
        loss_to_backward.backward()

        is_accum_step = (self.episode_count // self.train_cfg.batch_size) % self.train_cfg.gradient_accumulation_steps == 0
        if is_accum_step:
            if self.train_cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    (getattr(self.speaker, "model", None) or self.speaker).parameters(),
                    max_norm=self.train_cfg.max_grad_norm,
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.opt_step += 1

        step_metrics = {
            "loss": float(total_loss.item()),
            "policy_loss": float(pol_loss.item()),
            "entropy_loss": float(ent_loss.item()),
            "reward_mean": batch.mean_reward,
            "success_rate": batch.success_rate,
            "avg_tokens": float(batch.total_tokens / max(1, batch.size)),
            "opt_step": self.opt_step,
            "episode": self.episode_count,
            "lr": float(self.scheduler.get_last_lr()[0]) if self.scheduler else self.train_cfg.learning_rate,
            "kl": metrics.get("raw_kl", 0.0),
        }

        return step_metrics

    def train(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete RL training loop until num_episodes is reached.
        """
        target_episodes = num_episodes or self.train_cfg.num_episodes
        batch_size = self.train_cfg.batch_size
        logger.info(f"Starting GazeRL training: {target_episodes} target episodes (batch size={batch_size}).")

        dataset_len = len(self.train_dataset)
        sample_indices = list(range(dataset_len))
        curr_idx = 0

        while self.episode_count < target_episodes:
            # Batch extraction
            batch_items = []
            for _ in range(batch_size):
                if curr_idx >= dataset_len:
                    random.shuffle(sample_indices)
                    curr_idx = 0
                batch_items.append(self.train_dataset[sample_indices[curr_idx]])
                curr_idx += 1

            # Execute train step
            metrics = self.train_step(batch_items)

            # Log progress
            self.logger.log_metrics(step=self.opt_step, episode=self.episode_count, metrics=metrics)
            if self.opt_step % 5 == 0 or self.episode_count >= target_episodes:
                status_str = self.logger.format_status(
                    step=self.opt_step,
                    episode=self.episode_count,
                    total_episodes=target_episodes,
                    loss=metrics["loss"],
                    reward=metrics["reward_mean"],
                    success_rate=metrics["success_rate"],
                    avg_tokens=metrics["avg_tokens"],
                    kl=metrics["kl"],
                )
                print(status_str)

            # Periodic Validation
            if self.validator is not None and self.train_cfg.validation_frequency > 0:
                if self.episode_count % self.train_cfg.validation_frequency < batch_size:
                    val_metrics = self.validator.evaluate(current_episode=self.episode_count)
                    self.logger.log_metrics(step=self.opt_step, episode=self.episode_count, metrics=val_metrics)

                    # Save best checkpoint
                    val_succ = val_metrics.get("val_success_rate", 0.0)
                    if val_succ > self.best_success_rate:
                        self.best_success_rate = val_succ
                        best_dir = Path(self.train_cfg.output_dir) / "checkpoint_best"
                        self.save_checkpoint(best_dir)

            # Periodic Checkpoint
            if self.train_cfg.periodic_checkpoint_interval > 0:
                if self.episode_count % self.train_cfg.periodic_checkpoint_interval < batch_size:
                    periodic_dir = Path(self.train_cfg.output_dir) / f"checkpoint_ep_{self.episode_count:06d}"
                    self.save_checkpoint(periodic_dir)

        # Final checkpoint save
        final_dir = Path(self.train_cfg.output_dir) / "checkpoint_final"
        self.save_checkpoint(final_dir)
        self.logger.finish()

        logger.info(f"Training completed at episode {self.episode_count}. Final checkpoint saved to {final_dir}.")
        return {"final_episode": self.episode_count, "final_step": self.opt_step, "best_success_rate": self.best_success_rate}

    def save_checkpoint(self, checkpoint_dir: Union[str, Path]) -> str:
        """Save current training state and model weights."""
        return save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            speaker=self.speaker,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            opt_step=self.opt_step,
            episode_count=self.episode_count,
            best_success_rate=self.best_success_rate,
            config=self.config,
        )

    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> Dict[str, Any]:
        """Restore training state and model weights from checkpoint."""
        state = load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            speaker=self.speaker,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.opt_step = state.get("opt_step", 0)
        self.episode_count = state.get("episode_count", 0)
        self.best_success_rate = state.get("best_success_rate", 0.0)
        return state
