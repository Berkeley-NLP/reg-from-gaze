"""
Validation runner for periodic evaluation during RL training.
"""

import logging
from typing import Any, Dict, List, Optional
import torch

from models.base import BaseSpeaker, BaseListener
from rl.rewards.base import BaseRewardFunction
from rl.rollouts import collect_single_rollout

logger = logging.getLogger(__name__)


class Validator:
    """
    Evaluates policy speaker and listener comprehension on validation examples.
    """

    def __init__(
        self,
        speaker: BaseSpeaker,
        listener: BaseListener,
        reward_fn: BaseRewardFunction,
        val_dataset: Any,
        prompt: str = "Briefly describe the object in the red box.",
        max_val_samples: Optional[int] = 100,
    ):
        self.speaker = speaker
        self.listener = listener
        self.reward_fn = reward_fn
        self.val_dataset = val_dataset
        self.prompt = prompt
        self.max_val_samples = max_val_samples

    def evaluate(self, current_episode: int = 0) -> Dict[str, float]:
        """
        Run validation pass over dataset and return aggregated metrics.
        """
        if self.val_dataset is None or len(self.val_dataset) == 0:
            return {}

        num_samples = min(len(self.val_dataset), self.max_val_samples) if self.max_val_samples else len(self.val_dataset)
        logger.info(f"Running validation on {num_samples} samples at episode {current_episode}...")

        # Switch speaker to eval mode
        inner = getattr(self.speaker, "model", self.speaker)
        was_training = inner.training if hasattr(inner, "training") else True
        if hasattr(inner, "eval"):
            inner.eval()

        rewards: List[float] = []
        successes: List[bool] = []
        lengths: List[int] = []

        with torch.no_grad():
            for i in range(num_samples):
                item = self.val_dataset[i]
                if hasattr(item, "image"):
                    img = item.image
                    bbox = getattr(item, "bbox_minmax", getattr(item, "bbox", None))
                    speaker_img = getattr(item, "speaker_image", None)
                    listener_img = getattr(item, "listener_image", None)
                    sample_prompt = getattr(item, "prompt", self.prompt)
                    sample_id = getattr(item, "sample_id", f"val_{i}")
                else:
                    img = item.get("image")
                    bbox = item.get("bbox", item.get("bbox_minmax"))
                    speaker_img = item.get("speaker_image")
                    listener_img = item.get("listener_image")
                    sample_prompt = item.get("prompt", self.prompt)
                    sample_id = str(item.get("question_id", item.get("sample_id", f"val_{i}")))

                if img is None or bbox is None:
                    continue

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
                )

                rewards.append(rollout.total_reward)
                successes.append(rollout.is_success)
                lengths.append(len(rollout.tokens))

        # Restore training state
        if was_training and hasattr(inner, "train"):
            inner.train()

        val_metrics = {
            "val_reward_mean": float(sum(rewards) / len(rewards)) if rewards else 0.0,
            "val_success_rate": float(sum(1.0 if s else 0.0 for s in successes) / len(successes)) if successes else 0.0,
            "val_avg_tokens": float(sum(lengths) / len(lengths)) if lengths else 0.0,
            "val_samples_evaluated": len(rewards),
        }

        logger.info(
            f"Validation results [Ep {current_episode}]: "
            f"Reward: {val_metrics['val_reward_mean']:.3f} | "
            f"Success: {val_metrics['val_success_rate'] * 100:.1f}% | "
            f"Tokens: {val_metrics['val_avg_tokens']:.1f}"
        )

        return val_metrics
