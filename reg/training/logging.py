"""
Logging utilities: console progress formatting, JSONL metric logging, and WandB integration.
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class MetricLogger:
    """
    Unified logger for training progress, recording metrics locally to JSONL and optionally to WandB.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        use_wandb: bool = False,
        wandb_project: str = "reg-from-gaze",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.use_wandb = use_wandb

        # Initialize WandB if enabled
        if self.use_wandb:
            try:
                import wandb
                config_dict = config.to_dict() if hasattr(config, "to_dict") else config
                resolved_entity = wandb_entity or os.environ.get("WANDB_ENTITY", None)
                init_kwargs = {
                    "project": wandb_project,
                    "name": wandb_run_name,
                    "config": config_dict,
                    "dir": str(self.output_dir),
                }
                if resolved_entity:
                    init_kwargs["entity"] = resolved_entity
                wandb.init(**init_kwargs)
                logger.info(f"Initialized WandB project '{wandb_project}', run '{wandb_run_name}' (entity: {resolved_entity}).")
            except Exception as e:
                logger.warning(f"Could not initialize WandB: {e}; falling back to local logging.")
                self.use_wandb = False

    def log_metrics(self, step: int, episode: int, metrics: Dict[str, Any]):
        """
        Log metrics dictionary with timestamp to JSONL and WandB.
        """
        record = {
            "step": step,
            "episode": episode,
            "timestamp": time.time(),
            **metrics,
        }

        # Write to JSONL
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

        # Log to WandB
        if self.use_wandb:
            try:
                import wandb
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"Failed to log to WandB: {e}")

    @staticmethod
    def format_status(
        step: int,
        episode: int,
        total_episodes: int,
        loss: float,
        reward: float,
        success_rate: float,
        avg_tokens: float = 0.0,
        kl: float = 0.0,
    ) -> str:
        """
        Format clean console log string.
        """
        pct = (episode / total_episodes * 100) if total_episodes > 0 else 0.0
        return (
            f"[Step {step:05d} | Ep {episode:06d}/{total_episodes} ({pct:4.1f}%)] "
            f"Loss: {loss:7.4f} | "
            f"Reward: {reward:6.3f} | "
            f"Success: {success_rate * 100:5.1f}% | "
            f"Tokens: {avg_tokens:4.1f} | "
            f"KL: {kl:6.4f}"
        )

    def finish(self):
        """Clean up and close WandB run if active."""
        if self.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
