"""
Reward functions and factory for GazeRL.
"""

from typing import Any, Dict, Union

from configs.base import RewardConfig
from rl.rewards.base import BaseRewardFunction, RewardOutput
from rl.rewards.sparse import BFHReward, SeqAnyHitReward, SeqLPHitReward, SupervisedRECReward
from rl.rewards.shaping import DistanceShapingReward, GaussianShapingReward
from rl.rewards.composite import GazeRLReward


def get_reward_function(config: Union[RewardConfig, Dict[str, Any], str] = "sparse_decay", **kwargs) -> GazeRLReward:
    """
    Factory function to instantiate the appropriate reward function.

    Args:
        config: RewardConfig instance, config dict, or reward type string.
        **kwargs: Overrides for configuration parameters.

    Returns:
        Instantiated GazeRLReward instance.
    """
    if isinstance(config, str):
        cfg = RewardConfig(reward_type=config, **kwargs)
        return GazeRLReward.from_config(cfg)
    elif isinstance(config, dict):
        merged = dict(config)
        merged.update(kwargs)
        return GazeRLReward(**merged)
    elif isinstance(config, RewardConfig):
        return GazeRLReward.from_config(config)
    else:
        raise ValueError(f"Unsupported reward configuration type: {type(config)}")


__all__ = [
    "BaseRewardFunction",
    "RewardOutput",
    "BFHReward",
    "SeqAnyHitReward",
    "SeqLPHitReward",
    "SupervisedRECReward",
    "DistanceShapingReward",
    "GaussianShapingReward",
    "GazeRLReward",
    "get_reward_function",
]
