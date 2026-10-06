"""
Reinforcement Learning (RL) package for GazeRL.

Exports modular reward calculators, policy gradient loss functions, and rollout buffers.
"""

from rl.rewards import (
    BaseRewardFunction,
    RewardOutput,
    BFHReward,
    SeqAnyHitReward,
    SeqLPHitReward,
    SupervisedRECReward,
    DistanceShapingReward,
    GaussianShapingReward,
    GazeRLReward,
    get_reward_function,
)
from rl.loss import (
    calculate_policy_loss,
    standardize_advantages_batch,
    normalize_rewards_batch,
    calculate_ema_baseline,
)
from rl.rollouts import (
    RolloutSample,
    RolloutBatch,
    collect_single_rollout,
    collate_rollouts,
)

__all__ = [
    # Rewards
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
    # Loss
    "calculate_policy_loss",
    "standardize_advantages_batch",
    "normalize_rewards_batch",
    "calculate_ema_baseline",
    # Rollouts
    "RolloutSample",
    "RolloutBatch",
    "collect_single_rollout",
    "collate_rollouts",
]
