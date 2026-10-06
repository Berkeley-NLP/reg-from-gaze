"""
Shaping reward implementations: Continuous Distance Shaping and Gaussian Probability Mass Shaping.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch

from rl.rewards.base import BaseRewardFunction, RewardOutput
from data.processing import calculate_gaussian_bbox_probability


class DistanceShapingReward(BaseRewardFunction):
    """
    Gaze-Shaping / REC-Shaping Reward.
    
    Continuous step-wise progress towards the target bounding box:
      Delta_t = (d_{t-1} - d_t) / max(d_0, 1e-6)
      R_{shaping, t} = scale * clip(Delta_t, -clip_val, clip_val)
    """

    def __init__(
        self,
        shaping_scale: float = 1.0,
        shaping_clip: float = 1.0,
        shaping_reward_all_tokens: bool = False,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.shaping_scale = shaping_scale
        self.shaping_clip = shaping_clip
        self.shaping_reward_all_tokens = shaping_reward_all_tokens

    def compute_rewards(
        self,
        gaze_points: Sequence[Optional[Sequence[float]]],
        bbox: Optional[Tuple[float, float, float, float]],
        tokens: Optional[Sequence[str]] = None,
        word_to_tokens: Optional[Sequence[Sequence[int]]] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> RewardOutput:
        num_tokens = len(tokens) if tokens is not None else max(0, len(gaze_points) - 1)
        if num_tokens == 0:
            return RewardOutput(rewards=[], components=[], total_reward=0.0, is_success=False)

        if gaze_points is None or len(gaze_points) == 0:
            gaze_points = [None] * (num_tokens + 1)

        # Initial point (BOS)
        p0 = gaze_points[0] if len(gaze_points) > 0 else None
        d0 = self.distance_to_bbox(p0, bbox) if (p0 is not None and bbox is not None) else None
        prev_dist = d0

        # Find first hit index to gate shaping if needed
        hit_idx: Optional[int] = None
        if bbox is not None:
            for i in range(1, min(len(gaze_points), num_tokens + 1)):
                gp = gaze_points[i]
                if gp is not None and self.is_in_bbox(gp, bbox):
                    hit_idx = i
                    break

        rewards: List[float] = []
        components: List[Dict[str, Any]] = []
        EPS = 1e-6

        for i in range(1, num_tokens + 1):
            gp = gaze_points[i] if i < len(gaze_points) else None
            in_bbox = bool(gp is not None and bbox is not None and self.is_in_bbox(gp, bbox))
            cur_dist = self.distance_to_bbox(gp, bbox) if (gp is not None and bbox is not None) else None

            allow_shaping = self.shaping_reward_all_tokens or (hit_idx is None) or (i <= hit_idx)

            shaping_r = 0.0
            raw_delta = 0.0

            if allow_shaping and cur_dist is not None and prev_dist is not None and d0 is not None and d0 > EPS:
                raw_delta = (prev_dist - cur_dist) / d0
                clipped_delta = max(-self.shaping_clip, min(self.shaping_clip, raw_delta))
                shaping_r = float(self.shaping_scale * clipped_delta)

            if cur_dist is not None:
                prev_dist = cur_dist

            rewards.append(shaping_r)
            components.append({
                "shaping_reward": shaping_r,
                "raw_shaping_reward": raw_delta,
                "total_reward": shaping_r,
                "in_bbox": in_bbox,
                "distance": cur_dist,
                "initial_distance": d0,
                "token_idx": i - 1,
            })

        return RewardOutput(
            rewards=rewards,
            components=components,
            total_reward=float(sum(rewards)),
            hit_idx=(hit_idx - 1) if hit_idx is not None else None,
            is_success=(hit_idx is not None),
        )


class GaussianShapingReward(BaseRewardFunction):
    """
    Gaussian Probability Mass Shaping Reward.
    
    Computes change in probability mass of a 2D Gaussian gaze prediction
    inside the target bounding box:
      Phi_t = P(Gaze_t in bbox)
      Delta_t = Phi_t - Phi_{t-1}
      R_{gaussian, t} = scale * Delta_t
    """

    def __init__(
        self,
        shaping_scale: float = 1.0,
        gaussian_sigma_default: float = 10.0,
        use_absolute_prob_mass: bool = False,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.shaping_scale = shaping_scale
        self.gaussian_sigma_default = gaussian_sigma_default
        self.use_absolute_prob_mass = use_absolute_prob_mass

    def compute_rewards(
        self,
        gaze_points: Sequence[Optional[Sequence[float]]],
        bbox: Optional[Tuple[float, float, float, float]],
        tokens: Optional[Sequence[str]] = None,
        word_to_tokens: Optional[Sequence[Sequence[int]]] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> RewardOutput:
        num_tokens = len(tokens) if tokens is not None else max(0, len(gaze_points) - 1)
        if num_tokens == 0:
            return RewardOutput(rewards=[], components=[], total_reward=0.0, is_success=False)

        if gaze_points is None or len(gaze_points) == 0:
            gaze_points = [None] * (num_tokens + 1)

        # Initial probability mass (BOS)
        p0 = gaze_points[0] if len(gaze_points) > 0 else None
        prev_prob = 0.0
        if p0 is not None and bbox is not None:
            sigma0 = p0[2] if len(p0) >= 3 else self.gaussian_sigma_default
            prev_prob = calculate_gaussian_bbox_probability(p0[0], p0[1], sigma0, bbox)

        rewards: List[float] = []
        components: List[Dict[str, Any]] = []
        hit_found = False
        hit_idx: Optional[int] = None

        for i in range(1, num_tokens + 1):
            gp = gaze_points[i] if i < len(gaze_points) else None
            cur_prob = 0.0
            in_bbox = False

            if gp is not None and bbox is not None:
                sigma = gp[2] if len(gp) >= 3 else self.gaussian_sigma_default
                cur_prob = calculate_gaussian_bbox_probability(gp[0], gp[1], sigma, bbox)
                in_bbox = bool(cur_prob >= 0.5 or self.is_in_bbox(gp, bbox))

            if in_bbox and not hit_found:
                hit_found = True
                hit_idx = i - 1

            if self.use_absolute_prob_mass:
                r = float(self.shaping_scale * cur_prob)
            else:
                delta = cur_prob - prev_prob
                r = float(self.shaping_scale * delta)

            prev_prob = cur_prob
            rewards.append(r)
            components.append({
                "gaussian_shaping_reward": r,
                "prob_mass": cur_prob,
                "total_reward": r,
                "in_bbox": in_bbox,
                "token_idx": i - 1,
            })

        return RewardOutput(
            rewards=rewards,
            components=components,
            total_reward=float(sum(rewards)),
            hit_idx=hit_idx,
            is_success=hit_found,
        )
