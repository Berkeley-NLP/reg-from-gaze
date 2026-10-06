"""
Sparse reward implementations: Backward First Hit (BFH), Sequence Any Hit, Sequence Last Point Hit, and Supervised REC.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch

from rl.rewards.base import BaseRewardFunction, RewardOutput


class BFHReward(BaseRewardFunction):
    """
    Gaze-BFH / REC-BFH (Backward First Hit) Reward.
    
    Assigns credit backwards from the first token where the listener's gaze
    enters the target bounding box:
      R_{t^*} = hit_reward
      R_t = gamma^{t^* - t} * context_reward   (for t < t^*)
      R_t = 0.0                               (for t > t^*)
    """

    def __init__(
        self,
        gamma: float = 0.9,
        hit_reward: float = 1.0,
        miss_reward: float = 0.0,
        context_reward: float = 1.0,
        context_window: int = 100,
        ignore_stop_words: bool = False,
        last_hit: bool = False,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.gamma = gamma
        self.hit_reward = hit_reward
        self.miss_reward = miss_reward
        self.context_reward = context_reward
        self.context_window = context_window
        self.ignore_stop_words = ignore_stop_words
        self.last_hit = last_hit

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

        # Gaze point index 0 is BOS; index 1..num_tokens are token gaze points
        max_idx = min(len(gaze_points) - 1, num_tokens)
        hit_idx: Optional[int] = None
        first_hit_token: Optional[str] = None

        if bbox is not None:
            if self.last_hit:
                # Search backwards for the last hit
                for i in range(max_idx, 0, -1):
                    gp = gaze_points[i]
                    if gp is not None and self.is_in_bbox(gp, bbox):
                        hit_idx = i
                        if tokens and i - 1 < len(tokens):
                            first_hit_token = tokens[i - 1]
                        break
            else:
                # Search forwards for the first hit
                for i in range(1, max_idx + 1):
                    gp = gaze_points[i]
                    if gp is not None and self.is_in_bbox(gp, bbox):
                        if self.ignore_stop_words and tokens and i - 1 < len(tokens):
                            if self.is_stop_word(tokens[i - 1]):
                                continue
                        hit_idx = i
                        if tokens and i - 1 < len(tokens):
                            first_hit_token = tokens[i - 1]
                        break

        rewards: List[float] = []
        components: List[Dict[str, Any]] = []

        for i in range(1, num_tokens + 1):
            gp = gaze_points[i] if i < len(gaze_points) else None
            in_bbox = bool(gp is not None and bbox is not None and self.is_in_bbox(gp, bbox))
            dist = self.distance_to_bbox(gp, bbox) if (gp is not None and bbox is not None) else None

            if hit_idx is None:
                r = self.miss_reward
            elif i == hit_idx:
                r = self.hit_reward
            elif i < hit_idx:
                steps_back = hit_idx - i
                if steps_back <= self.context_window:
                    r = (self.gamma ** steps_back) * self.context_reward
                else:
                    r = 0.0
            else:
                r = 0.0

            rewards.append(float(r))
            components.append({
                "sparse_reward": float(r),
                "total_reward": float(r),
                "in_bbox": in_bbox,
                "distance": dist,
                "token_idx": i - 1,
                "hit_idx": (hit_idx - 1) if hit_idx is not None else None,
            })

        return RewardOutput(
            rewards=rewards,
            components=components,
            total_reward=float(sum(rewards)),
            hit_idx=(hit_idx - 1) if hit_idx is not None else None,
            is_success=(hit_idx is not None),
            first_hit_token=first_hit_token,
        )


class SeqAnyHitReward(BaseRewardFunction):
    """
    Gaze-SeqAnyHit / REC-SeqAnyHit Reward.
    
    If the listener's gaze ever enters the bounding box during the sequence,
    all tokens receive a uniform reward of 1.0; otherwise 0.0.
    """

    def __init__(
        self,
        hit_reward: float = 1.0,
        miss_reward: float = 0.0,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.hit_reward = hit_reward
        self.miss_reward = miss_reward

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

        hit_found = False
        hit_idx: Optional[int] = None
        first_hit_token: Optional[str] = None

        if bbox is not None and gaze_points is not None:
            for i in range(1, min(len(gaze_points), num_tokens + 1)):
                gp = gaze_points[i]
                if gp is not None and self.is_in_bbox(gp, bbox):
                    hit_found = True
                    hit_idx = i - 1
                    if tokens and hit_idx < len(tokens):
                        first_hit_token = tokens[hit_idx]
                    break

        reward_val = self.hit_reward if hit_found else self.miss_reward
        rewards = [float(reward_val)] * num_tokens
        components = []

        for i in range(1, num_tokens + 1):
            gp = gaze_points[i] if (gaze_points and i < len(gaze_points)) else None
            in_bbox = bool(gp is not None and bbox is not None and self.is_in_bbox(gp, bbox))
            dist = self.distance_to_bbox(gp, bbox) if (gp is not None and bbox is not None) else None

            components.append({
                "sparse_reward": float(reward_val),
                "total_reward": float(reward_val),
                "in_bbox": in_bbox,
                "distance": dist,
                "token_idx": i - 1,
            })

        return RewardOutput(
            rewards=rewards,
            components=components,
            total_reward=float(sum(rewards)),
            hit_idx=hit_idx,
            is_success=hit_found,
            first_hit_token=first_hit_token,
        )


class SeqLPHitReward(BaseRewardFunction):
    """
    Gaze-SeqLPHit Reward.
    
    Checks if the last non-null gaze point in the sequence hits the target bbox.
    If yes, all tokens receive 1.0; otherwise 0.0.
    """

    def __init__(
        self,
        hit_reward: float = 1.0,
        miss_reward: float = 0.0,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.hit_reward = hit_reward
        self.miss_reward = miss_reward

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

        last_hit = False
        last_idx: Optional[int] = None

        if bbox is not None and gaze_points is not None:
            max_idx = min(len(gaze_points) - 1, num_tokens)
            for i in range(max_idx, 0, -1):
                gp = gaze_points[i]
                if gp is not None:
                    last_idx = i - 1
                    last_hit = self.is_in_bbox(gp, bbox)
                    break

        reward_val = self.hit_reward if last_hit else self.miss_reward
        rewards = [float(reward_val)] * num_tokens
        components = []

        for i in range(1, num_tokens + 1):
            gp = gaze_points[i] if (gaze_points and i < len(gaze_points)) else None
            in_bbox = bool(gp is not None and bbox is not None and self.is_in_bbox(gp, bbox))
            dist = self.distance_to_bbox(gp, bbox) if (gp is not None and bbox is not None) else None

            components.append({
                "sparse_reward": float(reward_val),
                "total_reward": float(reward_val),
                "in_bbox": in_bbox,
                "distance": dist,
                "token_idx": i - 1,
            })

        return RewardOutput(
            rewards=rewards,
            components=components,
            total_reward=float(sum(rewards)),
            hit_idx=last_idx if last_hit else None,
            is_success=last_hit,
            first_hit_token=tokens[last_idx] if (last_hit and tokens and last_idx is not None and last_idx < len(tokens)) else None,
        )


class SupervisedRECReward(BaseRewardFunction):
    """
    REC-Success (Supervised REC Baseline) Reward.
    
    All tokens receive 1.0 if the predicted single-point REC coordinate falls
    inside the target bounding box; otherwise 0.0.
    """

    def __init__(
        self,
        hit_reward: float = 1.0,
        miss_reward: float = 0.0,
        debug: bool = False,
    ):
        super().__init__(debug=debug)
        self.hit_reward = hit_reward
        self.miss_reward = miss_reward

    def compute_rewards(
        self,
        gaze_points: Sequence[Optional[Sequence[float]]],
        bbox: Optional[Tuple[float, float, float, float]],
        tokens: Optional[Sequence[str]] = None,
        word_to_tokens: Optional[Sequence[Sequence[int]]] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> RewardOutput:
        num_tokens = len(tokens) if tokens is not None else (len(gaze_points) if gaze_points else 0)
        if num_tokens == 0:
            return RewardOutput(rewards=[], components=[], total_reward=0.0, is_success=False)

        hit = False
        pred_pt = None
        if gaze_points and len(gaze_points) > 0:
            # Look at the last available point (or point 0 for single-point prediction)
            for pt in reversed(gaze_points):
                if pt is not None:
                    pred_pt = pt
                    break

        if pred_pt is not None and bbox is not None:
            hit = self.is_in_bbox(pred_pt, bbox)

        reward_val = self.hit_reward if hit else self.miss_reward
        rewards = [float(reward_val)] * num_tokens
        components = [{
            "sparse_reward": float(reward_val),
            "total_reward": float(reward_val),
            "in_bbox": hit,
            "distance": self.distance_to_bbox(pred_pt, bbox) if pred_pt and bbox else None,
            "token_idx": i,
        } for i in range(num_tokens)]

        return RewardOutput(
            rewards=rewards,
            components=components,
            total_reward=float(sum(rewards)),
            hit_idx=0 if hit else None,
            is_success=hit,
        )
