"""
Base interfaces and data structures for GazeRL reward functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from data.processing import calculate_distance, is_point_in_bbox, calculate_gaussian_bbox_probability


@dataclass
class RewardOutput:
    """Standardized output container for RL reward computation."""
    rewards: List[float]
    components: List[Dict[str, Any]]
    total_reward: float = 0.0
    hit_idx: Optional[int] = None
    is_success: bool = False
    first_hit_token: Optional[str] = None

    def __post_init__(self):
        if self.rewards and self.total_reward == 0.0:
            self.total_reward = float(sum(self.rewards))


class BaseRewardFunction(ABC):
    """
    Abstract base class for referring expression RL reward functions.
    """

    PUNCTUATION_CHARS = frozenset('.,!?;:\'"-–—()[]{}/\\')
    STOP_WORDS = frozenset({
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'and', 'or', 'that', 'this',
        'it', 'its', 'there', 'here', 'which', 'who', 'whom'
    })

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.current_step = 0
        self._episode_initial_distance: Optional[float] = None

    @abstractmethod
    def compute_rewards(
        self,
        gaze_points: Sequence[Optional[Sequence[float]]],
        bbox: Optional[Tuple[float, float, float, float]],
        tokens: Optional[Sequence[str]] = None,
        word_to_tokens: Optional[Sequence[Sequence[int]]] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> RewardOutput:
        """
        Compute token-level rewards and breakdown components.

        Args:
            gaze_points: Gaze coordinates [BOS, tok_1, tok_2, ...].
                         Coordinates are in 0-100 normalized space.
            bbox: Target bounding box (x1, y1, x2, y2) in 0-100 space.
            tokens: Sequence of generated token strings.
            word_to_tokens: Optional mapping from word indices to token indices.
            logits: Optional tensor of logits of shape [num_tokens, vocab_size].

        Returns:
            RewardOutput containing token rewards and component breakdowns.
        """
        pass

    def __call__(
        self,
        gaze_points: Sequence[Optional[Sequence[float]]],
        bbox: Optional[Tuple[float, float, float, float]],
        tokens: Optional[Sequence[str]] = None,
        word_to_tokens: Optional[Sequence[Sequence[int]]] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> RewardOutput:
        return self.compute_rewards(
            gaze_points=gaze_points,
            bbox=bbox,
            tokens=tokens,
            word_to_tokens=word_to_tokens,
            logits=logits,
        )

    def calculate_comprehensive_reward(
        self,
        gaze_points: Sequence[Optional[Sequence[float]]],
        bbox: Optional[Tuple[float, float, float, float]],
        sentence_length: int = 0,
        target_length: Optional[int] = None,
        image_size: Tuple[int, int] = (400, 300),
        tokens: Optional[Sequence[str]] = None,
        word_to_tokens: Optional[Sequence[Sequence[int]]] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Backward-compatible method returning (rewards, reward_components)."""
        output = self.compute_rewards(
            gaze_points=gaze_points,
            bbox=bbox,
            tokens=tokens,
            word_to_tokens=word_to_tokens,
            logits=logits,
        )
        return output.rewards, output.components

    def update_step(self, step: int):
        self.current_step = int(step)

    def reset_episode(self):
        self._episode_initial_distance = None

    def distance_to_bbox(
        self,
        point: Optional[Sequence[float]],
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> float:
        """
        Euclidean distance from point (x, y) to the nearest boundary of bbox (x1, y1, x2, y2).
        Returns 0.0 if point is strictly inside the box.
        """
        if point is None or bbox is None:
            return float("inf")
        x, y = point[0], point[1]
        x1, y1, x2, y2 = bbox
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        cx = min(max(x, min_x), max_x)
        cy = min(max(y, min_y), max_y)
        return float(calculate_distance((x, y), (cx, cy)))

    def is_in_bbox(
        self,
        point: Optional[Sequence[float]],
        bbox: Optional[Tuple[float, float, float, float]],
    ) -> bool:
        """Check if 2D coordinate is inside bounding box."""
        if point is None or bbox is None:
            return False
        return is_point_in_bbox((point[0], point[1]), bbox)

    def is_punctuation_token(self, token: Optional[str]) -> bool:
        """Check if a token consists solely of punctuation."""
        if not token or not isinstance(token, str):
            return False
        stripped = token.strip()
        if not stripped:
            return False
        return all(c in self.PUNCTUATION_CHARS or c.isspace() for c in stripped)

    def is_stop_word(self, token: Optional[str]) -> bool:
        """Check if a token is a common English stop word."""
        if not token or not isinstance(token, str):
            return False
        cleaned = re.sub(r'[^\w]', '', token.lower())
        return cleaned in self.STOP_WORDS
