"""
Rollout collection, trajectory representation, and batch collation for RL training.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from PIL import Image
import torch

from models.base import BaseSpeaker, BaseListener, SpeakerOutput, ListenerOutput
from rl.rewards.base import BaseRewardFunction, RewardOutput
from data.processing import (
    create_speaker_bbox_overlay,
    create_listener_padded_image,
    normalize_bbox_to_100,
)


@dataclass
class RolloutSample:
    """A single rollout trajectory produced by speaker and evaluated by listener."""
    sample_id: str
    tokens: List[str]
    token_ids: List[int]
    generated_text: str
    bbox_100: Tuple[float, float, float, float]
    gaze_points: List[Optional[Sequence[float]]]
    rewards: List[float]
    reward_components: List[Dict[str, Any]]
    total_reward: float
    is_success: bool
    hit_idx: Optional[int] = None
    log_probs: List[torch.Tensor] = field(default_factory=list)
    entropies: Optional[List[torch.Tensor]] = None
    ref_log_probs: Optional[List[Optional[torch.Tensor]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutBatch:
    """A collection of rollout trajectories ready for policy gradient updates."""
    samples: List[RolloutSample]

    @property
    def size(self) -> int:
        return len(self.samples)

    @property
    def total_tokens(self) -> int:
        return sum(len(s.tokens) for s in self.samples)

    @property
    def mean_reward(self) -> float:
        if not self.samples:
            return 0.0
        return float(sum(s.total_reward for s in self.samples) / len(self.samples))

    @property
    def success_rate(self) -> float:
        if not self.samples:
            return 0.0
        return float(sum(1.0 if s.is_success else 0.0 for s in self.samples) / len(self.samples))

    def all_log_probs(self) -> List[torch.Tensor]:
        result = []
        for s in self.samples:
            result.extend(s.log_probs)
        return result

    def all_rewards(self) -> List[float]:
        result = []
        for s in self.samples:
            result.extend(s.rewards)
        return result

    def all_entropies(self) -> Optional[List[torch.Tensor]]:
        result = []
        has_any = False
        for s in self.samples:
            if s.entropies:
                result.extend(s.entropies)
                has_any = True
            else:
                result.extend([torch.tensor(0.0)] * len(s.tokens))
        return result if has_any else None

    def all_ref_log_probs(self) -> Optional[List[Optional[torch.Tensor]]]:
        result = []
        has_any = False
        for s in self.samples:
            if s.ref_log_probs:
                result.extend(s.ref_log_probs)
                has_any = True
            else:
                result.extend([None] * len(s.tokens))
        return result if has_any else None


def collect_single_rollout(
    speaker: BaseSpeaker,
    listener: BaseListener,
    reward_fn: BaseRewardFunction,
    image: Image.Image,
    bbox_orig: Tuple[float, float, float, float],
    prompt: str,
    sample_id: str = "sample",
    speaker_image: Optional[Image.Image] = None,
    listener_image: Optional[Image.Image] = None,
    ref_speaker: Optional[BaseSpeaker] = None,
) -> RolloutSample:
    """
    Execute a full rollout:
      1. Prepare images (speaker with red box overlay, listener letterboxed).
      2. Speaker generates referring expression, yielding tokens and log_probs.
      3. Listener predicts gaze path / target from image + expression.
      4. Reward function evaluates gaze trajectory against target bounding box.
      5. Reference speaker optionally computes baseline log_probs for KL penalty.

    Args:
        speaker: Active policy speaker model.
        listener: Comprehension listener model.
        reward_fn: Reward function instance.
        image: Original PIL Image.
        bbox_orig: Target bounding box (x1, y1, x2, y2) in original image pixels.
        prompt: Task instruction prompt string.
        sample_id: Identifier string for tracking.
        speaker_image: Pre-rendered speaker image (if cached).
        listener_image: Pre-rendered listener image (if cached).
        ref_speaker: Optional reference policy for KL divergence.

    Returns:
        RolloutSample containing all trajectory elements and rewards.
    """
    orig_w, orig_h = image.size

    # 1. Image preparation
    if speaker_image is None:
        speaker_image = create_speaker_bbox_overlay(image, bbox_orig)
    if listener_image is None:
        listener_image = create_listener_padded_image(image)

    # 2. Convert target bbox to 0-100 normalized coordinate space
    bbox_100 = normalize_bbox_to_100(bbox_orig, (orig_w, orig_h))

    # 3. Speaker rollout
    speaker_out: SpeakerOutput = speaker.generate(speaker_image, prompt)

    # 4. Listener prediction
    # Listener expects list of tokens; returns sequence of coordinates
    listener_out: ListenerOutput = listener.predict(
        images=[listener_image],
        tokens_list=[speaker_out.tokens],
        bboxes=[bbox_100],
    )[0]

    # 5. Compute rewards
    reward_out: RewardOutput = reward_fn.compute_rewards(
        gaze_points=listener_out.coordinates,
        bbox=bbox_100,
        tokens=speaker_out.tokens,
        logits=speaker_out.logits,
    )

    # 6. Reference log_probs for KL divergence (if ref_speaker is provided)
    ref_logps: Optional[List[Optional[torch.Tensor]]] = None
    if ref_speaker is not None and len(speaker_out.token_ids) > 0:
        with torch.no_grad():
            ref_logps = ref_speaker.compute_log_probs(
                image=speaker_image,
                prompt=prompt,
                token_ids=speaker_out.token_ids,
            )

    return RolloutSample(
        sample_id=sample_id,
        tokens=speaker_out.tokens,
        token_ids=speaker_out.token_ids,
        generated_text=speaker_out.text,
        bbox_100=bbox_100,
        gaze_points=listener_out.coordinates,
        rewards=reward_out.rewards,
        reward_components=reward_out.components,
        total_reward=reward_out.total_reward,
        is_success=reward_out.is_success,
        hit_idx=reward_out.hit_idx,
        log_probs=speaker_out.log_probs,
        entropies=speaker_out.entropies,
        ref_log_probs=ref_logps,
        metadata={
            "first_hit_token": reward_out.first_hit_token,
            "orig_image_size": (orig_w, orig_h),
        },
    )


def collate_rollouts(rollouts: Sequence[RolloutSample]) -> RolloutBatch:
    """Collate a sequence of RolloutSamples into a RolloutBatch."""
    return RolloutBatch(samples=list(rollouts))
