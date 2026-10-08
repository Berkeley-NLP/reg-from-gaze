"""
Unified GazeRL composite reward module combining sparse, shaping, penalties, and alignments.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch

from configs.base import RewardConfig
from rl.rewards.base import BaseRewardFunction, RewardOutput
from rl.rewards.sparse import BFHReward, SeqAnyHitReward, SeqLPHitReward, SupervisedRECReward
from rl.rewards.shaping import DistanceShapingReward, GaussianShapingReward
from rl.rewards.penalties import compute_logit_penalties, compute_length_penalties


class GazeRLReward(BaseRewardFunction):
    """
    Unified, configurable reward function for GazeRL and baseline reference games.
    
    Supports:
    - Gaze-BFH / REC-BFH: Backward first hit credit assignment decay
    - Gaze-Shaping / REC-Shaping: Continuous distance-based delta shaping
    - Gaze-SeqAnyHit / REC-SeqAnyHit: Binary sequence hit
    - Gaze-SeqLPHit: Last-point binary hit
    - REC-Success: Supervised REC target hit
    - Gaussian-Shaping: Analytical probability mass change
    - Logit penalty (GRPO-style) and Length penalty
    - Word-to-token reward projection modes ('all', 'normalize', 'first_only', 'last_only')
    """

    def __init__(
        self,
        reward_type: str = "sparse_decay",
        gamma: float = 0.9,
        hit_reward: float = 1.0,
        miss_reward: float = 0.0,
        context_reward: float = 1.0,
        context_window: int = 100,
        shaping_scale: float = 1.0,
        shaping_clip: float = 1.0,
        shaping_reward_all_tokens: bool = False,
        use_gaussian_gaze: bool = False,
        gaussian_sigma_default: float = 10.0,
        gaussian_use_absolute_prob_mass: bool = False,
        use_length_penalty: bool = False,
        length_penalty_beta: float = 0.01,
        logit_penalty_weight: float = 0.0,
        zero_reward_for_punctuation: bool = False,
        word_reward_mode: str = "all",
        ignore_stop_words: bool = False,
        eos_bonus: float = 0.0,
        eos_reward_on_success: float = 0.0,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(debug=debug)
        self.reward_type = reward_type.lower().replace("-", "_")
        self.gamma = gamma
        self.hit_reward = hit_reward
        self.miss_reward = miss_reward
        self.context_reward = context_reward
        self.context_window = context_window
        self.shaping_scale = shaping_scale
        self.shaping_clip = shaping_clip
        self.shaping_reward_all_tokens = shaping_reward_all_tokens
        self.use_gaussian_gaze = use_gaussian_gaze or (reward_type == "gaussian_shaping")
        self.gaussian_sigma_default = gaussian_sigma_default
        self.gaussian_use_absolute_prob_mass = gaussian_use_absolute_prob_mass
        self.use_length_penalty = use_length_penalty
        self.length_penalty_beta = length_penalty_beta
        self.logit_penalty_weight = logit_penalty_weight
        self.zero_reward_for_punctuation = zero_reward_for_punctuation
        self.word_reward_mode = word_reward_mode
        self.ignore_stop_words = ignore_stop_words
        self.eos_bonus = eos_bonus
        self.eos_reward_on_success = eos_reward_on_success

        # Underlying component calculators
        self._bfh = BFHReward(
            gamma=gamma,
            hit_reward=hit_reward,
            miss_reward=miss_reward,
            context_reward=context_reward,
            context_window=context_window,
            ignore_stop_words=ignore_stop_words,
            debug=debug,
        )
        self._seq_any = SeqAnyHitReward(hit_reward=hit_reward, miss_reward=miss_reward, debug=debug)
        self._seq_lp = SeqLPHitReward(hit_reward=hit_reward, miss_reward=miss_reward, debug=debug)
        self._supervised = SupervisedRECReward(hit_reward=hit_reward, miss_reward=miss_reward, debug=debug)
        self._dist_shaping = DistanceShapingReward(
            shaping_scale=shaping_scale,
            shaping_clip=shaping_clip,
            shaping_reward_all_tokens=shaping_reward_all_tokens,
            debug=debug,
        )
        self._gauss_shaping = GaussianShapingReward(
            shaping_scale=shaping_scale,
            gaussian_sigma_default=gaussian_sigma_default,
            use_absolute_prob_mass=gaussian_use_absolute_prob_mass,
            debug=debug,
        )

    @classmethod
    def from_config(cls, config: Union[RewardConfig, Dict[str, Any]]) -> "GazeRLReward":
        if isinstance(config, RewardConfig):
            params = {
                "reward_type": config.reward_type,
                "gamma": config.gamma,
                "hit_reward": config.hit_reward,
                "miss_reward": config.miss_reward,
                "eos_bonus": config.eos_bonus,
                "shaping_scale": config.shaping_scale,
                "use_length_penalty": config.use_length_penalty,
                "length_penalty_beta": config.length_penalty_beta,
                "logit_penalty_weight": config.logit_penalty_weight,
                "gaussian_sigma_default": config.gaussian_sigma_default,
                "zero_reward_for_punctuation": config.zero_reward_for_punctuation,
                "word_reward_mode": config.word_reward_mode,
                "ignore_stop_words": config.ignore_stop_words,
                "eos_reward_on_success": config.eos_reward_on_success,
                "shaping_reward_all_tokens": config.shaping_reward_all_tokens,
                "gaussian_use_absolute_prob_mass": config.gaussian_use_absolute_prob_mass,
            }
        else:
            params = dict(config)
        return cls(**params)

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

        # 1. Dispatch primary task reward calculation
        sparse_out: Optional[RewardOutput] = None
        shaping_out: Optional[RewardOutput] = None

        if self.reward_type in ("sparse_decay", "gaze_bfh", "rec_bfh", "sparse", "bfh", "before_first_hit", "gaze_before_first_hit", "rec_before_first_hit"):
            sparse_out = self._bfh.compute_rewards(gaze_points, bbox, tokens=tokens)
        elif self.reward_type in ("binary", "gaze_seq_any_hit", "rec_seq_any_hit", "seq_any_hit", "seqanyhit"):
            sparse_out = self._seq_any.compute_rewards(gaze_points, bbox, tokens=tokens)
        elif self.reward_type in ("binary_last_point", "gaze_seq_lp_hit", "rec_seq_lp_hit", "seq_lp_hit", "seq_lphit", "seqlphit"):
            sparse_out = self._seq_lp.compute_rewards(gaze_points, bbox, tokens=tokens)
        elif self.reward_type in ("supervised", "rec_success", "rec_supervised", "success"):
            sparse_out = self._supervised.compute_rewards(gaze_points, bbox, tokens=tokens)
        elif self.reward_type in ("distance_shaping", "gaze_shaping", "rec_shaping", "shaping"):
            shaping_out = self._dist_shaping.compute_rewards(gaze_points, bbox, tokens=tokens)
        elif self.reward_type in ("gaussian_shaping", "gaussian"):
            shaping_out = self._gauss_shaping.compute_rewards(gaze_points, bbox, tokens=tokens)
        elif self.reward_type in ("sparse_shaping", "combined"):
            sparse_out = self._bfh.compute_rewards(gaze_points, bbox, tokens=tokens)
            shaping_out = self._dist_shaping.compute_rewards(gaze_points, bbox, tokens=tokens)
        else:
            # Default fallback to BFH
            sparse_out = self._bfh.compute_rewards(gaze_points, bbox, tokens=tokens)

        # Merge primary reward values
        base_rewards = [0.0] * num_tokens
        is_success = False
        hit_idx = None
        first_hit_token = None

        if sparse_out is not None:
            is_success = sparse_out.is_success
            hit_idx = sparse_out.hit_idx
            first_hit_token = sparse_out.first_hit_token
            for i, r in enumerate(sparse_out.rewards):
                if i < num_tokens:
                    base_rewards[i] += r

        if shaping_out is not None:
            if not is_success and shaping_out.is_success:
                is_success = True
                hit_idx = shaping_out.hit_idx
            for i, r in enumerate(shaping_out.rewards):
                if i < num_tokens:
                    base_rewards[i] += r

        # 2. Add penalties
        logit_pens = compute_logit_penalties(logits, lambda_weight=self.logit_penalty_weight)
        len_pens = compute_length_penalties(tokens, beta=self.length_penalty_beta) if self.use_length_penalty else [0.0] * num_tokens

        # 3. Word-level reward mapping if specified
        if word_to_tokens and len(word_to_tokens) > 0 and tokens:
            aligned_rewards = [0.0] * num_tokens
            for w_idx, token_indices in enumerate(word_to_tokens):
                if not token_indices:
                    continue
                # Pick word reward from the first token of the word
                word_r = base_rewards[token_indices[0]] if token_indices[0] < len(base_rewards) else 0.0
                num_tok_in_w = len(token_indices)
                for t_pos, t_idx in enumerate(token_indices):
                    if t_idx >= num_tokens:
                        continue
                    if self.word_reward_mode == "normalize" and num_tok_in_w > 0:
                        aligned_rewards[t_idx] = word_r / num_tok_in_w
                    elif self.word_reward_mode == "first_only":
                        aligned_rewards[t_idx] = word_r if t_pos == 0 else 0.0
                    elif self.word_reward_mode == "last_only":
                        aligned_rewards[t_idx] = word_r if t_pos == (num_tok_in_w - 1) else 0.0
                    else:  # 'all'
                        aligned_rewards[t_idx] = word_r
            base_rewards = aligned_rewards

        # 4. Zero punctuation tokens if enabled
        if self.zero_reward_for_punctuation and tokens:
            for i in range(num_tokens):
                if i < len(tokens) and self.is_punctuation_token(tokens[i]):
                    base_rewards[i] = 0.0

        # 5. Assemble final rewards and components
        final_rewards: List[float] = []
        components: List[Dict[str, Any]] = []

        eos_tokens = {'<eos>', '</s>', '<|endoftext|>', 'eos'}
        has_eos = any(tok.strip().lower() in eos_tokens for tok in tokens) if tokens else False

        for i in range(num_tokens):
            r = base_rewards[i]

            # Length penalty subtraction
            lp = len_pens[i] if i < len(len_pens) else 0.0
            r -= lp

            # Logit penalty addition (logit_pens are negative)
            log_p = logit_pens[i] if i < len(logit_pens) else 0.0
            r += log_p

            # EOS completion bonus on last token if EOS reached
            eos_b = 0.0
            if i == num_tokens - 1 and has_eos and self.eos_bonus > 0.0:
                eos_b = self.eos_bonus
                r += eos_b

            # EOS reward on success
            eos_succ = 0.0
            if i == num_tokens - 1 and is_success and self.eos_reward_on_success > 0.0:
                eos_succ = self.eos_reward_on_success
                r += eos_succ

            final_rewards.append(float(r))

            gp = gaze_points[i + 1] if (gaze_points and i + 1 < len(gaze_points)) else None
            in_bbox = bool(gp is not None and bbox is not None and self.is_in_bbox(gp, bbox))
            dist = self.distance_to_bbox(gp, bbox) if (gp is not None and bbox is not None) else None

            components.append({
                "total_reward": float(r),
                "base_reward": float(base_rewards[i]),
                "length_penalty": float(lp),
                "logit_penalty": float(log_p),
                "eos_bonus": float(eos_b + eos_succ),
                "in_bbox": in_bbox,
                "distance": dist,
                "token_idx": i,
                "hit_idx": hit_idx,
            })

        return RewardOutput(
            rewards=final_rewards,
            components=components,
            total_reward=float(sum(final_rewards)),
            hit_idx=hit_idx,
            is_success=is_success,
            first_hit_token=first_hit_token,
        )
