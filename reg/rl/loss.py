"""
RL policy gradient loss functions, advantage standardization, and KL divergence penalties.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch


def normalize_rewards_batch(rewards: Sequence[float], reward_normalize: bool = True) -> List[float]:
    """
    Normalize scalar rewards using batch-wise standardization: (R - mean) / (std + eps).
    """
    if not reward_normalize or len(rewards) <= 1:
        return list(rewards)

    arr = np.array(rewards, dtype=np.float32)
    mean = np.mean(arr)
    std = np.std(arr, ddof=0)
    eps = 1e-8
    if std < eps:
        return list(rewards)

    return ((arr - mean) / (std + eps)).tolist()


def standardize_advantages_batch(advantages: torch.Tensor) -> torch.Tensor:
    """
    Standardize advantages across a batch: (A - mean) / (std + eps).
    """
    if advantages.numel() <= 1:
        return advantages
    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    eps = 1e-8
    return (advantages - mean) / (std + eps)


def calculate_ema_baseline(
    baseline_ema: Optional[float],
    current_rewards: Sequence[float],
    alpha: float = 0.9,
) -> float:
    """
    Update exponential moving average baseline with current rewards.
    """
    if not current_rewards:
        return baseline_ema if baseline_ema is not None else 0.0
    mean_r = float(np.mean(current_rewards))
    if baseline_ema is None:
        return mean_r
    return alpha * baseline_ema + (1.0 - alpha) * mean_r


def calculate_policy_loss(
    log_probs: Sequence[torch.Tensor],
    rewards: Sequence[float],
    kl_divergence_coef: float = 0.0,
    entropy_bonus_coef: float = 0.0,
    entropies: Optional[Sequence[torch.Tensor]] = None,
    ref_log_probs: Optional[Sequence[Optional[torch.Tensor]]] = None,
    length_penalties: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, Dict[str, Any]]:
    """
    Calculate REINFORCE policy gradient loss with advantage standardization,
    KL divergence penalty (from reference model), and entropy regularization.

    Formula:
      A_t = standardize(R_env) - beta_KL * (log pi_theta - log pi_ref) - beta_L * L
      Loss_policy = - sum_t (log pi_theta(t) * A_t)
      Loss_entropy = - beta_ent * sum_t H(pi_theta(t))
      Total_loss = Loss_policy + Loss_entropy

    Args:
        log_probs: List of per-token log probabilities from current policy [T].
        rewards: List of scalar rewards per token [T].
        kl_divergence_coef: Coefficient beta for KL penalty against reference policy.
        entropy_bonus_coef: Coefficient for entropy bonus.
        entropies: Optional list of per-token entropy tensors.
        ref_log_probs: Optional list of reference policy log probabilities [T].
        length_penalties: Optional list of per-token length penalties.

    Returns:
        total_loss, policy_loss, entropy_loss, kl_loss, num_kl_tokens, metrics_dict
    """
    device = log_probs[0].device if log_probs else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    if not log_probs or not rewards:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, zero.clone(), zero.clone(), zero.clone(), 0, {}

    rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
    logp_t = torch.stack(list(log_probs)).float()

    assert logp_t.shape[0] == rewards_tensor.shape[0], (
        f"Mismatch: {logp_t.shape[0]} log_probs vs {rewards_tensor.shape[0]} rewards"
    )

    # Compute KL penalty per token: KL_t = log pi_theta - log pi_ref
    kl_tensor = torch.zeros_like(rewards_tensor, device=device, dtype=torch.float32)
    num_kl_tokens = 0
    raw_kl_sum = 0.0

    if ref_log_probs is not None and len(ref_log_probs) == len(log_probs):
        valid_mask = torch.tensor([rlp is not None for rlp in ref_log_probs], device=device, dtype=torch.bool)
        if valid_mask.any():
            ref_stack = torch.stack([
                rlp.float().to(device).detach() if rlp is not None else torch.tensor(0.0, device=device)
                for rlp in ref_log_probs
            ]).float()
            logp_diff = (logp_t - ref_stack).detach()
            kl_tensor = torch.where(valid_mask, logp_diff, kl_tensor)
            num_kl_tokens = int(valid_mask.sum().item())
            raw_kl_sum = float(logp_diff[valid_mask].sum().item())

    # 1. Batch standardization of task rewards
    eps = 1e-8
    r_mean = rewards_tensor.mean()
    r_std = rewards_tensor.std(unbiased=False) + eps
    if rewards_tensor.numel() <= 1:
        r_std = torch.tensor(eps, device=device, dtype=rewards_tensor.dtype)
    normalized_rewards = (rewards_tensor - r_mean) / r_std

    # 2. Subtract post-normalization length penalty if provided
    if length_penalties is not None and len(length_penalties) == len(rewards):
        lp_tensor = torch.tensor(length_penalties, device=device, dtype=torch.float32)
        normalized_rewards = normalized_rewards - lp_tensor

    # 3. Form final advantage: A = standardize(R) - beta_KL * KL
    advantages = normalized_rewards - kl_divergence_coef * kl_tensor

    # Policy loss: - sum_t (log_prob * advantage)
    policy_loss = -(logp_t * advantages).sum()

    # Entropy loss (bonus)
    entropy_loss = torch.tensor(0.0, device=device)
    if entropies is not None and entropy_bonus_coef > 0.0:
        ent_t = torch.stack(list(entropies)).float().to(device)
        entropy_loss = -entropy_bonus_coef * ent_t.sum()

    total_loss = policy_loss + entropy_loss
    kl_loss = torch.tensor(0.0, device=device)

    metrics = {
        "policy_loss": float(policy_loss.item()),
        "entropy_loss": float(entropy_loss.item()),
        "raw_kl": raw_kl_sum,
        "kl_tokens": num_kl_tokens,
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
        "reward_mean": float(rewards_tensor.mean().item()),
    }

    return total_loss, policy_loss, entropy_loss, kl_loss, num_kl_tokens, metrics
