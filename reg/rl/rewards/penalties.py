"""
Penalty functions: GRPO-style logit norm penalty and sequence length penalties.
"""

from typing import List, Optional, Sequence
import torch


def compute_logit_penalties(
    logits: Optional[torch.Tensor],
    lambda_weight: float = 0.01,
) -> List[float]:
    """
    Compute GRPO-style L2 penalty on logits:
      penalty_t = -lambda * sum_v (z_{t, v}^2)

    Args:
        logits: Tensor of shape [num_tokens, vocab_size].
        lambda_weight: Scaling coefficient.

    Returns:
        List of negative penalty values per token.
    """
    if logits is None or lambda_weight <= 0.0:
        return []

    penalties: List[float] = []
    with torch.no_grad():
        for t in range(logits.shape[0]):
            vec = logits[t].float()
            l2_sq = float(torch.sum(vec ** 2).item())
            penalties.append(-float(lambda_weight * l2_sq))
    return penalties


def compute_length_penalties(
    tokens: Optional[Sequence[str]],
    beta: float = 0.01,
) -> List[float]:
    """
    Compute step-wise length penalty:
    Subtracts beta from each content token (skipping EOS and padding tokens).

    Args:
        tokens: Sequence of token strings.
        beta: Per-token penalty amount.

    Returns:
        List of length penalty amounts to subtract per token.
    """
    if not tokens or beta <= 0.0:
        return [0.0] * (len(tokens) if tokens else 0)

    eos_tokens = {'<eos>', '</s>', '<|endoftext|>', 'eos', '[eos]', '<pad>', 'pad', ''}
    penalties: List[float] = []
    for tok in tokens:
        cleaned = tok.strip().lower() if tok else ""
        if not cleaned or cleaned in eos_tokens or 'eos' in cleaned or 'endoftext' in cleaned:
            penalties.append(0.0)
        else:
            penalties.append(float(beta))
    return penalties
