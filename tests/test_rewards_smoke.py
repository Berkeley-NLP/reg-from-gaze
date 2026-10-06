"""
Smoke tests for RL rewards, penalties, and policy gradient loss functions.
"""

import pytest
import torch

from configs.base import ExperimentConfig, RewardConfig
from rl.rewards import (
    BFHReward,
    SeqAnyHitReward,
    SeqLPHitReward,
    SupervisedRECReward,
    DistanceShapingReward,
    GaussianShapingReward,
    GazeRLReward,
    get_reward_function,
)
from rl.loss import calculate_policy_loss, normalize_rewards_batch, standardize_advantages_batch


def test_bfh_reward_decay():
    """Test Backward First Hit (BFH) credit assignment with decay gamma=0.9."""
    rf = BFHReward(gamma=0.9, hit_reward=1.0, context_reward=1.0)

    bbox = (40.0, 40.0, 60.0, 60.0)  # Box in [40, 60] x [40, 60]
    tokens = ["the", "brown", "dog", "is", "running"]  # 5 tokens

    # BOS at (10, 10); tok1 at (20, 20); tok2 at (30, 30); tok3 at (50, 50) [HIT!]; tok4 at (50, 50); tok5 at (80, 80)
    gaze_points = [
        (10.0, 10.0),  # BOS (index 0)
        (20.0, 20.0),  # tok 1 ("the")
        (30.0, 30.0),  # tok 2 ("brown")
        (50.0, 50.0),  # tok 3 ("dog") -> FIRST HIT
        (50.0, 50.0),  # tok 4 ("is")
        (80.0, 80.0),  # tok 5 ("running")
    ]

    out = rf.compute_rewards(gaze_points=gaze_points, bbox=bbox, tokens=tokens)
    assert len(out.rewards) == 5
    assert out.is_success is True
    assert out.hit_idx == 2  # 0-indexed token 2 ("dog")
    assert out.first_hit_token == "dog"

    # Token 2 ("dog") was first hit -> 1.0
    assert pytest.approx(out.rewards[2], abs=1e-4) == 1.0
    # Token 1 ("brown") is 1 step back -> 0.9^1 * 1.0 = 0.9
    assert pytest.approx(out.rewards[1], abs=1e-4) == 0.9
    # Token 0 ("the") is 2 steps back -> 0.9^2 * 1.0 = 0.81
    assert pytest.approx(out.rewards[0], abs=1e-4) == 0.81
    # Tokens after hit get 0.0
    assert out.rewards[3] == 0.0
    assert out.rewards[4] == 0.0

    # Test Miss Case: no hit inside bbox
    miss_gaze = [(10.0, 10.0), (15.0, 15.0), (20.0, 20.0), (25.0, 25.0), (30.0, 30.0), (35.0, 35.0)]
    miss_out = rf.compute_rewards(gaze_points=miss_gaze, bbox=bbox, tokens=tokens)
    assert miss_out.is_success is False
    assert miss_out.hit_idx is None
    assert all(r == 0.0 for r in miss_out.rewards)


def test_distance_shaping_reward():
    """Test continuous distance delta shaping."""
    rf = DistanceShapingReward(shaping_scale=1.0)

    bbox = (50.0, 50.0, 60.0, 60.0)
    tokens = ["a", "red", "car"]

    # Initial distance from BOS (10, 50) to bbox (50, 50) is 40.0
    # tok 1 at (30, 50) -> dist = 20.0. delta = (40 - 20) / 40 = +0.5
    # tok 2 at (40, 50) -> dist = 10.0. delta = (20 - 10) / 40 = +0.25
    # tok 3 at (20, 50) -> dist = 30.0. delta = (10 - 30) / 40 = -0.5
    gaze_points = [
        (10.0, 50.0),  # BOS (dist=40)
        (30.0, 50.0),  # tok 1 (dist=20) -> +0.5
        (40.0, 50.0),  # tok 2 (dist=10) -> +0.25
        (20.0, 50.0),  # tok 3 (dist=30) -> -0.5
    ]

    out = rf.compute_rewards(gaze_points=gaze_points, bbox=bbox, tokens=tokens)
    assert len(out.rewards) == 3
    assert pytest.approx(out.rewards[0], abs=1e-4) == 0.5
    assert pytest.approx(out.rewards[1], abs=1e-4) == 0.25
    assert pytest.approx(out.rewards[2], abs=1e-4) == -0.5


def test_seq_any_hit_reward():
    """Test Gaze-SeqAnyHit uniform binary reward."""
    rf = SeqAnyHitReward(hit_reward=1.0, miss_reward=0.0)
    bbox = (40.0, 40.0, 60.0, 60.0)
    tokens = ["small", "cat"]

    # One hit in sequence -> all tokens get 1.0
    gaze_hit = [(0.0, 0.0), (10.0, 10.0), (50.0, 50.0)]
    out_hit = rf.compute_rewards(gaze_points=gaze_hit, bbox=bbox, tokens=tokens)
    assert out_hit.is_success is True
    assert out_hit.rewards == [1.0, 1.0]

    # No hits in sequence -> all tokens get 0.0
    gaze_miss = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0)]
    out_miss = rf.compute_rewards(gaze_points=gaze_miss, bbox=bbox, tokens=tokens)
    assert out_miss.is_success is False
    assert out_miss.rewards == [0.0, 0.0]


def test_seq_lp_hit_reward():
    """Test Gaze-SeqLPHit (last point hit only)."""
    rf = SeqLPHitReward(hit_reward=1.0, miss_reward=0.0)
    bbox = (40.0, 40.0, 60.0, 60.0)
    tokens = ["small", "cat", "sleeping"]

    # Hit at step 1, but last point (step 3) misses -> should fail
    gaze_early_hit = [(0.0, 0.0), (50.0, 50.0), (10.0, 10.0), (10.0, 10.0)]
    out1 = rf.compute_rewards(gaze_points=gaze_early_hit, bbox=bbox, tokens=tokens)
    assert out1.is_success is False
    assert out1.rewards == [0.0, 0.0, 0.0]

    # Last point hits -> should succeed
    gaze_last_hit = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0), (50.0, 50.0)]
    out2 = rf.compute_rewards(gaze_points=gaze_last_hit, bbox=bbox, tokens=tokens)
    assert out2.is_success is True
    assert out2.rewards == [1.0, 1.0, 1.0]


def test_supervised_rec_reward():
    """Test REC-Success supervised hit reward."""
    rf = SupervisedRECReward(hit_reward=1.0, miss_reward=0.0)
    bbox = (40.0, 40.0, 60.0, 60.0)
    tokens = ["big", "dog"]

    # REC prediction point is inside bbox
    gaze_in = [(50.0, 50.0)]
    out = rf.compute_rewards(gaze_points=gaze_in, bbox=bbox, tokens=tokens)
    assert out.is_success is True
    assert out.rewards == [1.0, 1.0]

    # REC prediction point is outside bbox
    gaze_out = [(10.0, 10.0)]
    out_miss = rf.compute_rewards(gaze_points=gaze_out, bbox=bbox, tokens=tokens)
    assert out_miss.is_success is False
    assert out_miss.rewards == [0.0, 0.0]


def test_gaussian_shaping_reward():
    """Test Gaussian probability mass shaping reward."""
    rf = GaussianShapingReward(shaping_scale=1.0, gaussian_sigma_default=5.0)
    bbox = (40.0, 40.0, 60.0, 60.0)
    tokens = ["white", "cup"]

    # Point moves from outside (10, 10) to center of bbox (50, 50)
    gaze_points = [
        (10.0, 10.0, 5.0),  # BOS
        (30.0, 30.0, 5.0),  # closer
        (50.0, 50.0, 5.0),  # right inside
    ]
    out = rf.compute_rewards(gaze_points=gaze_points, bbox=bbox, tokens=tokens)
    assert len(out.rewards) == 2
    # Probability increases as it moves closer and inside
    assert out.rewards[0] >= 0.0
    assert out.rewards[1] > 0.5


def test_logit_and_length_penalties():
    """Test GRPO-style logit norm squared penalty and length penalty."""
    cfg = RewardConfig(
        reward_type="sparse_decay",
        gamma=0.9,
        use_length_penalty=True,
        length_penalty_beta=0.05,
        logit_penalty_weight=0.01,
    )
    rf = get_reward_function(cfg)
    bbox = (40.0, 40.0, 60.0, 60.0)
    tokens = ["large", "building", "<eos>"]
    gaze_points = [(0.0, 0.0), (10.0, 10.0), (50.0, 50.0), (50.0, 50.0)]

    # Create dummy logits [3, 10]
    logits = torch.ones(3, 10, dtype=torch.float32) * 2.0
    # L2 norm squared for each vector is 10 * (2^2) = 40.0
    # Penalty is -0.01 * 40.0 = -0.4

    out = rf.compute_rewards(gaze_points=gaze_points, bbox=bbox, tokens=tokens, logits=logits)
    assert len(out.rewards) == 3

    # Token 1 ("large"): length penalty = 0.05, logit penalty = -0.4
    comp0 = out.components[0]
    assert pytest.approx(comp0["length_penalty"], abs=1e-4) == 0.05
    assert pytest.approx(comp0["logit_penalty"], abs=1e-4) == -0.4

    # Token 2 ("<eos>"): length penalty should be 0.0 for EOS
    comp2 = out.components[2]
    assert comp2["length_penalty"] == 0.0


def test_all_paper_presets_instantiate_and_run():
    """Verify that all 8 canonical preset configurations load and compute rewards."""
    import yaml
    from pathlib import Path

    presets_dir = Path("configs/presets")
    yaml_files = list(presets_dir.glob("*.yaml"))
    assert len(yaml_files) >= 8

    for yf in yaml_files:
        with open(yf) as f:
            data = yaml.safe_load(f)
        exp_cfg = ExperimentConfig.from_dict(data)
        rf = get_reward_function(exp_cfg.reward)

        bbox = (30.0, 30.0, 70.0, 70.0)
        tokens = ["test", "phrase", "here"]
        gaze = [(0.0, 0.0), (20.0, 20.0), (50.0, 50.0), (50.0, 50.0)]

        out = rf.compute_rewards(gaze_points=gaze, bbox=bbox, tokens=tokens)
        assert len(out.rewards) == 3
        assert len(out.components) == 3
        assert isinstance(out.is_success, bool)


def test_policy_loss_and_backward():
    """Verify REINFORCE policy loss calculation, advantage standardization, and gradient backprop."""
    # 4 tokens with differentiable log probs
    log_probs = [
        torch.tensor(-1.2, requires_grad=True),
        torch.tensor(-0.8, requires_grad=True),
        torch.tensor(-0.5, requires_grad=True),
        torch.tensor(-0.2, requires_grad=True),
    ]
    rewards = [0.1, 0.2, 0.8, 1.0]
    ref_log_probs = [
        torch.tensor(-1.1),
        torch.tensor(-0.9),
        torch.tensor(-0.4),
        torch.tensor(-0.3),
    ]

    total_loss, policy_loss, entropy_loss, kl_loss, num_kl, metrics = calculate_policy_loss(
        log_probs=log_probs,
        rewards=rewards,
        kl_divergence_coef=0.02,
        entropy_bonus_coef=0.01,
        ref_log_probs=ref_log_probs,
    )

    assert total_loss.requires_grad
    assert num_kl == 4
    assert "advantage_mean" in metrics
    assert "raw_kl" in metrics

    # Backward pass should produce valid gradients for all log_probs
    total_loss.backward()
    for lp in log_probs:
        assert lp.grad is not None
        assert not torch.isnan(lp.grad)
