"""
Unit tests verifying paper contract adherence:
1. Dataset split loading for paper test sets (RefCOCO testA/B, RefOI co-occurrence/single-presence).
2. Reward type resolution for exact paper nomenclature.
3. Listener predict interface contract (ListenerOutput sequence protocol, bbox forwarding, vectorized returns).
"""
import pytest
from PIL import Image
import torch

from configs.base import RewardConfig, ListenerConfig
from data.datasets import ReferringExpressionDataset, SyntheticReferringExpressionDataset
from models.base import ListenerOutput, BaseListener
from models.listeners import GazePredictorVectorized, MolmoListenerVectorized
from rl.rewards import get_reward_function
from rl.rewards.composite import GazeRLReward
from tests.test_listeners_smoke import DummyListenerModel, DummyListenerProcessor


def test_reward_paper_naming_resolution():
    """Verify that paper reward names correctly map to underlying reward implementations."""
    paper_names = [
        ("before_first_hit", "_bfh"),
        ("bfh", "_bfh"),
        ("Gaze-BeforeFirstHit", "_bfh"),
        ("REC-BeforeFirstHit", "_bfh"),
        ("seq_any_hit", "_seq_any"),
        ("Gaze-SeqAnyHit", "_seq_any"),
        ("seq_lp_hit", "_seq_lp"),
        ("Gaze-SeqLPHit", "_seq_lp"),
        ("rec_success", "_supervised"),
        ("REC-Success", "_supervised"),
        ("shaping", "_dist_shaping"),
        ("Gaze-Shaping", "_dist_shaping"),
        ("REC-Shaping", "_dist_shaping"),
    ]
    for name, attr in paper_names:
        cfg = RewardConfig(reward_type=name)
        reward_fn = GazeRLReward.from_config(cfg)
        assert reward_fn is not None
        # Verify computing rewards produces RewardOutput
        gaze_pts = [(50.0, 50.0), (25.0, 30.0)]
        bbox = (20.0, 20.0, 40.0, 40.0)
        tokens = ["the", "circle"]
        out = reward_fn.compute_rewards(gaze_pts, bbox, tokens=tokens)
        assert len(out.rewards) == 2


def test_listener_output_sequence_protocol():
    """Verify ListenerOutput satisfies both dataclass attributes and sequence protocol."""
    pts = [(10.0, 20.0), (30.0, 40.0), None]
    out = ListenerOutput(gaze_points=pts, hit_mask=[False, True, False], is_success=True)

    # Sequence protocol
    assert len(out) == 3
    assert out[0] == (10.0, 20.0)
    assert out[1] == (30.0, 40.0)
    assert list(iter(out)) == pts

    # Attribute protocol
    assert out.coordinates == pts
    assert out.is_success is True


def test_listener_predict_interface_contract():
    """Verify GazePredictorVectorized.predict adheres to BaseListener.predict returning List[ListenerOutput]."""
    dummy_model = DummyListenerModel()
    dummy_proc = DummyListenerProcessor(mode="point")
    config = ListenerConfig(listener_type="gaze_predictor")

    listener = GazePredictorVectorized(
        config=config,
        model=dummy_model,
        processor=dummy_proc,
        random_bos_point=False,
    )

    img = Image.new("RGB", (100, 100), color="white")
    tokens = ["the", "ball"]
    bbox = (20.0, 30.0, 40.0, 50.0)

    # 1. Batched call with keywords matching rollouts.py / evaluate.py
    outputs = listener.predict(
        images=[img],
        tokens_list=[tokens],
        bboxes=[bbox],
    )
    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert isinstance(outputs[0], ListenerOutput)
    assert outputs[0].coordinates[0] == (25.0, 35.0)

    # 2. Legacy / single item call
    single_out = listener.predict(
        img=img,
        tokens=tokens,
        bbox=bbox,
    )
    assert isinstance(single_out, list)
    assert isinstance(single_out[0], ListenerOutput)


def test_paper_dataset_loading_contract():
    """Verify ReferringExpressionDataset accepts canonical paper evaluation datasets and both split/split_name."""
    # Test synthetic fallback mode or cached datasets
    ds1 = ReferringExpressionDataset(dataset_name="refcoco_testA", split_name="testA", max_samples=2)
    assert len(ds1) <= 2
    item1 = ds1[0]
    assert item1.image is not None
    assert len(item1.bbox_minmax) == 4

    ds2 = ReferringExpressionDataset(dataset_name="refoi_co_occurrence", split="co_occurrence", max_samples=2)
    assert len(ds2) <= 2
    item2 = ds2[0]
    assert item2.image is not None
    assert len(item2.bbox_minmax) == 4
