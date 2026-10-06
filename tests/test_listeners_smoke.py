"""
Smoke tests for Listener models and factory helper.
Verifies BaseListener interface, get_listener factory dispatch,
GazePredictorVectorized rollouts, MolmoListenerVectorized pointing, and coordinate parsing.
"""
from typing import Dict, Any, List, Optional
from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn
from PIL import Image

from configs.base import ListenerConfig
from models.base import ListenerOutput, BaseListener
from models.listeners import (
    GazePredictorVectorized,
    MolmoListenerVectorized,
    QwenVLListener,
    CogVLMListener,
    get_listener,
    parse_molmo_coordinate,
)


class DummyListenerTokenizer:
    """Mock tokenizer that decodes generated token ids to coordinate strings."""
    def __init__(self, mode: str = "point"):
        self.mode = mode
        self.eos_token_id = 1
        self.pad_token_id = 0

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        if self.mode == "gaussian":
            return "<x=25.0 y=35.0 v=12.5>"
        elif self.mode == "molmo_point":
            return '<point x="30.0" y="40.0">'
        else:
            return "<x=25.0 y=35.0>"


class DummyListenerProcessor:
    """Mock processor for listener models."""
    def __init__(self, mode: str = "point"):
        self.tokenizer = DummyListenerTokenizer(mode=mode)

    def process(self, images=None, text=None, **kwargs):
        return {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
            "images": torch.zeros((1, 3, 64, 64), dtype=torch.float32),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class DummyListenerModel(nn.Module):
    """Mock listener CausalLM model with generate_from_batch support."""
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def generate_from_batch(self, batch, **kwargs):
        # Return batch of sequences with prompt tokens + 2 generated tokens
        batch_size = batch["input_ids"].shape[0]
        prompt_len = batch["input_ids"].shape[1]
        seqs = torch.zeros((batch_size, prompt_len + 2), dtype=torch.long)
        return SimpleNamespace(sequences=seqs)

    def generate(self, **kwargs):
        return torch.zeros((1, 10), dtype=torch.long)


def test_get_listener_factory_dispatch():
    dummy_model = DummyListenerModel()
    dummy_proc = DummyListenerProcessor()

    cfg_gaze = ListenerConfig(listener_type="gaze_predictor")
    l_gaze = get_listener(cfg_gaze, model=dummy_model, processor=dummy_proc)
    assert isinstance(l_gaze, GazePredictorVectorized)

    cfg_molmo_iter = ListenerConfig(listener_type="molmo_iterative")
    l_molmo_iter = get_listener(cfg_molmo_iter, model=dummy_model, processor=dummy_proc)
    assert isinstance(l_molmo_iter, MolmoListenerVectorized)
    assert l_molmo_iter.iterative_mode is True

    cfg_molmo_rec = ListenerConfig(listener_type="rec_single_point")
    l_molmo_rec = get_listener(cfg_molmo_rec, model=dummy_model, processor=dummy_proc)
    assert isinstance(l_molmo_rec, MolmoListenerVectorized)
    assert l_molmo_rec.iterative_mode is False

    # Dispatch with dict
    l_dict = get_listener({"listener_type": "gaze_predictor"}, model=dummy_model, processor=dummy_proc)
    assert isinstance(l_dict, GazePredictorVectorized)

    with pytest.raises(ValueError, match="Unsupported listener type"):
        get_listener({"listener_type": "unknown_type"})


def test_gaze_predictor_point_rollout():
    dummy_model = DummyListenerModel()
    dummy_proc = DummyListenerProcessor(mode="point")
    config = ListenerConfig(listener_type="gaze_predictor")

    listener = GazePredictorVectorized(
        config=config,
        model=dummy_model,
        processor=dummy_proc,
        gaussian_mode=False,
        random_bos_point=False,
    )

    img = Image.new("RGB", (100, 100), color="white")
    output = listener.predict_gaze_sequence(
        image=img,
        referring_expression="the red ball on the table",
        target_bbox_normalized=(20.0, 30.0, 40.0, 50.0),
    )

    assert isinstance(output, ListenerOutput)
    assert len(output.gaze_points) > 0
    # Dummy tokenizer produces (25.0, 35.0) which is inside (20, 30, 40, 50)
    assert output.gaze_points[0] == (25.0, 35.0)
    assert output.hit_mask[0] is True
    assert output.is_success is True
    assert output.first_hit_index == 0
    assert output.distances_to_bbox[0] >= 0.0


def test_gaze_predictor_gaussian_rollout():
    dummy_model = DummyListenerModel()
    dummy_proc = DummyListenerProcessor(mode="gaussian")
    config = ListenerConfig(listener_type="gaze_predictor")

    listener = GazePredictorVectorized(
        config=config,
        model=dummy_model,
        processor=dummy_proc,
        gaussian_mode=True,
        random_bos_point=False,
    )

    img = Image.new("RGB", (100, 100), color="white")
    output = listener.predict_gaze_sequence(
        image=img,
        referring_expression="the cup",
        target_bbox_normalized=(20.0, 30.0, 40.0, 50.0),
    )

    assert isinstance(output, ListenerOutput)
    assert output.gaussian_params is not None
    assert len(output.gaussian_params) > 0
    assert output.gaussian_params[0] == (25.0, 35.0, 12.5)


def test_molmo_listener_coordinate_parsing():
    p1 = parse_molmo_coordinate('<point x="15.2" y="32.8" alt="a cup">')
    assert p1 == [[15.2, 32.8]]

    p2 = parse_molmo_coordinate('<x="12.0" y="45.0">')
    assert p2 == [[12.0, 45.0]]

    p3 = parse_molmo_coordinate('<x=12.0 y=45.0>')
    assert p3 == [[12.0, 45.0]]

    p4 = parse_molmo_coordinate('<points x1="10" y1="20" x2="30" y2="40">')
    assert p4 == [[10.0, 20.0], [30.0, 40.0]]

    p5 = parse_molmo_coordinate("random unparseable text")
    assert p5 is None


def test_molmo_listener_rollout():
    dummy_model = DummyListenerModel()
    dummy_proc = DummyListenerProcessor(mode="molmo_point")
    config = ListenerConfig(listener_type="rec_single_point")

    listener = MolmoListenerVectorized(
        config=config,
        model=dummy_model,
        processor=dummy_proc,
        iterative_mode=False,
    )

    img = Image.new("RGB", (100, 100), color="white")
    output = listener.predict_gaze_sequence(
        image=img,
        referring_expression="the laptop",
        target_bbox_normalized=(25.0, 35.0, 45.0, 55.0),
    )

    assert isinstance(output, ListenerOutput)
    assert len(output.gaze_points) > 0
    # Dummy tokenizer produces (30.0, 40.0) which is inside (25, 35, 45, 55)
    assert output.gaze_points[0] == (30.0, 40.0)
    assert output.is_success is True
