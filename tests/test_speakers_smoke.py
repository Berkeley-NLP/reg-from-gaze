"""
Smoke tests for Speaker models and factory helper.
Verifies BaseSpeaker interface, get_speaker factory dispatch,
autoregressive generation format, and logprob/entropy computations.
"""
from typing import Dict, Any, List, Optional
from types import SimpleNamespace
import pytest
import torch
import torch.nn as nn
from PIL import Image

from configs.base import SpeakerConfig
from models.base import SpeakerOutput, BaseSpeaker
from models.speakers import (
    MolmoSpeaker,
    PaliGemmaSpeaker,
    LlavaSpeaker,
    get_speaker,
)


class DummyTokenizer:
    """Mock tokenizer for speaker smoke testing."""
    def __init__(self, vocab_size: int = 100, eos_token_id: int = 1, pad_token_id: int = 0):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in (self.eos_token_id, self.pad_token_id)]
        return " ".join(f"tok_{t}" for t in token_ids)


class DummyProcessor:
    """Mock multi-modal processor."""
    def __init__(self, vocab_size: int = 100):
        self.tokenizer = DummyTokenizer(vocab_size=vocab_size)

    def process(self, images=None, text=None, **kwargs):
        return {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
            "images": torch.zeros((1, 3, 64, 64), dtype=torch.float32),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }

    def __call__(self, images=None, text=None, return_tensors="pt", **kwargs):
        return {
            "input_ids": torch.tensor([[10, 11, 12]], dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 64, 64), dtype=torch.float32),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class DummyCausalLM(nn.Module):
    """Mock CausalLM with target projection layers for LoRA."""
    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, **kwargs):
        emb = self.embed(input_ids)
        h = self.q_proj(emb)
        logits = self.lm_head(h)
        return SimpleNamespace(logits=logits, past_key_values=None)


def test_get_speaker_factory_dispatch():
    dummy_model = DummyCausalLM()
    dummy_proc = DummyProcessor()

    molmo_cfg = SpeakerConfig(architecture="molmo")
    pali_cfg = SpeakerConfig(architecture="paligemma")
    llava_cfg = SpeakerConfig(architecture="llava")

    s1 = get_speaker(molmo_cfg, model=dummy_model, processor=dummy_proc)
    assert isinstance(s1, MolmoSpeaker)

    s2 = get_speaker(pali_cfg, model=dummy_model, processor=dummy_proc)
    assert isinstance(s2, PaliGemmaSpeaker)

    s3 = get_speaker(llava_cfg, model=dummy_model, processor=dummy_proc)
    assert isinstance(s3, LlavaSpeaker)

    # Test dictionary config and string fallback
    s_dict = get_speaker({"architecture": "molmo"}, model=dummy_model, processor=dummy_proc)
    assert isinstance(s_dict, MolmoSpeaker)

    with pytest.raises(ValueError, match="Unsupported speaker architecture"):
        get_speaker({"architecture": "unsupported_arch"})


def test_molmo_speaker_generation_and_logprobs():
    dummy_model = DummyCausalLM(vocab_size=50)
    dummy_proc = DummyProcessor(vocab_size=50)
    config = SpeakerConfig(architecture="molmo", max_new_tokens=4)

    speaker = MolmoSpeaker(config=config, model=dummy_model, processor=dummy_proc)
    dummy_image = Image.new("RGB", (100, 100), color="white")

    output = speaker.generate_referring_expression(
        image=dummy_image,
        prompt="Describe the object in red box",
        max_new_tokens=3,
        validation_mode=True,
    )

    assert isinstance(output, SpeakerOutput)
    assert isinstance(output.text, str)
    assert output.token_ids.shape[0] == 1
    assert output.token_ids.shape[1] <= 3
    assert len(output.tokens) == output.token_ids.shape[1]
    assert output.logprobs.shape[0] == output.token_ids.shape[1]
    assert output.entropy.shape[0] == output.token_ids.shape[1]

    # Test forward_logprobs
    inputs = {"input_ids": output.token_ids}
    log_probs, entropy = speaker.forward_logprobs(inputs, output.token_ids)
    assert log_probs.shape == output.token_ids.shape
    assert entropy.shape == output.token_ids.shape


def test_paligemma_speaker_generation_and_logprobs():
    dummy_model = DummyCausalLM(vocab_size=50)
    dummy_proc = DummyProcessor(vocab_size=50)
    config = SpeakerConfig(architecture="paligemma", max_new_tokens=4)

    speaker = PaliGemmaSpeaker(config=config, model=dummy_model, processor=dummy_proc)
    dummy_image = Image.new("RGB", (100, 100), color="white")

    output = speaker.generate_referring_expression(
        image=dummy_image,
        prompt="Describe the target",
        max_new_tokens=3,
        validation_mode=True,
    )

    assert isinstance(output, SpeakerOutput)
    assert output.token_ids.shape[0] == 1
    assert len(output.tokens) == output.token_ids.shape[1]
    assert output.logprobs.shape[0] == output.token_ids.shape[1]


def test_llava_speaker_generation_and_logprobs():
    dummy_model = DummyCausalLM(vocab_size=50)
    dummy_proc = DummyProcessor(vocab_size=50)
    config = SpeakerConfig(architecture="llava", max_new_tokens=4)

    speaker = LlavaSpeaker(config=config, model=dummy_model, processor=dummy_proc)
    dummy_image = Image.new("RGB", (100, 100), color="white")

    output = speaker.generate_referring_expression(
        image=dummy_image,
        prompt="Describe the object in red box",
        max_new_tokens=3,
        validation_mode=True,
    )

    assert isinstance(output, SpeakerOutput)
    assert output.token_ids.shape[0] == 1
    assert len(output.tokens) == output.token_ids.shape[1]
    assert output.logprobs.shape[0] == output.token_ids.shape[1]


def test_speaker_lora_attachment():
    dummy_model = DummyCausalLM(vocab_size=50)
    dummy_proc = DummyProcessor(vocab_size=50)
    config = SpeakerConfig(architecture="molmo")

    speaker = MolmoSpeaker(config=config, model=dummy_model, processor=dummy_proc)

    lora_config = {
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.0,
    }
    speaker.apply_lora(lora_config)

    # Verify LoRA parameters exist in the model
    trainable_params = [name for name, p in speaker.model.named_parameters() if p.requires_grad]
    assert len(trainable_params) > 0
    assert any("lora" in name.lower() for name in trainable_params)
