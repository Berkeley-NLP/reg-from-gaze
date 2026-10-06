"""
Smoke test for checkpoint save and resume operations.
"""

from pathlib import Path
import tempfile
import torch
import torch.nn as nn

from models.base import BaseSpeaker, SpeakerOutput
from configs.base import ExperimentConfig
from training.checkpoint import save_checkpoint, load_checkpoint


class DummySpeaker(BaseSpeaker):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def prepare_inputs(self, image, prompt, device=None):
        return {"dummy": torch.zeros(1)}

    def generate_referring_expression(self, image, prompt, **kwargs):
        return SpeakerOutput(text="dummy", tokens=["dum", "my"], token_ids=[1, 2], logprobs=torch.tensor([0.0, 0.0]))

    def forward_logprobs(self, inputs, token_ids):
        return torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])

    def apply_lora(self, lora_config):
        pass


def test_checkpoint_save_and_load():
    """Verify saving checkpoint and restoring training state and model parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        speaker = DummySpeaker()
        optimizer = torch.optim.AdamW(speaker.parameters(), lr=1e-4)
        config = ExperimentConfig(name="test_ckpt")

        # Mutate model weights slightly
        with torch.no_grad():
            speaker.linear.weight.fill_(4.2)

        ckpt_dir = Path(tmpdir) / "ckpt_test"
        saved_path = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            speaker=speaker,
            optimizer=optimizer,
            opt_step=42,
            episode_count=336,
            best_success_rate=0.85,
            config=config,
        )

        assert Path(saved_path).exists()
        assert (Path(saved_path) / "trainer_state.pt").exists()
        assert (Path(saved_path) / "config.json").exists()

        # Create new clean instance and restore
        new_speaker = DummySpeaker()
        new_optimizer = torch.optim.AdamW(new_speaker.parameters(), lr=1e-4)

        restored_state = load_checkpoint(
            checkpoint_dir=saved_path,
            speaker=new_speaker,
            optimizer=new_optimizer,
        )

        assert restored_state["opt_step"] == 42
        assert restored_state["episode_count"] == 336
        assert restored_state["best_success_rate"] == 0.85

        # Check model weights were restored
        assert torch.allclose(new_speaker.linear.weight, torch.tensor(4.2))
