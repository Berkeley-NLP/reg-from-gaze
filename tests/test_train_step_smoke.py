"""
Smoke test for mini-RL training step end-to-end.
"""

from pathlib import Path
import tempfile
import torch
import torch.nn as nn

from configs.base import ExperimentConfig, TrainingConfig
from data.datasets import SyntheticReferringExpressionDataset
from models.base import BaseSpeaker, BaseListener, SpeakerOutput, ListenerOutput
from rl.rewards import get_reward_function
from training.trainer import GazeRLTrainer


class DifferentiableMockSpeaker(BaseSpeaker):
    """Mock speaker where log_probs depend on trainable parameters."""
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([0.5, -0.5], requires_grad=True))

    def prepare_inputs(self, image, prompt, device=None):
        return {"dummy": torch.zeros(1)}

    def generate_referring_expression(self, image, prompt, **kwargs):
        probs = torch.softmax(self.param, dim=0)
        log_probs = torch.log_softmax(self.param, dim=0)
        ent = -(probs * log_probs).sum()
        return SpeakerOutput(
            text="red ball",
            tokens=["red", "ball"],
            token_ids=[101, 102],
            logprobs=log_probs,
            entropy=torch.stack([ent, ent]),
        )

    def forward_logprobs(self, inputs, token_ids):
        probs = torch.softmax(self.param, dim=0)
        log_probs = torch.log_softmax(self.param, dim=0)
        ent = -(probs * log_probs).sum()
        return log_probs, torch.stack([ent, ent])

    def apply_lora(self, lora_config):
        pass


class MockListener(BaseListener):
    """Mock listener returning gaze path that hits target on alternate samples."""
    def __init__(self):
        super().__init__()
        self.step = 0

    def predict_gaze_sequence(self, image, referring_expression, target_bbox_normalized):
        return ListenerOutput(
            gaze_points=[(10.0, 10.0), (30.0, 30.0), (65.0, 40.0)],
            raw_output_text="<point x=\"65\" y=\"40\"/>",
        )

    def predict(self, images, tokens_list, **kwargs):
        outs = []
        for _ in tokens_list:
            self.step += 1
            # Alternate between hit (65, 40) and miss (10, 10)
            pt = (65.0, 40.0) if self.step % 2 == 1 else (10.0, 10.0)
            outs.append(ListenerOutput(
                gaze_points=[(0.0, 0.0), (20.0, 20.0), pt],
                raw_output_text=f"<point x=\"{pt[0]}\" y=\"{pt[1]}\"/>",
            ))
        return outs


def test_mini_rl_train_step():
    """Verify single batch rollout, REINFORCE loss backward pass, and optimizer step."""
    with tempfile.TemporaryDirectory() as tmpdir:
        speaker = DifferentiableMockSpeaker()
        listener = MockListener()
        reward_fn = get_reward_function("sparse_decay", gamma=0.9)
        dataset = SyntheticReferringExpressionDataset(num_samples=8)

        config = ExperimentConfig(
            training=TrainingConfig(
                learning_rate=1e-3,
                batch_size=2,
                gradient_accumulation_steps=1,
                num_episodes=4,
                output_dir=tmpdir,
                warmup_steps=0,
                wandb=False,
            )
        )

        trainer = GazeRLTrainer(
            speaker=speaker,
            listener=listener,
            reward_fn=reward_fn,
            train_dataset=dataset,
            config=config,
        )

        batch_items = [dataset[0], dataset[1]]

        # Record initial parameter value
        initial_param = speaker.param.detach().clone()

        # Run 1 train step
        metrics = trainer.train_step(batch_items)

        assert "loss" in metrics
        assert "reward_mean" in metrics
        assert "success_rate" in metrics
        assert trainer.opt_step == 1
        assert trainer.episode_count == 2

        # Check parameter was updated by optimizer
        assert not torch.allclose(speaker.param, initial_param)

        # Run full train loop for remaining episodes
        result = trainer.train(num_episodes=4)
        assert result["final_episode"] >= 4
        assert (Path(tmpdir) / "checkpoint_final").exists()
