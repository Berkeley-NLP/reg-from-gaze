"""
Smoke test for evaluation pipeline, spatial metrics, and language quality metrics.
"""

from pathlib import Path
import tempfile
import pytest
import torch

from models.base import BaseSpeaker, BaseListener, SpeakerOutput, ListenerOutput
from data.datasets import SyntheticReferringExpressionDataset
from evaluation import (
    compute_bbox_iou,
    compute_grounding_accuracy,
    compute_mean_distance_to_bbox,
    compute_mean_sentence_length,
    compute_corpus_bleu,
    compute_rouge_l,
    evaluate_dataset,
)


class EvalDummySpeaker(BaseSpeaker):
    def prepare_inputs(self, image, prompt, device=None):
        return {}

    def generate_referring_expression(self, image, prompt, **kwargs):
        return SpeakerOutput(
            text="the blue circle",
            tokens=["the", "blue", "circle"],
            token_ids=[1, 2, 3],
            logprobs=torch.tensor([0.0, 0.0, 0.0]),
        )

    def forward_logprobs(self, inputs, token_ids):
        return torch.zeros(len(token_ids)), torch.zeros(len(token_ids))

    def apply_lora(self, lora_config):
        pass


class EvalDummyListener(BaseListener):
    def predict_gaze_sequence(self, image, referring_expression, target_bbox_normalized):
        return ListenerOutput(
            gaze_points=[(0.0, 0.0), (65.0, 40.0)],
            raw_output_text="<point x=\"65\" y=\"40\"/>",
        )

    def predict(self, images, tokens_list, **kwargs):
        return [
            ListenerOutput(
                gaze_points=[(0.0, 0.0), (65.0, 40.0)],
                raw_output_text="<point x=\"65\" y=\"40\"/>",
            )
            for _ in tokens_list
        ]


def test_spatial_metrics_smoke():
    """Verify IoU, grounding accuracy, and mean distance computations."""
    # Test IoU: perfect overlap -> 1.0
    boxA = (10.0, 10.0, 30.0, 30.0)
    assert pytest.approx(compute_bbox_iou(boxA, boxA)) == 1.0

    # Test IoU: disjoint -> 0.0
    boxB = (40.0, 40.0, 60.0, 60.0)
    assert compute_bbox_iou(boxA, boxB) == 0.0

    # Test Grounding Accuracy
    pts = [(20.0, 20.0), (50.0, 50.0), (0.0, 0.0)]
    boxes = [boxA, boxB, boxA]
    # pts[0] in boxA (hit), pts[1] in boxB (hit), pts[2] in boxA (miss) -> 2/3
    acc = compute_grounding_accuracy(pts, boxes)
    assert pytest.approx(acc, abs=1e-4) == 2.0 / 3.0

    # Test Mean Distance
    dist = compute_mean_distance_to_bbox(pts, boxes)
    assert dist > 0.0


def test_language_metrics_smoke():
    """Verify BLEU and ROUGE-L computations on hypotheses and references."""
    hyps = ["the red circle in the corner", "small yellow square"]
    refs = [
        ["the red circle on the right", "a circular red shape"],
        ["a tiny yellow square", "small yellow square object"],
    ]

    bleu = compute_corpus_bleu(hyps, refs)
    assert "bleu_1" in bleu
    assert "bleu_2" in bleu
    assert bleu["bleu_1"] > 0.0

    rouge = compute_rouge_l(hyps, refs)
    assert rouge > 0.0

    mean_len = compute_mean_sentence_length(hyps)
    assert pytest.approx(mean_len, abs=1e-4) == 4.5  # (6 + 3) / 2 = 4.5


def test_evaluate_dataset_end_to_end():
    """Verify evaluate_dataset runs on synthetic dataset and saves output JSON report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        speaker = EvalDummySpeaker()
        listener = EvalDummyListener()
        dataset = SyntheticReferringExpressionDataset(num_samples=4)
        report_path = Path(tmpdir) / "eval_results.json"

        output = evaluate_dataset(
            speaker=speaker,
            listener=listener,
            dataset=dataset,
            max_samples=4,
            output_path=report_path,
            show_progress=False,
        )

        assert "summary" in output
        assert "results" in output
        assert output["summary"]["num_evaluated"] == 4
        assert "grounding_accuracy" in output["summary"]
        assert report_path.exists()
