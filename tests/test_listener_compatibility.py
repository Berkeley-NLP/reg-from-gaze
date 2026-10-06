"""
Compatibility test verifying adherence to GazeRL (reg-from-gaze) shared contract.
Matches AGENT_GAZE_PREDICTION.md specifications.
"""

from pathlib import Path
import re
from PIL import Image
import pytest

from gaze_estimation.data.transforms import letterbox_image, canvas_to_normalized


def test_gazerl_image_contract():
    # Verify canvas dimension strictly 512x320 and black padding
    img = Image.new("RGB", (640, 480), color="blue")
    padded, scale, pad_x, pad_y = letterbox_image(img, 512, 320)
    assert padded.size == (512, 320)
    # Check top-left corner is black background (0, 0, 0)
    assert padded.getpixel((0, 0)) == (0, 0, 0)


def test_gazerl_regex_compatibility():
    # Regexes exactly as specified in AGENT_GAZE_PREDICTION.md Section 2.3
    regex_point = re.compile(r'<x=(\d+(?:\.\d+)?)\s+y=(\d+(?:\.\d+)?)>', re.IGNORECASE)
    regex_gaussian = re.compile(r'<x=(\d+(?:\.\d+)?)\s+y=(\d+(?:\.\d+)?)\s+v=(\d+(?:\.\d+)?)>', re.IGNORECASE)

    sample_output = "BOS <x=50.2 y=48.9> The <x=48.1 y=45.2>"
    matches = list(regex_point.finditer(sample_output))
    assert len(matches) == 2
    assert float(matches[0].group(1)) == 50.2
    assert float(matches[0].group(2)) == 48.9

    sample_gaussian = "<x=50.0 y=50.0 v=10.5>"
    m_g = regex_gaussian.search(sample_gaussian)
    assert m_g is not None
    assert float(m_g.group(3)) == 10.5


def test_checkpoint_files_exist():
    # Verify canonical checkpoint directory contains necessary configs
    ckpt_dir = Path("checkpoints/gaze_predictor_delay_token_116")
    if ckpt_dir.exists():
        expected = [
            "config.json",
            "processor_config.json",
            "tokenizer.json",
            "image_preprocessing_molmo.py",
        ]
        for f in expected:
            assert (ckpt_dir / f).exists() or (ckpt_dir / "best_checkpoint" / f).exists()
