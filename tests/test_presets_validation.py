"""
Tests for preset configuration validation, path resolution, and preflight checks.
Ensures that all YAML presets are structurally valid and that missing paths fail fast.
"""
from pathlib import Path
import pytest
import yaml

from configs.base import ExperimentConfig
from configs.constants import (
    resolve_gaze_predictor_path,
    resolve_molmo_listener_path,
    KNOWN_GAZE_PREDICTOR_CANDIDATES,
)
from rl.rewards import get_reward_function
from configs.validation import preflight_check


PRESET_DIR = Path(__file__).resolve().parent.parent / "configs" / "presets"


def get_all_presets():
    yaml_files = sorted(list(PRESET_DIR.glob("*.yaml")))
    assert len(yaml_files) >= 8, f"Expected at least 8 presets, found {len(yaml_files)}"
    return yaml_files


@pytest.mark.parametrize("preset_path", get_all_presets(), ids=lambda p: p.name)
def test_preset_structure_and_parsing(preset_path: Path):
    """Verify that every preset YAML can be loaded into an ExperimentConfig."""
    with open(preset_path) as f:
        data = yaml.safe_load(f)

    assert "speaker" in data, f"{preset_path.name} missing 'speaker'"
    assert "listener" in data, f"{preset_path.name} missing 'listener'"
    assert "reward" in data, f"{preset_path.name} missing 'reward'"

    config = ExperimentConfig.from_dict(data)
    assert config.name is not None
    assert config.speaker.architecture in ("molmo", "paligemma", "llava")
    assert config.listener.listener_type in (
        "gaze_predictor",
        "molmo_iterative",
        "rec_single_point",
        "eval_qwenvl",
    )


@pytest.mark.parametrize("preset_path", get_all_presets(), ids=lambda p: p.name)
def test_preset_reward_function_instantiation(preset_path: Path):
    """Verify that reward function can be instantiated directly from each preset config."""
    with open(preset_path) as f:
        data = yaml.safe_load(f)
    config = ExperimentConfig.from_dict(data)
    reward_fn = get_reward_function(config.reward)
    assert reward_fn is not None


@pytest.mark.parametrize("preset_path", get_all_presets(), ids=lambda p: p.name)
def test_preset_preflight_check(preset_path: Path):
    """Verify that preflight check succeeds on every paper preset."""
    with open(preset_path) as f:
        data = yaml.safe_load(f)
    config = ExperimentConfig.from_dict(data)
    # preflight_check should complete without raising FileNotFoundError
    preflight_check(config)


def test_path_resolution():
    """Verify that gaze predictor path resolves to an existing candidate if available."""
    resolved = resolve_gaze_predictor_path()
    assert resolved is not None
    existing_cands = [c for c in KNOWN_GAZE_PREDICTOR_CANDIDATES if Path(c).exists()]
    if existing_cands:
        assert Path(resolved).exists(), f"Resolved path '{resolved}' should exist on this system"


def test_preflight_check_fails_on_nonexistent_path():
    """Verify that preflight check catches non-existent local paths fast without loading models."""
    config = ExperimentConfig()
    config.listener.model_name_or_path = "/nonexistent/path/to/fake_gaze_predictor"
    config.listener.listener_type = "gaze_predictor"

    with pytest.raises(FileNotFoundError, match="PRE-FLIGHT CONFIGURATION CHECK FAILED"):
        preflight_check(config)
