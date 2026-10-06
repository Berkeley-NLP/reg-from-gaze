"""
Configuration pre-flight validation utilities for GazeRL.
Verifies model paths, candidate resolutions, and dataset presence before executing heavy operations.
"""
import logging
import os
from pathlib import Path

from configs.base import ExperimentConfig
from configs.constants import (
    resolve_gaze_predictor_path,
    resolve_molmo_listener_path,
)

logger = logging.getLogger(__name__)


def preflight_check(config: ExperimentConfig) -> None:
    """
    Perform fast pre-flight verification of configuration, paths, and datasets.
    Fails in milliseconds with actionable messages before loading multi-gigabyte models into GPU memory.
    """
    errors = []

    # 1. Listener model path check
    listener_path = config.listener.model_name_or_path
    if listener_path.startswith(("/", "./", "../")):
        if not os.path.exists(listener_path):
            if "vectorized" in listener_path:
                if "gaze" in config.listener.listener_type:
                    resolved = resolve_gaze_predictor_path(listener_path)
                else:
                    resolved = resolve_molmo_listener_path(listener_path)

                if os.path.exists(resolved):
                    logger.warning(
                        f"Configured listener path '{listener_path}' was not found. "
                        f"Auto-resolved to local candidate: '{resolved}'."
                    )
                    config.listener.model_name_or_path = resolved
                    listener_path = resolved

            if not os.path.exists(listener_path):
                errors.append(
                    f"Listener checkpoint path '{listener_path}' does not exist on disk.\n"
                    f"  -> Specify a valid directory via --listener-path <path> or set GAZERL_GAZE_PREDICTOR_PATH."
                )

    # 2. Speaker model check
    speaker_path = config.speaker.model_name_or_path
    if speaker_path.startswith(("/", "./", "../")) and not os.path.exists(speaker_path):
        errors.append(
            f"Speaker model path '{speaker_path}' does not exist on disk.\n"
            f"  -> Specify a valid directory or Hugging Face model identifier."
        )

    # 3. Dataset split files check
    try:
        from data.datasets import DatasetSplitManager
        split_mgr = DatasetSplitManager(config.dataset.splits_dir)
        split_mgr.load_splits(config.dataset.dataset_name)
    except Exception as e:
        errors.append(f"Failed to validate dataset splits: {e}")

    if errors:
        error_msg = "\n".join([f"  [{i+1}] {err}" for i, err in enumerate(errors)])
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"PRE-FLIGHT CONFIGURATION CHECK FAILED:\n"
            f"{'='*70}\n"
            f"{error_msg}\n"
            f"{'='*70}"
        )
