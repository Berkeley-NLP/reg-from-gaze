"""
Speaker models module for GazeRL.
Exports MolmoSpeaker, PaliGemmaSpeaker, LlavaSpeaker, and get_speaker factory helper.
"""
from typing import Any, Optional, Union
import torch.nn as nn

from models.base import BaseSpeaker
from models.speakers.molmo_speaker import MolmoSpeaker
from models.speakers.paligemma_speaker import PaliGemmaSpeaker
from models.speakers.llava_speaker import LlavaSpeaker


def get_speaker(
    config: Any,
    model: Optional[nn.Module] = None,
    processor: Optional[Any] = None,
) -> BaseSpeaker:
    """
    Factory function to instantiate the appropriate speaker model based on config.

    Args:
        config: SpeakerConfig instance, dict, or object with 'architecture' or 'model_name_or_path' attribute.
        model: Optional pre-loaded PyTorch model (useful for unit testing or custom injection).
        processor: Optional pre-loaded processor.

    Returns:
        BaseSpeaker instance (MolmoSpeaker, PaliGemmaSpeaker, or LlavaSpeaker).
    """
    arch = ""
    if hasattr(config, "architecture"):
        arch = str(config.architecture).lower()
    elif isinstance(config, dict) and "architecture" in config:
        arch = str(config["architecture"]).lower()

    if not arch:
        model_name = ""
        if hasattr(config, "model_name_or_path"):
            model_name = str(config.model_name_or_path).lower()
        elif isinstance(config, dict) and "model_name_or_path" in config:
            model_name = str(config["model_name_or_path"]).lower()
        elif isinstance(config, str):
            model_name = config.lower()
            arch = model_name

        if "paligemma" in model_name:
            arch = "paligemma"
        elif "llava" in model_name:
            arch = "llava"
        elif "molmo" in model_name:
            arch = "molmo"

    if arch == "molmo":
        return MolmoSpeaker(config=config, model=model, processor=processor)
    elif arch == "paligemma":
        return PaliGemmaSpeaker(config=config, model=model, processor=processor)
    elif arch == "llava":
        return LlavaSpeaker(config=config, model=model, processor=processor)
    else:
        raise ValueError(
            f"Unsupported speaker architecture: '{arch}'. Supported architectures: 'molmo', 'paligemma', 'llava'."
        )


__all__ = [
    "BaseSpeaker",
    "MolmoSpeaker",
    "PaliGemmaSpeaker",
    "LlavaSpeaker",
    "get_speaker",
]
