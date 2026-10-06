"""
Listeners module for GazeRL: GazePredictor, MolmoListener, and external evaluation listeners.
Provides get_listener factory function.
"""
from typing import Any, Optional, Union
import torch.nn as nn

from models.base import BaseListener
from models.listeners.gaze_predictor import GazePredictorVectorized
from models.listeners.molmo_listener import (
    MolmoListenerVectorized,
    parse_molmo_coordinate,
)
from models.listeners.eval import QwenVLListener, CogVLMListener


def get_listener(
    config: Any,
    model: Optional[nn.Module] = None,
    processor: Optional[Any] = None,
    **kwargs,
) -> BaseListener:
    """
    Factory function to instantiate the appropriate listener model based on config.

    Args:
        config: ListenerConfig instance, dict, or string listener identifier.
        model: Optional pre-loaded PyTorch model (useful for unit testing or custom injection).
        processor: Optional pre-loaded processor.

    Returns:
        BaseListener instance.
    """
    listener_type = ""
    if hasattr(config, "listener_type"):
        listener_type = str(config.listener_type).lower()
    elif isinstance(config, dict) and "listener_type" in config:
        listener_type = str(config["listener_type"]).lower()
    elif isinstance(config, dict) and "type" in config:
        listener_type = str(config["type"]).lower()
    elif isinstance(config, str):
        listener_type = config.lower()

    if not listener_type:
        model_name = ""
        if hasattr(config, "model_name_or_path"):
            model_name = str(config.model_name_or_path).lower()
        elif isinstance(config, dict) and "model_name_or_path" in config:
            model_name = str(config["model_name_or_path"]).lower()

        if "gaze" in model_name:
            listener_type = "gaze_predictor"
        elif "qwen" in model_name:
            listener_type = "eval_qwenvl"
        elif "cogvlm" in model_name:
            listener_type = "eval_cogvlm"
        elif "molmo" in model_name:
            listener_type = "molmo_iterative"
        else:
            listener_type = "gaze_predictor"

    if listener_type in ("gaze_predictor", "gazepredictor", "gaze_predictor_vectorized", "gazepredictorvectorized"):
        return GazePredictorVectorized(config=config, model=model, processor=processor, **kwargs)
    elif listener_type in ("molmo_iterative", "molmo_listener_vectorized", "molmolistenervectorized"):
        return MolmoListenerVectorized(config=config, model=model, processor=processor, iterative_mode=True, **kwargs)
    elif listener_type in ("molmo_single_point", "rec_single_point", "molmo_bbox_center", "molmolistener", "molmo_listener"):
        return MolmoListenerVectorized(config=config, model=model, processor=processor, iterative_mode=False, **kwargs)
    elif listener_type in ("eval_qwenvl", "qwenvl", "qwen_vl"):
        return QwenVLListener(config=config, model=model, processor=processor, **kwargs)
    elif listener_type in ("eval_cogvlm", "cogvlm"):
        return CogVLMListener(config=config, model=model, **kwargs)
    else:
        raise ValueError(
            f"Unsupported listener type: '{listener_type}'. Supported types: "
            f"'gaze_predictor', 'molmo_iterative', 'rec_single_point', 'eval_qwenvl', 'eval_cogvlm'."
        )


__all__ = [
    "BaseListener",
    "GazePredictorVectorized",
    "MolmoListenerVectorized",
    "QwenVLListener",
    "CogVLMListener",
    "parse_molmo_coordinate",
    "get_listener",
]
