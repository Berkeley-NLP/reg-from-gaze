"""
Model and processor loading utilities for Molmo-REC-Gaze.
"""

from typing import Tuple, Any, Optional
from pathlib import Path
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


def load_molmo_model_and_processor(
    model_name_or_path: str = "allenai/Molmo-7B-D-0924",
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    freeze_vision_backbone: bool = True,
) -> Tuple[Any, Any]:
    """
    Load Molmo model and multimodal processor.

    Args:
        model_name_or_path: Hugging Face hub ID or path to checkpoint directory
        device_map: Accelerate device mapping ("auto", "cuda", etc.)
        torch_dtype: Precision (default: torch.bfloat16)
        freeze_vision_backbone: Whether to freeze vision ViT parameters for training

    Returns:
        (model, processor)
    """
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        tokenizer_kwargs={"chat_template": None, "add_special_tokens": False},
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if freeze_vision_backbone:
        for name, param in model.named_parameters():
            if name.startswith("vision_backbone"):
                param.requires_grad = False

    return model, processor


def export_model_and_processor(
    model: Any,
    processor: Any,
    output_dir: str,
) -> None:
    """
    Save complete weights and processor configurations into output_dir.
    Ensures zero-code drop-in loading via AutoModelForCausalLM.from_pretrained.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    processor.save_pretrained(str(out_path))
