"""
CogVLM external evaluation listener for GazeRL.
Evaluates referring expression comprehension and bounding box grounding.
"""
from typing import List, Tuple, Optional, Dict, Any
import logging
import re
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from models.base import BaseListener, ListenerOutput
from models.listeners.eval.qwenvl_listener import calculate_iou

logger = logging.getLogger(__name__)


class CogVLMListener(BaseListener):
    """
    CogVLM evaluation listener model.
    """

    def __init__(
        self,
        config: Any = None,
        model_path: str = "THUDM/cogvlm-grounding-general-v1.4",
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__(config)
        self.model_path = model_path or getattr(config, "model_name_or_path", "THUDM/cogvlm-grounding-general-v1.4")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self) -> None:
        logger.info("Loading CogVLM listener from %s", self.model_path)
        from transformers import AutoModelForCausalLM, LlamaTokenizer

        self.tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    def parse_bbox_from_response(self, text: str, width: int, height: int) -> Optional[List[float]]:
        """Parse [[x0, y0, x1, y1]] from CogVLM response into pixel coords."""
        match = re.search(r"\[\[(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?),(\d+(?:\.\d+)?)\]\]", text)
        if match:
            try:
                x0, y0, x1, y1 = map(float, match.groups())
                # 0-1000 normalized to pixels
                px0 = (x0 / 1000.0) * width
                py0 = (y0 / 1000.0) * height
                px1 = (x1 / 1000.0) * width
                py1 = (y1 / 1000.0) * height
                return [px0, py0, px1, py1]
            except Exception:
                pass
        return None

    def predict_gaze_sequence(
        self,
        image: Image.Image,
        referring_expression: str,
        target_bbox_normalized: Tuple[float, float, float, float],
    ) -> ListenerOutput:
        w, h = image.size
        tx1, ty1, tx2, ty2 = target_bbox_normalized
        gold_bbox_px = [
            (tx1 / 100.0) * w,
            (ty1 / 100.0) * h,
            (tx2 / 100.0) * w,
            (ty2 / 100.0) * h,
        ]

        query = f"Question: Where is the {referring_expression}? Answer:"
        inputs = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=query,
            history=[],
            images=[image],
        )

        dev = next(self.model.parameters()).device
        inputs = {
            "input_ids": inputs["input_ids"].unsqueeze(0).to(dev),
            "token_type_ids": inputs["token_type_ids"].unsqueeze(0).to(dev),
            "attention_mask": inputs["attention_mask"].unsqueeze(0).to(dev),
            "images": [[inputs["images"][0].to(dev).to(torch.bfloat16)]],
        }

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
            in_len = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(outputs[0, in_len:])

        pred_bbox_px = self.parse_bbox_from_response(response, w, h)
        if pred_bbox_px:
            iou = calculate_iou(pred_bbox_px, gold_bbox_px)
            center_x = ((pred_bbox_px[0] + pred_bbox_px[2]) / 2.0 / w) * 100.0
            center_y = ((pred_bbox_px[1] + pred_bbox_px[3]) / 2.0 / h) * 100.0
            point = (center_x, center_y)
            hit = iou > 0.5
        else:
            point = None
            hit = False

        return ListenerOutput(
            gaze_points=[point],
            hit_mask=[hit],
            is_success=hit,
            first_hit_index=0 if hit else None,
            distances_to_bbox=[0.0 if hit else 100.0],
            raw_output_text=response,
        )
