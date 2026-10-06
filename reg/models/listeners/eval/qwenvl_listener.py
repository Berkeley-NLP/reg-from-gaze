"""
Qwen2.5-VL external evaluation listener for GazeRL.
Evaluates referring expression comprehension and bounding box grounding.
"""
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
import re
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from models.base import BaseListener, ListenerOutput
from data.processing import is_point_in_bbox

logger = logging.getLogger(__name__)


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate IoU between two bboxes [x1, y1, x2, y2]."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class QwenVLListener(BaseListener):
    """
    Qwen2.5-VL evaluation listener model.
    """

    def __init__(
        self,
        config: Any = None,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        model: Optional[nn.Module] = None,
        processor: Optional[Any] = None,
    ):
        super().__init__(config)
        self.model_path = model_path or getattr(config, "model_name_or_path", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        else:
            self._load_model_and_processor()

    def _load_model_and_processor(self) -> None:
        logger.info("Loading Qwen-VL listener from %s", self.model_path)
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

    def parse_bbox_from_response(self, text: str, width: int, height: int) -> Optional[List[float]]:
        """Parse predicted bounding box from Qwen-VL response in pixel coordinates."""
        # Check for JSON format
        json_match = re.search(r"\{.*?\}", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if "bbox_2d" in data:
                    b = data["bbox_2d"]
                    return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
            except Exception:
                pass

        # Check for [ymin, xmin, ymax, xmax] or [x1, y1, x2, y2]
        match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", text)
        if match:
            coords = [float(x) for x in match.groups()]
            # Qwen often outputs 0-1000 scale: [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = coords
            x1 = (xmin / 1000.0) * width
            y1 = (ymin / 1000.0) * height
            x2 = (xmax / 1000.0) * width
            y2 = (ymax / 1000.0) * height
            return [x1, y1, x2, y2]

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

        prompt = f"Locate '{referring_expression}' in the image and output the bounding box in [ymin, xmin, ymax, xmax] 0-1000 format."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        dev = next(self.model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=64)
            in_len = inputs["input_ids"].shape[1]
            out_tokens = generated_ids[0, in_len:]
            response = self.processor.decode(out_tokens, skip_special_tokens=True)

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
