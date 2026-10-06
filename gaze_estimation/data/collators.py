"""
Multimodal data collator for Molmo-REC-Gaze training.
Pads sequences, processes letterboxed images, and masks non-target tokens.
"""

from typing import List, Dict, Any
from pathlib import Path
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence

from gaze_estimation.data.transforms import letterbox_image


class GazeDataCollator:
    """
    Collator for batching images and text sequences for Molmo models.
    Masks padding and image patch tokens so cross-entropy loss is only computed on gaze & token predictions.
    """

    def __init__(self, processor: Any, max_length: int = 256):
        self.processor = processor
        self.max_length = max_length
        self.pad_id = processor.tokenizer.pad_token_id
        # Molmo image patch token ID
        self.patch_id = processor.tokenizer.convert_tokens_to_ids("<im_patch>")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        processed_items = []
        for sample in batch:
            img_path = sample.get("image_path")
            if img_path and Path(img_path).exists():
                raw_img = Image.open(img_path).convert("RGB")
            else:
                # Fallback blank black canvas if image missing
                raw_img = Image.new("RGB", (512, 320), color=(0, 0, 0))

            padded_img, _, _, _ = letterbox_image(raw_img, target_width=512, target_height=320)

            enc = self.processor.process(
                images=padded_img,
                text=sample["sequence"],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                message_format="none",
            )
            processed_items.append(enc)

        batch_enc = {}
        for k in processed_items[0]:
            tensors = [item[k] for item in processed_items]
            if all(t.shape == tensors[0].shape for t in tensors):
                batch_enc[k] = torch.stack(tensors)
            else:
                batch_enc[k] = pad_sequence(
                    tensors, batch_first=True, padding_value=self.pad_id if self.pad_id is not None else 0
                )

        # Labels for causal LM: mask patch tokens and pad tokens
        labels = batch_enc["input_ids"].clone()
        if self.patch_id is not None:
            labels.masked_fill_(labels == self.patch_id, -100)
        if self.pad_id is not None:
            labels.masked_fill_(labels == self.pad_id, -100)

        batch_enc["labels"] = labels
        return batch_enc
