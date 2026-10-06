"""
Molmo Listener implementation for GazeRL (Iterative and Single-Point REC modes).
Uses AllenAI Molmo pointing capabilities to localize referring expressions in images.
"""
from typing import List, Tuple, Optional, Union, Dict, Any
import logging
import os
import random
import time
import re
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from models.base import BaseListener, ListenerOutput
from data.processing import is_point_in_bbox
from configs.constants import DEFAULT_MOLMO_MODEL

logger = logging.getLogger(__name__)


def parse_molmo_coordinate(text: str) -> Optional[List[List[float]]]:
    """
    Parse coordinate(s) from Molmo pointing output.
    Supports:
    1. <x="11.1" y="22.2"> or <x=11.1 y=22.2>
    2. <point x="11.1" y="22.2" alt="...">
    3. <points x1="11.1" y1="22.2" x2="..." y2="...">
    4. Fallback number extraction.

    Returns:
        List of [x, y] coordinates in 0-100 scale, or None if parsing fails.
    """
    if not text or not isinstance(text, str):
        return None

    # Pattern 1: <x=.. y=..>
    pattern0 = r'<x=["\']?\s*([0-9]+(?:\.[0-9]+)?)\s*["\']?\s+y=["\']?\s*([0-9]+(?:\.[0-9]+)?)\s*["\']?>'
    match0 = re.search(pattern0, text)
    if match0:
        try:
            return [[float(match0.group(1)), float(match0.group(2))]]
        except (ValueError, TypeError):
            pass

    # Pattern 2: <point x=".." y="..">
    pattern1 = r'<point\s+x=["\']([^"\']+)["\']\s+y=["\']([^"\']+)["\']'
    match1 = re.search(pattern1, text)
    if match1:
        try:
            return [[float(match1.group(1)), float(match1.group(2))]]
        except (ValueError, TypeError):
            pass

    # Pattern 3: <points ...>
    pattern2 = r'<points[^>]*>'
    points_match = re.search(pattern2, text)
    if points_match:
        coord_pattern = r'x\d*=["\']?\s*([0-9]+(?:\.[0-9]+)?)\s*["\']?[^>]*?y\d*=["\']?\s*([0-9]+(?:\.[0-9]+)?)\s*["\']?'
        found = []
        for m in re.finditer(coord_pattern, points_match.group(0)):
            try:
                found.append([float(m.group(1)), float(m.group(2))])
            except (ValueError, TypeError):
                continue
        if found:
            return found

    # Pattern 4: Fallback pair of floats
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(numbers) >= 2:
        try:
            return [[float(numbers[0]), float(numbers[1])]]
        except (ValueError, TypeError):
            pass

    return None


class MolmoListenerVectorized(BaseListener):
    """
    Vectorized Molmo Listener supporting both iterative step-by-step rollouts and single-point REC grounding.
    """

    def __init__(
        self,
        config: Any = None,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        processor: Optional[Any] = None,
        iterative_mode: bool = False,
        random_bos_point: bool = False,
        prompt_prefix: str = "locate",
    ):
        super().__init__(config)
        self.model_path = (
            model_path
            or getattr(config, "model_name_or_path", DEFAULT_MOLMO_MODEL)
            if config is not None
            else (model_path or DEFAULT_MOLMO_MODEL)
        )
        self.iterative_mode = (
            getattr(config, "iterative_mode", iterative_mode)
            if config is not None
            else iterative_mode
        )
        self.random_bos_point = (
            getattr(config, "random_bos_point", random_bos_point)
            if config is not None
            else random_bos_point
        )
        self.prompt_prefix = prompt_prefix

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = (self.device.type == "cuda")

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        else:
            self._load_model_and_processor()

    def _load_model_and_processor(self) -> None:
        # Validate path and try candidate fallback if local path doesn't exist
        from configs.constants import resolve_molmo_listener_path
        if self.model_path.startswith(("/", "./", "../")) and not os.path.exists(self.model_path):
            resolved = resolve_molmo_listener_path(self.model_path)
            if os.path.exists(resolved):
                logger.warning(
                    "MolmoListener path '%s' not found on disk. Auto-resolving to existing candidate: '%s'.",
                    self.model_path,
                    resolved,
                )
                self.model_path = resolved
            else:
                raise FileNotFoundError(
                    f"MolmoListener checkpoint directory not found at '{self.model_path}'. "
                    f"Please verify the path, set GAZERL_MOLMO_LISTENER_PATH, or pass --listener-path."
                )

        logger.info("Loading MolmoListener model from %s", self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not getattr(self.model, "hf_device_map", None):
            self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

    def _generate_random_point_outside_bbox(
        self,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        max_attempts: int = 100,
        validation_mode: bool = False,
    ) -> Optional[List[float]]:
        if validation_mode:
            rng = random
        else:
            try:
                rng = random.Random(int.from_bytes(os.urandom(4), "big"))
            except Exception:
                rng = random.Random()

        if bbox is None:
            return [float(f"{rng.uniform(0.0, 100.0):.2f}"), float(f"{rng.uniform(0.0, 100.0):.2f}")]

        x1, y1, x2, y2 = bbox
        for _ in range(max_attempts):
            rx = float(f"{rng.uniform(0.0, 100.0):.2f}")
            ry = float(f"{rng.uniform(0.0, 100.0):.2f}")
            if not is_point_in_bbox((rx, ry), bbox):
                return [rx, ry]
        return None

    def _build_sequences(self, tokens_list: List[List[str]]) -> Tuple[List[List[str]], int]:
        processed_sequences = []
        max_length = 0
        for tokens in tokens_list:
            processed_tokens = []
            has_eos = False
            for token in tokens:
                if token in ["<|endoftext|>", "</s>", "<eos>", "<EOS>"]:
                    processed_tokens.append("EOS")
                    has_eos = True
                else:
                    processed_tokens.append(token)
            if not has_eos:
                processed_tokens.append("EOS")
            seq = ["BOS"] + processed_tokens
            processed_sequences.append(seq)
            max_length = max(max_length, len(seq))
        return processed_sequences, max_length

    def predict_vectorized(
        self,
        images: List[Image.Image],
        tokens_list: List[List[str]],
        bboxes: Optional[List[Optional[Tuple[float, float, float, float]]]] = None,
        sample: bool = False,
        validation_mode: bool = False,
    ) -> List[List[Optional[List[float]]]]:
        """Run vectorized Molmo listener prediction across batch."""
        if bboxes is None:
            bboxes = [None] * len(images)
        if len(images) != len(tokens_list) or len(images) != len(bboxes):
            raise ValueError(
                f"Mismatched batch sizes: {len(images)} images, {len(tokens_list)} token lists, {len(bboxes)} bboxes"
            )

        if not self.iterative_mode:
            # Single-point REC mode: each prompt is f"{prompt_prefix} {' '.join(tokens)}"
            results: List[List[Optional[List[float]]]] = []
            for img, toks in zip(images, tokens_list):
                cleaned_text = " ".join([t for t in toks if t not in ["BOS", "EOS", "<|endoftext|>"]]).strip()
                prompt = f"{self.prompt_prefix} {cleaned_text}"
                inputs = self.processor.process(images=[img], text=prompt, return_tensors="pt")
                if "pixel_values" in inputs and "images" not in inputs:
                    inputs["images"] = inputs.pop("pixel_values")

                dev = next(self.model.parameters()).device
                tensor_inputs = {k: v.to(dev) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                if "images" in tensor_inputs and torch.cuda.is_available():
                    tensor_inputs["images"] = tensor_inputs["images"].to(torch.bfloat16)

                with torch.no_grad():
                    if hasattr(self.model, "generate_from_batch"):
                        out = self.model.generate_from_batch(
                            batch=tensor_inputs,
                            tokenizer=self.processor.tokenizer,
                            max_new_tokens=24,
                        )
                    else:
                        out = self.model.generate(**tensor_inputs, max_new_tokens=24)

                seqs = getattr(out, "sequences", out)
                gen_text = self.processor.tokenizer.decode(seqs[0], skip_special_tokens=False)
                parsed = parse_molmo_coordinate(gen_text)
                pt = parsed[0] if parsed else None
                # Return single point list aligned with tokens length
                results.append([pt] * max(len(toks), 1))
            return results

        # Iterative mode
        processed_sequences, max_length = self._build_sequences(tokens_list)
        base_sequences = processed_sequences
        results = [[] for _ in range(len(images))]

        transformed_images = [ImageOps.exif_transpose(img).convert("RGB") for img in images]
        cached_img_arrays = [np.array(img) for img in transformed_images]

        tok = self.processor.tokenizer
        gen_cfg = GenerationConfig(
            do_sample=sample,
            max_new_tokens=24,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )

        for token_idx in range(max_length):
            if token_idx == 0 and self.random_bos_point:
                for i, (img, bbox) in enumerate(zip(images, bboxes)):
                    if token_idx < len(base_sequences[i]):
                        coord = self._generate_random_point_outside_bbox(bbox, validation_mode=validation_mode)
                        results[i].append(coord)
                continue

            active_indices, active_images, active_prompts, active_bboxes = [], [], [], []
            for i, (img, bbox) in enumerate(zip(images, bboxes)):
                if token_idx < len(base_sequences[i]):
                    active_indices.append(i)
                    prompt_tokens = [self.prompt_prefix]
                    coords_so_far = results[i]

                    for j in range(token_idx):
                        w = base_sequences[i][j + 1]
                        prompt_tokens.append(w)
                        coord_idx = j
                        if coord_idx < len(coords_so_far) and coords_so_far[coord_idx] is not None:
                            cx, cy = coords_so_far[coord_idx][:2]
                            prompt_tokens.append(f'<point x="{cx:.1f}" y="{cy:.1f}">')

                    if token_idx < len(base_sequences[i]):
                        prompt_tokens.append(base_sequences[i][token_idx])

                    prompt_str = " ".join([t for t in prompt_tokens if t])
                    active_prompts.append(prompt_str)
                    active_bboxes.append(bbox)

            if not active_indices:
                break

            all_processor_outputs = []
            for i_local, active_idx in enumerate(active_indices):
                single_img_array = [cached_img_arrays[active_idx]]
                single_prompt = active_prompts[i_local]
                single_output = self.processor.process(
                    images=single_img_array,
                    text=single_prompt,
                    message_format="none",
                    return_tensors="pt",
                    sequence_length=1024,
                )
                if "pixel_values" in single_output and "images" not in single_output:
                    single_output["images"] = single_output.pop("pixel_values")
                all_processor_outputs.append(single_output)

            batch = {}
            all_input_ids = [out["input_ids"].squeeze(0) for out in all_processor_outputs]
            pad_token_id = self.processor.tokenizer.pad_token_id or 0

            reversed_input_ids = [t.flip(dims=[0]) for t in all_input_ids]
            padded_reversed_ids = torch.nn.utils.rnn.pad_sequence(
                reversed_input_ids,
                batch_first=True,
                padding_value=pad_token_id,
            )
            batch["input_ids"] = padded_reversed_ids.flip(dims=[1])

            all_attention_tensors = [torch.ones_like(ids, dtype=torch.long) for ids in all_input_ids]
            reversed_attention = [t.flip(dims=[0]) for t in all_attention_tensors]
            padded_reversed_attention = torch.nn.utils.rnn.pad_sequence(
                reversed_attention,
                batch_first=True,
                padding_value=0,
            )
            batch["attention_mask"] = padded_reversed_attention.flip(dims=[1])

            image_key = "images" if "images" in all_processor_outputs[0] else "pixel_values"
            if image_key in all_processor_outputs[0]:
                image_list = [out[image_key].squeeze(0) for out in all_processor_outputs]
                batch[image_key] = torch.stack(image_list, dim=0)

            target_device = next(self.model.parameters()).device
            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(target_device)

            if "images" in batch and torch.cuda.is_available():
                batch["images"] = batch["images"].to(torch.bfloat16)

            with torch.no_grad():
                if hasattr(self.model, "generate_from_batch"):
                    out = self.model.generate_from_batch(
                        batch=batch,
                        generation_config=gen_cfg,
                        tokenizer=self.processor.tokenizer,
                        return_dict_in_generate=True,
                    )
                else:
                    out = self.model.generate(
                        **batch,
                        generation_config=gen_cfg,
                        return_dict_in_generate=True,
                    )

            padded_prompt_len = batch["input_ids"].shape[1]
            generated_sequences = getattr(out, "sequences", out)

            for i_local, active_idx in enumerate(active_indices):
                gen_tokens = generated_sequences[i_local, padded_prompt_len:]
                generated_text = self.processor.tokenizer.decode(gen_tokens, skip_special_tokens=False)
                parsed = parse_molmo_coordinate(generated_text)
                coord = parsed[0] if parsed else None
                results[active_idx].append(coord)

        return results

    def predict_batch(
        self,
        images: List[Image.Image],
        tokens_list: List[List[str]],
        bboxes: Optional[List[Optional[Tuple[float, float, float, float]]]] = None,
        validation_mode: bool = False,
    ) -> List[List[Optional[List[float]]]]:
        return self.predict_vectorized(images, tokens_list, bboxes=bboxes, validation_mode=validation_mode)

    def predict(
        self,
        img: Image.Image,
        tokens: List[str],
        sample: bool = False,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        validation_mode: bool = False,
    ) -> List[Optional[List[float]]]:
        results = self.predict_vectorized([img], [tokens], [bbox], sample=sample, validation_mode=validation_mode)
        return results[0]

    def predict_iterative(
        self,
        img: Image.Image,
        tokens: List[str],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        validation_mode: bool = False,
    ) -> List[Optional[List[float]]]:
        return self.predict(img, tokens, bbox=bbox, validation_mode=validation_mode)

    def predict_gaze_sequence(
        self,
        image: Image.Image,
        referring_expression: str,
        target_bbox_normalized: Tuple[float, float, float, float],
    ) -> ListenerOutput:
        words = referring_expression.strip().split()
        if not words:
            words = ["."]

        coords = self.predict(image, words, bbox=target_bbox_normalized, validation_mode=True)

        gaze_points: List[Optional[Tuple[float, float]]] = []
        hit_mask: List[bool] = []
        distances: List[float] = []
        first_hit_idx: Optional[int] = None

        tx1, ty1, tx2, ty2 = target_bbox_normalized
        cx = (tx1 + tx2) / 2.0
        cy = (ty1 + ty2) / 2.0

        for idx, c in enumerate(coords):
            if c is not None and len(c) >= 2:
                px, py = c[0], c[1]
                gaze_points.append((px, py))
                hit = is_point_in_bbox((px, py), target_bbox_normalized)
                hit_mask.append(hit)
                dist = float(np.hypot(px - cx, py - cy))
                distances.append(dist)
                if hit and first_hit_idx is None:
                    first_hit_idx = idx
            else:
                gaze_points.append(None)
                hit_mask.append(False)
                distances.append(100.0)

        is_success = any(hit_mask)

        return ListenerOutput(
            gaze_points=gaze_points,
            hit_mask=hit_mask,
            is_success=is_success,
            first_hit_index=first_hit_idx,
            distances_to_bbox=distances,
            raw_output_text=referring_expression,
        )
