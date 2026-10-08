"""
Vectorized Gaze Predictor listener for GazeRL.
Processes multiple images and expression token sequences simultaneously using batched left-padded generation.
Supports both point predictions and Gaussian uncertainty distributions.
"""
from typing import List, Tuple, Optional, Union, Dict, Any, Sequence
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
from data.processing import (
    parse_coordinate,
    parse_gaussian_coordinate,
    is_point_in_bbox,
    calculate_gaussian_bbox_probability,
)
from configs.constants import DEFAULT_GAZE_PREDICTOR_WEIGHTS

logger = logging.getLogger(__name__)


class GazePredictorVectorized(BaseListener):
    """
    Batched Gaze Predictor listener model.
    Generates incremental gaze scanpaths conditioned on image and reference sequence.
    """

    def __init__(
        self,
        config: Any = None,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
        processor: Optional[Any] = None,
        gaussian_mode: Optional[bool] = None,
        log_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        random_bos_point: bool = True,
    ):
        super().__init__(config)
        self.model_path = (
            model_path
            or getattr(config, "model_name_or_path", DEFAULT_GAZE_PREDICTOR_WEIGHTS)
            if config is not None
            else (model_path or DEFAULT_GAZE_PREDICTOR_WEIGHTS)
        )

        if gaussian_mode is not None:
            self.gaussian_mode = gaussian_mode
        elif config is not None:
            self.gaussian_mode = getattr(config, "gaussian_mode", False) or (
                getattr(config, "reward_type", "") == "gaussian_shaping"
            )
        else:
            self.gaussian_mode = False

        self.random_bos_point = (
            getattr(config, "random_bos_point", random_bos_point)
            if config is not None
            else random_bos_point
        )
        self.log_dir = log_dir
        self.run_name = run_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = (self.device.type == "cuda")

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
        else:
            self._load_model_and_processor()

        if self.processor is not None and hasattr(self.processor, "tokenizer"):
            tok = self.processor.tokenizer
            if tok.pad_token_id is None:
                if tok.eos_token_id is not None:
                    tok.pad_token = tok.eos_token
                    tok.pad_token_id = tok.eos_token_id
                else:
                    tok.add_special_tokens({"pad_token": "<|pad|>"})

    def _load_model_and_processor(self) -> None:
        # Validate path and try candidate fallback if local path doesn't exist
        from configs.constants import resolve_gaze_predictor_path
        if self.model_path.startswith(("/", "./", "../")) and not os.path.exists(self.model_path):
            resolved = resolve_gaze_predictor_path(self.model_path)
            if os.path.exists(resolved) or (not resolved.startswith(("/", "./", "../"))):
                logger.warning(
                    "GazePredictor path '%s' not found on disk. Auto-resolving to: '%s'.",
                    self.model_path,
                    resolved,
                )
                self.model_path = resolved
            else:
                raise FileNotFoundError(
                    f"GazePredictor checkpoint directory not found at '{self.model_path}'. "
                    f"Please verify the path, set GAZERL_GAZE_PREDICTOR_PATH, or use 'Berkeley-NLP/Molmo-REC-Gaze'."
                )

        logger.info("Loading GazePredictor model from %s", self.model_path)
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

    def _is_point_in_bbox_internal(
        self, point: Optional[Sequence[float]], bbox: Optional[Tuple[float, float, float, float]]
    ) -> bool:
        if bbox is None or point is None:
            return False
        if len(point) >= 2:
            return is_point_in_bbox((point[0], point[1]), bbox)
        return False

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
            rx = float(f"{rng.uniform(0.0, 100.0):.2f}")
            ry = float(f"{rng.uniform(0.0, 100.0):.2f}")
            if self.gaussian_mode:
                std_dev = float(f"{rng.uniform(20.0, 40.0):.2f}")
                return [rx, ry, std_dev]
            return [rx, ry]

        x1, y1, x2, y2 = bbox
        MARGIN = 0.1
        outside_regions = []
        if x1 > MARGIN:
            outside_regions.append(("left", 0.0, x1 - MARGIN, 0.0, 100.0))
        if x2 < 100.0 - MARGIN:
            outside_regions.append(("right", x2 + MARGIN, 100.0, 0.0, 100.0))
        if y1 > MARGIN:
            if x1 > 0:
                outside_regions.append(("below_left", 0.0, x1 - MARGIN, 0.0, y1 - MARGIN))
            if x2 < 100.0:
                outside_regions.append(("below_right", x2 + MARGIN, 100.0, 0.0, y1 - MARGIN))
        if y2 < 100.0 - MARGIN:
            if x1 > 0:
                outside_regions.append(("above_left", 0.0, x1 - MARGIN, y2 + MARGIN, 100.0))
            if x2 < 100.0:
                outside_regions.append(("above_right", x2 + MARGIN, 100.0, y2 + MARGIN, 100.0))

        if not outside_regions:
            corners = [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (100.0, 100.0)]
            for cx, cy in corners:
                if not self._is_point_in_bbox_internal([cx, cy], bbox):
                    if self.gaussian_mode:
                        std_dev = float(f"{rng.uniform(20.0, 40.0):.2f}")
                        return [cx, cy, std_dev]
                    return [cx, cy]
            return None

        for _ in range(min(max_attempts, 8)):
            _, min_x, max_x, min_y, max_y = rng.choice(outside_regions)
            rx = float(f"{rng.uniform(min_x, max_x):.2f}")
            ry = float(f"{rng.uniform(min_y, max_y):.2f}")
            p = [rx, ry]
            if not self._is_point_in_bbox_internal(p, bbox):
                if self.gaussian_mode:
                    std_dev = float(f"{rng.uniform(20.0, 40.0):.2f}")
                    return [rx, ry, std_dev]
                return p
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
        """
        Batched gaze rollout over multi-token sequences.

        Args:
            images: List of PIL images.
            tokens_list: List of token string sequences.
            bboxes: Optional bounding boxes in 0-100 coordinates.
            sample: Whether to sample during generation.
            validation_mode: Deterministic mode for validation.

        Returns:
            List of trajectory coordinates per example.
        """
        if bboxes is None:
            bboxes = [None] * len(images)
        if len(images) != len(tokens_list) or len(images) != len(bboxes):
            raise ValueError(
                f"Mismatched batch sizes: {len(images)} images, {len(tokens_list)} token lists, {len(bboxes)} bboxes"
            )

        processed_sequences, max_length = self._build_sequences(tokens_list)
        base_sequences = processed_sequences
        results: List[List[Optional[List[float]]]] = [[] for _ in range(len(images))]

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
                    prompt_tokens = ["BOS"]
                    coords_so_far = results[i]

                    if token_idx == 0 and (self.gaussian_mode or not self.random_bos_point):
                        active_images.append(img)
                        active_prompts.append("BOS")
                        active_bboxes.append(bbox)
                        continue

                    if len(coords_so_far) > 0 and coords_so_far[0] is not None:
                        if self.gaussian_mode and len(coords_so_far[0]) == 3:
                            cx, cy, std = coords_so_far[0]
                            prompt_tokens.append(f"<x={cx} y={cy} v={std}>")
                        elif (not self.gaussian_mode) and len(coords_so_far[0]) >= 2:
                            cx, cy = coords_so_far[0][:2]
                            prompt_tokens.append(f"<x={cx} y={cy}>")
                        else:
                            prompt_tokens.append("<x= y= v=>" if self.gaussian_mode else "<x= y=>")
                    else:
                        prompt_tokens.append("<x= y= v=>" if self.gaussian_mode else "<x= y=>")

                    words_up_to_current = base_sequences[i][1:token_idx]
                    for j, word in enumerate(words_up_to_current):
                        prompt_tokens.append(word)
                        coord_idx = j + 1
                        if coord_idx < len(coords_so_far) and coords_so_far[coord_idx] is not None:
                            coord_val = coords_so_far[coord_idx]
                            if self.gaussian_mode and len(coord_val) == 3:
                                cx, cy, std = coord_val
                                prompt_tokens.append(f"<x={cx} y={cy} v={std}>")
                            elif (not self.gaussian_mode) and len(coord_val) >= 2:
                                cx, cy = coord_val[:2]
                                prompt_tokens.append(f"<x={cx} y={cy}>")
                            else:
                                prompt_tokens.append("<x= y= v=>" if self.gaussian_mode else "<x= y=>")
                        else:
                            prompt_tokens.append("<x= y= v=>" if self.gaussian_mode else "<x= y=>")

                    if token_idx > 0:
                        current_word = base_sequences[i][token_idx] if token_idx < len(base_sequences[i]) else None
                        if current_word:
                            prompt_tokens.append(current_word)

                    cleaned_tokens = [tok.strip() for tok in prompt_tokens if tok.strip()]
                    prompt_str = re.sub(r" +", " ", " ".join(cleaned_tokens))
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

            # Left-pad input_ids
            reversed_input_ids = [t.flip(dims=[0]) for t in all_input_ids]
            padded_reversed_ids = torch.nn.utils.rnn.pad_sequence(
                reversed_input_ids,
                batch_first=True,
                padding_value=pad_token_id,
            )
            batch["input_ids"] = padded_reversed_ids.flip(dims=[1])

            # Left-pad attention_mask
            all_attention_tensors = [torch.ones_like(ids, dtype=torch.long) for ids in all_input_ids]
            reversed_attention = [t.flip(dims=[0]) for t in all_attention_tensors]
            padded_reversed_attention = torch.nn.utils.rnn.pad_sequence(
                reversed_attention,
                batch_first=True,
                padding_value=0,
            )
            batch["attention_mask"] = padded_reversed_attention.flip(dims=[1])

            # Stack images
            image_key = "images" if "images" in all_processor_outputs[0] else "pixel_values"
            if image_key in all_processor_outputs[0]:
                image_list = []
                for out in all_processor_outputs:
                    img_t = out[image_key]
                    if img_t.dim() > 3:
                        img_t = img_t.squeeze(0)
                    image_list.append(img_t)
                batch[image_key] = torch.stack(image_list, dim=0)

            # Target device mapping
            if not getattr(self.model, "hf_device_map", None):
                target_device = getattr(self.model, "device", self.device)
            else:
                try:
                    target_device = next(self.model.parameters()).device
                except Exception:
                    target_device = self.device

            for k, v in list(batch.items()):
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(target_device)

            if "images" in batch and torch.cuda.is_available():
                batch["images"] = batch["images"].to(torch.bfloat16)

            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=torch.bfloat16)
                if self.is_cuda
                else torch.autocast(enabled=False, device_type="cpu")
            )

            with torch.no_grad():
                with autocast_ctx:
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

                coord = None
                if self.gaussian_mode:
                    parsed_g = parse_gaussian_coordinate(generated_text)
                    if parsed_g is not None:
                        coord = list(parsed_g)
                else:
                    parsed_p = parse_coordinate(generated_text)
                    if parsed_p is not None:
                        coord = list(parsed_p)

                results[active_idx].append(coord)

        return results

    def _coords_to_listener_output(
        self,
        coords: List[Optional[List[float]]],
        target_bbox_normalized: Optional[Tuple[float, float, float, float]] = None,
        raw_text: Optional[str] = None,
    ) -> ListenerOutput:
        gaze_points: List[Optional[Tuple[float, float]]] = []
        gaussian_params: List[Optional[Tuple[float, float, float]]] = []
        hit_mask: List[bool] = []
        distances: List[float] = []
        first_hit_idx: Optional[int] = None

        if target_bbox_normalized is not None:
            tx1, ty1, tx2, ty2 = target_bbox_normalized
            cx = (tx1 + tx2) / 2.0
            cy = (ty1 + ty2) / 2.0
        else:
            cx, cy = 0.0, 0.0

        for idx, c in enumerate(coords):
            if c is not None and len(c) >= 2:
                px, py = c[0], c[1]
                gaze_points.append((px, py))
                hit = is_point_in_bbox((px, py), target_bbox_normalized) if target_bbox_normalized is not None else False
                hit_mask.append(hit)
                dist = float(np.hypot(px - cx, py - cy)) if target_bbox_normalized is not None else 0.0
                distances.append(dist)
                if hit and first_hit_idx is None:
                    first_hit_idx = idx

                if len(c) == 3:
                    gaussian_params.append((px, py, c[2]))
                else:
                    gaussian_params.append(None)
            else:
                gaze_points.append(None)
                gaussian_params.append(None)
                hit_mask.append(False)
                distances.append(100.0)

        is_success = any(hit_mask)
        return ListenerOutput(
            gaze_points=gaze_points,
            gaussian_params=gaussian_params if self.gaussian_mode else None,
            hit_mask=hit_mask,
            is_success=is_success,
            first_hit_index=first_hit_idx,
            distances_to_bbox=distances,
            raw_output_text=raw_text,
        )

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
        images: Optional[Union[Image.Image, List[Image.Image]]] = None,
        tokens_list: Optional[Union[List[str], List[List[str]]]] = None,
        bboxes: Optional[Union[Tuple[float, float, float, float], List[Tuple[float, float, float, float]]]] = None,
        img: Optional[Image.Image] = None,
        tokens: Optional[List[str]] = None,
        sample: bool = False,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        validation_mode: bool = False,
        **kwargs,
    ) -> List[ListenerOutput]:
        """
        Batched or single listener prediction returning List[ListenerOutput].
        Fully compatible with BaseListener.predict and legacy calls.
        """
        if images is None and img is not None:
            images = [img]
        elif isinstance(images, Image.Image):
            images = [images]
        elif images is None:
            images = kwargs.get("image")
            if isinstance(images, Image.Image):
                images = [images]

        if tokens_list is None and tokens is not None:
            tokens_list = [tokens]
        elif isinstance(tokens_list, list) and (len(tokens_list) == 0 or isinstance(tokens_list[0], str)):
            tokens_list = [tokens_list]
        elif tokens_list is None:
            tok = kwargs.get("tokens")
            if tok is not None:
                tokens_list = [tok]

        if bboxes is None and bbox is not None:
            bboxes_list = [bbox]
        elif bboxes is not None and not isinstance(bboxes, list):
            bboxes_list = [bboxes]
        elif bboxes is not None:
            bboxes_list = bboxes
        else:
            cand = kwargs.get("target_bboxes_normalized") or kwargs.get("target_bboxes") or kwargs.get("bboxes")
            if cand is not None:
                bboxes_list = cand if isinstance(cand, list) else [cand]
            else:
                bboxes_list = [None] * (len(images) if images else 1)

        raw_batch = self.predict_vectorized(
            images,
            tokens_list,
            bboxes=bboxes_list,
            sample=sample,
            validation_mode=validation_mode,
        )

        outputs = []
        for coords, tgt_bbox, toks in zip(raw_batch, bboxes_list, tokens_list):
            raw_text = " ".join(toks) if isinstance(toks, list) else str(toks)
            outputs.append(self._coords_to_listener_output(coords, tgt_bbox, raw_text=raw_text))
        return outputs

    def predict_iterative(
        self,
        img: Image.Image,
        tokens: List[str],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        validation_mode: bool = False,
    ) -> List[Optional[List[float]]]:
        return self.predict_vectorized([img], [tokens], [bbox], validation_mode=validation_mode)[0]

    def predict_gaze_sequence(
        self,
        image: Image.Image,
        referring_expression: str,
        target_bbox_normalized: Tuple[float, float, float, float],
    ) -> ListenerOutput:
        """
        Implementation of abstract BaseListener method.
        Computes structured ListenerOutput for single episode evaluation.
        """
        words = referring_expression.strip().split()
        if not words:
            words = ["."]

        coords = self.predict_vectorized([image], [words], [target_bbox_normalized], validation_mode=True)[0]
        return self._coords_to_listener_output(coords, target_bbox_normalized, raw_text=referring_expression)
