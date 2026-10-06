"""
PyTorch Dataset classes and split loaders for RefCOCO and COCO datasets.
"""
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

from configs.prompts import get_prompts
from data.processing import (
    convert_bbox_to_minmax,
    normalize_bbox_for_gaze_predictor,
    create_speaker_bbox_overlay,
    create_listener_padded_image,
)

logger = logging.getLogger(__name__)


@dataclass
class ReferringExpressionSample:
    """A single referring expression training or evaluation sample."""
    sample_id: str
    image: Image.Image
    bbox_minmax: Tuple[float, float, float, float]
    bbox_normalized: Tuple[float, float, float, float]
    speaker_image: Image.Image
    listener_image: Image.Image
    prompt: str
    gold_references: Optional[List[str]] = None


class DatasetSplitManager:
    """Manages loading and filtering by deterministic split JSON files."""

    def __init__(self, splits_dir: Union[str, Path] = "data/splits"):
        self.splits_dir = Path(splits_dir)

    def load_splits(self, dataset_name: str = "coco_2014") -> Tuple[List[str], List[str]]:
        """
        Load train and validation split IDs for the given dataset.
        
        Args:
            dataset_name: 'refcoco', 'coco_2014', or 'coco_2017'

        Returns:
            Tuple of (train_ids, val_ids)
        """
        name_clean = dataset_name.lower().replace("-", "_")

        if "refcoco" in name_clean:
            train_file = self.splits_dir / "refcoco_train_ids.json"
            val_file = self.splits_dir / "refcoco_val_ids.json"
        elif "2017" in name_clean:
            train_file = self.splits_dir / "coco_2017_train_ids.json"
            val_file = self.splits_dir / "coco_2017_val_ids.json"
        else:
            # Default to COCO 2014
            train_file = self.splits_dir / "coco_2014_train_ids.json"
            val_file = self.splits_dir / "coco_2014_val_ids.json"

        if not train_file.exists() or not val_file.exists():
            raise FileNotFoundError(
                f"Split files not found in {self.splits_dir} for {dataset_name}. "
                f"Expected: {train_file.name} and {val_file.name}"
            )

        with open(train_file, "r") as f:
            train_ids = json.load(f)
        with open(val_file, "r") as f:
            val_ids = json.load(f)

        logger.info("Loaded %d train and %d val split IDs for %s", len(train_ids), len(val_ids), dataset_name)
        return train_ids, val_ids

    def create_custom_split(
        self,
        dataset_name: str,
        all_ids: List[str],
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate and persist a deterministic train/validation split for a new dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g. 'my_dataset').
            all_ids: Full list of sample or image IDs.
            train_ratio: Fraction of data allocated to training (default 0.90).
            seed: Random seed for deterministic shuffling.

        Returns:
            Tuple of (train_ids, val_ids)
        """
        rng = random.Random(seed)
        shuffled = list(all_ids)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        train_ids = sorted(shuffled[:split_idx])
        val_ids = sorted(shuffled[split_idx:])

        name_clean = dataset_name.lower().replace("-", "_")
        train_file = self.splits_dir / f"{name_clean}_train_ids.json"
        val_file = self.splits_dir / f"{name_clean}_val_ids.json"
        meta_file = self.splits_dir / f"{name_clean}_split_metadata.json"

        self.splits_dir.mkdir(parents=True, exist_ok=True)
        with open(train_file, "w") as f:
            json.dump(train_ids, f, indent=2)
        with open(val_file, "w") as f:
            json.dump(val_ids, f, indent=2)
        with open(meta_file, "w") as f:
            json.dump({
                "dataset_name": dataset_name,
                "train_count": len(train_ids),
                "val_count": len(val_ids),
                "train_ratio": train_ratio,
                "seed": seed,
            }, f, indent=2)

        logger.info("Generated %d train and %d val IDs for %s in %s", len(train_ids), len(val_ids), dataset_name, self.splits_dir)
        return train_ids, val_ids


class SyntheticReferringExpressionDataset(Dataset):
    """
    Synthetic dataset generating clean PIL images with random geometric shapes and boxes.
    Designed for fast, zero-dependency unit tests, CI, and smoke testing.
    """

    def __init__(
        self,
        num_samples: int = 20,
        image_size: Tuple[int, int] = (640, 480),
        prompt_type: str = "all",
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.prompts = get_prompts(prompt_type)
        self.seed = seed

        # Generate deterministic synthetic samples
        rng = random.Random(seed)
        self.samples = []
        for i in range(num_samples):
            w, h = image_size
            # Random box occupying between 10% and 40% of image dimensions
            bw = rng.randint(int(w * 0.15), int(w * 0.40))
            bh = rng.randint(int(h * 0.15), int(h * 0.40))
            x1 = rng.randint(10, w - bw - 10)
            y1 = rng.randint(10, h - bh - 10)
            x2 = x1 + bw
            y2 = y1 + bh
            self.samples.append({
                "sample_id": f"synthetic_{i:04d}",
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "color": (rng.randint(50, 220), rng.randint(50, 220), rng.randint(50, 220)),
                "prompt_idx": i % len(self.prompts),
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> ReferringExpressionSample:
        data = self.samples[idx]
        w, h = self.image_size

        # Create synthetic base image with gradient background and shape
        img = Image.new("RGB", (w, h), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = data["bbox"]
        draw.ellipse([x1, y1, x2, y2], fill=data["color"], outline="black")

        bbox_minmax = (x1, y1, x2, y2)
        bbox_norm = normalize_bbox_for_gaze_predictor(bbox_minmax, (w, h))
        speaker_img = create_speaker_bbox_overlay(img, bbox_minmax)
        listener_img = create_listener_padded_image(img)
        prompt = self.prompts[data["prompt_idx"]]

        return ReferringExpressionSample(
            sample_id=data["sample_id"],
            image=img,
            bbox_minmax=bbox_minmax,
            bbox_normalized=bbox_norm,
            speaker_image=speaker_img,
            listener_image=listener_img,
            prompt=prompt,
            gold_references=["the colored circle in the image"],
        )


class ReferringExpressionDataset(Dataset):
    """
    Standard Referring Expression Dataset wrapping RefCOCO or COCO datasets.
    """

    def __init__(
        self,
        dataset_name: str = "coco_2014",
        split: str = "train",
        splits_dir: Union[str, Path] = "data/splits",
        prompt_type: str = "all",
        max_samples: Optional[int] = None,
        hf_dataset: Optional[Any] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.prompt_type = prompt_type
        self.prompts = get_prompts(prompt_type)
        self.split_manager = DatasetSplitManager(splits_dir)

        if hf_dataset is not None:
            self.hf_dataset = hf_dataset
        else:
            # Lazy load Hugging Face dataset if available
            try:
                from datasets import load_dataset
                hf_name = (
                    "lmms-lab/refcoco" if "refcoco" in dataset_name.lower()
                    else "NaiveDev/coco-2014-instance"
                )
                raw_ds = load_dataset(hf_name, split="train")
                train_ids, val_ids = self.split_manager.load_splits(dataset_name)
                target_ids = set(train_ids if split == "train" else val_ids)
                id_col = "ref_id" if "ref_id" in raw_ds.column_names else "id"
                self.hf_dataset = raw_ds.filter(lambda ex: str(ex.get(id_col, "")) in target_ids)
            except Exception as e:
                logger.warning(
                    "Could not load HuggingFace dataset %s: %s. Falling back to synthetic mode.",
                    dataset_name, e
                )
                self.hf_dataset = None

        if self.hf_dataset is None:
            # Fallback to synthetic
            self._synthetic_fallback = SyntheticReferringExpressionDataset(
                num_samples=max_samples or 20,
                prompt_type=prompt_type,
            )
        else:
            self._synthetic_fallback = None

        self.max_samples = max_samples

    def __len__(self) -> int:
        if self._synthetic_fallback is not None:
            return len(self._synthetic_fallback)
        total = len(self.hf_dataset)
        return min(total, self.max_samples) if self.max_samples else total

    def __getitem__(self, idx: int) -> ReferringExpressionSample:
        if self._synthetic_fallback is not None:
            return self._synthetic_fallback[idx]

        example = self.hf_dataset[idx]
        raw_image = example["image"]
        if not isinstance(raw_image, Image.Image):
            raw_image = Image.fromarray(raw_image)
        raw_image = raw_image.convert("RGB")

        orig_w, orig_h = raw_image.size
        raw_bbox = example.get("bbox", [0, 0, orig_w, orig_h])
        bbox_minmax = convert_bbox_to_minmax(raw_bbox, (orig_w, orig_h))
        bbox_norm = normalize_bbox_for_gaze_predictor(bbox_minmax, (orig_w, orig_h))

        speaker_img = create_speaker_bbox_overlay(raw_image, bbox_minmax)
        listener_img = create_listener_padded_image(raw_image)

        prompt = random.choice(self.prompts)
        sample_id = str(example.get("ref_id", example.get("id", f"sample_{idx}")))
        gold_refs = example.get("sentences", example.get("captions", None))
        if gold_refs and isinstance(gold_refs, list) and isinstance(gold_refs[0], dict):
            gold_refs = [s.get("raw", s.get("sent", "")) for s in gold_refs]

        return ReferringExpressionSample(
            sample_id=sample_id,
            image=raw_image,
            bbox_minmax=bbox_minmax,
            bbox_normalized=bbox_norm,
            speaker_image=speaker_img,
            listener_image=listener_img,
            prompt=prompt,
            gold_references=gold_refs,
        )


def collate_referring_expression_batch(
    batch: List[ReferringExpressionSample],
) -> Dict[str, Any]:
    """
    Collate a list of ReferringExpressionSample items into a batch dict.
    """
    return {
        "sample_ids": [s.sample_id for s in batch],
        "images": [s.image for s in batch],
        "bbox_minmax": [s.bbox_minmax for s in batch],
        "bbox_normalized": torch.tensor([s.bbox_normalized for s in batch], dtype=torch.float32),
        "speaker_images": [s.speaker_image for s in batch],
        "listener_images": [s.listener_image for s in batch],
        "prompts": [s.prompt for s in batch],
        "gold_references": [s.gold_references for s in batch],
    }
