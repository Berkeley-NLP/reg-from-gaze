"""
PyTorch Dataset implementations for Molmo-REC-Gaze.
Supports both token-level (subword alignment) and word-level sequence generation.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import torch
from torch.utils.data import Dataset

from gaze_estimation.data.transforms import letterbox_image, normalize_bbox


def make_token_level_sequence(
    record: Dict[str, Any],
    tokenizer: Any,
    orig_width: int = 1680,
    orig_height: int = 1050,
) -> str:
    """
    Construct a token-level interleaved sequence:
        BOS <x=.. y=..> tok1 <x=.. y=..> tok2 <x=.. y=..> ... EOS <x=.. y=..>

    Each subword token generated from word w_k inherits the exact same fixation coordinate as w_k.
    """
    ref_words = record.get("REF_WORDS") or record.get("ref_words", [])
    xs = record.get("FIX_X") or record.get("fix_x", [])
    ys = record.get("FIX_Y") or record.get("fix_y", [])
    wis = record.get("FIX_WORDINDEX") or record.get("fix_wordindex", [])

    # Map word index to last fixation index:
    # 0 = BOS (-99), 1..n = ref_words, n+1 = EOS (99)
    last_map = {}
    for i, widx in enumerate(wis):
        if widx == -99:
            if 0 not in last_map:
                last_map[0] = i
        elif 0 <= widx < len(ref_words):
            last_map[widx + 1] = i
        elif widx == 99:
            last_map[len(ref_words) + 1] = i

    def get_gaze_tag(word_idx: int) -> str:
        if word_idx in last_map and len(xs) > 0 and len(ys) > 0:
            fi = last_map[word_idx]
            nx = round(xs[fi] / orig_width * 100.0, 1)
            ny = round(ys[fi] / orig_height * 100.0, 1)
            return f"<x={nx} y={ny}>"
        return "<x= y=>"

    toks = ["BOS", get_gaze_tag(0)]

    for word_idx, word in enumerate(ref_words):
        # Tokenize word with leading space for standard BPE subword splitting
        word_tokens = tokenizer.tokenize(" " + word)
        cleaned_tokens = []
        for t in word_tokens:
            clean_t = t.replace("Ġ", "").replace("▁", "").replace(" ", "")
            if clean_t:
                cleaned_tokens.append(clean_t)

        if not cleaned_tokens:
            cleaned_tokens = [word]

        gaze_tag = get_gaze_tag(word_idx + 1)
        for subword in cleaned_tokens:
            toks.append(subword)
            toks.append(gaze_tag)

    toks.append("EOS")
    toks.append(get_gaze_tag(len(ref_words) + 1))

    return " ".join(toks)


def make_word_level_sequence(
    record: Dict[str, Any],
    orig_width: int = 1680,
    orig_height: int = 1050,
) -> str:
    """
    Construct a word-level interleaved sequence:
        BOS <x=.. y=..> word1 <x=.. y=..> word2 <x=.. y=..> ... EOS <x=.. y=..>
    """
    ref_words = record.get("REF_WORDS") or record.get("ref_words", [])
    xs = record.get("FIX_X") or record.get("fix_x", [])
    ys = record.get("FIX_Y") or record.get("fix_y", [])
    wis = record.get("FIX_WORDINDEX") or record.get("fix_wordindex", [])

    last_map = {}
    for i, widx in enumerate(wis):
        if widx == -99:
            if 0 not in last_map:
                last_map[0] = i
        elif 0 <= widx < len(ref_words):
            last_map[widx + 1] = i
        elif widx == 99:
            last_map[len(ref_words) + 1] = i

    def get_gaze_tag(word_idx: int) -> str:
        if word_idx in last_map and len(xs) > 0 and len(ys) > 0:
            fi = last_map[word_idx]
            nx = round(xs[fi] / orig_width * 100.0, 1)
            ny = round(ys[fi] / orig_height * 100.0, 1)
            return f"<x={nx} y={ny}>"
        return "<x= y=>"

    toks = ["BOS", get_gaze_tag(0)]
    for word_idx, word in enumerate(ref_words):
        toks.append(word)
        toks.append(get_gaze_tag(word_idx + 1))

    toks.append("EOS")
    toks.append(get_gaze_tag(len(ref_words) + 1))

    return " ".join(toks)


class TokenLevelGazeDataset(Dataset):
    """
    Dataset producing token-level (subword aligned) gaze prediction sequences.
    This is the canonical training dataset format used for Molmo-REC-Gaze.
    """

    def __init__(
        self,
        json_path: str,
        split: Optional[str] = None,
        image_base: str = "",
        tokenizer: Any = None,
        use_letterbox: bool = True,
    ):
        with open(json_path, "r") as f:
            raw = json.load(f)

        if split:
            self.samples = [
                r for r in raw
                if r.get("REFCOCO_SPLIT") == split or r.get("REFCOCO_GAZE_SPLIT") == split
            ]
            if len(self.samples) == 0:
                self.samples = raw
        else:
            self.samples = raw

        self.image_base = Path(image_base) if image_base else None
        self.tokenizer = tokenizer
        self.use_letterbox = use_letterbox

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.samples[idx])

        # Resolve image path
        img_filename = Path(sample.get("IMAGEFILE") or sample.get("imagefile", "")).name
        if self.image_base:
            sample["image_path"] = str(self.image_base / img_filename)
        else:
            sample["image_path"] = img_filename

        # Generate sequence
        if self.tokenizer:
            sample["sequence"] = make_token_level_sequence(sample, self.tokenizer)
        else:
            sample["sequence"] = make_word_level_sequence(sample)

        # Normalize bounding box
        bbox = sample.get("BBOX") or sample.get("bbox")
        sample["normalized_bbox"] = normalize_bbox(bbox)

        return sample


class WordLevelGazeDataset(Dataset):
    """
    Dataset producing word-level gaze prediction sequences.
    """

    def __init__(
        self,
        json_path: str,
        split: Optional[str] = None,
        image_base: str = "",
        use_letterbox: bool = True,
    ):
        with open(json_path, "r") as f:
            raw = json.load(f)

        if split:
            self.samples = [
                r for r in raw
                if r.get("REFCOCO_SPLIT") == split or r.get("REFCOCO_GAZE_SPLIT") == split
            ]
            if len(self.samples) == 0:
                self.samples = raw
        else:
            self.samples = raw

        self.image_base = Path(image_base) if image_base else None
        self.use_letterbox = use_letterbox

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = dict(self.samples[idx])

        img_filename = Path(sample.get("IMAGEFILE") or sample.get("imagefile", "")).name
        if self.image_base:
            sample["image_path"] = str(self.image_base / img_filename)
        else:
            sample["image_path"] = img_filename

        sample["sequence"] = make_word_level_sequence(sample)
        bbox = sample.get("BBOX") or sample.get("bbox")
        sample["normalized_bbox"] = normalize_bbox(bbox)

        return sample
