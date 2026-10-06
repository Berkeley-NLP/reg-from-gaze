"""
Multi-modal processing, coordinate geometry, bounding-box normalization, and gaze parsing.
"""
from typing import Optional, Tuple, List, Union, Sequence
import re
import math
import numpy as np
import torch
from PIL import Image, ImageDraw

from configs.constants import (
    LISTENER_IMAGE_WIDTH,
    LISTENER_IMAGE_HEIGHT,
    SPEAKER_MAX_DIM,
    COORDINATE_SCALE,
)


def parse_coordinate(text: str) -> Optional[Tuple[float, float]]:
    """
    Parse point gaze coordinates with format <x=xx.x y=yy.y> (0-100 scale).

    Args:
        text: String containing coordinate tag.

    Returns:
        (x, y) tuple of floats or None if parsing fails.
    """
    text = text.strip()
    if re.match(r'<x=\s*y=>', text, re.IGNORECASE) or text in ('<x=', '<x= '):
        return None

    pattern = r'<x=(\d+(?:\.\d+)?)\s+y=(\d+(?:\.\d+)?)>'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except ValueError:
            return None
    return None


def parse_gaussian_coordinate(text: str) -> Optional[Tuple[float, float, float]]:
    """
    Parse Gaussian gaze coordinates with format <x=xx.x y=yy.y v=vv.v>
    where v is standard deviation of the 2D isotropic Gaussian.

    Args:
        text: String containing coordinate tag.

    Returns:
        (x, y, std_dev) tuple of floats or None if parsing fails.
    """
    text = text.strip()
    if re.match(r'<x=\s*y=\s*v=>', text, re.IGNORECASE) or text in ('<x=', '<x= '):
        return None

    pattern = r'<x=(\d+(?:\.\d+)?)\s+y=(\d+(?:\.\d+)?)\s+v=(\d+(?:\.\d+)?)>'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)), float(match.group(2)), float(match.group(3))
        except ValueError:
            return None
    return None


def convert_bbox_to_minmax(
    bbox: Sequence[float],
    image_size: Tuple[int, int],
    is_minmax: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box to (x1, y1, x2, y2) minmax pixel coordinates.

    Args:
        bbox: [x, y, width, height] (standard COCO) or [x1, y1, x2, y2]
        image_size: (width, height) of original image
        is_minmax: If True, input is already in minmax format

    Returns:
        (x1, y1, x2, y2) in pixel coordinates clamped to image dimensions
    """
    img_width, img_height = image_size
    if len(bbox) != 4:
        return (float(img_width // 4), float(img_height // 4),
                float(3 * img_width // 4), float(3 * img_height // 4))

    if is_minmax:
        x1, y1, x2, y2 = bbox
    else:
        x, y, width, height = bbox
        x1, y1, x2, y2 = x, y, x + width, y + height

    x1 = max(0.0, min(float(x1), float(img_width)))
    y1 = max(0.0, min(float(y1), float(img_height)))
    x2 = max(0.0, min(float(x2), float(img_width)))
    y2 = max(0.0, min(float(y2), float(img_height)))

    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def normalize_bbox_for_gaze_predictor(
    bbox: Tuple[float, float, float, float],
    original_image_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    """
    Normalize bbox coordinates to 0-100 scale relative to the listener's
    aspect-ratio padded image (512x320).

    Args:
        bbox: (x1, y1, x2, y2) in original image pixel coordinates
        original_image_size: (width, height) of original image

    Returns:
        (x1, y1, x2, y2) in 0-100 coordinate scale
    """
    x1, y1, x2, y2 = bbox
    orig_width, orig_height = original_image_size
    target_width, target_height = LISTENER_IMAGE_WIDTH, LISTENER_IMAGE_HEIGHT

    scale = min(target_width / orig_width, target_height / orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2

    x1_scaled = x1 * scale + pad_x
    y1_scaled = y1 * scale + pad_y
    x2_scaled = x2 * scale + pad_x
    y2_scaled = y2 * scale + pad_y

    x1_norm = (x1_scaled / target_width) * COORDINATE_SCALE
    y1_norm = (y1_scaled / target_height) * COORDINATE_SCALE
    x2_norm = (x2_scaled / target_width) * COORDINATE_SCALE
    y2_norm = (y2_scaled / target_height) * COORDINATE_SCALE

    return (x1_norm, y1_norm, x2_norm, y2_norm)


# Alias for concise reference
normalize_bbox_to_100 = normalize_bbox_for_gaze_predictor


def create_speaker_bbox_overlay(
    image: Union[Image.Image, np.ndarray],
    bbox_minmax: Tuple[float, float, float, float],
    max_dim: Optional[int] = None,
    box_color: str = "red",
    box_width: int = 3,
) -> Image.Image:
    """
    Create an image with a red bounding box overlay for the speaker model.
    Resizes image preserving aspect ratio such that the longer side is max_dim.

    Args:
        image: PIL Image or NumPy array
        bbox_minmax: (x1, y1, x2, y2) in original image pixels
        max_dim: Maximum size of longest side (defaults to SPEAKER_MAX_DIM = 336)
        box_color: Outline color (default 'red')
        box_width: Line width in pixels (default 3)

    Returns:
        PIL Image with red bounding box overlay
    """
    if max_dim is None:
        max_dim = int(SPEAKER_MAX_DIM)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    orig_width, orig_height = image.size
    scale = min(max_dim / orig_width, max_dim / orig_height)
    new_width = max(1, int(round(orig_width * scale)))
    new_height = max(1, int(round(orig_height * scale)))

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    x1, y1, x2, y2 = bbox_minmax
    x1_scaled = max(0, min(int(round(x1 * scale)), new_width - 1))
    y1_scaled = max(0, min(int(round(y1 * scale)), new_height - 1))
    x2_scaled = max(0, min(int(round(x2 * scale)), new_width - 1))
    y2_scaled = max(0, min(int(round(y2 * scale)), new_height - 1))

    x0 = min(x1_scaled, x2_scaled)
    y0 = min(y1_scaled, y2_scaled)
    x1p = max(x1_scaled, x2_scaled)
    y1p = max(y1_scaled, y2_scaled)

    draw = ImageDraw.Draw(resized_image)
    draw.rectangle([x0, y0, x1p, y1p], outline=box_color, width=box_width)

    return resized_image


def create_listener_padded_image(
    image: Union[Image.Image, np.ndarray],
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Image.Image:
    """
    Create a letterboxed padded image for the listener/gaze predictor (no bbox).

    Args:
        image: PIL Image or NumPy array
        target_width: Target image width (defaults to 512)
        target_height: Target image height (defaults to 320)

    Returns:
        PIL Image centered on black background with dimensions (target_width, target_height)
    """
    if target_width is None:
        target_width = int(LISTENER_IMAGE_WIDTH)
    if target_height is None:
        target_height = int(LISTENER_IMAGE_HEIGHT)

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    orig_width, orig_height = image.size
    scale = min(target_width / orig_width, target_height / orig_height)
    new_width = max(1, int(orig_width * scale))
    new_height = max(1, int(orig_height * scale))

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    padded_image = Image.new("RGB", (target_width, target_height), color="black")

    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2
    padded_image.paste(resized_image, (pad_x, pad_y))

    return padded_image


def calculate_distance(coord1: Optional[Tuple[float, float]], coord2: Optional[Tuple[float, float]]) -> float:
    """Calculate Euclidean distance between two coordinates in 0-100 space."""
    if coord1 is None or coord2 is None:
        return float("inf")
    return math.hypot(coord2[0] - coord1[0], coord2[1] - coord1[1])


def is_point_in_bbox(
    point: Optional[Tuple[float, float]],
    bbox: Tuple[float, float, float, float],
) -> bool:
    """Check if point (x, y) lies inside bounding box (x1, y1, x2, y2)."""
    if point is None:
        return False
    x, y = point
    x1, y1, x2, y2 = bbox
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)
    return min_x <= x <= max_x and min_y <= y <= max_y


def calculate_gaussian_bbox_probability(
    x: float,
    y: float,
    std_dev: float,
    bbox: Tuple[float, float, float, float],
    n_samples: int = 1000,
) -> float:
    """
    Calculate the proportion of probability mass of a 2D isotropic Gaussian
    that falls within a bounding box.

    Uses the analytical normal CDF when possible, with Monte Carlo fallback.
    """
    x1, y1, x2, y2 = bbox
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    if std_dev <= 1e-4:
        return 1.0 if (min_x <= x <= max_x and min_y <= y <= max_y) else 0.0

    # Analytical solution using error function for isotropic 2D Gaussian
    try:
        def norm_cdf(val: float, mean: float, sigma: float) -> float:
            return 0.5 * (1.0 + math.erf((val - mean) / (sigma * math.sqrt(2.0))))

        px = norm_cdf(max_x, x, std_dev) - norm_cdf(min_x, x, std_dev)
        py = norm_cdf(max_y, y, std_dev) - norm_cdf(min_y, y, std_dev)
        return float(max(0.0, min(1.0, px * py)))
    except Exception:
        # Fallback to Monte Carlo
        samples_x = np.random.normal(x, std_dev, n_samples)
        samples_y = np.random.normal(y, std_dev, n_samples)
        in_bbox = (samples_x >= min_x) & (samples_x <= max_x) & (samples_y >= min_y) & (samples_y <= max_y)
        return float(np.mean(in_bbox))


def reconstruct_text_from_tokens(
    tokens: List[str],
    word_to_tokens: Optional[List[List[int]]] = None,
) -> str:
    """
    Reconstruct natural text string from subword tokens.
    """
    if not tokens:
        return ""

    if word_to_tokens is None or len(word_to_tokens) == 0:
        result: List[str] = []
        for i, token in enumerate(tokens):
            if i > 0 and not token.startswith(("<",)):
                prev = tokens[i - 1]
                if not prev.endswith(("-",)):
                    result.append(" ")
            result.append(token)
        return "".join(result)

    words = []
    for token_indices in word_to_tokens:
        if not token_indices:
            continue
        word_tokens = [tokens[i] for i in token_indices if i < len(tokens)]
        if word_tokens:
            word = "".join(word_tokens).strip()
            words.append(word)

    return " ".join(words) if words else "".join(tokens)
