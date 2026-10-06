"""
Image and coordinate transforms for Molmo-REC-Gaze.
Adheres strictly to the GazeRL shared contract (512x320 letterbox canvas, 0-100 normalized scale).
"""

from typing import Tuple, Optional, List
from PIL import Image

TARGET_CANVAS_WIDTH = 512
TARGET_CANVAS_HEIGHT = 320


def letterbox_image(
    img: Image.Image,
    target_width: int = TARGET_CANVAS_WIDTH,
    target_height: int = TARGET_CANVAS_HEIGHT,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, float, int, int]:
    """
    Scale image to fit within target canvas preserving aspect ratio, centered on black background.
    
    Returns:
        padded_img: Resulting PIL Image (target_width x target_height)
        scale: Scale factor applied
        pad_x: Left horizontal padding in pixels
        pad_y: Top vertical padding in pixels
    """
    orig_w, orig_h = img.size
    scale = min(target_width / orig_w, target_height / orig_h)
    new_w, new_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
    
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    padded = Image.new("RGB", (target_width, target_height), color=bg_color)
    pad_x = (target_width - new_w) // 2
    pad_y = (target_height - new_h) // 2
    padded.paste(resized, (pad_x, pad_y))
    
    return padded, scale, pad_x, pad_y


def scale_coords_to_canvas(
    x_orig: float,
    y_orig: float,
    scale: float,
    pad_x: int,
    pad_y: int,
) -> Tuple[float, float]:
    """Map coordinate from original image space to letterboxed canvas space."""
    x_canvas = x_orig * scale + pad_x
    y_canvas = y_orig * scale + pad_y
    return x_canvas, y_canvas


def canvas_to_normalized(
    x_canvas: float,
    y_canvas: float,
    canvas_w: int = TARGET_CANVAS_WIDTH,
    canvas_h: int = TARGET_CANVAS_HEIGHT,
) -> Tuple[float, float]:
    """
    Convert canvas pixel coordinates to 0.0-100.0 continuous normalized scale.
    """
    x_norm = (x_canvas / canvas_w) * 100.0
    y_norm = (y_canvas / canvas_h) * 100.0
    return x_norm, y_norm


def normalized_to_canvas(
    x_norm: float,
    y_norm: float,
    canvas_w: int = TARGET_CANVAS_WIDTH,
    canvas_h: int = TARGET_CANVAS_HEIGHT,
) -> Tuple[float, float]:
    """Convert 0.0-100.0 continuous normalized coordinates back to canvas pixels."""
    x_canvas = (x_norm / 100.0) * canvas_w
    y_canvas = (y_norm / 100.0) * canvas_h
    return x_canvas, y_canvas


def normalize_coords(
    x_orig: float,
    y_orig: float,
    scale: float,
    pad_x: int,
    pad_y: int,
    canvas_w: int = TARGET_CANVAS_WIDTH,
    canvas_h: int = TARGET_CANVAS_HEIGHT,
) -> Tuple[float, float]:
    """Directly map coordinate from original image space to 0-100 normalized space."""
    x_c, y_c = scale_coords_to_canvas(x_orig, y_orig, scale, pad_x, pad_y)
    return canvas_to_normalized(x_c, y_c, canvas_w, canvas_h)


def normalize_bbox(
    bbox: List[float],
    orig_w: int = 1680,
    orig_h: int = 1050,
) -> Optional[List[float]]:
    """
    Normalize bounding box [x, y, w, h] from original coordinate space to 0-100 scale.
    """
    if bbox is None or len(bbox) != 4:
        return None
    x, y, w, h = bbox
    return [
        (x / orig_w) * 100.0,
        (y / orig_h) * 100.0,
        (w / orig_w) * 100.0,
        (h / orig_h) * 100.0,
    ]


def is_point_in_bbox(
    point: Optional[Tuple[float, float]],
    bbox: Optional[List[float]],
) -> bool:
    """
    Check whether normalized point (x, y) falls inside normalized bbox [x, y, w, h].
    Both point and bbox must be on the same coordinate scale (e.g. 0-100).
    """
    if point is None or bbox is None or len(bbox) != 4:
        return False
    px, py = point
    bx, by, bw, bh = bbox
    return (bx <= px <= bx + bw) and (by <= py <= by + bh)
