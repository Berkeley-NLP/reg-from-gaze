"""
Static scanpath visualization and trajectory overlay on images.
"""

from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from gaze_estimation.data.transforms import letterbox_image, normalized_to_canvas


def draw_scanpath_on_image(
    image: Image.Image,
    scanpath: List[Optional[Tuple[float, float]]],
    words: Optional[List[str]] = None,
    color: str = "cyan",
    radius: int = 4,
    line_width: int = 2,
    canvas_w: int = 512,
    canvas_h: int = 320,
) -> Image.Image:
    """
    Draw a gaze scanpath on an image canvas. Coordinates in scanpath are normalized 0-100.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    canvas_pts = []
    for p in scanpath:
        if p is not None:
            cx, cy = normalized_to_canvas(p[0], p[1], canvas_w, canvas_h)
            canvas_pts.append((cx, cy))
        else:
            canvas_pts.append(None)

    # Draw saccade lines between consecutive valid points
    prev_pt = None
    for pt in canvas_pts:
        if pt is not None:
            if prev_pt is not None:
                draw.line([prev_pt, pt], fill=color, width=line_width)
            prev_pt = pt

    # Draw fixation circles
    for idx, pt in enumerate(canvas_pts):
        if pt is not None:
            x, y = pt
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color, outline="white")
            if words and idx < len(words):
                draw.text((x + radius + 2, y - radius), words[idx], fill=color)

    return img_copy
