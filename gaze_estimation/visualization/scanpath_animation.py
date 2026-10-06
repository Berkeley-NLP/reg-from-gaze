"""
Animated GIF visualization for unfolding gaze scanpaths.
"""

from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image

from gaze_estimation.visualization.scanpath_overlay import draw_scanpath_on_image


def generate_scanpath_gif(
    base_image: Image.Image,
    scanpath: List[Optional[Tuple[float, float]]],
    words: Optional[List[str]] = None,
    output_path: str = "results/figures/scanpath.gif",
    duration_ms: int = 400,
    color: str = "cyan",
) -> str:
    """
    Create animated GIF showing step-by-step fixation progression across words.
    """
    frames = []
    for step in range(1, len(scanpath) + 1):
        partial_scanpath = scanpath[:step]
        partial_words = words[:step] if words else None
        frame = draw_scanpath_on_image(
            base_image,
            partial_scanpath,
            words=partial_words,
            color=color,
        )
        frames.append(frame)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if frames:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )

    return output_path
