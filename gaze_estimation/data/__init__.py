"""
Data handling, alignment, transforms, and datasets for Molmo-REC-Gaze.
"""

from gaze_estimation.data.transforms import (
    letterbox_image,
    scale_coords_to_canvas,
    canvas_to_normalized,
    normalized_to_canvas,
    normalize_coords,
    normalize_bbox,
    is_point_in_bbox,
)
from gaze_estimation.data.alignment import (
    apply_auditory_latency_delay,
    extract_word_aligned_scanpath,
)
from gaze_estimation.data.dataset import (
    TokenLevelGazeDataset,
    WordLevelGazeDataset,
    make_token_level_sequence,
    make_word_level_sequence,
)
from gaze_estimation.data.collators import GazeDataCollator
