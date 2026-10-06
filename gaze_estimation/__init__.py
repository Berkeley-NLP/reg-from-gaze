"""
Molmo-REC-Gaze: Incremental listener gaze prediction for referring expressions.
Based on Wright et al. (2026), "Learning to Refer from Estimated Listener Gaze".
"""

__version__ = "1.0.0"

from gaze_estimation.data.transforms import letterbox_image, scale_coords_to_canvas, normalize_coords
from gaze_estimation.models.sequence import parse_gaze_point, format_gaze_point

# Backwards compatibility alias
import sys
sys.modules['molmo_gaze'] = sys.modules[__name__]
