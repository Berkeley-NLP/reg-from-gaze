"""
Model loading, sequence parsing, and baseline listeners for Molmo-REC-Gaze.
"""

from gaze_estimation.models.sequence import (
    POINT_REGEX,
    GAUSSIAN_REGEX,
    NULL_REGEX,
    parse_gaze_point,
    parse_gaussian_gaze_point,
    parse_gaze_sequence,
    format_gaze_point,
    build_incremental_prompt,
)
from gaze_estimation.models.loader import (
    load_molmo_model_and_processor,
    export_model_and_processor,
)
from gaze_estimation.models.baselines import (
    ARTBaseline,
    BboxCenterBaseline,
    LastPointBaseline,
)
