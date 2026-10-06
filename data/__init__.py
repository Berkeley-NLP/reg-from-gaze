"""
Data processing, datasets, and split management package for GazeRL.
"""
from data.processing import (
    parse_coordinate,
    parse_gaussian_coordinate,
    convert_bbox_to_minmax,
    normalize_bbox_for_gaze_predictor,
    create_speaker_bbox_overlay,
    create_listener_padded_image,
    calculate_distance,
    is_point_in_bbox,
    calculate_gaussian_bbox_probability,
    reconstruct_text_from_tokens,
)
from data.datasets import (
    ReferringExpressionSample,
    ReferringExpressionDataset,
    SyntheticReferringExpressionDataset,
    DatasetSplitManager,
    collate_referring_expression_batch,
)

__all__ = [
    "parse_coordinate",
    "parse_gaussian_coordinate",
    "convert_bbox_to_minmax",
    "normalize_bbox_for_gaze_predictor",
    "create_speaker_bbox_overlay",
    "create_listener_padded_image",
    "calculate_distance",
    "is_point_in_bbox",
    "calculate_gaussian_bbox_probability",
    "reconstruct_text_from_tokens",
    "ReferringExpressionSample",
    "ReferringExpressionDataset",
    "SyntheticReferringExpressionDataset",
    "DatasetSplitManager",
    "collate_referring_expression_batch",
]
