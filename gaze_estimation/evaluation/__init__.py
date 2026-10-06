"""
Evaluation metrics and benchmark runners for Molmo-REC-Gaze.
"""

from gaze_estimation.evaluation.dtw import (
    euclidean_dist,
    compute_dtw_distance,
    align_sequence_lengths,
)
from gaze_estimation.evaluation.rec_accuracy import (
    compute_final_gaze_in_bbox_accuracy,
    compute_any_gaze_in_bbox_accuracy,
    compute_distance_to_bbox_center,
)
from gaze_estimation.evaluation.per_position import (
    compute_per_position_stats,
    export_per_position_table,
)
from gaze_estimation.evaluation.benchmark import run_comparative_benchmark
