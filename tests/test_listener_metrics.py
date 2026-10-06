"""
Unit tests for evaluation metrics: DTW, Bbox accuracy, and per-position aggregation.
"""

import pytest
from gaze_estimation.evaluation.dtw import compute_dtw_distance, align_sequence_lengths
from gaze_estimation.evaluation.rec_accuracy import (
    compute_final_gaze_in_bbox_accuracy,
    compute_any_gaze_in_bbox_accuracy,
)
from gaze_estimation.evaluation.per_position import compute_per_position_stats


def test_dtw_identical_sequences():
    seq = [(10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]
    dist = compute_dtw_distance(seq, seq)
    assert dist == 0.0


def test_dtw_with_nulls():
    seq1 = [(10.0, 10.0), None, (30.0, 30.0)]
    seq2 = [(10.0, 10.0), (30.0, 30.0)]
    dist = compute_dtw_distance(seq1, seq2)
    assert dist == 0.0


def test_bbox_accuracy_calculation():
    entries = [
        {"ref_id": "1", "predicted_gaze": [(10, 10), (55, 55)], "normalized_bbox": [50, 50, 10, 10]}, # hit
        {"ref_id": "2", "predicted_gaze": [(10, 10), (20, 20)], "normalized_bbox": [50, 50, 10, 10]}, # miss
    ]
    acc, hits, total = compute_final_gaze_in_bbox_accuracy(entries)
    assert hits == 1
    assert total == 2
    assert acc == 0.5


def test_per_position_stats():
    entries = [
        {"predicted_gaze": [(10.0, 10.0), (20.0, 20.0)], "gold_gaze": [(10.0, 10.0), (20.0, 20.0)]},
        {"predicted_gaze": [(10.0, 10.0), (20.0, 25.0)], "gold_gaze": [(10.0, 10.0), (20.0, 20.0)]},
    ]
    stats = compute_per_position_stats(entries)
    assert stats[0]["mean"] == 0.0
    assert stats[0]["count"] == 2
    assert stats[1]["mean"] == 2.5
    assert stats[1]["count"] == 2
