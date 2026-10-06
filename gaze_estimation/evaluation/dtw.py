"""
Dynamic Time Warping (DTW) distance metric computation for gaze scanpaths.
"""

import math
from typing import List, Optional, Tuple
try:
    from fastdtw import fastdtw
except ImportError:
    def fastdtw(s1, s2, dist=None):
        """Pure-Python dynamic programming fallback for DTW."""
        dist_fn = dist or euclidean_dist
        n, m = len(s1), len(s2)
        dtw_matrix = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dtw_matrix[0][0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = dist_fn(s1[i - 1], s2[j - 1])
                dtw_matrix[i][j] = cost + min(
                    dtw_matrix[i - 1][j],
                    dtw_matrix[i][j - 1],
                    dtw_matrix[i - 1][j - 1],
                )
        return dtw_matrix[n][m], []


def euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute 2D Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_dtw_distance(
    seq1: List[Optional[Tuple[float, float]]],
    seq2: List[Optional[Tuple[float, float]]],
) -> float:
    """
    Compute Dynamic Time Warping (DTW) distance between two scanpaths using Euclidean distance.
    Filters out null fixations (None). Returns float('inf') if either sequence has no valid points.
    """
    filtered_seq1 = [p for p in seq1 if p is not None]
    filtered_seq2 = [p for p in seq2 if p is not None]

    if not filtered_seq1 or not filtered_seq2:
        return float('inf')

    dist, _ = fastdtw(filtered_seq1, filtered_seq2, dist=euclidean_dist)
    return dist


def align_sequence_lengths(
    pred_seq: List[Optional[Tuple[float, float]]],
    gold_seq: List[Optional[Tuple[float, float]]],
) -> List[Optional[Tuple[float, float]]]:
    """Pad prediction with None or truncate to match gold sequence length."""
    if len(pred_seq) < len(gold_seq):
        return pred_seq + [None] * (len(gold_seq) - len(pred_seq))
    return pred_seq[:len(gold_seq)]
