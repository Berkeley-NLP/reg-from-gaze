"""
Per-position Euclidean distance analysis for gaze scanpaths.
Directly reproduces the Appendix A.3 table (gaze_appendix_eval_metrics.csv).
"""

import csv
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np


def compute_per_position_stats(
    entries: List[Dict[str, Any]],
    null_penalty: float = 0.0,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute per-word-position average Euclidean distance from predictions to gold fixations.
    Matches the exact legacy and Appendix A.3 calculation:
      - when both pred and gold are present: Euclidean distance
      - when one is None and the other is not: null_penalty (default 0.0)
      - when both are None: skip
    """
    position_distances = defaultdict(list)

    for entry in entries:
        pred_seq = entry.get("predicted_gaze", [])
        gold_seq = entry.get("gold_gaze", [])

        for pos, (pred, gold) in enumerate(zip(pred_seq, gold_seq)):
            if pred is not None and gold is not None:
                dist = math.sqrt((pred[0] - gold[0]) ** 2 + (pred[1] - gold[1]) ** 2)
            elif pred is None and gold is not None:
                dist = null_penalty
            elif pred is not None and gold is None:
                dist = null_penalty
            else:
                continue
            position_distances[pos].append(dist)

    avg_by_pos = {}
    for pos in sorted(position_distances.keys()):
        dists = position_distances[pos]
        if dists:
            avg_by_pos[pos] = {
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists)),
                "count": len(dists),
            }

    return avg_by_pos


def compute_sum_distances_per_example(
    entries: List[Dict[str, Any]],
    null_penalty: float = 0.0,
) -> Tuple[float, List[float], Dict[str, int]]:
    """
    Compute total sum of Euclidean distances per example scanpath.
    Matches the exact legacy / Appendix A.3 Overall row calculation:
      Best Match overall: Molmo 43.21 vs ART 45.75
      All Gold overall: Molmo 62.24 vs ART 72.86
    """
    sum_distances = []
    tp = fp = fn = 0
    total_gold_nulls = 0
    total_pred_nulls = 0

    for entry in entries:
        pred_seq = entry.get("predicted_gaze", [])
        gold_seq = entry.get("gold_gaze", [])

        total_gold_nulls += sum(g is None for g in gold_seq)
        total_pred_nulls += sum(p is None for p in pred_seq)

        example_distances = []
        for pred, gold in zip(pred_seq, gold_seq):
            if pred is None and gold is None:
                tp += 1
            elif pred is None and gold is not None:
                fp += 1
                example_distances.append(null_penalty)
            elif pred is not None and gold is None:
                fn += 1
                example_distances.append(null_penalty)
            elif pred is not None and gold is not None:
                example_distances.append(math.sqrt((pred[0] - gold[0]) ** 2 + (pred[1] - gold[1]) ** 2))

        if example_distances:
            sum_distances.append(sum(example_distances))

    mean_sum = float(np.mean(sum_distances)) if sum_distances else 0.0
    stats = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gold_nulls": total_gold_nulls,
        "total_pred_nulls": total_pred_nulls,
    }
    return mean_sum, sum_distances, stats


def export_per_position_table(
    all_stats: Dict[str, Dict[int, Dict[str, Any]]],
    output_path: str = "results/tables/gaze_appendix_eval_metrics.csv",
) -> None:
    """
    Export per-position metrics for multiple models to a CSV file matching the Appendix A.3 table format.

    all_stats: dict mapping label (e.g. 'ART', 'Molmo') -> {pos -> {'mean': m, 'std': s, 'count': c}}
    """
    if not all_stats:
        return

    labels = sorted(all_stats.keys())
    # Ensure ART comes first if present, to match the paper appendix table format
    if "ART" in labels:
        labels.remove("ART")
        labels.insert(0, "ART")

    all_positions = sorted(set(pos for label in labels for pos in all_stats[label].keys()))

    headers = ["Position"]
    for label in labels:
        headers.extend([f"{label}_Mean", f"{label}_Std", f"{label}_Count"])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for pos in all_positions:
            row = [pos]
            for label in labels:
                stats = all_stats[label].get(pos, {"mean": float("nan"), "std": float("nan"), "count": 0})
                row.extend([stats["mean"], stats["std"], stats["count"]])
            writer.writerow(row)
