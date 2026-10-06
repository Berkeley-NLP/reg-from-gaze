"""
Referring Expression Comprehension (REC) and Bounding Box Accuracy Metrics.
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from gaze_estimation.data.transforms import is_point_in_bbox, normalize_bbox


def compute_final_gaze_in_bbox_accuracy(
    entries: List[Dict[str, Any]],
    raw_data: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, int, int]:
    """
    Compute percentage of sequences whose final non-null predicted fixation lands in the target bounding box.

    Args:
        entries: List of dicts containing 'predicted_gaze' and either 'normalized_bbox' or 'ref_id'
        raw_data: Optional raw dataset to look up bboxes by ref_id / imagefile

    Returns:
        (accuracy, hits, total_valid)
    """
    hits = 0
    total = 0

    bbox_by_ref = {}
    if raw_data:
        for r in raw_data:
            ref = str(r.get("REF_ID") or r.get("imagefile", "").split(".")[0])
            bbox = r.get("normalized_bbox") or r.get("BBOX") or r.get("bbox")
            if bbox:
                bbox_by_ref[ref] = bbox

    seen_refs = set()

    for entry in entries:
        ref_id = str(entry.get("ref_id", ""))
        if ref_id in seen_refs:
            continue
        seen_refs.add(ref_id)

        pred_seq = entry.get("predicted_gaze", [])
        raw_bbox = entry.get("normalized_bbox")
        if not raw_bbox and raw_data:
            raw_bbox = bbox_by_ref.get(ref_id)

        if not raw_bbox:
            continue

        # Normalize bbox to 0-100 scale (stored as raw coords in legacy preds)
        if raw_bbox[0] > 100.0 or raw_bbox[1] > 100.0 or raw_bbox[2] > 100.0 or raw_bbox[3] > 100.0:
            bbox = normalize_bbox(raw_bbox)
        else:
            bbox = raw_bbox

        if not bbox:
            continue

        # Find final non-null point
        final_pt = None
        for pt in reversed(pred_seq):
            if pt is not None:
                final_pt = pt
                break

        if final_pt is not None:
            total += 1
            if is_point_in_bbox(final_pt, bbox):
                hits += 1

    acc = (hits / total) if total > 0 else 0.0
    return acc, hits, total


def compute_any_gaze_in_bbox_accuracy(
    entries: List[Dict[str, Any]],
    raw_data: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, int, int]:
    """
    Compute percentage of sequences where ANY predicted fixation lands in the target bounding box.
    """
    hits = 0
    total = 0

    bbox_by_ref = {}
    if raw_data:
        for r in raw_data:
            ref = str(r.get("REF_ID") or r.get("imagefile", "").split(".")[0])
            bbox = r.get("normalized_bbox") or r.get("BBOX") or r.get("bbox")
            if bbox:
                bbox_by_ref[ref] = bbox

    seen_refs = set()

    for entry in entries:
        ref_id = str(entry.get("ref_id", ""))
        if ref_id in seen_refs:
            continue
        seen_refs.add(ref_id)

        pred_seq = entry.get("predicted_gaze", [])
        raw_bbox = entry.get("normalized_bbox")
        if not raw_bbox and raw_data:
            raw_bbox = bbox_by_ref.get(ref_id)

        if not raw_bbox:
            continue

        if raw_bbox[0] > 100.0 or raw_bbox[1] > 100.0 or raw_bbox[2] > 100.0 or raw_bbox[3] > 100.0:
            bbox = normalize_bbox(raw_bbox)
        else:
            bbox = raw_bbox

        if not bbox:
            continue

        valid_points = [p for p in pred_seq if p is not None]
        if valid_points:
            total += 1
            if any(is_point_in_bbox(pt, bbox) for pt in valid_points):
                hits += 1

    acc = (hits / total) if total > 0 else 0.0
    return acc, hits, total


def compute_distance_to_bbox_center(
    point: Tuple[float, float],
    bbox: List[float],
) -> float:
    """Euclidean distance from point (x, y) to bbox center (bx + bw/2, by + bh/2)."""
    cx = bbox[0] + bbox[2] / 2.0
    cy = bbox[1] + bbox[3] / 2.0
    return math.sqrt((point[0] - cx) ** 2 + (point[1] - cy) ** 2)
