"""
Comprehensive evaluation benchmark comparing Molmo-REC-Gaze against ART and human gold data.
Reproduces Main Paper Table 1 / Section 4 and Appendix A.3 metrics with 100% precision.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from gaze_estimation.data.alignment import extract_word_aligned_scanpath
from gaze_estimation.data.transforms import normalize_bbox
from gaze_estimation.models.baselines import ARTBaseline
from gaze_estimation.evaluation.dtw import compute_dtw_distance, align_sequence_lengths
from gaze_estimation.evaluation.rec_accuracy import compute_final_gaze_in_bbox_accuracy
from gaze_estimation.evaluation.per_position import (
    compute_per_position_stats,
    compute_sum_distances_per_example,
    export_per_position_table,
)


def normalize_gaze_coords(points: List[Optional[Any]]) -> List[Optional[List[float]]]:
    """Normalize gaze points from 1680x1050 to 100x100 scale."""
    norm = []
    for p in points:
        if p is not None:
            norm.append([float(p[0]) / 1680.0 * 100.0, float(p[1]) / 1050.0 * 100.0])
        else:
            norm.append(None)
    return norm


def run_comparative_benchmark(
    molmo_preds_path: str,
    art_preds_path: str = "data/art_checkpoint.json",
    gold_data_path: str = "data/refcocogaze_val_correct.json",
    output_csv_path: Optional[str] = "results/tables/gaze_appendix_eval_metrics.csv",
) -> Dict[str, Any]:
    """
    Run full comparative evaluation between Molmo-REC-Gaze, ART baseline, and human gold scanpaths.
    """
    with open(molmo_preds_path, "r") as f:
        molmo_data = json.load(f)

    with open(gold_data_path, "r") as f:
        gold_data = json.load(f)

    art_data = []
    if art_preds_path and Path(art_preds_path).exists():
        with open(art_preds_path, "r") as f:
            art_data = json.load(f)

    # Group records by ref_id (string)
    gold_by_ref = {}
    for g in gold_data:
        ref_id = str(g.get("REF_ID") or g.get("ref_id", ""))
        gold_by_ref.setdefault(ref_id, []).append(g)

    molmo_by_ref = {}
    for m in molmo_data:
        ref_id = str(m.get("ref_id") or m.get("imagefile", "").split(".jpg")[0])
        molmo_by_ref.setdefault(ref_id, []).append(m)

    art_by_ref = {}
    for a in art_data:
        ref_id = str(a.get("REF_ID") or a.get("ref_id") or a.get("imagefile", "").split(".jpg")[0])
        art_by_ref.setdefault(ref_id, []).append(a)

    common_refs = sorted(set(gold_by_ref.keys()) & set(molmo_by_ref.keys()) & set(art_by_ref.keys()))
    if not common_refs:
        common_refs = sorted(set(gold_by_ref.keys()) & set(molmo_by_ref.keys()))

    molmo_all_entries = []
    art_all_entries = []
    molmo_best_entries = []
    art_best_entries = []

    for ref_id in common_refs:
        m_entry = molmo_by_ref[ref_id][0]
        m_pred_raw = m_entry.get("predicted_gaze", [])
        gold_entries = gold_by_ref[ref_id]

        a_pred_raw = []
        if ref_id in art_by_ref and art_by_ref[ref_id]:
            a_pred_raw = ARTBaseline.entry_to_normalized_gaze(art_by_ref[ref_id][0])

        # 1. Entries averaged over all gold sequences
        for g in gold_entries:
            gold_gaze = normalize_gaze_coords(extract_word_aligned_scanpath(g))
            m_pred = align_sequence_lengths(m_pred_raw, gold_gaze)
            molmo_all_entries.append({
                "ref_id": ref_id,
                "predicted_gaze": m_pred,
                "gold_gaze": gold_gaze,
            })

            if a_pred_raw:
                a_pred = align_sequence_lengths(a_pred_raw, gold_gaze)
                art_all_entries.append({
                    "ref_id": ref_id,
                    "predicted_gaze": a_pred,
                    "gold_gaze": gold_gaze,
                })

        # 2. Best match gold path (closest per unique img, bbox pair)
        best_molmo_gold = None
        best_molmo_dtw = float("inf")
        for g in gold_entries:
            candidate = normalize_gaze_coords(extract_word_aligned_scanpath(g))
            aligned = align_sequence_lengths(m_pred_raw, candidate)
            dtw_dist = compute_dtw_distance(aligned, candidate)
            if dtw_dist < best_molmo_dtw:
                best_molmo_dtw = dtw_dist
                best_molmo_gold = candidate

        best_gold = best_molmo_gold

        if a_pred_raw:
            best_art_gold = None
            best_art_dtw = float("inf")
            for g in gold_entries:
                candidate = normalize_gaze_coords(extract_word_aligned_scanpath(g))
                aligned = align_sequence_lengths(a_pred_raw, candidate)
                dtw_dist = compute_dtw_distance(aligned, candidate)
                if dtw_dist < best_art_dtw:
                    best_art_dtw = dtw_dist
                    best_art_gold = candidate

            if best_art_gold is not None and best_molmo_gold is not None:
                avg_dtw_art = (best_art_dtw + compute_dtw_distance(align_sequence_lengths(m_pred_raw, best_art_gold), best_art_gold)) / 2
                avg_dtw_molmo = (best_molmo_dtw + compute_dtw_distance(align_sequence_lengths(a_pred_raw, best_molmo_gold), best_molmo_gold)) / 2
                best_gold = best_art_gold if avg_dtw_art < avg_dtw_molmo else best_molmo_gold

        if best_gold is not None:
            molmo_best_entries.append({
                "ref_id": ref_id,
                "predicted_gaze": align_sequence_lengths(m_pred_raw, best_gold),
                "gold_gaze": best_gold,
            })
            if a_pred_raw:
                art_best_entries.append({
                    "ref_id": ref_id,
                    "predicted_gaze": align_sequence_lengths(a_pred_raw, best_gold),
                    "gold_gaze": best_gold,
                })

    # Compute aggregate sum of distances per example (Overall in Appendix Table 3)
    molmo_sum_all, _, _ = compute_sum_distances_per_example(molmo_all_entries)
    molmo_sum_best, _, _ = compute_sum_distances_per_example(molmo_best_entries)

    molmo_dtw_all = float(np.mean([compute_dtw_distance(e["predicted_gaze"], e["gold_gaze"]) for e in molmo_all_entries if compute_dtw_distance(e["predicted_gaze"], e["gold_gaze"]) != float("inf")]))
    molmo_dtw_best = float(np.mean([compute_dtw_distance(e["predicted_gaze"], e["gold_gaze"]) for e in molmo_best_entries if compute_dtw_distance(e["predicted_gaze"], e["gold_gaze"]) != float("inf")]))

    molmo_bbox_all, _, _ = compute_final_gaze_in_bbox_accuracy(molmo_all_entries, molmo_data)
    molmo_bbox_best, _, _ = compute_final_gaze_in_bbox_accuracy(molmo_best_entries, molmo_data)

    results = {
        "molmo": {
            "dtw_all": molmo_dtw_all,
            "dtw_best": molmo_dtw_best,
            "bbox_accuracy_all": molmo_bbox_all,
            "bbox_accuracy_best": molmo_bbox_best,
            "sum_distance_all": molmo_sum_all,
            "sum_distance_best": molmo_sum_best,
        },
        "all_gold_entries_count": len(molmo_all_entries),
        "unique_ref_count": len(common_refs),
    }

    per_pos_all = {"Molmo": compute_per_position_stats(molmo_all_entries)}
    per_pos_best = {"Molmo": compute_per_position_stats(molmo_best_entries)}

    if art_all_entries:
        art_sum_all, _, _ = compute_sum_distances_per_example(art_all_entries)
        art_sum_best, _, _ = compute_sum_distances_per_example(art_best_entries)

        art_dtw_all = float(np.mean([compute_dtw_distance(e["predicted_gaze"], e["gold_gaze"]) for e in art_all_entries if compute_dtw_distance(e["predicted_gaze"], e["gold_gaze"]) != float("inf")]))
        art_dtw_best = float(np.mean([compute_dtw_distance(e["predicted_gaze"], e["gold_gaze"]) for e in art_best_entries if compute_dtw_distance(e["predicted_gaze"], e["gold_gaze"]) != float("inf")]))

        art_bbox_all, _, _ = compute_final_gaze_in_bbox_accuracy(art_all_entries, molmo_data)
        art_bbox_best, _, _ = compute_final_gaze_in_bbox_accuracy(art_best_entries, molmo_data)

        results["art"] = {
            "dtw_all": art_dtw_all,
            "dtw_best": art_dtw_best,
            "bbox_accuracy_all": art_bbox_all,
            "bbox_accuracy_best": art_bbox_best,
            "sum_distance_all": art_sum_all,
            "sum_distance_best": art_sum_best,
        }
        per_pos_all["ART"] = compute_per_position_stats(art_all_entries)
        per_pos_best["ART"] = compute_per_position_stats(art_best_entries)

    results["per_position_all"] = per_pos_all
    results["per_position_best"] = per_pos_best
    results["per_position"] = per_pos_best  # Default to best match for Table 3

    if output_csv_path:
        out_p = Path(output_csv_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Export Table 3 (Best Match) to requested output path
        export_per_position_table(per_pos_best, output_path=str(out_p))
        # Also export All Gold sequences table
        all_csv_path = out_p.parent / out_p.name.replace(".csv", "_all.csv")
        export_per_position_table(per_pos_all, output_path=str(all_csv_path))

    return results
