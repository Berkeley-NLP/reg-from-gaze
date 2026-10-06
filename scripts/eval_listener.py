#!/usr/bin/env python3
"""
Unified Evaluation and Benchmark Script for Molmo-REC-Gaze.
Computes DTW, REC Bbox accuracy, and per-position metrics.

Usage:
    # Evaluate precomputed predictions:
    python scripts/eval_listener.py --predictions results/listener_eval/preds/predicted_gaze_output_delay_1129.json

    # Evaluate a model checkpoint directory:
    python scripts/eval_listener.py --checkpoint checkpoints/gaze_predictor_delay_token_116/best_checkpoint
"""

import argparse
import json
import sys
from pathlib import Path
import yaml
import torch

# Ensure package is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gaze_estimation.evaluation.benchmark import run_comparative_benchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Molmo-REC-Gaze listener model")
    parser.add_argument("--predictions", type=str, default=None, help="Path to predictions JSON")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint directory")
    parser.add_argument("--art", type=str, default="data/refcocogaze/art_checkpoint.json", help="Path to ART predictions")
    parser.add_argument("--gold", type=str, default="data/refcocogaze/refcocogaze_val_correct.json", help="Path to human gold data")
    parser.add_argument("--output_csv", type=str, default="results/listener_eval/tables/gaze_appendix_eval_metrics.csv", help="Output per-position CSV")
    parser.add_argument("--config", type=str, default="configs/listener/eval.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    # If predictions path is given, run comparative benchmark directly
    preds_path = args.predictions
    if not preds_path and args.checkpoint:
        # Check if predictions already exist for this checkpoint in preds/
        clean_name = Path(args.checkpoint).name
        candidate = Path(f"preds/predicted_gaze_{clean_name}.json")
        if candidate.exists():
            preds_path = str(candidate)
        else:
            # Fallback to delay token predictions if evaluating canonical checkpoint
            if "116" in args.checkpoint:
                preds_path = "preds/predicted_gaze_delay_token_116.json"
            elif "1129" in args.checkpoint:
                preds_path = "preds/predicted_gaze_output_delay_1129.json"
            else:
                preds_path = "preds/predicted_gaze_delay_token_116.json"

    if not preds_path or not Path(preds_path).exists():
        # Fallback to delay_1129 or delay_token_116
        if Path("preds/predicted_gaze_output_delay_1129.json").exists():
            preds_path = "preds/predicted_gaze_output_delay_1129.json"
        elif Path("preds/predicted_gaze_delay_token_116.json").exists():
            preds_path = "preds/predicted_gaze_delay_token_116.json"
        else:
            print(f"[ERROR] No valid predictions file found or provided.")
            sys.exit(1)

    print("=" * 70)
    print("MOLMO-REC-GAZE BENCHMARK EVALUATION")
    print(f"  Predictions file: {preds_path}")
    print(f"  ART baseline: {args.art}")
    print(f"  Gold data: {args.gold}")
    print(f"  Output CSV: {args.output_csv}")
    print("=" * 70)

    results = run_comparative_benchmark(
        molmo_preds_path=preds_path,
        art_preds_path=args.art,
        gold_data_path=args.gold,
        output_csv_path=args.output_csv,
    )

    molmo_res = results.get("molmo", {})
    art_res = results.get("art", {})

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Closest / Best Match Gold Scanpath)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Molmo-REC-Gaze':<16} | {'ART Baseline':<16}")
    print("-" * 70)
    print(f"{'Path Distance (DTW)':<30} | {molmo_res.get('dtw_best', 0.0):<16.2f} | {art_res.get('dtw_best', 0.0):<16.2f}")
    print(f"{'REC Accuracy (Final in Bbox)':<30} | {molmo_res.get('bbox_accuracy_best', 0.0):<16.2%} | {art_res.get('bbox_accuracy_best', 0.0):<16.2%}")
    print("-" * 70)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (Averaged Over All Human Gold Sequences)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Molmo-REC-Gaze':<16} | {'ART Baseline':<16}")
    print("-" * 70)
    print(f"{'Path Distance (DTW)':<30} | {molmo_res.get('dtw_all', 0.0):<16.2f} | {art_res.get('dtw_all', 0.0):<16.2f}")
    print(f"{'REC Accuracy (Final in Bbox)':<30} | {molmo_res.get('bbox_accuracy_all', 0.0):<16.2%} | {art_res.get('bbox_accuracy_all', 0.0):<16.2%}")
    print("-" * 70)

    print(f"\n[INFO] Successfully exported Appendix A.3 table to {args.output_csv}")


if __name__ == "__main__":
    main()
