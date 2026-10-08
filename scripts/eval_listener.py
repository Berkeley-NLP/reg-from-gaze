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
    search_dirs = [
        Path("results/listener_eval/preds"),
        Path("preds"),
        REPO_ROOT / "results/listener_eval/preds",
        REPO_ROOT / "preds",
    ]

    if not preds_path and args.checkpoint:
        clean_name = Path(args.checkpoint).name
        for sdir in search_dirs:
            cand = sdir / f"predicted_gaze_{clean_name}.json"
            if cand.exists():
                preds_path = str(cand)
                break

        if not preds_path:
            for sdir in search_dirs:
                target_file = (
                    "predicted_gaze_output_delay_1129.json" if "1129" in args.checkpoint
                    else "predicted_gaze_delay_token_116.json"
                )
                cand = sdir / target_file
                if cand.exists():
                    preds_path = str(cand)
                    break

    if not preds_path or not Path(preds_path).exists():
        for sdir in search_dirs:
            for fallback_name in [
                "predicted_gaze_delay_token_116.json",
                "predicted_gaze_output_delay_1129.json",
            ]:
                cand = sdir / fallback_name
                if cand.exists():
                    preds_path = str(cand)
                    break
            if preds_path:
                break

    if not preds_path or not Path(preds_path).exists():
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
