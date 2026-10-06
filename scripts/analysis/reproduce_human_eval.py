#!/usr/bin/env python3
"""
scripts/analysis/reproduce_human_eval.py

Reproduces the Human Evaluation Benchmark results from the paper:
  - Table 4: Overall Performance Metrics (Accuracy, Length, Resolution Time, d_NEG)
  - Table 5: Behavioral Metrics (Resolution Time, Reveal Rate, Early Click Rate)
  - Appendix B.1: Dataset-level breakdown across:
      * RefCOCO TestA
      * RefCOCO TestB
      * RefCOCOg / RefOI Co-occurrence
      * RefCOCOg / RefOI Single-presence

Usage:
  python scripts/analysis/reproduce_human_eval.py
  python scripts/analysis/reproduce_human_eval.py --verify
"""

import os
import re
import math
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Canonical Mappings
# -----------------------------------------------------------------------------
SPEAKER_NAME_MAP = {
    "gold": "Gold",
    "molmo_vanilla": "Molmo",
    "sparse_constant_kl_02": "Gaze-BFH",
    "shaping": "Gaze-Shaping",
    "binary": "Gaze-SeqAnyHit",
    "binary_last_point": "Gaze-SeqLPHit",
    "supervised": "REC-Success",
    "iterative_sparse": "REC-BFH",
    "iterative_shaping": "REC-Shaping",
    "iterative_binary": "REC-SeqAnyHit",
}

# Legacy CSV names mapping
CANONICAL_TO_LEGACY_MAP = {
    "Gaze-BFH": "Sparse Gaze",
    "Gaze-Shaping": "Shaping Gaze",
    "Gaze-SeqAnyHit": "Binary Gaze",
    "Gaze-SeqLPHit": "Binary LP Gaze",
    "REC-BFH": "Sparse Molmo",
    "REC-Shaping": "Iterative Shaping",
    "REC-SeqAnyHit": "Binary Molmo",
    "REC-Success": "Binary LP Molmo",
    "Molmo": "Molmo",
    "Gold": "Gold",
}

DATASETS = [
    "refcoco_testA",
    "refcoco_testB",
    "refoi_co_occurrence",
    "refoi_single_presence",
]

DATASET_DISPLAY_NAMES = {
    "refcoco_testA": "RefCOCO Test A",
    "refcoco_testB": "RefCOCO Test B",
    "refoi_co_occurrence": "RefOI Co-occurrence",
    "refoi_single_presence": "RefOI Single Presence",
}


def get_clean_ref_len(ref: Any) -> int:
    """
    Compute reference length matching the paper's standard word tokenizer.
    Filters out code fences, extracts words, and includes single-letter words 'a' and 'i'.
    """
    if not ref:
        return 0
    if isinstance(ref, list):
        ref = ref[0] if len(ref) > 0 else ""
    if isinstance(ref, dict):
        ref = ref.get("generated_reference", ref.get("reference", ""))
    if not isinstance(ref, str):
        return 0
    ref = re.sub(r"```json.*?```", "", ref, flags=re.DOTALL)
    ref = re.sub(r"```.*?```", "", ref, flags=re.DOTALL)
    words = re.findall(r"\w+", ref.lower())
    words = [w for w in words if len(w) > 1 or w in ["a", "i"]]
    return len(words)


def compute_metrics_for_trials(
    trials: List[Dict[str, Any]],
    gold_acc: float,
    gold_len: float,
    base_acc: float,
    base_len: float,
    model_name: str,
) -> Dict[str, Any]:
    """Compute performance and behavioral metrics for a set of trials."""
    acc = np.mean([1.0 if t.get("is_hit", False) else 0.0 for t in trials]) * 100.0
    ref_lens = [get_clean_ref_len(t.get("reference", "")) for t in trials]
    mean_len = float(np.mean(ref_lens)) if ref_lens else 0.0

    # Resolution time (duration_ms on successful hits)
    hit_trials = [t for t in trials if t.get("is_hit", False)]
    hit_times = [t.get("duration_ms", 0.0) for t in hit_trials if t.get("duration_ms") is not None]
    mean_hit_time = float(np.mean(hit_times)) if hit_times else 0.0

    # Behavioral metrics
    revealed = float(np.mean([1.0 if t.get("text_was_revealed", False) else 0.0 for t in trials]))
    early_clicks = float(np.mean([len(t.get("early_clicks", [])) for t in trials]))

    # Communicative efficiency distance d_NEG from Gold with Molmo baseline
    if model_name == "Gold":
        d_neg = 0.0
    elif model_name == "Molmo":
        d_neg = math.sqrt(2.0)
    else:
        acc_term = (gold_acc - acc) / (gold_acc - base_acc) if (gold_acc != base_acc) else 0.0
        len_term = (mean_len - gold_len) / (base_len - gold_len) if (base_len != gold_len) else 0.0
        d_neg = float(math.sqrt(acc_term**2 + len_term**2))

    return {
        "Accuracy": acc,
        "RefLen": mean_len,
        "HitTime": mean_hit_time,
        "Revealed": revealed,
        "NumEarlyClicks": early_clicks,
        "d_NEG": d_neg,
        "TotalTrials": len(trials),
    }


def reproduce(trials_path: Path, output_dir: Path, verify: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load trials and compute overall and per-dataset tables."""
    if not trials_path.exists():
        gz_path = Path(str(trials_path) + ".gz")
        if gz_path.exists():
            trials_path = gz_path
        else:
            raise FileNotFoundError(f"Human evaluation trials file not found at: {trials_path}")

    print(f"Loading human evaluation dataset from {trials_path}...")
    if str(trials_path).endswith(".gz"):
        import gzip
        with gzip.open(trials_path, "rt", encoding="utf-8") as f:
            trials = json.load(f)
    else:
        with open(trials_path, "r", encoding="utf-8") as f:
            trials = json.load(f)

    print(f"Loaded {len(trials)} trials across {len(set(t['participant_id'] for t in trials))} participants.")

    # Group trials by (dataset, speaker) and overall by speaker
    overall_groups = defaultdict(list)
    dataset_groups = defaultdict(lambda: defaultdict(list))

    for t in trials:
        raw_spk = t.get("speaker", "")
        img_name = t.get("img_name", "")
        canonical_spk = SPEAKER_NAME_MAP.get(raw_spk, raw_spk)

        overall_groups[canonical_spk].append(t)

        for ds in DATASETS:
            if ds in img_name:
                dataset_groups[ds][canonical_spk].append(t)
                break

    # -------------------------------------------------------------------------
    # 1. OVERALL METRICS (Table 4 & Table 5 in Paper)
    # -------------------------------------------------------------------------
    gold_trials = overall_groups["Gold"]
    molmo_trials = overall_groups["Molmo"]

    gold_acc = np.mean([1.0 if t["is_hit"] else 0.0 for t in gold_trials]) * 100.0
    gold_len = np.mean([get_clean_ref_len(t["reference"]) for t in gold_trials])
    base_acc = np.mean([1.0 if t["is_hit"] else 0.0 for t in molmo_trials]) * 100.0
    base_len = np.mean([get_clean_ref_len(t["reference"]) for t in molmo_trials])

    canonical_order = [
        "Gold",
        "Molmo",
        "Gaze-BFH",
        "Gaze-Shaping",
        "Gaze-SeqAnyHit",
        "Gaze-SeqLPHit",
        "REC-BFH",
        "REC-Shaping",
        "REC-SeqAnyHit",
        "REC-Success",
    ]

    overall_rows = []
    for model in canonical_order:
        if model not in overall_groups:
            continue
        m_dict = compute_metrics_for_trials(
            overall_groups[model], gold_acc, gold_len, base_acc, base_len, model
        )
        overall_rows.append({
            "Model": model,
            "LegacyName": CANONICAL_TO_LEGACY_MAP.get(model, model),
            "Accuracy": m_dict["Accuracy"],
            "RefLen": m_dict["RefLen"],
            "HitTime": m_dict["HitTime"],
            "Revealed": m_dict["Revealed"],
            "NumEarlyClicks": m_dict["NumEarlyClicks"],
            "d_NEG": m_dict["d_NEG"],
            "Dataset": "Overall",
        })

    df_overall = pd.DataFrame(overall_rows)

    # -------------------------------------------------------------------------
    # 2. PER-DATASET METRICS (Appendix B.1 in Paper)
    # -------------------------------------------------------------------------
    per_dataset_rows = []
    for ds in DATASETS:
        ds_gold = dataset_groups[ds]["Gold"]
        ds_molmo = dataset_groups[ds]["Molmo"]

        ds_gold_acc = np.mean([1.0 if t["is_hit"] else 0.0 for t in ds_gold]) * 100.0
        ds_gold_len = np.mean([get_clean_ref_len(t["reference"]) for t in ds_gold])
        ds_base_acc = np.mean([1.0 if t["is_hit"] else 0.0 for t in ds_molmo]) * 100.0
        ds_base_len = np.mean([get_clean_ref_len(t["reference"]) for t in ds_molmo])

        for model in canonical_order:
            if model not in dataset_groups[ds]:
                continue
            m_dict = compute_metrics_for_trials(
                dataset_groups[ds][model], ds_gold_acc, ds_gold_len, ds_base_acc, ds_base_len, model
            )
            per_dataset_rows.append({
                "Dataset": ds,
                "DatasetName": DATASET_DISPLAY_NAMES.get(ds, ds),
                "Model": model,
                "LegacyName": CANONICAL_TO_LEGACY_MAP.get(model, model),
                "Accuracy": m_dict["Accuracy"],
                "RefLen": m_dict["RefLen"],
                "HitTime": m_dict["HitTime"],
                "Revealed": m_dict["Revealed"],
                "NumEarlyClicks": m_dict["NumEarlyClicks"],
                "d_NEG": m_dict["d_NEG"],
            })

    df_per_dataset = pd.DataFrame(per_dataset_rows)

    # -------------------------------------------------------------------------
    # Save CSV outputs
    # -------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    overall_csv = output_dir / "human_metrics_reproduced_overall.csv"
    per_ds_csv = output_dir / "human_metrics_per_dataset.csv"

    df_overall.to_csv(overall_csv, index=False)
    df_per_dataset.to_csv(per_ds_csv, index=False)
    print(f"\nSaved overall metrics table to: {overall_csv}")
    print(f"Saved per-dataset metrics table to: {per_ds_csv}")

    # -------------------------------------------------------------------------
    # Print Formatted Reproduction Tables
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TABLE 4: HUMAN EVALUATION REG PERFORMANCE (OVERALL)")
    print("=" * 80)
    display_t4 = df_overall[["Model", "Accuracy", "RefLen", "HitTime", "d_NEG"]].copy()
    display_t4["Accuracy"] = display_t4["Accuracy"].map(lambda x: f"{x:.1f}%")
    display_t4["RefLen"] = display_t4["RefLen"].map(lambda x: f"{x:.2f} w")
    display_t4["HitTime"] = display_t4["HitTime"].map(lambda x: f"{x:.0f} ms")
    display_t4["d_NEG"] = display_t4["d_NEG"].map(lambda x: f"{x:.3f}")
    print(display_t4.to_string(index=False))

    print("\n" + "=" * 80)
    print("TABLE 5: BEHAVIORAL METRICS DURING COMPREHENSION (OVERALL)")
    print("=" * 80)
    display_t5 = df_overall[["Model", "HitTime", "Revealed", "NumEarlyClicks"]].copy()
    display_t5["HitTime"] = display_t5["HitTime"].map(lambda x: f"{x:.0f} ms")
    display_t5["Revealed"] = display_t5["Revealed"].map(lambda x: f"{x*100:.1f}%")
    display_t5["NumEarlyClicks"] = display_t5["NumEarlyClicks"].map(lambda x: f"{x:.3f}")
    print(display_t5.to_string(index=False))

    print("\n" + "=" * 80)
    print("APPENDIX B.1: HUMAN EVALUATION BREAKDOWN PER DATASET")
    print("=" * 80)
    for ds in DATASETS:
        print(f"\n--- {DATASET_DISPLAY_NAMES[ds]} ---")
        sub_df = df_per_dataset[df_per_dataset["Dataset"] == ds][
            ["Model", "Accuracy", "RefLen", "HitTime", "d_NEG"]
        ].copy()
        sub_df["Accuracy"] = sub_df["Accuracy"].map(lambda x: f"{x:.1f}%")
        sub_df["RefLen"] = sub_df["RefLen"].map(lambda x: f"{x:.2f}")
        sub_df["HitTime"] = sub_df["HitTime"].map(lambda x: f"{x:.0f} ms")
        sub_df["d_NEG"] = sub_df["d_NEG"].map(lambda x: f"{x:.3f}")
        print(sub_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # Verification mode
    # -------------------------------------------------------------------------
    if verify:
        print("\nVerifying against target reference values...")
        # Check Gold: Acc == 92.5%, RefLen == 3.43
        gold_row = df_overall[df_overall["Model"] == "Gold"].iloc[0]
        assert abs(gold_row["Accuracy"] - 92.54) < 0.2, f"Gold accuracy mismatch: {gold_row['Accuracy']}"
        assert abs(gold_row["RefLen"] - 3.43) < 0.1, f"Gold length mismatch: {gold_row['RefLen']}"

        # Check Binary Gaze: Acc == 80.0%, RefLen == 9.45, d_NEG == 0.882
        bg_row = df_overall[df_overall["Model"] == "Gaze-SeqAnyHit"].iloc[0]
        assert abs(bg_row["Accuracy"] - 80.0) < 0.1, f"Binary Gaze accuracy mismatch: {bg_row['Accuracy']}"
        assert abs(bg_row["RefLen"] - 9.45) < 0.1, f"Binary Gaze length mismatch: {bg_row['RefLen']}"
        assert abs(bg_row["d_NEG"] - 0.882) < 0.05, f"Binary Gaze d_NEG mismatch: {bg_row['d_NEG']}"

        # Check Shaping Gaze: Acc == 75.5%, RefLen == 3.98
        sg_row = df_overall[df_overall["Model"] == "Gaze-Shaping"].iloc[0]
        assert abs(sg_row["Accuracy"] - 75.46) < 0.1, f"Shaping Gaze accuracy mismatch: {sg_row['Accuracy']}"
        assert abs(sg_row["RefLen"] - 3.98) < 0.1, f"Shaping Gaze length mismatch: {sg_row['RefLen']}"

        print("Verification PASSED: All reproduced values match paper results!")

    return df_overall, df_per_dataset


def main():
    parser = argparse.ArgumentParser(description="Reproduce Human Evaluation Results (Paper & Appendix)")
    default_trials = Path("results/human_eval/human_eval_trials_24000.json")
    if not default_trials.exists():
        default_trials = Path("results/human_eval/human_eval_trials_21600.json")

    parser.add_argument(
        "--trials-path",
        type=Path,
        default=default_trials,
        help="Path to anonymized human eval trials JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/human_eval"),
        help="Output directory for generated tables",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run assertions verifying paper numbers",
    )
    args = parser.parse_args()

    reproduce(args.trials_path, args.output_dir, verify=args.verify)


if __name__ == "__main__":
    main()
