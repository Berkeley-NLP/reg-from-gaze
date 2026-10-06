#!/usr/bin/env python3
"""
scripts/plotting/plot_machine_efficiency_variants.py

Generates publication-quality charts of Reference Length vs. Communicative Accuracy
for Machine Listeners (Qwen-VL Listener), averaged over all datasets to display
exactly one dot per model with tightly fitted y-axis to minimize empty white space.

Produces:
  1. qwenvl_efficiency_averaged.png / .pdf:
     Standalone averaged chart for Qwen-VL listener (1 dot per model, fitted y-axis).
  2. efficiency_comparison_averaged.png / .pdf:
     Side-by-side 2-panel chart comparing Human Listener (left) vs. QwenVL Listener (right),
     both averaged over all datasets with fitted y-axes.

All outputs are saved as both high-resolution PNG (300 DPI) and vector PDF to:
  - Workspace root directory (./)
  - figures/ directory
  - Antigravity brain artifact directory
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from collections import defaultdict
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
ARTIFACT_DIR = Path("/accounts/projects/berkeleynlp/teaywright/.gemini/antigravity-cli/brain/3d861f69-aeb0-408b-b998-392882dfe878")
HUMAN_TRIALS_PATH = REPO_ROOT / "results" / "human_eval" / "human_eval_trials_21600.json"
QWENVL_CSV_PATH = REPO_ROOT / "results" / "model_eval" / "qwenvl_metrics_per_dataset.csv"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling matching original paper figure
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Palatino", "Palatino Linotype", "Times New Roman", "Liberation Serif", "serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.usetex": False,
    "figure.autolayout": False,
})

# Canonical Models and Colors (exact match with paper Figure 5)
MODEL_ORDER = [
    "Human",
    "Gaze-BFH",
    "Gaze-Shaping",
    "REC-BFH",
    "REC-SeqAnyHit",
    "REC-Success",
    "Gaze-SeqLPHit",
    "Gaze-SeqAnyHit",
    "Molmo",
]

COLOR_MAP = {
    "Human": "#DAA520",          # Goldenrod / Gold
    "Gaze-BFH": "#22C55E",       # Vibrant emerald green
    "Gaze-Shaping": "#1B5E20",   # Dark forest green
    "REC-BFH": "#8B1A1A",        # Dark Red / Brown
    "REC-SeqAnyHit": "#FF00FF",  # Magenta
    "REC-Success": "#800080",    # Purple
    "Gaze-SeqLPHit": "#87CEEB",  # Sky Blue
    "Gaze-SeqAnyHit": "#000080", # Navy
    "Molmo": "#000000",          # Black
}

CSV_MODEL_MAP = {
    "Gold": "Human",
    "Gaze-BFH": "Gaze-BFH",
    "Gaze-Shaping": "Gaze-Shaping",
    "REC-BFH": "REC-BFH",
    "REC-SeqAnyHit": "REC-SeqAnyHit",
    "REC-Success": "REC-Success",
    "Gaze-SeqLPHit": "Gaze-SeqLPHit",
    "Gaze-SeqAnyHit": "Gaze-SeqAnyHit",
    "Molmo": "Molmo",
}

SPEAKER_NAME_MAP = {
    "gold": "Human",
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

from scripts.plotting.plot_utils import get_clean_ref_len


def confidence_ellipse(x, y, ax, n_std=2.447, facecolor="none", alpha=0.35, zorder=2, is_mean=True, stroke_color="white", center=None):
    """
    Draw 95% Confidence Ellipse (chi2(2, 0.95) = 5.991, sqrt = 2.447).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or x.size != y.size:
        return None
    cov = np.cov(x, y)
    if is_mean and len(x) > 0:
        cov = cov / len(x)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvals, 0))
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    if center is None:
        center = (np.mean(x), np.mean(y))
    ellipse = Ellipse(
        xy=center, width=width, height=height, angle=angle,
        facecolor=facecolor, alpha=alpha, zorder=zorder
    )
    if stroke_color:
        ellipse.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground=stroke_color, alpha=0.5)])
    return ax.add_patch(ellipse)


def load_qwenvl_data():
    """
    Load Qwen-VL metrics:
    Uses the exact published dataset-level metrics for canonical means,
    and extracts trial-level data from the canonical evaluation runs to compute
    the exact bivariate sampling distribution covariance of the mean.
    """
    df = pd.read_csv(QWENVL_CSV_PATH)
    canonical_runs = {
        "Human": "results/eval_runs/gold_refs",
        "Gaze-BFH": "results/eval_runs/gaze_bfh_seed187357",
        "Gaze-Shaping": "results/eval_runs/gaze_shaping_seed187357",
        "REC-BFH": "results/eval_runs/rec_bfh_seed187357",
        "REC-SeqAnyHit": "results/eval_runs/rec_seq_any_hit_seed838319",
        "REC-Success": "results/eval_runs/rec_success_seed187357",
        "Gaze-SeqLPHit": "results/eval_runs/gaze_seq_lp_hit_seed187357",
        "Gaze-SeqAnyHit": "results/eval_runs/gaze_seq_any_hit_seed187357",
        "Molmo": "results/eval_runs/molmo_vanilla",
    }
    datasets = ["refcoco_testA", "refcoco_testB", "refoi_co_occurrence", "refoi_single_presence"]

    qwenvl_stats = {}
    for csv_model, disp_model in CSV_MODEL_MAP.items():
        sub = df[df["Model"] == csv_model]
        mean_acc = float(sub["Accuracy"].mean())
        mean_len = float(sub["RefLen"].mean())

        # Extract pooled trials for covariance of the mean
        run_dir = REPO_ROOT / canonical_runs[disp_model]
        all_lens = []
        all_hits = []
        for ds in datasets:
            eval_files = list(run_dir.glob(f"evaluation/*{ds}*qwenvl/*evaluation.json"))
            ref_files = list(run_dir.glob(f"references/*{ds}*/*references.json"))
            if not eval_files or not ref_files:
                continue
            with open(eval_files[0]) as fp:
                e_data = json.load(fp)
            with open(ref_files[0]) as fp:
                r_data = json.load(fp)
            for r, e in zip(r_data, e_data.get("results", [])):
                ref_str = r.get("generated_reference", r.get("original_sentence", "")).replace("<|endoftext|>", "").strip()
                all_lens.append(len(ref_str.split()))
                ind_res = e.get("qwenvl_result", {}).get("individual_results", [])
                all_hits.append(float(any(x.get("hit", False) for x in ind_res)))

        qwenvl_stats[disp_model] = {
            "mean_len": mean_len,
            "mean_acc": mean_acc,
            "lens": np.array(all_lens) if all_lens else sub["RefLen"].values,
            "accs": (np.array(all_hits) * 100.0) if all_hits else sub["Accuracy"].values,
        }
    return qwenvl_stats


def load_human_data():
    """Load human trial records for comparison."""
    with open(HUMAN_TRIALS_PATH, "r") as f:
        trials = json.load(f)
    overall = defaultdict(lambda: {"len": [], "acc": []})
    for t in trials:
        raw_spk = t.get("speaker", "")
        spk = SPEAKER_NAME_MAP.get(raw_spk, raw_spk)
        if spk not in MODEL_ORDER:
            continue
        w_len = get_clean_ref_len(t.get("reference", ""))
        hit = 100.0 if t.get("is_hit", False) else 0.0
        overall[spk]["len"].append(w_len)
        overall[spk]["acc"].append(hit)
    return overall


def create_qwenvl_averaged(qwenvl_stats):
    """
    Generates standalone chart for Qwen-VL Listener averaged over all datasets
    with y-axis fitted to data (eliminating the bottom empty white space).
    """
    fig, ax = plt.subplots(figsize=(13.0, 8.5), dpi=300)

    # Qwen-VL accuracies range from 45.65% (Molmo) to 88.98% (Human)
    # Fitting y-axis to [40, 94] cuts out 0-40% dead space completely.
    ax.set_xlim(-0.5, 28.5)
    ax.set_ylim(40, 94)

    ax.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.tick_params(axis="both", which="both", labelsize=14)

    for model_name in MODEL_ORDER:
        color = COLOR_MAP[model_name]
        data = qwenvl_stats[model_name]
        mean_x = data["mean_len"]
        mean_y = data["mean_acc"]
        lens = data["lens"]
        accs = data["accs"]

        # Draw 95% confidence ellipse based on dataset sampling distribution
        confidence_ellipse(lens, accs, ax, n_std=2.447, facecolor=color, alpha=0.35, zorder=2, center=(mean_x, mean_y))

        # Plot single dot per model
        ax.scatter(
            mean_x, mean_y,
            color=color,
            marker="o",
            s=280,
            alpha=0.95,
            edgecolors="white",
            linewidths=1.8,
            zorder=5
        )

        # Offsets chosen specifically for QwenVL coordinates to prevent overlaps:
        # Human (3.52, 88.98), REC-BFH (3.05, 46.18), Gaze-Shaping (4.88, 51.36),
        # Gaze-BFH (5.15, 54.20), Gaze-SeqAnyHit (8.87, 51.78), Gaze-SeqLPHit (11.90, 54.25),
        # Molmo (16.35, 45.65), REC-Success (19.70, 61.94), REC-SeqAnyHit (25.29, 57.96)
        if model_name == "Human":
            offset = (12, -4)
        elif model_name == "REC-BFH":
            offset = (-14, -18)
        elif model_name == "Gaze-Shaping":
            offset = (12, -16)
        elif model_name == "Gaze-BFH":
            offset = (12, 10)
        elif model_name == "Gaze-SeqAnyHit":
            offset = (12, -14)
        elif model_name == "Gaze-SeqLPHit":
            offset = (12, 10)
        elif model_name == "Molmo":
            offset = (12, -4)
        elif model_name == "REC-Success":
            offset = (12, 4)
        elif model_name == "REC-SeqAnyHit":
            offset = (12, 2)
        else:
            offset = (12, 0)

        txt = ax.annotate(
            model_name,
            (mean_x, mean_y),
            xytext=offset,
            textcoords="offset points",
            fontsize=12.5,
            fontweight="bold",
            color="#111111",
            zorder=6
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=3.2, foreground="white", alpha=0.95)])

    ax.set_xlabel("Average Length (Words)", fontsize=18, fontweight="normal", labelpad=10)
    ax.set_ylabel("Communicative Success Rate (%)", fontsize=18, fontweight="normal", labelpad=10)
    ax.set_title("QwenVL Listener (Dataset Average)", fontsize=26, fontweight="normal", pad=15)

    # Models legend placed cleanly in the upper right
    model_handles = [
        Line2D([0], [0], marker="o", color="w", label=m, markerfacecolor=COLOR_MAP[m], markersize=11)
        for m in MODEL_ORDER
    ]
    ax.legend(
        handles=model_handles,
        title="Models",
        loc="upper right",
        bbox_to_anchor=(0.985, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=12,
        title_fontsize=13,
    )

    plt.tight_layout()

    out_png = FIGURES_DIR / "qwenvl_efficiency_averaged.png"
    out_pdf = FIGURES_DIR / "qwenvl_efficiency_averaged.pdf"
    root_png = REPO_ROOT / "qwenvl_efficiency_averaged.png"
    root_pdf = REPO_ROOT / "qwenvl_efficiency_averaged.pdf"
    art_png = ARTIFACT_DIR / "qwenvl_efficiency_averaged.png"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(root_png, dpi=300, bbox_inches="tight")
    fig.savefig(root_pdf, bbox_inches="tight")
    fig.savefig(art_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Qwen-VL Averaged to:\n  - {out_png}\n  - {root_png}\n  - {out_pdf}")


def create_efficiency_comparison_averaged(human_overall, qwenvl_stats):
    """
    Generates side-by-side 2-panel chart comparing Human Listener and QwenVL Listener,
    both averaged over all datasets with fitted y-axes to leave minimal white space.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22.0, 9.0), dpi=300, sharex=True)

    # Shared or tailored fitted limits
    ax1.set_xlim(-0.5, 28.5)
    ax2.set_xlim(-0.5, 28.5)

    # Human ranges from 66.8% to 92.5%, Qwen ranges from 45.6% to 89.0%
    # Using shared y-axis [42, 97]% allows direct comparison across listeners while cutting 0-42% dead space!
    ax1.set_ylim(42, 97)
    ax2.set_ylim(42, 97)

    for ax, title in [(ax1, "Human Listener (Dataset Average)"), (ax2, "QwenVL Listener (Dataset Average)")]:
        ax.grid(True, linestyle="--", alpha=0.45, zorder=0)
        ax.tick_params(axis="both", which="both", labelsize=14)
        ax.set_title(title, fontsize=24, fontweight="normal", pad=15)
        ax.set_xlabel("Average Length (Words)", fontsize=17, fontweight="normal", labelpad=10)

    ax1.set_ylabel("Communicative Success Rate (%)", fontsize=17, fontweight="normal", labelpad=10)

    # --- Panel 1: Human Listener ---
    for model_name in MODEL_ORDER:
        color = COLOR_MAP[model_name]
        h_data = human_overall[model_name]
        lens = np.array(h_data["len"])
        accs = np.array(h_data["acc"])
        mean_x, mean_y = np.mean(lens), np.mean(accs)

        confidence_ellipse(lens, accs, ax1, n_std=2.447, facecolor=color, alpha=0.35, zorder=2)
        ax1.scatter(mean_x, mean_y, color=color, marker="o", s=250, alpha=0.95, edgecolors="white", linewidths=1.8, zorder=5)

        if model_name == "REC-BFH":
            offset = (12, -4)
        elif model_name == "Gaze-BFH":
            offset = (12, -14)
        elif model_name == "Gaze-Shaping":
            offset = (12, 6)
        elif model_name == "Human":
            offset = (12, -4)
        elif model_name == "Gaze-SeqAnyHit":
            offset = (12, 6)
        elif model_name == "Gaze-SeqLPHit":
            offset = (12, -10)
        elif model_name == "Molmo":
            offset = (14, -4)
        elif model_name == "REC-Success":
            offset = (12, 6)
        elif model_name == "REC-SeqAnyHit":
            offset = (12, 2)
        else:
            offset = (12, 0)

        txt = ax1.annotate(model_name, (mean_x, mean_y), xytext=offset, textcoords="offset points",
                           fontsize=11.5, fontweight="bold", color="#111111", zorder=6)
        txt.set_path_effects([path_effects.withStroke(linewidth=3.0, foreground="white", alpha=0.95)])

    # --- Panel 2: QwenVL Listener ---
    for model_name in MODEL_ORDER:
        color = COLOR_MAP[model_name]
        q_data = qwenvl_stats[model_name]
        mean_x = q_data["mean_len"]
        mean_y = q_data["mean_acc"]
        lens = q_data["lens"]
        accs = q_data["accs"]

        confidence_ellipse(lens, accs, ax2, n_std=2.447, facecolor=color, alpha=0.35, zorder=2, center=(mean_x, mean_y))
        ax2.scatter(mean_x, mean_y, color=color, marker="o", s=250, alpha=0.95, edgecolors="white", linewidths=1.8, zorder=5)

        if model_name == "Human":
            offset = (12, -4)
        elif model_name == "REC-BFH":
            offset = (-14, -18)
        elif model_name == "Gaze-Shaping":
            offset = (12, -16)
        elif model_name == "Gaze-BFH":
            offset = (12, 10)
        elif model_name == "Gaze-SeqAnyHit":
            offset = (12, -14)
        elif model_name == "Gaze-SeqLPHit":
            offset = (12, 10)
        elif model_name == "Molmo":
            offset = (12, -4)
        elif model_name == "REC-Success":
            offset = (12, 4)
        elif model_name == "REC-SeqAnyHit":
            offset = (12, 2)
        else:
            offset = (12, 0)

        txt = ax2.annotate(model_name, (mean_x, mean_y), xytext=offset, textcoords="offset points",
                           fontsize=11.5, fontweight="bold", color="#111111", zorder=6)
        txt.set_path_effects([path_effects.withStroke(linewidth=3.0, foreground="white", alpha=0.95)])

    # Clean model legend on right panel
    model_handles = [
        Line2D([0], [0], marker="o", color="w", label=m, markerfacecolor=COLOR_MAP[m], markersize=10)
        for m in MODEL_ORDER
    ]
    ax2.legend(
        handles=model_handles,
        title="Models",
        loc="upper right",
        bbox_to_anchor=(0.985, 0.98),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=11,
        title_fontsize=12,
    )

    plt.tight_layout()

    out_png = FIGURES_DIR / "efficiency_comparison_averaged.png"
    out_pdf = FIGURES_DIR / "efficiency_comparison_averaged.pdf"
    root_png = REPO_ROOT / "efficiency_comparison_averaged.png"
    root_pdf = REPO_ROOT / "efficiency_comparison_averaged.pdf"
    art_png = ARTIFACT_DIR / "efficiency_comparison_averaged.png"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(root_png, dpi=300, bbox_inches="tight")
    fig.savefig(root_pdf, bbox_inches="tight")
    fig.savefig(art_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Side-by-Side Comparison Averaged to:\n  - {out_png}\n  - {root_png}\n  - {out_pdf}")


def main():
    print("=" * 80)
    print(" Generating Averaged Efficiency Plots for Machine Listeners (Qwen-VL) ")
    print("=" * 80)
    qwenvl_stats = load_qwenvl_data()
    human_overall = load_human_data()

    print("\n--- Generating Qwen-VL Averaged Chart ---")
    create_qwenvl_averaged(qwenvl_stats)

    print("\n--- Generating Side-by-Side (Human vs Qwen-VL) Averaged Comparison ---")
    create_efficiency_comparison_averaged(human_overall, qwenvl_stats)

    print("\n" + "=" * 80)
    print(" All Machine Listener Averaged figures successfully generated! ")
    print("=" * 80)


if __name__ == "__main__":
    main()
