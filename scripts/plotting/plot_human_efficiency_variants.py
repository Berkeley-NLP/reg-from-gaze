#!/usr/bin/env python3
"""
scripts/plotting/plot_human_efficiency_variants.py

Generates publication-quality charts of Reference Length vs. Communicative Accuracy
for Human Listeners, fitting the y-axis to minimize empty white space.

Produces 3 requested versions:
  1. Version 1: "as it is now" - Human listener only, all 4 datasets with dataset shapes,
     model colors, and 95% CI confidence ellipses, with y-axis tightly fitted to data.
  2. Version 2: "averaged over all datasets" - One dot per model, with 95% CI confidence
     ellipses, model color coding, and y-axis tightly fitted to the overall range.
  3. Version 3: "2x2 grid of subplots" - 4 separate plots, one for each evaluation dataset
     (RefCOCO Test A, RefCOCO Test B, RefOI Co-occurrence, RefOI Single Presence),
     each with y-axis fitted to minimize white space.

All outputs are saved as both high-resolution PNG (300 DPI) and vector PDF to:
  - gazeRL_publish/figures/
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
TRIALS_PATH = REPO_ROOT / "results" / "human_eval" / "human_eval_trials_21600.json"
FIGURES_DIR = REPO_ROOT / "figures"
ARTIFACT_DIR = Path("/accounts/projects/berkeleynlp/teaywright/.gemini/antigravity-cli/brain/3d861f69-aeb0-408b-b998-392882dfe878")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Publication styling matching original paper figure
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Palatino", "Palatino Linotype", "Times New Roman", "Liberation Serif", "serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
})

# Canonical Models and Colors (exact match with paper Figure 5 / efficiency_comparison_95CI_final)
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
    "Human": "#DAA520",          # Goldenrod
    "Gaze-BFH": "#22C55E",       # Vibrant emerald green
    "Gaze-Shaping": "#1B5E20",   # Dark forest green
    "REC-BFH": "#8B1A1A",        # Dark Red / Brown
    "REC-SeqAnyHit": "#FF00FF",  # Magenta / Fuchsia
    "REC-Success": "#800080",    # Purple
    "Gaze-SeqLPHit": "#87CEEB",  # Sky Blue
    "Gaze-SeqAnyHit": "#000080", # Navy Blue
    "Molmo": "#000000",          # Black
}

# Raw speaker key in json to canonical display name
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

DATASETS = [
    "refcoco_testA",
    "refcoco_testB",
    "refoi_co_occurrence",
    "refoi_single_presence",
]

DATASET_DISPLAY = {
    "refcoco_testA": "RefCOCO Test A",
    "refcoco_testB": "RefCOCO Test B",
    "refoi_co_occurrence": "RefOI Co-occurrence",
    "refoi_single_presence": "RefOI Single Presence",
}

DATASET_SHAPES = {
    "refcoco_testA": "o",          # circle
    "refcoco_testB": "s",          # square
    "refoi_co_occurrence": "^",    # triangle up
    "refoi_single_presence": "D",  # diamond
}


from scripts.plotting.plot_utils import get_clean_ref_len


def confidence_ellipse(x, y, ax, n_std=2.447, facecolor="none", alpha=0.35, zorder=2, is_mean=True, stroke_color="white"):
    """
    Draw 95% Confidence Ellipse (chi2(2, 0.95) = 5.991, sqrt = 2.447).
    Matches paper implementation with white path effect stroke.
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
    center = (np.mean(x), np.mean(y))
    ellipse = Ellipse(
        xy=center, width=width, height=height, angle=angle,
        facecolor=facecolor, alpha=alpha, zorder=zorder
    )
    if stroke_color:
        ellipse.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground=stroke_color, alpha=0.5)])
    return ax.add_patch(ellipse)


def load_human_data():
    """Load and process trial records from human_eval_trials_21600.json."""
    print(f"Loading trials from {TRIALS_PATH}...")
    with open(TRIALS_PATH, "r") as f:
        trials = json.load(f)

    # Per-dataset data: (model, dataset) -> {"len": [...], "acc": [...]}
    per_ds = defaultdict(lambda: {"len": [], "acc": []})
    # Overall data: model -> {"len": [...], "acc": [...]}
    overall = defaultdict(lambda: {"len": [], "acc": []})

    for t in trials:
        raw_spk = t.get("speaker", "")
        spk = SPEAKER_NAME_MAP.get(raw_spk, raw_spk)
        if spk not in MODEL_ORDER:
            continue
        img = t.get("img_name", "")
        for ds in DATASETS:
            if ds in img:
                l = get_clean_ref_len(t.get("reference", ""))
                a = 100.0 if t.get("is_hit", False) else 0.0
                per_ds[(spk, ds)]["len"].append(l)
                per_ds[(spk, ds)]["acc"].append(a)
                overall[spk]["len"].append(l)
                overall[spk]["acc"].append(a)
                break

    print(f"Successfully processed {len(trials)} trials.")
    return per_ds, overall


def create_version1_as_is(per_ds):
    """
    Version 1: "as it is now"
    Human listener only, points per dataset with shapes, colors, 95% CI ellipses,
    fitting the y-axis to the data to minimize empty white space.
    """
    print("\n--- Generating Version 1: As It Is Now (Fitted Y-axis) ---")
    fig, ax = plt.subplots(figsize=(13.5, 9.0), dpi=300)

    # Fitted Y-axis: data points range from 52.0% to 96.2%, ellipses extend from 47.0% to 98.2%
    ax.set_xlim(-0.5, 27)
    ax.set_ylim(45, 101)

    ax.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.tick_params(axis="both", which="both", labelsize=14)

    # Plot points and ellipses for all models and datasets
    for model_name in MODEL_ORDER:
        color = COLOR_MAP[model_name]
        for ds in DATASETS:
            if (model_name, ds) not in per_ds:
                continue
            data = per_ds[(model_name, ds)]
            lens = np.array(data["len"])
            accs = np.array(data["acc"])
            mean_x, mean_y = np.mean(lens), np.mean(accs)

            # Draw 95% confidence ellipse
            confidence_ellipse(lens, accs, ax, n_std=2.447, facecolor=color, alpha=0.35, zorder=2)

            # Plot marker
            ax.scatter(
                mean_x, mean_y,
                color=color,
                marker=DATASET_SHAPES[ds],
                s=240,
                alpha=0.95,
                edgecolors="white",
                linewidths=1.6,
                zorder=5
            )

    ax.set_xlabel("Average Length (Words)", fontsize=18, fontweight="normal", labelpad=10)
    ax.set_ylabel("Communicative Success Rate (%)", fontsize=18, fontweight="normal", labelpad=10)
    ax.set_title("Human Listener", fontsize=26, fontweight="normal", pad=15)

    # Legends: clean dual legends placed outside or neatly positioned
    # Models legend
    model_handles = [
        Line2D([0], [0], marker="o", color="w", label=m, markerfacecolor=COLOR_MAP[m], markersize=11)
        for m in MODEL_ORDER
    ]
    # Datasets legend
    dataset_handles = [
        Line2D([0], [0], marker=DATASET_SHAPES[d], color="w", label=d, markerfacecolor="gray", markersize=11)
        for d in DATASETS
    ]

    leg_models = ax.legend(
        handles=model_handles,
        title="Models",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=12,
        title_fontsize=13,
    )
    ax.add_artist(leg_models)

    leg_datasets = ax.legend(
        handles=dataset_handles,
        title="Datasets",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=12,
        title_fontsize=13,
    )

    plt.tight_layout()

    out_png = FIGURES_DIR / "human_efficiency_as_is.png"
    out_pdf = FIGURES_DIR / "human_efficiency_as_is.pdf"
    root_png = REPO_ROOT / "human_efficiency_as_is.png"
    root_pdf = REPO_ROOT / "human_efficiency_as_is.pdf"
    art_png = ARTIFACT_DIR / "human_efficiency_as_is.png"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(root_png, dpi=300, bbox_inches="tight")
    fig.savefig(root_pdf, bbox_inches="tight")
    fig.savefig(art_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Version 1 to:\n  - {out_png}\n  - {root_png}\n  - {out_pdf}")


def create_version2_averaged(overall):
    """
    Version 2: "averaging over all datasets to have one dot per model"
    One dot per model, 95% CI confidence ellipses, model color coding,
    and y-axis tightly fitted to the overall range.
    """
    print("\n--- Generating Version 2: Averaged Across Datasets (One Dot Per Model) ---")
    fig, ax = plt.subplots(figsize=(13.0, 8.5), dpi=300)

    # Overall accuracy ranges from 66.75% to 92.54%, ellipses extend from 64.4% to 93.8%
    ax.set_xlim(-0.5, 28.5)
    ax.set_ylim(60, 97)

    ax.grid(True, linestyle="--", alpha=0.45, zorder=0)
    ax.tick_params(axis="both", which="both", labelsize=14)

    # Plot points and ellipses for each model
    for model_name in MODEL_ORDER:
        color = COLOR_MAP[model_name]
        data = overall[model_name]
        lens = np.array(data["len"])
        accs = np.array(data["acc"])
        mean_x, mean_y = np.mean(lens), np.mean(accs)

        # Draw 95% confidence ellipse for the overall mean
        confidence_ellipse(lens, accs, ax, n_std=2.447, facecolor=color, alpha=0.35, zorder=2)

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

        # Direct text annotation to the right of each dot without overlapping
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
    ax.set_title("Human Listener (Dataset Average)", fontsize=26, fontweight="normal", pad=15)

    # Models legend placed cleanly in the lower right
    model_handles = [
        Line2D([0], [0], marker="o", color="w", label=m, markerfacecolor=COLOR_MAP[m], markersize=11)
        for m in MODEL_ORDER
    ]
    ax.legend(
        handles=model_handles,
        title="Models",
        loc="lower right",
        bbox_to_anchor=(0.98, 0.03),
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=12,
        title_fontsize=13,
    )

    plt.tight_layout()

    out_png = FIGURES_DIR / "human_efficiency_averaged.png"
    out_pdf = FIGURES_DIR / "human_efficiency_averaged.pdf"
    root_png = REPO_ROOT / "human_efficiency_averaged.png"
    root_pdf = REPO_ROOT / "human_efficiency_averaged.pdf"
    art_png = ARTIFACT_DIR / "human_efficiency_averaged.png"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(root_png, dpi=300, bbox_inches="tight")
    fig.savefig(root_pdf, bbox_inches="tight")
    fig.savefig(art_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Version 2 to:\n  - {out_png}\n  - {root_png}\n  - {out_pdf}")


def create_version3_2x2_grid(per_ds, shared_y=False):
    """
    Version 3: "having 4 separate plots for each dataset in a 2x2 grid of subplots"
    Subplots: RefCOCO Test A, RefCOCO Test B, RefOI Co-occurrence, RefOI Single Presence.
    Each subplot has the 9 models, 95% CI ellipses, and fitted y-axis to minimize white space.

    Args:
        per_ds: dictionary of per-dataset trial stats.
        shared_y: if True, uses a shared fitted y-axis [45, 101] across all subplots;
                  if False, individually fits each subplot's y-axis tightly.
    """
    suffix = "_shared_y" if shared_y else ""
    desc = "Shared Fitted Y-axis" if shared_y else "Individually Fitted Y-axes"
    print(f"\n--- Generating Version 3: 2x2 Grid of Subplots ({desc}) ---")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=300, sharex=True, sharey=shared_y)

    grid_order = [
        ("refcoco_testA", 0, 0),
        ("refcoco_testB", 0, 1),
        ("refoi_co_occurrence", 1, 0),
        ("refoi_single_presence", 1, 1),
    ]

    for ds, r, c in grid_order:
        ax = axes[r, c]
        shape = DATASET_SHAPES[ds]

        all_y_min, all_y_max = 100.0, 0.0

        for model_name in MODEL_ORDER:
            if (model_name, ds) not in per_ds:
                continue
            data = per_ds[(model_name, ds)]
            lens = np.array(data["len"])
            accs = np.array(data["acc"])
            mean_x, mean_y = np.mean(lens), np.mean(accs)
            color = COLOR_MAP[model_name]

            # 95% CI ellipse
            el = confidence_ellipse(lens, accs, ax, n_std=2.447, facecolor=color, alpha=0.35, zorder=2)

            # Scatter dot
            ax.scatter(
                mean_x, mean_y,
                color=color,
                marker=shape,
                s=220,
                alpha=0.95,
                edgecolors="white",
                linewidths=1.6,
                zorder=5
            )

            # Track bounds
            if el is not None:
                cov = np.cov(lens, accs) / len(lens)
                eigenvals = np.linalg.eigvalsh(cov)
                semi_h = 2.447 * np.sqrt(max(eigenvals))
                all_y_min = min(all_y_min, mean_y - semi_h)
                all_y_max = max(all_y_max, mean_y + semi_h)

        if shared_y:
            ax.set_ylim(45, 101)
        else:
            y_bottom = np.floor(all_y_min - 2.0)
            y_top = min(101.0, np.ceil(all_y_max + 2.0))
            y_bottom = max(0.0, float(int(y_bottom / 5.0) * 5.0))
            ax.set_ylim(y_bottom, y_top)

        ax.set_xlim(-0.5, 27)
        ax.set_title(DATASET_DISPLAY[ds], fontsize=20, fontweight="bold", pad=10)
        ax.grid(True, linestyle="--", alpha=0.45, zorder=0)
        ax.tick_params(axis="both", which="both", labelsize=13)

        if c == 0:
            ax.set_ylabel("Communicative Success Rate (%)", fontsize=15, fontweight="normal", labelpad=8)
        elif not shared_y:
            # Also show y-axis label on right column when y-scales differ
            ax.set_ylabel("Communicative Success Rate (%)", fontsize=15, fontweight="normal", labelpad=8)

        if r == 1:
            ax.set_xlabel("Average Length (Words)", fontsize=15, fontweight="normal", labelpad=8)

    # Shared Models Legend at top
    model_handles = [
        Line2D([0], [0], marker="o", color="w", label=m, markerfacecolor=COLOR_MAP[m], markersize=11)
        for m in MODEL_ORDER
    ]
    fig.legend(
        handles=model_handles,
        title="Models",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=5,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        fontsize=12,
        title_fontsize=13,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = FIGURES_DIR / f"human_efficiency_2x2_grid{suffix}.png"
    out_pdf = FIGURES_DIR / f"human_efficiency_2x2_grid{suffix}.pdf"
    root_png = REPO_ROOT / f"human_efficiency_2x2_grid{suffix}.png"
    root_pdf = REPO_ROOT / f"human_efficiency_2x2_grid{suffix}.pdf"
    art_png = ARTIFACT_DIR / f"human_efficiency_2x2_grid{suffix}.png"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(root_png, dpi=300, bbox_inches="tight")
    fig.savefig(root_pdf, bbox_inches="tight")
    fig.savefig(art_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Version 3 ({desc}) to:\n  - {out_png}\n  - {root_png}\n  - {out_pdf}")


def main():
    print("=" * 80)
    print(" Generating 3 Publication Variants of Human Efficiency Plot ")
    print("=" * 80)
    per_ds, overall = load_human_data()

    create_version1_as_is(per_ds)
    create_version2_averaged(overall)
    create_version3_2x2_grid(per_ds, shared_y=False)  # Minimal white space per dataset
    create_version3_2x2_grid(per_ds, shared_y=True)   # Shared scale across datasets

    print("\n" + "=" * 80)
    print(" All Human Efficiency figures successfully generated! ")
    print("=" * 80)


if __name__ == "__main__":
    main()
