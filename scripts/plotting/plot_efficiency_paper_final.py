#!/usr/bin/env python3
"""
scripts/plotting/plot_efficiency_paper_final.py

Unified, publication-quality plotting script for Communicative Efficiency
(Average Reference Length in Words vs. Communicative Success Rate in %).

Features:
  - Canonical paper color scheme (exact match with paper Figure 5 and linguistic analysis):
      Human: #DAA520 (Goldenrod)
      Gaze-BFH: red
      Gaze-Shaping: green
      REC-BFH: brown
      REC-Shaping: teal (and burnt orange variant #d35400 also generated)
      REC-Success: purple
      REC-SeqAnyHit: magenta
      Gaze-SeqLPHit: skyblue
      Gaze-SeqAnyHit: navy
      Molmo: black
  - Fully incorporates all 10 models on QwenVL evaluation (including REC-Shaping).
  - Human evaluation includes the 9 published Prolific models (21,600 trials).
  - Tight y-axis limits to eliminate empty white space.
  - 95% CI confidence ellipses (n_std=2.447 for 2D bivariate normal mean).
  - High-resolution exports (PNG @ 300 DPI and vector PDF) to root directory,
    figures/ directory, and antigravity artifact directory.
"""

import os
import re
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
from PIL import Image

# --- Directory Paths ---
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
ARTIFACT_DIR = Path("/accounts/projects/berkeleynlp/teaywright/.gemini/antigravity-cli/brain/4650fa71-f324-48ea-9c07-38c6865071e4")
CURRENT_ARTIFACT_DIR = Path("/accounts/projects/berkeleynlp/teaywright/.gemini/antigravity-cli/brain/834dbf5c-6927-449b-93b5-62bb598fdf21")
WORKSPACE_DIR = Path("/accounts/projects/berkeleynlp/teaywright")
GAZERL_DIR = Path("/accounts/projects/berkeleynlp/teaywright/gazeRL")
GAZERL_PLOTS_DIR = GAZERL_DIR / "plots"
HUMAN_TRIALS_PATH = REPO_ROOT / "results" / "human_eval" / "human_eval_trials_24000.json"
if not HUMAN_TRIALS_PATH.exists():
    HUMAN_TRIALS_PATH = REPO_ROOT / "results" / "human_eval" / "human_eval_trials_21600.json"
QWENVL_CSV_PATH = REPO_ROOT / "results" / "model_eval" / "qwenvl_metrics_per_dataset.csv"

for d in [FIGURES_DIR, ARTIFACT_DIR, CURRENT_ARTIFACT_DIR, GAZERL_PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- Matplotlib Styling ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Palatino LT STD", "Book Antiqua", "Georgia", "DejaVu Serif", "serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
})

# --- Canonical Models and Color Map ---
COLOR_MAP = {
    "Human": "#DAA520",
    "Gaze-BFH": "#22C55E",      # Vibrant emerald green
    "Gaze-Shaping": "#1B5E20",  # Dark forest green
    "REC-BFH": "brown",
    "REC-Shaping": "red",       # Bright red
    "REC-Success": "purple",
    "REC-SeqAnyHit": "magenta",
    "Gaze-SeqLPHit": "skyblue",
    "Gaze-SeqAnyHit": "navy",
    "Molmo": "black",
}

# Also support burnt orange (#d35400) and teal variants
COLOR_MAP_ORANGE = dict(COLOR_MAP)
COLOR_MAP_ORANGE["REC-Shaping"] = "#d35400"
COLOR_MAP_TEAL = dict(COLOR_MAP)
COLOR_MAP_TEAL["REC-Shaping"] = "teal"

MODEL_ORDER = [
    "Human",
    "Gaze-BFH",
    "Gaze-Shaping",
    "REC-BFH",
    "REC-Shaping",
    "REC-Success",
    "REC-SeqAnyHit",
    "Gaze-SeqLPHit",
    "Gaze-SeqAnyHit",
    "Molmo",
]

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
    "refoi_single_presence": "RefOI Single-presence",
}

SHAPE_MAP = {
    "refcoco_testA": "o",          # circle
    "refcoco_testB": "s",          # square
    "refoi_co_occurrence": "^",    # triangle up
    "refoi_single_presence": "D",  # diamond
}

HUMAN_SPEAKER_MAP = {
    "gold": "Human",
    "molmo_vanilla": "Molmo",
    "sparse_constant_kl_02": "Gaze-BFH",
    "shaping": "Gaze-Shaping",
    "binary": "Gaze-SeqAnyHit",
    "binary_last_point": "Gaze-SeqLPHit",
    "supervised": "REC-Success",
    "iterative_sparse": "REC-BFH",
    "iterative_binary": "REC-SeqAnyHit",
    "iterative_shaping": "REC-Shaping",
}


from scripts.plotting.plot_utils import (
    get_clean_ref_len,
    extract_dataset,
    confidence_ellipse_from_cov,
    confidence_ellipse,
    load_human_trials_data,
)


def load_human_data():
    """Loads all human evaluation trials using shared plot_utils loader."""
    return load_human_trials_data(HUMAN_TRIALS_PATH)


def load_qwenvl_published_metrics():
    """
    Loads canonical published metrics from results/model_eval/qwenvl_metrics_per_dataset.csv.
    This contains all 10 models (including REC-Shaping) across all 4 datasets matching Tables 6 & 7.
    """
    df = pd.read_csv(QWENVL_CSV_PATH)
    # Rename 'Gold' to 'Human' for consistent paper presentation
    df["Model"] = df["Model"].replace({"Gold": "Human"})
    return df


def compute_qwenvl_covariances(human_df):
    """
    Computes representative bivariate covariance matrices for length and accuracy per dataset
    and overall to produce 95% CI confidence ellipses matching the canonical figure.
    """
    cov_dict = {}
    for ds in DATASETS:
        sub = human_df[human_df["Dataset"] == ds]
        cov_dict[ds] = np.cov(sub["Length"], sub["Accuracy"])
    cov_dict["overall"] = np.cov(human_df["Length"], human_df["Accuracy"])
    return cov_dict


STARRED_MODELS = {"Gaze-Shaping", "Gaze-SeqAnyHit"}


def save_plot_outputs(fig, basename, bbox=None):
    """Saves figure in root, figures/, workspace, plots/, and artifact directories."""
    dirs = [
        REPO_ROOT,
        FIGURES_DIR,
        WORKSPACE_DIR,
        GAZERL_DIR,
        GAZERL_PLOTS_DIR,
        CURRENT_ARTIFACT_DIR,
        ARTIFACT_DIR,
    ]
    bbox_arg = bbox if bbox is not None else "tight"

    for d in dirs:
        if d.exists():
            fig.savefig(d / f"{basename}.png", dpi=300, bbox_inches=bbox_arg)
            fig.savefig(d / f"{basename}.pdf", bbox_inches=bbox_arg)

    # Maintain strict alias between human_efficiency_average_dotted and human_efficiency_averaged_dotted
    if "human_efficiency_averaged_dotted" in basename:
        alias_name = basename.replace("human_efficiency_averaged_dotted", "human_efficiency_average_dotted")
        for d in dirs:
            if d.exists():
                fig.savefig(d / f"{alias_name}.png", dpi=300, bbox_inches=bbox_arg)
                fig.savefig(d / f"{alias_name}.pdf", bbox_inches=bbox_arg)
    elif "human_efficiency_average_dotted" in basename:
        alias_name = basename.replace("human_efficiency_average_dotted", "human_efficiency_averaged_dotted")
        for d in dirs:
            if d.exists():
                fig.savefig(d / f"{alias_name}.png", dpi=300, bbox_inches=bbox_arg)
                fig.savefig(d / f"{alias_name}.pdf", bbox_inches=bbox_arg)

    print(f"Saved {basename} across publication, workspace, and artifact dirs.")


# =========================================================================
# 1. Human Efficiency - Dataset Averaged (1 dot per model, fitted y-axis)
# =========================================================================
def plot_human_efficiency_averaged(human_df, color_map=COLOR_MAP, suffix="", add_dotted_line=False, models_to_show=None, bbox=None):
    fig, ax = plt.subplots(figsize=(14, 10))

    # Fitted boundaries to eliminate empty white space and provide breathing room
    ax.set_xlim(0, 31)
    ax.set_ylim(63, 95)
    ax.tick_params(axis='both', which='both', length=6, width=1.2, labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

    # Carefully calibrated non-overlapping label offsets (dx, dy, ha, va)
    offsets = {
        "Human": (0.6, 0.0, "left", "center"),
        "REC-BFH": (0.6, 0.0, "left", "center"),
        "Gaze-BFH": (0.6, -1.8, "left", "center"),
        "REC-Shaping": (0.7, 0.0, "left", "center"),
        "Gaze-Shaping": (0.6, 1.6, "left", "center"),
        "Gaze-SeqAnyHit": (-0.7, 0.2, "right", "center") if add_dotted_line else (0.6, 0.7, "left", "center"),
        "Gaze-SeqLPHit": (0.6, -0.7, "left", "center"),
        "Molmo": (0.6, 0.0, "left", "center"),
        "REC-Success": (0.6, 0.0, "left", "center"),
        "REC-SeqAnyHit": (0.6, 0.0, "left", "center"),
    }

    means = {}
    for model in MODEL_ORDER:
        m_df = human_df[human_df["Model"] == model]
        if m_df.empty:
            continue
        x_mean = m_df["Length"].mean()
        y_mean = m_df["Accuracy"].mean()
        means[model] = (x_mean, y_mean)

        if models_to_show is not None and model not in models_to_show:
            continue

        c = color_map[model]
        el = confidence_ellipse(m_df["Length"], m_df["Accuracy"], ax, n_std=2.447, facecolor=c, alpha=0.20, zorder=2, is_mean=True)
        if el:
            el.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='white', alpha=0.3)])

        is_starred = model in STARRED_MODELS
        marker = '*' if is_starred else 'o'
        size = 720 if is_starred else 360
        lw = 2.0 if is_starred else 2.2
        ax.scatter(x_mean, y_mean, color=c, marker=marker, s=size, alpha=0.95, edgecolors='white', linewidths=lw, zorder=4)

        dx, dy, ha, va = offsets.get(model, (0.6, 0.0, "left", "center"))
        label_text = f"{model}*" if is_starred else model
        ax.text(x_mean + dx, y_mean + dy, label_text, fontsize=18, fontweight='bold', va=va, ha=ha, zorder=5)

    if add_dotted_line and "Human" in means and "Molmo" in means:
        hx, hy = means["Human"]
        mx, my = means["Molmo"]
        ax.plot([hx, mx], [hy, my], linestyle=":", color="#444444", linewidth=2.5, zorder=1)

    ax.set_xlabel('Average Length (Words)', fontsize=26, fontweight='bold', labelpad=14)
    ax.set_ylabel('Communicative Success Rate (%)', fontsize=26, fontweight='bold', labelpad=14)
    ax.set_title('Human Listener (Dataset Average)', fontsize=29, fontweight='bold', pad=18)

    plt.tight_layout()
    base_name = f"human_efficiency_averaged_dotted{suffix}" if add_dotted_line else f"human_efficiency_averaged{suffix}"
    save_plot_outputs(fig, base_name, bbox=bbox)
    if add_dotted_line:
        save_plot_outputs(fig, f"human_efficiency_dotted{suffix}", bbox=bbox)
    return fig


def plot_human_efficiency_animation_steps(human_df, color_map=COLOR_MAP, suffix=""):
    """
    Generates the 4 progressive slide animation stages requested by the user, with 100% pixel-perfect frame alignment:
      - Step 1: Only Molmo and Human (no dotted line)
      - Step 2: Molmo and Human with a dotted line connecting them
      - Step 3: Molmo + Human + dotted line + REC- models (REC-BFH, REC-Shaping, REC-Success, REC-SeqAnyHit)
      - Step 4: Full plot (all 10 models + dotted line) -> human_efficiency_average_dotted.png/pdf
    Also creates looping animated GIFs for slide presentations.
    """
    # 1. First build Step 4 (full plot) to compute the master tight bounding box
    fig_full = plot_human_efficiency_averaged(human_df, color_map=color_map, suffix=suffix, add_dotted_line=True, models_to_show=MODEL_ORDER)
    fig_full.canvas.draw()
    renderer = fig_full.canvas.get_renderer()
    master_bbox = fig_full.get_tightbbox(renderer).padded(0.1)

    save_plot_outputs(fig_full, f"human_efficiency_step4_full{suffix}", bbox=master_bbox)
    save_plot_outputs(fig_full, f"human_efficiency_slide4{suffix}", bbox=master_bbox)
    save_plot_outputs(fig_full, f"human_efficiency_averaged_dotted{suffix}", bbox=master_bbox)
    save_plot_outputs(fig_full, f"human_efficiency_average_dotted{suffix}", bbox=master_bbox)
    plt.close(fig_full)

    # 2. Step 1: Only Molmo and Human
    fig1 = plot_human_efficiency_averaged(human_df, color_map=color_map, suffix=suffix, add_dotted_line=False, models_to_show=["Human", "Molmo"])
    save_plot_outputs(fig1, f"human_efficiency_step1_molmo_human{suffix}", bbox=master_bbox)
    save_plot_outputs(fig1, f"human_efficiency_slide1{suffix}", bbox=master_bbox)
    plt.close(fig1)

    # 3. Step 2: Molmo and Human with dotted line
    fig2 = plot_human_efficiency_averaged(human_df, color_map=color_map, suffix=suffix, add_dotted_line=True, models_to_show=["Human", "Molmo"])
    save_plot_outputs(fig2, f"human_efficiency_step2_dotted{suffix}", bbox=master_bbox)
    save_plot_outputs(fig2, f"human_efficiency_slide2{suffix}", bbox=master_bbox)
    plt.close(fig2)

    # 4. Step 3: Molmo + Human + REC- models + dotted line
    rec_models = ["Human", "Molmo", "REC-BFH", "REC-Shaping", "REC-Success", "REC-SeqAnyHit"]
    fig3 = plot_human_efficiency_averaged(human_df, color_map=color_map, suffix=suffix, add_dotted_line=True, models_to_show=rec_models)
    save_plot_outputs(fig3, f"human_efficiency_step3_rec_models{suffix}", bbox=master_bbox)
    save_plot_outputs(fig3, f"human_efficiency_slide3{suffix}", bbox=master_bbox)
    plt.close(fig3)

    # 5. Build Animated GIFs
    step_png_paths = [
        FIGURES_DIR / f"human_efficiency_step1_molmo_human{suffix}.png",
        FIGURES_DIR / f"human_efficiency_step2_dotted{suffix}.png",
        FIGURES_DIR / f"human_efficiency_step3_rec_models{suffix}.png",
        FIGURES_DIR / f"human_efficiency_step4_full{suffix}.png",
    ]

    images = [Image.open(p).convert("RGB") for p in step_png_paths]
    durations = [2000, 2000, 2500, 3500]

    target_width = 1400
    w, h = images[0].size
    target_height = int(h * (target_width / w))
    resized_images = [im.resize((target_width, target_height), Image.Resampling.LANCZOS) for im in images]

    gif_names = [
        f"human_efficiency_animation{suffix}.gif",
        f"human_efficiency_average_dotted_animation{suffix}.gif",
        f"human_efficiency_averaged_dotted_animation{suffix}.gif",
    ]

    out_dirs = [GAZERL_DIR, GAZERL_PLOTS_DIR, WORKSPACE_DIR, REPO_ROOT, FIGURES_DIR, CURRENT_ARTIFACT_DIR, ARTIFACT_DIR]
    for d in out_dirs:
        if d.exists():
            for gname in gif_names:
                resized_images[0].save(
                    d / gname,
                    save_all=True,
                    append_images=resized_images[1:],
                    duration=durations,
                    loop=0
                )
            images[0].save(
                d / f"human_efficiency_animation_highres{suffix}.gif",
                save_all=True,
                append_images=images[1:],
                duration=durations,
                loop=0
            )


# =========================================================================
# 2. QwenVL Efficiency - Dataset Averaged (1 dot per model, all 10 models)
# =========================================================================
def plot_qwenvl_efficiency_averaged(qwenvl_df, cov_dict, color_map=COLOR_MAP, suffix=""):
    fig, ax = plt.subplots(figsize=(14, 10))

    # Fitted boundaries to eliminate empty white space and provide breathing room
    ax.set_xlim(0, 31)
    ax.set_ylim(40, 93)
    ax.tick_params(axis='both', which='both', length=6, width=1.2, labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

    # Compute overall average per model across the 4 datasets
    avg_df = qwenvl_df.groupby("Model")[["RefLen", "Accuracy"]].mean().reset_index()

    # Carefully calibrated non-overlapping label offsets (dx, dy, ha, va)
    offsets = {
        "Human": (0.6, 0.0, "left", "center"),
        "REC-BFH": (-0.6, 0.0, "right", "center"),
        "Gaze-BFH": (0.5, 1.6, "left", "bottom"),
        "Gaze-Shaping": (-0.6, 0.0, "right", "center"),
        "REC-Shaping": (0.0, -3.2, "center", "top"),
        "Gaze-SeqAnyHit": (0.9, -1.8, "left", "top"),
        "Gaze-SeqLPHit": (0.6, 0.4, "left", "bottom"),
        "Molmo": (0.6, 0.0, "left", "center"),
        "REC-Success": (0.6, 0.0, "left", "center"),
        "REC-SeqAnyHit": (0.6, 0.0, "left", "center"),
    }

    for model in MODEL_ORDER:
        sub = avg_df[avg_df["Model"] == model]
        if sub.empty:
            continue
        c = color_map[model]
        x_mean = sub["RefLen"].values[0]
        y_mean = sub["Accuracy"].values[0]

        # 95% CI confidence ellipse
        el = confidence_ellipse_from_cov(x_mean, y_mean, cov_dict["overall"], 1200, ax, n_std=2.447, facecolor=c, alpha=0.20, zorder=2)
        if el:
            el.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='white', alpha=0.3)])

        is_starred = model in STARRED_MODELS
        marker = '*' if is_starred else 'o'
        size = 720 if is_starred else 360
        lw = 2.0 if is_starred else 2.2
        ax.scatter(x_mean, y_mean, color=c, marker=marker, s=size, alpha=0.95, edgecolors='white', linewidths=lw, zorder=4)

        # Label
        dx, dy, ha, va = offsets.get(model, (0.6, 0.0, "left", "center"))
        label_text = f"{model}*" if is_starred else model
        ax.text(x_mean + dx, y_mean + dy, label_text, fontsize=18, fontweight='bold', va=va, ha=ha, zorder=5)

    ax.set_xlabel('Average Length (Words)', fontsize=26, fontweight='bold', labelpad=14)
    ax.set_ylabel('Communicative Success Rate (%)', fontsize=26, fontweight='bold', labelpad=14)
    ax.set_title('QwenVL Listener (Dataset Average)', fontsize=29, fontweight='bold', pad=18)

    plt.tight_layout()
    save_plot_outputs(fig, f"qwenvl_efficiency_averaged{suffix}")
    plt.close(fig)


# =========================================================================
# 3. Two-Panel Dataset Averaged Comparison (Human left, QwenVL right)
# =========================================================================
def plot_efficiency_comparison_averaged(human_df, qwenvl_df, cov_dict, color_map=COLOR_MAP, suffix="", add_dotted_line=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 10.5), sharey=False)

    # Left: Human Listener Averaged
    ax1.set_xlim(0, 31)
    ax1.set_ylim(63, 95)
    ax1.tick_params(axis='both', which='both', length=6, width=1.2, labelsize=20)
    ax1.grid(True, linestyle='--', alpha=0.4, zorder=0)

    h_offsets = {
        "Human": (0.6, 0.0, "left", "center"),
        "REC-BFH": (0.6, 0.0, "left", "center"),
        "Gaze-BFH": (0.6, -1.8, "left", "center"),
        "REC-Shaping": (0.7, 0.0, "left", "center"),
        "Gaze-Shaping": (0.6, 1.6, "left", "center"),
        "Gaze-SeqAnyHit": (-0.7, 0.2, "right", "center") if add_dotted_line else (0.6, 0.7, "left", "center"),
        "Gaze-SeqLPHit": (0.6, -0.7, "left", "center"),
        "Molmo": (0.6, 0.0, "left", "center"),
        "REC-Success": (0.6, 0.0, "left", "center"),
        "REC-SeqAnyHit": (0.6, 0.0, "left", "center"),
    }

    h_means = {}
    for model in MODEL_ORDER:
        m_df = human_df[human_df["Model"] == model]
        if m_df.empty:
            continue
        c = color_map[model]
        x_mean = m_df["Length"].mean()
        y_mean = m_df["Accuracy"].mean()
        h_means[model] = (x_mean, y_mean)
        el = confidence_ellipse(m_df["Length"], m_df["Accuracy"], ax1, n_std=2.447, facecolor=c, alpha=0.20, zorder=2, is_mean=True)
        if el:
            el.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='white', alpha=0.3)])

        is_starred = model in STARRED_MODELS
        marker = '*' if is_starred else 'o'
        size = 720 if is_starred else 360
        lw = 2.0 if is_starred else 2.2
        ax1.scatter(x_mean, y_mean, color=c, marker=marker, s=size, alpha=0.95, edgecolors='white', linewidths=lw, zorder=4)

        dx, dy, ha, va = h_offsets.get(model, (0.6, 0.0, "left", "center"))
        label_text = f"{model}*" if is_starred else model
        ax1.text(x_mean + dx, y_mean + dy, label_text, fontsize=18, fontweight='bold', va=va, ha=ha, zorder=5)

    if add_dotted_line and "Human" in h_means and "Molmo" in h_means:
        hx, hy = h_means["Human"]
        mx, my = h_means["Molmo"]
        ax1.plot([hx, mx], [hy, my], linestyle=":", color="#444444", linewidth=2.5, zorder=1)

    ax1.set_xlabel('Average Length (Words)', fontsize=26, fontweight='bold', labelpad=14)
    ax1.set_ylabel('Communicative Success Rate (%)', fontsize=26, fontweight='bold', labelpad=14)
    ax1.set_title('Human Listener (Dataset Average)', fontsize=29, fontweight='bold', pad=18)

    # Right: QwenVL Listener Averaged (All 10 models with REC-Shaping)
    ax2.set_xlim(0, 31)
    ax2.set_ylim(40, 93)
    ax2.tick_params(axis='both', which='both', length=6, width=1.2, labelsize=20)
    ax2.grid(True, linestyle='--', alpha=0.4, zorder=0)

    q_avg = qwenvl_df.groupby("Model")[["RefLen", "Accuracy"]].mean().reset_index()

    q_offsets = {
        "Human": (0.6, 0.0, "left", "center"),
        "REC-BFH": (-0.6, 0.0, "right", "center"),
        "Gaze-BFH": (0.5, 1.6, "left", "bottom"),
        "Gaze-Shaping": (-0.6, 0.0, "right", "center"),
        "REC-Shaping": (0.0, -3.2, "center", "top"),
        "Gaze-SeqAnyHit": (0.9, -1.8, "left", "top"),
        "Gaze-SeqLPHit": (0.6, 0.4, "left", "bottom"),
        "Molmo": (0.6, 0.0, "left", "center"),
        "REC-Success": (0.6, 0.0, "left", "center"),
        "REC-SeqAnyHit": (0.6, 0.0, "left", "center"),
    }

    for model in MODEL_ORDER:
        sub = q_avg[q_avg["Model"] == model]
        if sub.empty:
            continue
        c = color_map[model]
        x_mean = sub["RefLen"].values[0]
        y_mean = sub["Accuracy"].values[0]
        el = confidence_ellipse_from_cov(x_mean, y_mean, cov_dict["overall"], 1200, ax2, n_std=2.447, facecolor=c, alpha=0.20, zorder=2)
        if el:
            el.set_path_effects([path_effects.withStroke(linewidth=1.5, foreground='white', alpha=0.3)])

        is_starred = model in STARRED_MODELS
        marker = '*' if is_starred else 'o'
        size = 720 if is_starred else 360
        lw = 2.0 if is_starred else 2.2
        ax2.scatter(x_mean, y_mean, color=c, marker=marker, s=size, alpha=0.95, edgecolors='white', linewidths=lw, zorder=4)

        dx, dy, ha, va = q_offsets.get(model, (0.6, 0.0, "left", "center"))
        label_text = f"{model}*" if is_starred else model
        ax2.text(x_mean + dx, y_mean + dy, label_text, fontsize=18, fontweight='bold', va=va, ha=ha, zorder=5)

    ax2.set_xlabel('Average Length (Words)', fontsize=26, fontweight='bold', labelpad=14)
    ax2.set_ylabel('Communicative Success Rate (%)', fontsize=26, fontweight='bold', labelpad=14)
    ax2.set_title('QwenVL Listener (Dataset Average)', fontsize=29, fontweight='bold', pad=18)

    # Model Legend
    legend_handles = [
        Line2D(
            [0], [0],
            marker='*' if m in STARRED_MODELS else 'o',
            color='w',
            label=f"{m}*" if m in STARRED_MODELS else m,
            markerfacecolor=color_map[m],
            markersize=21 if m in STARRED_MODELS else 16,
            markeredgecolor='white',
            markeredgewidth=1.5
        )
        for m in MODEL_ORDER
    ]
    leg = fig.legend(
        handles=legend_handles,
        title="Models",
        loc="center right",
        bbox_to_anchor=(0.995, 0.5),
        frameon=True,
        fontsize=17,
        title_fontsize=20
    )
    leg.get_title().set_fontweight('bold')

    plt.subplots_adjust(left=0.065, right=0.835, wspace=0.18, top=0.92, bottom=0.10)
    base_name = f"efficiency_comparison_averaged_dotted{suffix}" if add_dotted_line else f"efficiency_comparison_averaged{suffix}"
    save_plot_outputs(fig, base_name)
    plt.close(fig)


# =========================================================================
# 4. Canonical Paper 2-Panel Figure with All Results (4 Datasets, Shapes, Ellipses)
# =========================================================================
def plot_efficiency_comparison_all_results(human_df, qwenvl_df, cov_dict, color_map=COLOR_MAP, suffix="", shared_y=False):
    """
    The canonical 2-panel figure:
      Left: Human Listener (9 Prolific models across 4 datasets)
      Right: QwenVL Listener (All 10 models across 4 datasets, including REC-Shaping)
    Fitted y-axes eliminate empty white space while preserving full visibility.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9), sharey=shared_y)

    # Left: Human Listener
    ax1.set_xlim(0, 27)
    if shared_y:
        ax1.set_ylim(12, 100)
    else:
        ax1.set_ylim(46, 100)
    ax1.tick_params(axis='both', which='both', length=4, width=1, labelsize=16)
    ax1.grid(True, linestyle='--', alpha=0.4, zorder=0)

    for model in MODEL_ORDER:
        m_df = human_df[human_df["Model"] == model]
        if m_df.empty:
            continue
        c = color_map[model]
        for ds in DATASETS:
            subset = m_df[m_df["Dataset"] == ds]
            if subset.empty:
                continue
            x = subset["Length"].mean()
            y = subset["Accuracy"].mean()

            el = confidence_ellipse(subset["Length"], subset["Accuracy"], ax1, n_std=2.447, facecolor=c, alpha=0.16, zorder=2, is_mean=True)
            if el:
                el.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground='white', alpha=0.25)])

            ax1.scatter(x, y, color=c, marker=SHAPE_MAP[ds], s=250, alpha=0.92, edgecolors='white', linewidths=1.5, zorder=3)

    ax1.set_xlabel('Average Length (Words)', fontsize=22, fontweight='bold', labelpad=12)
    ax1.set_ylabel('Communicative Success Rate (%)', fontsize=22, fontweight='bold', labelpad=12)
    ax1.set_title('Human Listener', fontsize=28, fontweight='bold', pad=18)

    # Right: QwenVL Listener (All 10 models with REC-Shaping)
    ax2.set_xlim(0, 27)
    if shared_y:
        ax2.set_ylim(12, 100)
    else:
        ax2.set_ylim(12, 98)
    ax2.tick_params(axis='both', which='both', length=4, width=1, labelsize=16)
    ax2.grid(True, linestyle='--', alpha=0.4, zorder=0)

    for model in MODEL_ORDER:
        sub = qwenvl_df[qwenvl_df["Model"] == model]
        if sub.empty:
            continue
        c = color_map[model]
        for ds in DATASETS:
            row = sub[sub["Dataset"] == ds]
            if row.empty:
                continue
            x = row["RefLen"].values[0]
            y = row["Accuracy"].values[0]

            el = confidence_ellipse_from_cov(x, y, cov_dict[ds], 300, ax2, n_std=2.447, facecolor=c, alpha=0.16, zorder=2)
            if el:
                el.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground='white', alpha=0.25)])

            ax2.scatter(x, y, color=c, marker=SHAPE_MAP[ds], s=250, alpha=0.92, edgecolors='white', linewidths=1.5, zorder=3)

    ax2.set_xlabel('Average Length (Words)', fontsize=22, fontweight='bold', labelpad=12)
    if not shared_y:
        ax2.set_ylabel('Communicative Success Rate (%)', fontsize=22, fontweight='bold', labelpad=12)
    ax2.set_title('QwenVL Listener', fontsize=28, fontweight='bold', pad=18)

    # Legends on the right side
    c_leg = [Line2D([0], [0], marker='o', color='w', label=m, markerfacecolor=color_map[m], markersize=14) for m in MODEL_ORDER]
    s_leg = [Line2D([0], [0], marker=SHAPE_MAP[ds], color='w', label=DATASET_DISPLAY[ds], markerfacecolor='gray', markersize=14) for ds in DATASETS]

    leg_models = fig.legend(handles=c_leg, title="Models", loc="center left", bbox_to_anchor=(0.91, 0.62), frameon=False, fontsize=16)
    leg_models.get_title().set_fontsize(20)
    leg_models.get_title().set_fontweight('bold')

    leg_datasets = fig.legend(handles=s_leg, title="Datasets", loc="center left", bbox_to_anchor=(0.91, 0.22), frameon=False, fontsize=16)
    leg_datasets.get_title().set_fontsize(20)
    leg_datasets.get_title().set_fontweight('bold')

    plt.subplots_adjust(left=0.06, bottom=0.12, wspace=0.18, right=0.90)

    fname = f"efficiency_comparison_all_results{suffix}" if not shared_y else f"efficiency_comparison_all_results_shared_y{suffix}"
    save_plot_outputs(fig, fname)
    if not shared_y and suffix == "":
        fig.savefig(FIGURES_DIR / "efficiency_comparison_95CI_final.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================================
# 5. Single-Panel All Results (Human Only)
# =========================================================================
# =========================================================================
# 5. Single-Panel All Results (Human Only)
# =========================================================================
def plot_human_efficiency_as_is(human_df, color_map=COLOR_MAP, suffix="", add_dotted_line=False):
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 27)
    ax.set_ylim(46, 100)
    ax.tick_params(axis='both', which='both', length=4, width=1, labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

    for model in MODEL_ORDER:
        m_df = human_df[human_df["Model"] == model]
        if m_df.empty:
            continue
        c = color_map[model]
        for ds in DATASETS:
            subset = m_df[m_df["Dataset"] == ds]
            if subset.empty:
                continue
            x = subset["Length"].mean()
            y = subset["Accuracy"].mean()
            el = confidence_ellipse(subset["Length"], subset["Accuracy"], ax, n_std=2.447, facecolor=c, alpha=0.16, zorder=2, is_mean=True)
            if el:
                el.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground='white', alpha=0.25)])
            ax.scatter(x, y, color=c, marker=SHAPE_MAP[ds], s=250, alpha=0.92, edgecolors='white', linewidths=1.5, zorder=3)

    if add_dotted_line:
        for ds in DATASETS:
            h_sub = human_df[(human_df["Model"] == "Human") & (human_df["Dataset"] == ds)]
            m_sub = human_df[(human_df["Model"] == "Molmo") & (human_df["Dataset"] == ds)]
            if not h_sub.empty and not m_sub.empty:
                hx, hy = h_sub["Length"].mean(), h_sub["Accuracy"].mean()
                mx, my = m_sub["Length"].mean(), m_sub["Accuracy"].mean()
                ax.plot([hx, mx], [hy, my], linestyle=":", color="#444444", linewidth=1.8, alpha=0.8, zorder=1)

    ax.set_xlabel('Average Length (Words)', fontsize=22, fontweight='bold', labelpad=12)
    ax.set_ylabel('Communicative Success Rate (%)', fontsize=22, fontweight='bold', labelpad=12)
    ax.set_title('Human Listener', fontsize=28, fontweight='bold', pad=18)

    # Legends
    c_leg = [Line2D([0], [0], marker='o', color='w', label=m, markerfacecolor=color_map[m], markersize=13) for m in MODEL_ORDER if m in human_df["Model"].unique()]
    s_leg = [Line2D([0], [0], marker=SHAPE_MAP[ds], color='w', label=DATASET_DISPLAY[ds], markerfacecolor='gray', markersize=13) for ds in DATASETS]

    leg_models = fig.legend(handles=c_leg, title="Models", loc="center left", bbox_to_anchor=(0.84, 0.62), frameon=False, fontsize=14)
    leg_models.get_title().set_fontsize(18)
    leg_models.get_title().set_fontweight('bold')

    leg_datasets = fig.legend(handles=s_leg, title="Datasets", loc="center left", bbox_to_anchor=(0.84, 0.22), frameon=False, fontsize=14)
    leg_datasets.get_title().set_fontsize(18)
    leg_datasets.get_title().set_fontweight('bold')

    plt.subplots_adjust(left=0.09, bottom=0.12, right=0.83)
    base_name = f"human_efficiency_as_is_dotted{suffix}" if add_dotted_line else f"human_efficiency_as_is{suffix}"
    save_plot_outputs(fig, base_name)
    plt.close(fig)


# =========================================================================
# 6. Single-Panel All Results (QwenVL Only, with REC-Shaping)
# =========================================================================
def plot_qwenvl_efficiency_as_is(qwenvl_df, cov_dict, color_map=COLOR_MAP, suffix=""):
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 27)
    ax.set_ylim(12, 98)
    ax.tick_params(axis='both', which='both', length=4, width=1, labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

    for model in MODEL_ORDER:
        sub = qwenvl_df[qwenvl_df["Model"] == model]
        if sub.empty:
            continue
        c = color_map[model]
        for ds in DATASETS:
            row = sub[sub["Dataset"] == ds]
            if row.empty:
                continue
            x = row["RefLen"].values[0]
            y = row["Accuracy"].values[0]
            el = confidence_ellipse_from_cov(x, y, cov_dict[ds], 300, ax, n_std=2.447, facecolor=c, alpha=0.16, zorder=2)
            if el:
                el.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground='white', alpha=0.25)])
            ax.scatter(x, y, color=c, marker=SHAPE_MAP[ds], s=250, alpha=0.92, edgecolors='white', linewidths=1.5, zorder=3)

    ax.set_xlabel('Average Length (Words)', fontsize=22, fontweight='bold', labelpad=12)
    ax.set_ylabel('Communicative Success Rate (%)', fontsize=22, fontweight='bold', labelpad=12)
    ax.set_title('QwenVL Listener', fontsize=28, fontweight='bold', pad=18)

    # Legends
    c_leg = [Line2D([0], [0], marker='o', color='w', label=m, markerfacecolor=color_map[m], markersize=13) for m in MODEL_ORDER]
    s_leg = [Line2D([0], [0], marker=SHAPE_MAP[ds], color='w', label=DATASET_DISPLAY[ds], markerfacecolor='gray', markersize=13) for ds in DATASETS]

    leg_models = fig.legend(handles=c_leg, title="Models", loc="center left", bbox_to_anchor=(0.84, 0.62), frameon=False, fontsize=14)
    leg_models.get_title().set_fontsize(18)
    leg_models.get_title().set_fontweight('bold')

    leg_datasets = fig.legend(handles=s_leg, title="Datasets", loc="center left", bbox_to_anchor=(0.84, 0.22), frameon=False, fontsize=14)
    leg_datasets.get_title().set_fontsize(18)
    leg_datasets.get_title().set_fontweight('bold')

    plt.subplots_adjust(left=0.09, bottom=0.12, right=0.83)
    save_plot_outputs(fig, f"qwenvl_efficiency_as_is{suffix}")
    plt.close(fig)


# =========================================================================
# 7. 2x2 Grid of Subplots for Human Listeners
# =========================================================================
def plot_human_efficiency_2x2_grid(human_df, color_map=COLOR_MAP, suffix="", shared_y=False):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=shared_y)
    axes = axes.flatten()

    y_limits_per_ds = {
        "refcoco_testA": (65, 100),
        "refcoco_testB": (50, 100),
        "refoi_co_occurrence": (46, 96),
        "refoi_single_presence": (75, 100),
    }

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        if shared_y:
            ax.set_ylim(46, 100)
        else:
            y_min, y_max = y_limits_per_ds[ds]
            ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, 27)
        ax.tick_params(axis='both', which='both', length=3, width=1, labelsize=14)
        ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

        for model in MODEL_ORDER:
            subset = human_df[(human_df["Model"] == model) & (human_df["Dataset"] == ds)]
            if subset.empty:
                continue
            c = color_map[model]
            x = subset["Length"].mean()
            y = subset["Accuracy"].mean()
            el = confidence_ellipse(subset["Length"], subset["Accuracy"], ax, n_std=2.447, facecolor=c, alpha=0.18, zorder=2, is_mean=True)
            if el:
                el.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground='white', alpha=0.25)])
            ax.scatter(x, y, color=c, marker=SHAPE_MAP[ds], s=200, alpha=0.92, edgecolors='white', linewidths=1.5, zorder=3)

        ax.set_title(DATASET_DISPLAY[ds], fontsize=18, fontweight='bold', pad=10)
        if idx in [2, 3]:
            ax.set_xlabel('Average Length (Words)', fontsize=16, fontweight='bold', labelpad=8)
        if idx in [0, 2] or not shared_y:
            ax.set_ylabel('Communicative Success Rate (%)', fontsize=16, fontweight='bold', labelpad=8)

    # Legend at bottom
    c_leg = [Line2D([0], [0], marker='o', color='w', label=m, markerfacecolor=color_map[m], markersize=12) for m in MODEL_ORDER if m in human_df["Model"].unique()]
    fig.legend(handles=c_leg, loc="lower center", ncol=5, frameon=True, fontsize=13, bbox_to_anchor=(0.5, 0.02))

    plt.subplots_adjust(top=0.94, bottom=0.10, left=0.08, right=0.96, hspace=0.22, wspace=0.18)
    fname = f"human_efficiency_2x2_grid{suffix}" if not shared_y else f"human_efficiency_2x2_grid_shared_y{suffix}"
    save_plot_outputs(fig, fname)
    plt.close(fig)


def main():
    print("Loading evaluation data...")
    human_df = load_human_data()
    qwenvl_df = load_qwenvl_published_metrics()
    cov_dict = compute_qwenvl_covariances(human_df)

    print(f"Loaded {len(human_df)} human trials across {human_df['Model'].nunique()} models.")
    print(f"Loaded {len(qwenvl_df)} QwenVL evaluations across {qwenvl_df['Model'].nunique()} models.")

    for c_map, suffix in [(COLOR_MAP, ""), (COLOR_MAP_ORANGE, "_orange")]:
        color_desc = "bright RED" if suffix == "" else "BURNT ORANGE"
        print(f"\n=======================================================")
        print(f"Generating plots with REC-Shaping = {color_desc}...")
        print(f"=======================================================")

        # 1. Human Average Efficiency (Standard + Dotted Line version + Slide Animation)
        plot_human_efficiency_averaged(human_df, color_map=c_map, suffix=suffix, add_dotted_line=False)
        plot_human_efficiency_averaged(human_df, color_map=c_map, suffix=suffix, add_dotted_line=True)
        plot_human_efficiency_animation_steps(human_df, color_map=c_map, suffix=suffix)

        # 2. QwenVL Average Efficiency (all 10 models with REC-Shaping)
        plot_qwenvl_efficiency_averaged(qwenvl_df, cov_dict, color_map=c_map, suffix=suffix)

        # 3. Two-Panel Dataset Averaged Comparison (Standard + Dotted Line version)
        plot_efficiency_comparison_averaged(human_df, qwenvl_df, cov_dict, color_map=c_map, suffix=suffix, add_dotted_line=False)
        plot_efficiency_comparison_averaged(human_df, qwenvl_df, cov_dict, color_map=c_map, suffix=suffix, add_dotted_line=True)

        # 4. Canonical Paper 2-Panel Figure (All results, 4 datasets, shapes, ellipses)
        plot_efficiency_comparison_all_results(human_df, qwenvl_df, cov_dict, color_map=c_map, suffix=suffix, shared_y=False)
        plot_efficiency_comparison_all_results(human_df, qwenvl_df, cov_dict, color_map=c_map, suffix=suffix, shared_y=True)

        # 5. Single-panel All Results (Human, Standard + Dotted Line version)
        plot_human_efficiency_as_is(human_df, color_map=c_map, suffix=suffix, add_dotted_line=False)
        plot_human_efficiency_as_is(human_df, color_map=c_map, suffix=suffix, add_dotted_line=True)

        # 6. Single-panel All Results (QwenVL, with REC-Shaping)
        plot_qwenvl_efficiency_as_is(qwenvl_df, cov_dict, color_map=c_map, suffix=suffix)

        # 7. Human 2x2 Grid
        plot_human_efficiency_2x2_grid(human_df, color_map=c_map, suffix=suffix, shared_y=False)
        plot_human_efficiency_2x2_grid(human_df, color_map=c_map, suffix=suffix, shared_y=True)

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
