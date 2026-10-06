#!/usr/bin/env python3
"""
scripts/plotting/generate_animated_human_efficiency.py

Generates progressive animation frames and animated GIFs for the Human Listener
Communicative Efficiency plot (Average Length in Words vs. Communicative Success Rate in %):
  - Stage 1: Only Molmo and Human
  - Stage 2: Molmo and Human with a dotted line connecting them
  - Stage 3: Molmo + Human + dotted line + REC- models (REC-BFH, REC-Shaping, REC-Success, REC-SeqAnyHit)
  - Stage 4: Full plot (all models + dotted line) -> human_efficiency_average_dotted.png/pdf

All 4 frames are rendered with 100% pixel-perfect frame-to-frame alignment so that when
advancing slides in presentation software (Keynote, PowerPoint, Google Slides, LaTeX Beamer),
there is zero jumping, jitter, or coordinate shifting.
"""

import os
import sys
import json
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

# Ensure gazeRL_publish is on sys.path
PUBLISH_DIR = Path(__file__).resolve().parent.parent.parent
if str(PUBLISH_DIR) not in sys.path:
    sys.path.insert(0, str(PUBLISH_DIR))

from scripts.plotting.plot_utils import confidence_ellipse, load_human_trials_data

# Directories
GAZERL_DIR = Path("/accounts/projects/berkeleynlp/teaywright/gazeRL")
GAZERL_PLOTS_DIR = GAZERL_DIR / "plots"
WORKSPACE_DIR = Path("/accounts/projects/berkeleynlp/teaywright")
FIGURES_DIR = PUBLISH_DIR / "figures"
ARTIFACT_DIR = Path("/accounts/projects/berkeleynlp/teaywright/.gemini/antigravity-cli/brain/834dbf5c-6927-449b-93b5-62bb598fdf21")

HUMAN_TRIALS_PATH = PUBLISH_DIR / "results" / "human_eval" / "human_eval_trials_24000.json"
if not HUMAN_TRIALS_PATH.exists():
    HUMAN_TRIALS_PATH = PUBLISH_DIR / "results" / "human_eval" / "human_eval_trials_21600.json"

for d in [GAZERL_PLOTS_DIR, FIGURES_DIR, ARTIFACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Matplotlib styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Palatino LT STD", "Book Antiqua", "Georgia", "DejaVu Serif", "serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
})

# Canonical color mapping
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

COLOR_MAP_ORANGE = dict(COLOR_MAP)
COLOR_MAP_ORANGE["REC-Shaping"] = "#d35400"

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

STARRED_MODELS = {"Gaze-Shaping", "Gaze-SeqAnyHit"}


def render_plot_frame(human_df, models_to_show, add_dotted_line, color_map=COLOR_MAP):
    """
    Renders a single frame of the Human Listener efficiency plot with fixed axes boundaries.
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Strict fixed boundaries to eliminate empty white space and ensure absolute framing stability
    ax.set_xlim(0, 31)
    ax.set_ylim(63, 95)
    ax.tick_params(axis='both', which='both', length=6, width=1.2, labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

    # Label offsets
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

        if model not in models_to_show:
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
    return fig


def save_multi_target(fig, basenames, bbox):
    """Saves figure across all destination directories."""
    dirs = [
        GAZERL_DIR,
        GAZERL_PLOTS_DIR,
        WORKSPACE_DIR,
        PUBLISH_DIR,
        FIGURES_DIR,
        ARTIFACT_DIR,
    ]
    if isinstance(basenames, str):
        basenames = [basenames]

    for d in dirs:
        if d.exists():
            for name in basenames:
                fig.savefig(d / f"{name}.png", dpi=300, bbox_inches=bbox)
                fig.savefig(d / f"{name}.pdf", bbox_inches=bbox)


def generate_animation_sequence(human_df, color_map=COLOR_MAP, suffix=""):
    """
    Generates all 4 frames with identical bounding box and creates smooth animated GIFs.
    """
    print(f"\n--- Generating Animation Sequence (suffix='{suffix}') ---")

    # Step 4 definitions: all models
    all_models = list(MODEL_ORDER)
    # Step 3 definitions: molmo + human + rec- models
    rec_models = ["Human", "Molmo", "REC-BFH", "REC-Shaping", "REC-Success", "REC-SeqAnyHit"]
    # Step 1 & 2 definitions: only molmo and human
    base_models = ["Human", "Molmo"]

    # Compute master bounding box from the full plot (Step 4)
    print("Computing master tight bounding box from the complete plot...")
    fig_full = render_plot_frame(human_df, all_models, add_dotted_line=True, color_map=color_map)
    fig_full.canvas.draw()
    renderer = fig_full.canvas.get_renderer()
    master_bbox = fig_full.get_tightbbox(renderer).padded(0.1)

    # Save Step 4 (Full plot)
    print("Saving Step 4 (Full plot)...")
    step4_names = [
        f"human_efficiency_step4_full{suffix}",
        f"human_efficiency_slide4{suffix}",
        f"human_efficiency_averaged_dotted{suffix}",
        f"human_efficiency_average_dotted{suffix}",
        f"human_efficiency_dotted{suffix}",
    ]
    save_multi_target(fig_full, step4_names, master_bbox)
    plt.close(fig_full)

    # Save Step 1 (Only Molmo and Human, no dotted line)
    print("Saving Step 1 (Only Molmo and Human)...")
    fig1 = render_plot_frame(human_df, base_models, add_dotted_line=False, color_map=color_map)
    step1_names = [
        f"human_efficiency_step1_molmo_human{suffix}",
        f"human_efficiency_slide1{suffix}",
    ]
    save_multi_target(fig1, step1_names, master_bbox)
    plt.close(fig1)

    # Save Step 2 (Molmo and Human + dotted line)
    print("Saving Step 2 (Molmo and Human with dotted line)...")
    fig2 = render_plot_frame(human_df, base_models, add_dotted_line=True, color_map=color_map)
    step2_names = [
        f"human_efficiency_step2_dotted{suffix}",
        f"human_efficiency_slide2{suffix}",
    ]
    save_multi_target(fig2, step2_names, master_bbox)
    plt.close(fig2)

    # Save Step 3 (Molmo + Human + REC- models)
    print("Saving Step 3 (Molmo + Human + REC- models)...")
    fig3 = render_plot_frame(human_df, rec_models, add_dotted_line=True, color_map=color_map)
    step3_names = [
        f"human_efficiency_step3_rec_models{suffix}",
        f"human_efficiency_slide3{suffix}",
    ]
    save_multi_target(fig3, step3_names, master_bbox)
    plt.close(fig3)

    # Create Animated GIFs
    print("Generating animated GIFs...")
    step_png_paths = [
        FIGURES_DIR / f"human_efficiency_step1_molmo_human{suffix}.png",
        FIGURES_DIR / f"human_efficiency_step2_dotted{suffix}.png",
        FIGURES_DIR / f"human_efficiency_step3_rec_models{suffix}.png",
        FIGURES_DIR / f"human_efficiency_step4_full{suffix}.png",
    ]

    images = [Image.open(p).convert("RGB") for p in step_png_paths]
    durations = [2000, 2000, 2500, 3500]  # Frame display times in ms

    # 1. Standard web/presentation resolution (~1400px wide)
    target_width = 1400
    w, h = images[0].size
    target_height = int(h * (target_width / w))
    resized_images = [im.resize((target_width, target_height), Image.Resampling.LANCZOS) for im in images]

    gif_names = [
        f"human_efficiency_animation{suffix}.gif",
        f"human_efficiency_average_dotted_animation{suffix}.gif",
        f"human_efficiency_averaged_dotted_animation{suffix}.gif",
    ]

    out_dirs = [GAZERL_DIR, GAZERL_PLOTS_DIR, WORKSPACE_DIR, PUBLISH_DIR, FIGURES_DIR, ARTIFACT_DIR]
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

    # 2. High-res GIF (full 300 DPI)
    for d in out_dirs:
        if d.exists():
            images[0].save(
                d / f"human_efficiency_animation_highres{suffix}.gif",
                save_all=True,
                append_images=images[1:],
                duration=durations,
                loop=0
            )

    print(f"Animation generation complete for suffix='{suffix}'.")


def main():
    print("Loading human trials data...")
    human_df = load_human_trials_data(HUMAN_TRIALS_PATH)
    print(f"Loaded {len(human_df)} human trials across {human_df['Model'].nunique()} models.")

    # 1. Standard version (REC-Shaping = Red)
    generate_animation_sequence(human_df, color_map=COLOR_MAP, suffix="")

    # 2. Burnt Orange version (REC-Shaping = Burnt Orange #d35400)
    generate_animation_sequence(human_df, color_map=COLOR_MAP_ORANGE, suffix="_orange")

    print("\n=======================================================")
    print("All slide animation frames and animated GIFs generated successfully!")
    print("=======================================================")


if __name__ == "__main__":
    main()
