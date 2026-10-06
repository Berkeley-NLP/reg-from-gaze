#!/usr/bin/env python3
"""
scripts/plotting/plot_strategy_comparison.py

Generates publication-quality Referring Strategy Comparison stacked bar chart:
  - "Both" placed in the middle between Spatial Only and Attribute Only.
  - Additive color scheme: Blue (Spatial Only) + Red (Attribute Only) -> Purple (Both), plus Neutral Grey (None).
  - Enlarged text for readability in paper figures.
  - Exports to gazeRL_publish/, figures/, workspace root, gazeRL/plots/, and artifact dir.
"""

import os
import glob
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Directory Configuration
EVAL_DIRS = [
    "/data/teaywright/gazerl_main_experiments/eval/",
    "/scratch/users/teaywright/gazerl_eval/",
    "/scratch/users/teaywright/gazerl_ablations_eval/",
    "/scratch/users/teaywright/gazerl_eval_old/",
    "/accounts/projects/berkeleynlp/teaywright/gazeRL_publish/results/eval_runs",
]

DATASETS = [
    "refcoco_testA",
    "refcoco_testB",
    "refoi_co_occurrence",
    "refoi_single_presence",
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "Palatino Linotype", "Palatino LT STD", "Book Antiqua", "Georgia", "DejaVu Serif", "serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.0,
})

MODEL_MAP = {
    "gold_refs": "Human",
    "gold": "Human",
    "sparse_constant_kl_02": "Gaze-BFH",
    "shaping": "Gaze-Shaping",
    "iterative_sparse": "REC-BFH",
    "iterative_binary": "REC-SeqAnyHit",
    "supervised": "REC-Success",
    "binary_last_point": "Gaze-SeqLPHit",
    "binary": "Gaze-SeqAnyHit",
    "molmo_vanilla": "Molmo",
    "iterative_shaping": "REC-Shaping",
}

MODEL_ORDER = [
    "Human",
    "Gaze-Shaping",
    "Gaze-SeqAnyHit",
    "REC-Success",
    "Gaze-BFH",
    "REC-Shaping",
    "REC-BFH",
    "Gaze-SeqLPHit",
    "REC-SeqAnyHit",
    "Molmo",
]

# Order with "Both" in the middle:
# Spatial Only (Soft Blue) + Attribute Only (Soft Coral Red) -> Both (Soft Purple, in the middle), followed by None (Soft Slate Grey)
CATEGORIES = ["Spatial Only", "Both", "Attribute Only", "None"]

# Additive Rich Pastel / Soft Muted Colors (solid, high contrast, not washed out):
# Spatial Only: Rich Soft Blue (#5684D4)
# Both: Soft Amethyst / Purple (#9B6CC7) - additive blend of Blue & Coral Red
# Attribute Only: Soft Coral Red (#E66464)
# None: Solid Soft Slate Grey (#BFC8D2)
CAT_COLORS = ["#5684D4", "#9B6CC7", "#E66464", "#BFC8D2"]

# Keywords for categorization
SPATIAL_KEYWORDS = [
    "left", "right", "top", "bottom", "middle", "center", "above", "below",
    "next", "near", "far", "between", "behind", "front", "centered", "side",
    "corner", "upper", "lower", "background", "foreground", "back", "overhead",
]

ATTRIBUTE_KEYWORDS = [
    # Colors
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "grey", "silver", "gold", "tan", "beige",
    "maroon", "navy", "teal", "cyan", "magenta", "violet",
    # Descriptive
    "large", "small", "big", "little", "tall", "short", "long", "metal",
    "wooden", "plastic", "glass", "striped", "checkered", "patterned",
    "wearing", "holding", "carrying", "sitting", "standing", "walking",
    "running", "smiling", "elderly", "young", "old", "new", "broken", "dirty",
]


def categorize_reference(text):
    if not isinstance(text, str):
        return "None"
    text = text.lower().replace("<|endoftext|>", "").strip()
    text = re.sub(r"```json.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    words = re.findall(r"\w+", text)
    has_spatial = any(kw in words for kw in SPATIAL_KEYWORDS)
    has_attribute = any(kw in words for kw in ATTRIBUTE_KEYWORDS)

    if has_spatial and has_attribute:
        return "Both"
    elif has_spatial:
        return "Spatial Only"
    elif has_attribute:
        return "Attribute Only"
    else:
        return "None"


def load_all_references():
    data = []
    processed_files = set()

    sorted_model_map = sorted(
        MODEL_MAP.items(), key=lambda x: len(x[0]), reverse=True
    )

    for eval_dir in EVAL_DIRS:
        if not os.path.exists(eval_dir):
            continue

        folders = [
            d
            for d in os.listdir(eval_dir)
            if os.path.isdir(os.path.join(eval_dir, d))
            and "refcoco_sft" not in d
        ]

        for folder in folders:
            model_name = "Other"
            for key, val in sorted_model_map:
                if key.lower() in folder.lower():
                    model_name = val
                    break

            if model_name == "Other":
                continue

            folder_path = os.path.join(eval_dir, folder)

            for dataset in DATASETS:
                pattern = f"{folder_path}/**/*{dataset}*.json"
                files = glob.glob(pattern, recursive=True)

                for file_path in files:
                    # Avoid double-counting duplicate symlinks or multiple matching paths
                    file_key = (model_name, dataset, os.path.basename(file_path))
                    if file_key in processed_files:
                        continue
                    processed_files.add(file_key)

                    try:
                        with open(file_path, "r") as f:
                            ref_json = json.load(f)

                            if isinstance(ref_json, dict):
                                ref_json = [ref_json]

                            for item in ref_json:
                                ref_text = item.get(
                                    "generated_reference",
                                    item.get(
                                        "reference",
                                        item.get("ref", item.get("text", "")),
                                    ),
                                )

                                if isinstance(ref_text, list) and ref_text:
                                    ref_text = ref_text[0]

                                category = categorize_reference(ref_text)
                                data.append(
                                    {"Model": model_name, "Category": category}
                                )
                    except Exception:
                        continue

    df = pd.DataFrame(data)
    if not df.empty:
        print("\nSuccessfully loaded reference counts per model:")
        print(df["Model"].value_counts())
    return df


CACHE_CSV = Path("/accounts/projects/berkeleynlp/teaywright/gazeRL_publish/results/model_eval/referential_strategies_percentages.csv")


def main():
    print("Loading and categorizing references for strategy comparison...")
    if CACHE_CSV.exists():
        print(f"Loading cached percentages from {CACHE_CSV}...")
        percentages = pd.read_csv(CACHE_CSV, index_col=0)
        for cat in CATEGORIES:
            if cat not in percentages.columns:
                percentages[cat] = 0.0
    else:
        df = load_all_references()
        if df.empty:
            print("No data found to plot.")
            return
        df = df[df["Model"] != "REC-Success (LP=0.5)"]
        counts = df.groupby(["Model", "Category"]).size().unstack(fill_value=0)
        percentages = counts.div(counts.sum(axis=1), axis=0) * 100
        for cat in CATEGORIES:
            if cat not in percentages.columns:
                percentages[cat] = 0.0
        CACHE_CSV.parent.mkdir(parents=True, exist_ok=True)
        percentages[CATEGORIES].to_csv(CACHE_CSV)

    # Filter out REC-Success (LP=0.5) if present
    percentages = percentages[percentages.index != "REC-Success (LP=0.5)"]

    # Calculate Total Variation Distance (TVD) to Human distribution
    human_dist = percentages.loc["Human"][CATEGORIES].values / 100.0
    distances = {}
    for m in percentages.index:
        if m == "Human":
            continue
        dist = percentages.loc[m][CATEGORIES].values / 100.0
        distances[m] = 0.5 * np.sum(np.abs(dist - human_dist))

    # Order from most similar to least similar (Human benchmark first)
    order_desc = ["Human"] + sorted(distances.keys(), key=lambda m: distances[m])
    # Order from least similar building up to most similar (Human benchmark last)
    order_asc = sorted(distances.keys(), key=lambda m: distances[m], reverse=True) + ["Human"]

    print("\n--- DISTANCE / SIMILARITY TO HUMAN (TVD Ascending, Closest to Human First) ---")
    for m in sorted(distances.keys(), key=lambda m: distances[m]):
        print(f"  {m:18s}: TVD = {distances[m]:.4f}")

    print("\n--- STRATEGY PERCENTAGES PER MODEL (%) ---")
    print(percentages.loc[order_desc, CATEGORIES].round(2))

    out_dirs = [
        Path("/accounts/projects/berkeleynlp/teaywright/gazeRL_publish"),
        Path("/accounts/projects/berkeleynlp/teaywright/gazeRL_publish/figures"),
        Path("/accounts/projects/berkeleynlp/teaywright"),
        Path("/accounts/projects/berkeleynlp/teaywright/gazeRL/plots"),
        Path("/accounts/projects/berkeleynlp/teaywright/.gemini/antigravity-cli/brain/4650fa71-f324-48ea-9c07-38c6865071e4"),
    ]
    for p in out_dirs:
        p.mkdir(exist_ok=True, parents=True)

    def render_order(order, filename_base, title):
        fig, ax = plt.subplots(figsize=(17, 8.5), dpi=300)
        sub_pcts = percentages.loc[order]
        bottom = np.zeros(len(order))
        for i, cat in enumerate(CATEGORIES):
            ax.bar(
                order,
                sub_pcts[cat],
                bottom=bottom,
                label=cat,
                color=CAT_COLORS[i],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.92,
            )
            bottom += sub_pcts[cat]

        ax.set_ylabel(
            "Percentage of References (%)", fontsize=22, fontweight="bold", labelpad=14
        )
        ax.set_title(title, fontsize=26, fontweight="bold", pad=22)
        ax.tick_params(axis="y", labelsize=17)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            order, fontsize=17, fontweight="bold", rotation=35, ha="right"
        )

        leg = ax.legend(
            title="Strategy",
            fontsize=17,
            title_fontsize=19,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            frameon=True,
            edgecolor="#cccccc",
        )
        leg.get_title().set_fontweight("bold")

        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()

        for p in out_dirs:
            fig.savefig(p / f"{filename_base}.pdf", dpi=300, bbox_inches="tight")
            fig.savefig(p / f"{filename_base}.png", dpi=300, bbox_inches="tight")
        print(f"[SUCCESS] Saved {filename_base}.pdf and .png across all target directories.")
        plt.close(fig)

    # 1. Primary figure: Ordered from Human -> Most to least similar
    render_order(order_desc, "model_strategy_comparison", "Referring Strategy Comparison")
    # 2. Variant figure: Ordered ascending similarity (Least to most similar -> Human on right)
    render_order(order_asc, "model_strategy_comparison_ascending", "Referring Strategy Comparison (Increasing Similarity to Human)")



if __name__ == "__main__":
    main()
