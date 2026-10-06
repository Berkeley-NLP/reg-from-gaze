#!/usr/bin/env python3
"""
Regenerates the compact qualitative examples figure used in the paper:
'Learning to Refer from Estimated Listener Gaze' (Wright & Suhr, COLM 2026).

Compares referring expressions across:
  - Molmo (base model)
  - REC-SeqAnyHit (magenta)
  - Gaze-SeqAnyHit (navy)
  - Gaze-Shaping (green)
  - Human (gold)
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def generate_figure(output_dir: str = "."):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Asset paths
    assets_dir = Path(__file__).resolve().parent / "figure_assets"
    img1_path = assets_dir / "example1_refcoco_testA.png"
    img2_path = assets_dir / "example2_refcoco_testB.png"

    if not img1_path.exists() or not img2_path.exists():
        # Fallback to GAZERL_COLM if available
        base_colm = Path("/accounts/projects/berkeleynlp/teaywright/GAZERL_COLM")
        img1_path = base_colm / "html/data/human_eval_data/images/refcoco_testA_1309.jpg"
        img2_path = base_colm / "html/data/human_eval_data/images/refcoco_testB_203.jpg"

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # PDF MediaBox dimensions: 1118.88 x 511.282875 pt
    fig_w_pt = 1118.88
    fig_h_pt = 511.282875
    fig_w_in = fig_w_pt / 72.0
    fig_h_in = fig_h_pt / 72.0

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Palatino", "Palatino Linotype", "DejaVu Serif", "Times New Roman"]

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig_w_pt)
    ax.set_ylim(0, fig_h_pt)
    ax.axis("off")

    # Background white fill
    ax.add_patch(patches.Rectangle((0, 0), fig_w_pt, fig_h_pt, color="white", zorder=0))

    # --- TOP: IMAGES & TITLES ---
    # Image 1 (RefCOCO Test A)
    # Coordinate in PDF: x: 240.397 to 530.189, y: 295.123 to 488.016
    im1_x, im1_y = 240.397339, 295.122977
    im1_w, im1_h = 289.792335, 192.893023
    ax.imshow(img1, extent=[im1_x, im1_x + im1_w, im1_y, im1_y + im1_h], zorder=1)
    ax.add_patch(patches.Rectangle((im1_x, im1_y), im1_w, im1_h, fill=False, edgecolor="black", linewidth=1.8, zorder=2))
    ax.text(im1_x + im1_w / 2.0, 496.016, "Example 1: RefCOCO Test A", fontsize=20, ha="center", va="baseline", color="black", zorder=3)

    # Image 2 (RefCOCO Test B)
    # Coordinate in PDF: x: 734.826 to 1022.591, y: 295.123 to 488.016
    im2_x, im2_y = 734.825664, 295.122977
    im2_w, im2_h = 287.765816, 192.893023
    ax.imshow(img2, extent=[im2_x, im2_x + im2_w, im2_y, im2_y + im2_h], zorder=1)
    ax.add_patch(patches.Rectangle((im2_x, im2_y), im2_w, im2_h, fill=False, edgecolor="black", linewidth=1.8, zorder=2))
    ax.text(im2_x + im2_w / 2.0, 496.016, "Example 2: RefCOCO Test B", fontsize=20, ha="center", va="baseline", color="black", zorder=3)

    # --- TABLE SHADING ---
    shading_color = (0.95686, 0.95686, 0.95686)  # #F4F4F4
    table_left = 11.8944
    table_right = 1106.9856
    table_w = table_right - table_left

    # Row 2 (REC-SeqAnyHit) shading
    ax.add_patch(patches.Rectangle((table_left, 104.882233), table_w, 180.399851 - 104.882233, facecolor=shading_color, edgecolor="none", zorder=1))

    # Row 4 (Gaze-Shaping) shading
    ax.add_patch(patches.Rectangle((table_left, 31.968670), table_w, 60.613284 - 31.968670, facecolor=shading_color, edgecolor="none", zorder=1))

    # Header divider rule
    ax.plot([table_left, table_right], [242.897191, 242.897191], color="black", linewidth=2.0, zorder=2)

    # --- TABLE HEADERS ---
    col1_center = 100.0
    col2_center = 420.0
    col3_center = 890.0

    ax.text(col1_center, 250.709, "Speaker", fontsize=18.5, ha="center", va="bottom", color="black", zorder=3)
    ax.text(col2_center, 250.709, "Reference: Example 1", fontsize=18.5, ha="center", va="bottom", color="black", zorder=3)
    ax.text(col3_center, 250.709, "Reference: Example 2", fontsize=18.5, ha="center", va="bottom", color="black", zorder=3)

    # --- TABLE ROWS ---
    rows = [
        {
            "speaker": "Molmo",
            "speaker_color": "black",
            "ref1": "Man in pink shirt with short brown hair, looking\nleft.",
            "ref2": "A white bowl with a red border containing a\nbrownish-red sauce, located in the bottom left\ncorner of the image.",
            "y_center": 206.503,
            "y_ref1": 206.33,
            "y_ref2": 206.34,
        },
        {
            "speaker": "REC-SeqAnyHit",
            "speaker_color": "#FF00FF",  # Magenta
            "ref1": "A man wearing a pink shirt and gray pants, standing\non the far right, with his left hand raised to his\nchin.",
            "ref2": "In the bottom left corner, there's a small white\nbowl containing a red sauce with a white spoon\ninside. This bowl is highlighted by a red box,\nmaking it stand out from the rest of the image.",
            "y_center": 138.829,
            "y_ref1": 138.63,
            "y_ref2": 139.03,
        },
        {
            "speaker": "Gaze-SeqAnyHit",
            "speaker_color": "#000080",  # Navy
            "ref1": "Man wearing pink shirt and gray pants.",
            "ref2": "Bottom-left corner, white bowl with red border\ncontaining brownish-orange sauce.",
            "y_center": 78.935,
            "y_ref1": 78.75,
            "y_ref2": 79.14,
        },
        {
            "speaker": "Gaze-Shaping",
            "speaker_color": "#008000",  # Green
            "ref1": "Man in pink shirt.",
            "ref2": "Bottom-left",
            "y_center": 42.033,
            "y_ref1": 42.28,
            "y_ref2": 42.67,
        },
        {
            "speaker": "Human",
            "speaker_color": (0.8549, 0.6471, 0.1255),  # Goldenrod
            "ref1": "pink shirt",
            "ref2": "bowl on bottom left",
            "y_center": 13.803,
            "y_ref1": 13.63,
            "y_ref2": 14.03,
        },
    ]

    for row in rows:
        # Speaker column
        ax.text(col1_center, row["y_center"], row["speaker"], fontsize=16.5, ha="center", va="center", color=row["speaker_color"], zorder=3)
        # Reference 1 column
        ax.text(col2_center, row["y_ref1"], row["ref1"], fontsize=15.5, ha="center", va="center", color="black", multialignment="center", linespacing=1.2, zorder=3)
        # Reference 2 column
        ax.text(col3_center, row["y_ref2"], row["ref2"], fontsize=15.5, ha="center", va="center", color="black", multialignment="center", linespacing=1.2, zorder=3)

    # Save outputs
    pdf_out = output_path / "examples_comparison_fig_pretty.pdf"
    png_out = output_path / "examples_comparison_fig_pretty.png"

    plt.savefig(pdf_out, format="pdf", dpi=300, bbox_inches=None)
    plt.savefig(png_out, format="png", dpi=300, bbox_inches=None)
    plt.close(fig)

    print(f"Generated: {pdf_out}")
    print(f"Generated: {png_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate qualitative examples figure")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output figures")
    args = parser.parse_args()
    generate_figure(args.output_dir)
