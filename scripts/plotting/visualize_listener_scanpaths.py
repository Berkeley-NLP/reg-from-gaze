#!/usr/bin/env python3
"""
Scanpath Visualization CLI for Molmo-REC-Gaze.
Generates static scanpath overlays and animated GIFs.

Usage:
    python scripts/visualize.py \
        --ref_id 34573 \
        --image_dir ../../../../scratch/current/teaywright/images \
        --preds preds/predicted_gaze_output_delay_1129.json \
        --output_overlay results/figures/scanpath_34573.png \
        --output_gif results/figures/scanpath_34573.gif
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image

# Ensure package is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gaze_estimation.data.transforms import letterbox_image
from gaze_estimation.visualization.scanpath_overlay import draw_scanpath_on_image
from gaze_estimation.visualization.scanpath_animation import generate_scanpath_gif


def main():
    parser = argparse.ArgumentParser(description="Visualize gaze scanpaths on images")
    parser.add_argument("--ref_id", type=str, default="34573", help="REF_ID to visualize")
    parser.add_argument("--image_dir", type=str, default="../../../../scratch/current/teaywright/images")
    parser.add_argument("--preds", type=str, default="preds/predicted_gaze_output_delay_1129.json")
    parser.add_argument("--output_overlay", type=str, default="results/figures/scanpath_overlay.png")
    parser.add_argument("--output_gif", type=str, default=None)
    args = parser.parse_args()

    with open(args.preds, "r") as f:
        preds = json.load(f)

    match = next((e for e in preds if str(e.get("ref_id") or e.get("imagefile", "").split(".")[0]) == args.ref_id), None)
    if not match:
        print(f"[ERROR] REF_ID {args.ref_id} not found in {args.preds}")
        sys.exit(1)

    img_filename = match.get("imagefile", f"{args.ref_id}.jpg")
    img_path = Path(args.image_dir) / img_filename
    if img_path.exists():
        img = Image.open(img_path).convert("RGB")
    else:
        print(f"[WARNING] Image {img_path} not found. Using black canvas.")
        img = Image.new("RGB", (512, 320), color=(0, 0, 0))

    padded_img, _, _, _ = letterbox_image(img, 512, 320)

    scanpath = match.get("predicted_gaze", [])
    words = ["BOS"] + match.get("ref_words", []) + ["EOS"]

    # Save static overlay
    out_overlay = draw_scanpath_on_image(padded_img, scanpath, words=words, color="cyan")
    Path(args.output_overlay).parent.mkdir(parents=True, exist_ok=True)
    out_overlay.save(args.output_overlay)
    print(f"✓ Saved scanpath overlay to: {args.output_overlay}")

    # Save animated GIF if requested
    if args.output_gif:
        generate_scanpath_gif(padded_img, scanpath, words=words, output_path=args.output_gif)
        print(f"✓ Saved animated GIF to: {args.output_gif}")


if __name__ == "__main__":
    main()
