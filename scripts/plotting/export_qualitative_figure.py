
import re
import os
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, List, Dict
from collections import defaultdict

# Higher resolution for "polished" look
SCALE_FACTOR = 4
TARGET_W, TARGET_H = 512 * SCALE_FACTOR, 320 * SCALE_FACTOR

def letterbox_to_canvas(img: Image.Image, target_w=TARGET_W, target_h=TARGET_H) -> Tuple[Image.Image, Tuple[int, int], float]:
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (25, 25, 25))
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas, (off_x, off_y), scale

def export_gaze_example_manual_subscript():
    target_id = "000000298151"
    bbox_norm = (74.7160400104746, 3.1170961170062528, 96.80181275486109, 52.98594914778055)
    
    # Gaze points from row 62840
    raw_gaze_pts = [(57.71, 28.8), (0.0, 0.0), (71.8, 20.8), (81.8, 18.8), (86.8, 18.8), (0.0, 0.0), (86.8, 18.8)]
    
    blue_shades = [
        (190, 230, 255), (150, 210, 255), (100, 185, 255), 
        (60, 155, 255), (20, 110, 255), (0, 70, 220), (0, 40, 150)
    ]

    print(f"Loading COCO dataset to find {target_id}...")
    ds = load_dataset("NaiveDev/coco-2014-instance", split="validation")
    
    found_item = None
    for item in ds:
        url = item.get("coco_url", "")
        if url and target_id in url:
            found_item = item
            break
            
    if not found_item:
        print(f"Could not find {target_id}.")
        return

    orig_img = found_item["image"].convert("RGB")
    canvas, (off_x, off_y), scale = letterbox_to_canvas(orig_img, TARGET_W, TARGET_H)
    draw = ImageDraw.Draw(canvas, "RGBA")

    # 1. Draw BBox
    xmin, ymin, xmax, ymax = [v * TARGET_W / 100.0 if i%2==0 else v * TARGET_H / 100.0 for i, v in enumerate(bbox_norm)]
    draw.rectangle([xmin, ymin, xmax, ymax], outline=(255, 0, 0, 255), width=4)

    # 2. Setup Fonts
    try:
        font_path = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        font = ImageFont.truetype(font_path, 14 * SCALE_FACTOR)
        sub_font = ImageFont.truetype(font_path, 10 * SCALE_FACTOR)
    except:
        font = ImageFont.load_default()
        sub_font = ImageFont.load_default()

    # 3. Group points by coordinate (skip 0,0)
    coord_groups = defaultdict(list)
    for i, pt in enumerate(raw_gaze_pts):
        if pt[0] != 0 or pt[1] != 0:
            coord_groups[tuple(pt)].append(i)

    # 4. Draw Path
    path_pts = []
    for pt in raw_gaze_pts:
        if pt[0] != 0 or pt[1] != 0:
            path_pts.append((pt[0] * TARGET_W / 100.0, pt[1] * TARGET_H / 100.0))
    if len(path_pts) >= 2:
        draw.line(path_pts, fill=(255, 255, 255, 100), width=2 * SCALE_FACTOR)

    # 5. Draw Grouped Points and Labels with Manual Subscripts
    for coord, indices in coord_groups.items():
        px = coord[0] * TARGET_W / 100.0
        py = coord[1] * TARGET_H / 100.0
        
        # Prepare label parts: [('g', '1'), (', ', None), ('g', '2')]
        label_parts = []
        for i, idx in enumerate(indices):
            label_parts.append(('g', str(idx+1)))
            if i < len(indices) - 1:
                label_parts.append((', ', None))
        
        # Calculate total width and height for text box
        tw = 0
        for base, sub in label_parts:
            tw += draw.textlength(base, font=font)
            if sub:
                tw += draw.textlength(sub, font=sub_font)
        
        # Base height
        line_bbox = draw.textbbox((0, 0), "gp", font=font)
        th = line_bbox[3] - line_bbox[1]
        
        color = blue_shades[max(indices)]
        
        # Dot with glow
        for r_off in range(1, 4):
            draw.ellipse([px-8*SCALE_FACTOR-r_off, py-8*SCALE_FACTOR-r_off, px+8*SCALE_FACTOR+r_off, py+8*SCALE_FACTOR+r_off], outline=(*color, 50), width=1)
        
        r = 7 * SCALE_FACTOR
        draw.ellipse([px-r, py-r, px+r, py+r], fill=(*color, 255), outline=(255, 255, 255, 255), width=2)
        
        # Text Box Position
        text_padding = 4 * SCALE_FACTOR
        tx, ty = px + r + 5, py - r - th - (6 * SCALE_FACTOR) # offset up a bit for subscripts
        
        if tx + tw > TARGET_W: tx = px - r - tw - 10
        if ty < 0: ty = py + r + 10

        # Background Box
        box = [tx - text_padding, ty - text_padding, tx + tw + text_padding, ty + th + (6 * SCALE_FACTOR)]
        draw.rectangle(box, fill=(0, 0, 0, 200), outline=(255, 255, 255, 255), width=1)
        
        # Render parts manually
        curr_x = tx
        for base, sub in label_parts:
            draw.text((curr_x, ty), base, font=font, fill="white")
            curr_x += draw.textlength(base, font=font)
            if sub:
                # Draw subscript slightly lower and smaller
                draw.text((curr_x, ty + (5 * SCALE_FACTOR)), sub, font=sub_font, fill="white")
                curr_x += draw.textlength(sub, font=sub_font)

    output_path = f"Q{target_id}_gaze_final_polished.jpg"
    canvas.save(output_path, quality=95, subsampling=0)
    print(f"Exported final polished gaze view to {output_path}")

if __name__ == "__main__":
    export_gaze_example_manual_subscript()
