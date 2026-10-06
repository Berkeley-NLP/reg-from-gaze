import ast
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import colors as pc

import requests
import streamlit as st
from PIL import Image

# Optional HF datasets for lookup
HF_IMPORT_ERROR = None
try:
    import datasets
    from datasets import load_dataset
    HF_AVAILABLE = True
except Exception as e:
    HF_IMPORT_ERROR = str(e)
    HF_AVAILABLE = False

# -------------------------
# Utilities
# -------------------------

TARGET_W, TARGET_H = 512, 320  # final canvas size for plotting

COL_IN_BBOX = "in bbox "
COL_SHAPING = "shaping _rewards"
COL_SPARSE = "sparse_rewards"
COL_TOTAL = "total_rewards"
COL_GAZE = "gaze"
COL_TOKENS = "tokens"
COL_LPROBS = "log_probs"
COL_HIT_IDX = "hit_idx"

LISTY_COLS = [
    "gold_bbox_normalized_minmax",
    COL_TOKENS,
    COL_LPROBS,
    COL_GAZE,
    COL_IN_BBOX,
    COL_SPARSE,
    COL_SHAPING,
    COL_TOTAL,
    "answer",
    "reference",
]

SPATIAL_KEYWORDS = [
    'left', 'right', 'top', 'bottom', 'middle', 'center', 'above', 'below', 
    'next', 'near', 'far', 'between', 'behind', 'front', 'centered', 'side', 
    'corner', 'upper', 'lower', 'background', 'foreground', 'back', 'overhead'
]

ATTRIBUTE_KEYWORDS = [
    'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 
    'black', 'white', 'gray', 'grey', 'silver', 'gold', 'tan', 'beige', 
    'maroon', 'navy', 'teal', 'cyan', 'magenta', 'violet',
    'large', 'small', 'big', 'little', 'tall', 'short', 'long', 'metal', 
    'wooden', 'plastic', 'glass', 'striped', 'checkered', 'patterned', 
    'wearing', 'holding', 'carrying', 'sitting', 'standing', 'walking', 
    'running', 'smiling', 'elderly', 'young', 'old', 'new', 'broken', 'dirty'
]

EOS_TOKEN = "<|endoftext|>"

def has_terms(text, keywords):
    if not text: return False
    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text)
    if not isinstance(text, str): return False
    
    text = text.lower()
    text = text.replace(EOS_TOKEN, "").strip()
    text = re.sub(r'```json.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL).strip()
    words = re.findall(r'\w+', text)
    return any(kw in words for kw in keywords)

def _parse_listish(x):
    if pd.isna(x): return []
    if isinstance(x, (list, tuple)): return list(x)
    s = str(x).strip()
    if not s: return []
    try:
        v = json.loads(s)
        if isinstance(v, (list, tuple)): return list(v)
    except Exception: pass
    s2 = re.sub(r'\btrue\b', 'True', s, flags=re.IGNORECASE)
    s2 = re.sub(r'\bfalse\b', 'False', s2, flags=re.IGNORECASE)
    try:
        v = ast.literal_eval(s2)
        if isinstance(v, (list, tuple)): return list(v)
        return [v]
    except Exception: return [s]

@st.cache_data(show_spinner=False)
def load_csv(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, engine="python", quotechar='"')
    except Exception as e:
        st.error(f"Pandas failed to read CSV: {e}")
        return None
    for col in LISTY_COLS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_listish)
    return df

def letterbox_to_canvas(img: Image.Image, target_w=TARGET_W, target_h=TARGET_H) -> Tuple[Image.Image, Tuple[int, int], float]:
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new("RGB", (target_w, target_h), (20, 20, 20))
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas.paste(resized, (off_x, off_y))
    return canvas, (off_x, off_y), scale

def denorm_point_0_100_to_canvas(pt: Tuple[float, float], off_x: int = 0, off_y: int = 0) -> Tuple[float, float]:
    new_w = TARGET_W - 2 * off_x
    new_h = TARGET_H - 2 * off_y
    cx = (pt[0] / 100.0) * new_w + off_x
    cy = (pt[1] / 100.0) * new_h + off_y
    return cx, cy

def denorm_bbox_minmax_0_100_to_canvas(b: List[float], off_x: int = 0, off_y: int = 0) -> Tuple[float, float, float, float]:
    if not b or len(b) < 4: return 0, 0, 0, 0
    try:
        new_w = TARGET_W - 2 * off_x
        new_h = TARGET_H - 2 * off_y
        x0 = (float(b[0]) / 100.0) * new_w + off_x
        y0 = (float(b[1]) / 100.0) * new_h + off_y
        x1 = (float(b[2]) / 100.0) * new_w + off_x
        y1 = (float(b[3]) / 100.0) * new_h + off_y
        return x0, y0, x1, y1
    except: return 0, 0, 0, 0

def safe_list(L, n, default=None):
    L = list(L) if isinstance(L, (list, tuple, np.ndarray, pd.Series)) else []
    if len(L) < n:
        L = list(L) + [default] * (n - len(L))
    return L

def calculate_average_reward(total_rewards):
    if not total_rewards or not isinstance(total_rewards, (list, tuple)): return 0.0
    valid_rewards = []
    for v in total_rewards:
        if v is not None:
            try: valid_rewards.append(float(v))
            except: continue
    return float(np.mean(valid_rewards)) if valid_rewards else 0.0

# -------------------------
# Dataset lookup (HF)
# -------------------------

@st.cache_resource(show_spinner=True)
def _coco_dataset():
    if not HF_AVAILABLE: return None
    try:
        return load_dataset("NaiveDev/coco-2014-instance", split="validation")
    except Exception as e:
        print(f"DEBUG: Error loading COCO split 'validation': {e}")
        return None

@st.cache_resource(show_spinner=True)
def _coco_id_index() -> Optional[Dict[str, int]]:
    ds = _coco_dataset()
    if ds is None: return None
    index = {}
    import re
    urls = ds["coco_url"]
    for i, url in enumerate(urls):
        if url:
            match = re.search(r'(\d+)\.jpg$', url)
            if match:
                coco_id = match.group(1).zfill(12)
                index[coco_id] = i
    return index

@st.cache_data(show_spinner=False)
def coco_lookup(coco_id: str) -> Optional[Dict[str, Any]]:
    if not HF_AVAILABLE: return None
    idx_map = _coco_id_index()
    ds = _coco_dataset()
    if not idx_map or not ds: return None
    padded_id = str(coco_id).zfill(12)
    row_idx = idx_map.get(padded_id)
    if row_idx is None: return None
    ex = ds[row_idx]
    img = ex.get("image")
    if img and isinstance(img, Image.Image):
        return {"image": img.convert("RGB"), "orig_size": img.size}
    return None

# -------------------------
# Rendering (Gaze Episodes)
# -------------------------

def build_fig(img: Image.Image,
              bbox_canvas_xyxy: Optional[Tuple[float, float, float, float]],
              pts_px: List[Tuple[float, float]],
              pts_meta: List[Dict[str, Any]],
              show_numbers: bool = True,
              use_gradient_color: bool = True,
              point_size: int = 10) -> go.Figure:

    canvas, (off_x, off_y), scale = letterbox_to_canvas(img, TARGET_W, TARGET_H)
    fig = go.Figure()
    fig.add_layout_image(dict(source=canvas, xref="x", yref="y", x=0, y=TARGET_H, sizex=TARGET_W, sizey=TARGET_H, sizing="stretch", layer="below"))

    if bbox_canvas_xyxy:
        bx0, by0, bx1, by1 = bbox_canvas_xyxy
        fig.add_shape(type="rect", x0=bx0, y0=TARGET_H-by0, x1=bx1, y1=TARGET_H-by1, line=dict(width=3, color="red"), fillcolor="rgba(0,0,0,0)")

    if pts_px:
        xs = [p[0] for p in pts_px]
        ys = [TARGET_H - p[1] for p in pts_px]
        hover_text = []
        for i, meta in enumerate(pts_meta):
            hover = (f"<b>step</b>: {meta['Step']}<br><b>token</b>: {meta['Token']}<br><b>log p</b>: {meta['Log Prob']:.3f}<br>"
                     f"<b>in_bbox</b>: {meta['In Bbox']}<br><b>total</b>: {meta['Total']:.3f}")
            hover_text.append(hover)
        if len(xs) >= 2:
            denom = max(1, len(xs) - 1)
            for i in range(len(xs) - 1):
                frac = i / denom
                seg_color = pc.sample_colorscale("Viridis", [frac])[0] if use_gradient_color else "white"
                fig.add_trace(go.Scatter(x=[xs[i], xs[i+1]], y=[ys[i], ys[i+1]], mode="lines", line=dict(width=3, color=seg_color), hoverinfo="skip", showlegend=False))
        marker_kwargs = dict(size=point_size)
        if use_gradient_color:
            marker_kwargs.update(dict(color=list(range(len(xs))), colorscale="Viridis", showscale=True))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text" if show_numbers else "markers", text=[str(m['Step']) for m in pts_meta] if show_numbers else None,
                                 textposition="top center", hovertext=hover_text, hoverinfo="text", marker=marker_kwargs, name="gaze"))

    fig.update_xaxes(range=[0, TARGET_W], visible=False)
    fig.update_yaxes(range=[0, TARGET_H], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(width=TARGET_W, height=TARGET_H, margin=dict(l=0, r=0, t=0, b=0), dragmode=False)
    return fig

def prepare_row_fields(row: pd.Series, off_x: int = 0, off_y: int = 0) -> Dict[str, Any]:
    tokens = row.get(COL_TOKENS) or []
    lprobs = row.get(COL_LPROBS) or []
    gaze_pts = row.get(COL_GAZE) or []
    in_bbox = row.get(COL_IN_BBOX) or []
    sparse = row.get(COL_SPARSE) or []
    shaping = row.get(COL_SHAPING) or []
    total = row.get(COL_TOTAL) or []
    max_len = len(tokens)
    all_tokens_data = []
    pts_px = []
    pts_meta = []
    for i in range(max_len):
        t = tokens[i] if i < len(tokens) else ""
        lp = lprobs[i] if (lprobs and i < len(lprobs)) else 0.0
        ib = in_bbox[i] if (in_bbox and i < len(in_bbox)) else False
        s = sparse[i] if (sparse and i < len(sparse)) else 0.0
        sh = shaping[i] if (shaping and i < len(shaping)) else 0.0
        tot = total[i] if (total and i < len(total)) else 0.0
        g = gaze_pts[i] if (gaze_pts and i < len(gaze_pts)) else None
        g_str, has_gaze = "None", False
        if isinstance(g, (list, tuple)) and len(g) == 2:
            try:
                gx, gy = float(g[0]), float(g[1])
                if not (np.isnan(gx) or np.isnan(gy)) and not (gx == 0.0 and gy == 0.0):
                    px_pt = denorm_point_0_100_to_canvas((gx, gy), off_x, off_y)
                    pts_px.append(px_pt)
                    has_gaze = True
                    g_str = f"({gx:.1f}, {gy:.1f})"
            except: pass
        meta = {"Step": i, "Token": t, "Log Prob": lp, "In Bbox": ib, "Sparse": s, "Shaping": sh, "Total": tot, "Gaze": g_str}
        all_tokens_data.append(meta)
        if has_gaze: pts_meta.append(meta)
    return {"all_tokens_data": all_tokens_data, "pts_px": pts_px, "pts_meta": pts_meta, "bbox_norm": row.get("gold_bbox_normalized_minmax")}

def render_example(row: pd.Series, show_numbers: bool, use_gradient_color: bool, point_size: int):
    id_val = row.get("question_id")
    qid = str(id_val).split('.')[0].zfill(12) if not pd.isna(id_val) else "000000000000"
    hf = coco_lookup(qid)
    img = hf["image"] if hf else Image.new("RGB", (TARGET_W, TARGET_H), (128, 128, 128))
    
    # Calculate letterboxing parameters first to get padding offsets
    canvas, (off_x, off_y), scale = letterbox_to_canvas(img, TARGET_W, TARGET_H)
    
    fields = prepare_row_fields(row, off_x, off_y)
    bbox_px = denorm_bbox_minmax_0_100_to_canvas(fields["bbox_norm"], off_x, off_y) if hf else None
    fig = build_fig(img, bbox_px, fields["pts_px"], fields["pts_meta"], show_numbers, use_gradient_color, point_size)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Status indicators
        flags = [m["In Bbox"] for m in fields["all_tokens_data"]]
        entered = any(flags)
        ended = flags[-1] if flags else False
        status = "🟢" if (entered and ended) else ("🟡" if entered else "🔴")
        
        # Strategy tags
        ref = row.get("reference")
        is_spatial = has_terms(ref, SPATIAL_KEYWORDS)
        is_attr = has_terms(ref, ATTRIBUTE_KEYWORDS)
        strat_tags = []
        if is_spatial: strat_tags.append("🏠 Spatial")
        if is_attr: strat_tags.append("🎨 Attribute")
        strat_str = " | ".join(strat_tags) if strat_tags else "General"
        
        st.markdown(f"### {status} Ep {row.get('episode','?')} | Q{qid}")
        st.caption(f"**Strategy:** {strat_str} | **Hit Idx:** {row.get(COL_HIT_IDX, 'N/A')}")
        
        if ref:
            if isinstance(ref, (list, tuple)): ref = " ".join(str(r) for r in ref)
            st.markdown(f"**Generated Reference:**\n{ref}")
            
        st.plotly_chart(fig, use_container_width=False)
        st.info(f"**Average Reward:** {calculate_average_reward(row.get(COL_TOTAL)):.4f}")
        with st.expander("More details", expanded=True):
            if fields["all_tokens_data"]: st.dataframe(pd.DataFrame(fields["all_tokens_data"]), use_container_width=True)
    with col2: st.empty()

# -------------------------
# Human vs Model Utilities
# -------------------------

DATASETS = ["refcoco_testA", "refcoco_testB", "refoi_co_occurrence", "refoi_single_presence"]

def load_human_image(img_name: str) -> Image.Image:
    path = os.path.join("/Users/teaywright/Projects/ToM/GAZERL_COLM", "html", "data", "human_eval_data", "images", img_name)
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    else:
        return Image.new("RGB", (TARGET_W, TARGET_H), (128, 128, 128))

@st.cache_data(show_spinner=True)
def load_comparison_data(cache_path: str) -> pd.DataFrame:
    with open(cache_path, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def cohen_kappa_score(y1, y2):
    y1 = np.array(y1)
    y2 = np.array(y2)
    if len(y1) != len(y2) or len(y1) == 0:
        return 0.0
    po = np.mean(y1 == y2)
    p1_true = np.mean(y1 == 1)
    p2_true = np.mean(y2 == 1)
    p1_false = 1 - p1_true
    p2_false = 1 - p2_true
    pe = (p1_true * p2_true) + (p1_false * p2_false)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)

def get_browser_image_scale(natural_w: float, natural_h: float, screen_info: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    if not screen_info:
        return 1.0, 1.0
    
    window_w = screen_info.get("windowWidth")
    window_h = screen_info.get("windowHeight")
    if not window_w or not window_h or window_w <= 0 or window_h <= 0 or natural_w <= 0 or natural_h <= 0:
        return 1.0, 1.0
        
    max_w = 0.95 * window_w
    max_h = 0.85 * window_h
    
    # Calculate scale factor to fit within max_w and max_h (matching CSS contain aspect ratio fitting)
    scale = 1.0
    if natural_w > max_w or natural_h > max_h:
        scale = min(max_w / natural_w, max_h / natural_h)
        
    scale_factor = 1.0 / scale
    return scale_factor, scale_factor

def build_comparison_fig(img: Image.Image,
                         bbox: Optional[List[float]],
                         predicted_bbox: Optional[List[float]],
                         human_trials: List[Dict[str, Any]],
                         show_mouse_movements: bool = True) -> go.Figure:
    
    canvas, (off_x, off_y), scale = letterbox_to_canvas(img, TARGET_W, TARGET_H)
    natural_w, natural_h = img.size
    
    fig = go.Figure()
    fig.add_layout_image(dict(
        source=canvas,
        xref="x", yref="y",
        x=0, y=TARGET_H,
        sizex=TARGET_W, sizey=TARGET_H,
        sizing="stretch",
        layer="below"
    ))

    # 1. Gold Bounding Box (bbox format: [x0, y0, x1, y1] in natural image coords)
    if bbox and len(bbox) == 4:
        bx0 = bbox[0] * scale + off_x
        by0 = bbox[1] * scale + off_y
        bx1 = bbox[2] * scale + off_x
        by1 = bbox[3] * scale + off_y
        fig.add_shape(
            type="rect",
            x0=bx0, y0=TARGET_H - by0,
            x1=bx1, y1=TARGET_H - by1,
            line=dict(width=3, color="red"),
            fillcolor="rgba(0,0,0,0)",
            name="Target (Gold)"
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="square", color="red", size=10),
            name="Target (Gold)"
        ))

    # 2. Predicted Bbox (predicted_bbox format: [x0, y0, x1, y1] in natural image coords)
    if predicted_bbox and len(predicted_bbox) == 4:
        px0 = predicted_bbox[0] * scale + off_x
        py0 = predicted_bbox[1] * scale + off_y
        px1 = predicted_bbox[2] * scale + off_x
        py1 = predicted_bbox[3] * scale + off_y
        fig.add_shape(
            type="rect",
            x0=px0, y0=TARGET_H - py0,
            x1=px1, y1=TARGET_H - py1,
            line=dict(width=3, color="cyan", dash="dash"),
            fillcolor="rgba(0,0,0,0)",
            name="Model Predicted"
        )
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(symbol="square", color="cyan", size=10),
            name="Model Predicted"
        ))

    # Listener colors (Emerald Green, Peter River Blue, Amethyst Purple)
    colors = [
        "rgb(46, 204, 113)",  # L1
        "rgb(52, 152, 219)",  # L2
        "rgb(155, 89, 182)"   # L3
    ]
    colors_alpha = [
        "rgba(46, 204, 113, 0.4)",
        "rgba(52, 152, 219, 0.4)",
        "rgba(155, 89, 182, 0.4)"
    ]

    # Plot each listener
    for idx, t in enumerate(human_trials):
        color = colors[idx % len(colors)]
        color_alpha = colors_alpha[idx % len(colors_alpha)]
        l_name = f"Listener {idx+1}"
        
        click_x = t.get("click_x")
        click_y = t.get("click_y")
        is_hit = bool(t.get("is_hit", False))
        early_clicks = t.get("early_clicks") or []
        mouse_movements = t.get("mouse_movements") or []
        screen_info = t.get("screen_info")
        
        # Calculate scale factor for browser-to-natural conversion of mouse movements
        scale_x, scale_y = get_browser_image_scale(natural_w, natural_h, screen_info)
        
        # 3. Mouse movements
        if show_mouse_movements and mouse_movements:
            m_xs = [(m["x"] * scale_x) * scale + off_x for m in mouse_movements]
            m_ys = [TARGET_H - ((m["y"] * scale_y) * scale + off_y) for m in mouse_movements]
            m_ts = [m["t"] for m in mouse_movements]
            
            fig.add_trace(go.Scatter(
                x=m_xs, y=m_ys,
                mode="lines+markers",
                line=dict(width=2, color=color_alpha),
                marker=dict(size=4, color=color),
                hovertext=[f"{l_name} Gaze | Time: {t}ms" for t in m_ts],
                hoverinfo="text",
                name=f"{l_name} Path"
            ))

        # 4. Early clicks
        if early_clicks:
            ec_xs = [c["x"] * scale + off_x for c in early_clicks]
            ec_ys = [TARGET_H - (c["y"] * scale + off_y) for c in early_clicks]
            fig.add_trace(go.Scatter(
                x=ec_xs, y=ec_ys,
                mode="markers",
                marker=dict(symbol="circle-open", size=10, color=color, line=dict(width=2)),
                hovertext=[f"{l_name} Early Click: {c['t']}ms" for c in early_clicks],
                hoverinfo="text",
                name=f"{l_name} Early Click"
            ))

        # 5. Final Click
        if click_x is not None and click_y is not None:
            cx = click_x * scale + off_x
            cy = TARGET_H - (click_y * scale + off_y)
            symbol = "circle" if is_hit else "x"
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers",
                marker=dict(symbol=symbol, size=12, color=color, line=dict(width=2, color="white")),
                hovertext=[f"{l_name} Final Click: {click_x:.1f}, {click_y:.1f} ({'Hit' if is_hit else 'Miss'})"],
                hoverinfo="text",
                name=f"{l_name} Final Click ({'✓' if is_hit else '✗'})"
            ))

    fig.update_xaxes(range=[0, TARGET_W], visible=False)
    fig.update_yaxes(range=[0, TARGET_H], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(width=TARGET_W, height=TARGET_H, margin=dict(l=0, r=0, t=0, b=0), dragmode=False)
    return fig

def render_comparison_example(row: pd.Series, evaluator: str, show_mouse_move: bool):
    img_name = row.get("img_name")
    dataset = row.get("dataset")
    speaker = row.get("speaker")
    bbox = row.get("bbox")
    reference = row.get("reference")
    
    # Model hit column name
    hit_col = "qwenvl_hit" if evaluator == "QwenVL" else "cogvlm_hit"
    iou_col = "qwenvl_avg_iou" if evaluator == "QwenVL" else "cogvlm_avg_iou"
    bbox_col = "qwenvl_predicted_bbox" if evaluator == "QwenVL" else "cogvlm_predicted_bbox"
    
    eval_hit = row.get(hit_col)
    eval_iou = row.get(iou_col)
    predicted_bbox = row.get(bbox_col)
    
    # Human aggregated info
    human_is_hit_majority = row.get("human_is_hit_majority", False)
    human_trials = row.get("human_trials") or []
    
    # Load image
    img = load_human_image(img_name)
    
    # Build figure with all 3 human trials
    fig = build_comparison_fig(
        img=img,
        bbox=bbox,
        predicted_bbox=predicted_bbox,
        human_trials=human_trials,
        show_mouse_movements=show_mouse_move
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        maj_emoji = "🟢 Correct" if human_is_hit_majority else "🔴 Incorrect"
        status_m = "🟢" if eval_hit else ("🔴" if eval_hit is not None else "⚪")
        st.markdown(f"### Image: {img_name}")
        st.caption(f"**Speaker Model:** `{speaker}` | **Dataset:** `{dataset}`")
        
        if reference:
            st.info(f"**Reference:**\n{reference}")
            
        st.plotly_chart(fig, use_container_width=False)
        
    with col2:
        st.markdown("#### Comparison Metrics")
        
        # Human performance
        st.markdown(f"##### Humans (Majority: {maj_emoji})")
        
        # Display details for each listener
        colors_hex = ["#2ecc71", "#3498db", "#9b59b6"]
        for idx, t in enumerate(human_trials):
            l_status = "🟢 Correct" if t['is_hit'] else "🔴 Incorrect"
            color_dot = f"<span style='color:{colors_hex[idx]}; font-size: 1.5em;'>■</span>"
            st.markdown(f"{color_dot} **Listener {idx+1}:** {l_status}", unsafe_allow_html=True)
            st.markdown(f"""
            * Duration: `{t['duration_ms']/1000.0:.2f} s`
            * Early Clicks: `{len(t['early_clicks'])}`
            * Revealed text: `{'Yes' if t['text_was_revealed'] else 'No'}`
            """)
        
        st.divider()
        
        # Model performance
        st.markdown(f"##### Model Evaluator ({evaluator}): {status_m} {'Correct' if eval_hit else ('Incorrect' if eval_hit is not None else 'Not Evaluated')}")
        if eval_hit is not None:
            st.markdown(f"""
            - **Average IoU:** `{eval_iou:.4f}`
            - **Predicted Bbox:** `{predicted_bbox}`
            """)
        else:
            st.write("No matching model evaluation found.")

# -------------------------
# App Main Setup
# -------------------------

st.set_page_config(page_title="Gaze & Error Viewer", layout="wide")
st.title("Gaze & Error Analysis Viewer")

# Main selection for viewer mode
app_mode = st.sidebar.radio("App Mode", ["Gaze Episode Viewer (RL)", "Human vs Model Error Analyzer"])

if app_mode == "Gaze Episode Viewer (RL)":
    st.sidebar.header("Data")
    csv_path = st.sidebar.text_input("CSV path", value="episodes.csv")

    st.sidebar.header("Filter")
    success_only = st.sidebar.checkbox("Comm Success Only", value=False)
    high_gaze_density = st.sidebar.checkbox("Gaze Density > 60%", value=False)
    spatial_only = st.sidebar.checkbox("Spatial Strategy Only", value=False)
    attribute_only = st.sidebar.checkbox("Attribute Strategy Only", value=False)
    exclude_eos_hit = st.sidebar.checkbox("Exclude EOS Hits", value=False)
    min_length = st.sidebar.number_input("Min Ref Length", 0, 500, 0)
    max_length = st.sidebar.number_input("Max Ref Length", 0, 500, 100)
    min_hit_idx = st.sidebar.number_input("Min Hit Index", 0, 500, 0)

    st.sidebar.header("Display")
    display_mode = st.sidebar.radio("Mode", ["Single example", "Gallery"])

    st.sidebar.header("Plot options")
    show_numbers = st.sidebar.checkbox("Indices", value=True)
    use_gradient = st.sidebar.checkbox("Gradient", value=True)
    point_size = st.sidebar.slider("Size", 4, 16, 10)

    if not csv_path: st.stop()
    try: df = load_csv(csv_path)
    except Exception as e: st.error(f"CSV Error: {e}"); st.stop()
    if df is None or len(df) == 0: st.warning("CSV is empty."); st.stop()

    # APPLY FILTERS
    if success_only:
        def is_success(row):
            flags = row.get(COL_IN_BBOX) or []
            entered = any(bool(v) for v in flags if v is not None)
            ended = bool(flags[-1]) if flags else False
            return entered and ended
        df = df[df.apply(is_success, axis=1)]

    if high_gaze_density:
        def is_dense(row):
            tokens = row.get(COL_TOKENS) or []
            gaze = row.get(COL_GAZE) or []
            if not tokens: return False
            valid_gaze_count = 0
            for g in gaze:
                if isinstance(g, (list, tuple)) and len(g) == 2:
                    try:
                        gx, gy = float(g[0]), float(g[1])
                        if not (np.isnan(gx) or np.isnan(gy)) and not (gx == 0.0 and gy == 0.0):
                            valid_gaze_count += 1
                    except: pass
            return (valid_gaze_count / len(tokens)) > 0.6
        df = df[df.apply(is_dense, axis=1)]

    if spatial_only:
        df = df[df['reference'].apply(lambda x: has_terms(x, SPATIAL_KEYWORDS))]

    if attribute_only:
        df = df[df['reference'].apply(lambda x: has_terms(x, ATTRIBUTE_KEYWORDS))]

    if exclude_eos_hit:
        def not_eos_hit(row):
            hit_idx = row.get(COL_HIT_IDX)
            tokens = row.get(COL_TOKENS) or []
            if pd.isna(hit_idx) or not tokens: return True
            try:
                hit_idx_int = int(float(hit_idx))
                if 1 <= hit_idx_int <= len(tokens):
                    target_token = str(tokens[hit_idx_int - 1])
                    return EOS_TOKEN not in target_token
                return True
            except: return True
        df = df[df.apply(not_eos_hit, axis=1)]

    if min_hit_idx > 0:
        df = df[df[COL_HIT_IDX] >= min_hit_idx]

    if min_length > 0:
        df = df[df[COL_TOKENS].apply(lambda x: len(x) >= min_length)]

    if max_length > 0:
        df = df[df[COL_TOKENS].apply(lambda x: len(x) <= max_length)]

    st.sidebar.info(f"Showing {len(df)} filtered episodes.")

    with st.sidebar.expander("Debug"):
        st.write(f"PYTHON: {sys.executable}")
        st.write(f"HF AVAILABLE: {HF_AVAILABLE}")
        idx = _coco_id_index()
        st.write(f"COCO Index: {'READY' if idx else 'FAILED'}")
        if idx: st.write(f"Size: {len(idx)}")

    if len(df) == 0:
        st.warning("No episodes match the selected filters.")
        st.stop()

    if display_mode == "Single example":
        selected_row = None
        with st.sidebar.expander("Selection", expanded=True):
            eps = sorted(map(int, pd.Series(df.get("episode", [])).dropna().unique().tolist()))
            if eps:
                sel_ep = st.slider("Episode Slider", eps[0], eps[-1], eps[0])
                sel_ep_num = st.number_input("Exact Episode", eps[0], eps[-1], int(sel_ep))
                sel_ep_val = min(eps, key=lambda v: abs(v - sel_ep_num))
                selected_row = df.loc[df["episode"] == sel_ep_val].iloc[0]
            else:
                idx = st.number_input("Row index", 0, len(df)-1, 0)
                selected_row = df.iloc[int(idx)]
        if selected_row is not None:
            render_example(selected_row, show_numbers, use_gradient, point_size)
    else:
        page_size = st.sidebar.slider("Items per page", 1, 100, 10)
        total_pages = max(1, int(np.ceil(len(df) / page_size)))
        page_num = st.sidebar.number_input("Page", 1, total_pages, 1)
        start_idx = (page_num - 1) * page_size
        for i in range(start_idx, min(start_idx + page_size, len(df))):
            render_example(df.iloc[i], show_numbers, use_gradient, point_size)
            st.divider()

else:
    # Human vs Model Error Analyzer Mode
    cache_file = "human_model_comparison_cache.json"
    if not os.path.exists(cache_file):
        st.warning(f"Comparison cache file '{cache_file}' not found. Please compile it first.")
        st.stop()
        
    comp_df = load_comparison_data(cache_file)
    
    st.sidebar.header("Filter Comparison")
    evaluator = st.sidebar.selectbox("Model Evaluator", ["QwenVL", "CogVLM"])
    
    comp_category = st.sidebar.selectbox("Comparison Category (Human Majority)", [
        "All",
        "Human Correct, Model Incorrect",
        "Human Incorrect, Model Correct",
        "Both Correct",
        "Both Incorrect"
    ])
    
    sel_dataset = st.sidebar.selectbox("Dataset", ["All"] + DATASETS)
    
    speaker_map = {
        "All": "All",
        "Gold": "gold",
        "Molmo": "molmo_vanilla",
        "Binary Gaze": "binary",
        "Binary LP Gaze": "binary_last_point",
        "Shaping Gaze": "shaping",
        "Sparse Gaze": "sparse_constant_kl_02",
        "Sparse Molmo": "iterative_sparse",
        "Binary LP Molmo": "supervised"
    }
    sel_speaker_label = st.sidebar.selectbox("Speaker Model", list(speaker_map.keys()))
    sel_speaker = speaker_map[sel_speaker_label]

    ref_search = st.sidebar.text_input("Search Reference Text", "")
    
    # Reset index if filters change to avoid out of bounds or mismatch from previous filters
    current_filter_key = f"{evaluator}_{comp_category}_{sel_dataset}_{sel_speaker_label}_{ref_search}"
    if "last_filter_key" not in st.session_state or st.session_state.last_filter_key != current_filter_key:
        st.session_state.comparison_idx = 0
        st.session_state.last_filter_key = current_filter_key

    st.sidebar.header("Display Options")
    display_mode = st.sidebar.radio("Mode", ["Single example", "Gallery"])
    show_mouse_move = st.sidebar.checkbox("Show Human Mouse Path", value=True)
    
    # Apply filters
    filtered_df = comp_df.copy()
    if sel_dataset != "All":
        filtered_df = filtered_df[filtered_df["dataset"] == sel_dataset]
        
    if sel_speaker != "All":
        if sel_speaker == "gold":
            filtered_df = filtered_df[filtered_df["speaker"].isin(["gold", "gold_refs"])]
        else:
            filtered_df = filtered_df[filtered_df["speaker"] == sel_speaker]
            
    if ref_search:
        filtered_df = filtered_df[filtered_df["reference"].str.contains(ref_search, case=False, na=False)]
        
    hit_col = "qwenvl_hit" if evaluator == "QwenVL" else "cogvlm_hit"
    
    if comp_category != "All":
        filtered_df = filtered_df[filtered_df[hit_col].notna()]
        
    if comp_category == "Human Correct, Model Incorrect":
        filtered_df = filtered_df[(filtered_df["human_is_hit_majority"] == True) & (filtered_df[hit_col] == False)]
    elif comp_category == "Human Incorrect, Model Correct":
        filtered_df = filtered_df[(filtered_df["human_is_hit_majority"] == False) & (filtered_df[hit_col] == True)]
    elif comp_category == "Both Correct":
        filtered_df = filtered_df[(filtered_df["human_is_hit_majority"] == True) & (filtered_df[hit_col] == True)]
    elif comp_category == "Both Incorrect":
        filtered_df = filtered_df[(filtered_df["human_is_hit_majority"] == False) & (filtered_df[hit_col] == False)]
        
    total_in_subset = len(filtered_df)
    st.sidebar.info(f"Showing {total_in_subset} matching examples.")
    
    # Display Stats Header
    st.subheader(f"Subset Summary Statistics ({comp_category} | Evaluator: {evaluator})")
    if total_in_subset > 0:
        human_acc_val = filtered_df["human_is_hit_majority"].mean() * 100
        valid_eval = filtered_df[filtered_df[hit_col].notna()]
        if len(valid_eval) > 0:
            model_acc_val = valid_eval[hit_col].mean() * 100
            agreement_val = (valid_eval["human_is_hit_majority"] == valid_eval[hit_col]).mean() * 100
            kappa_val = cohen_kappa_score(valid_eval["human_is_hit_majority"], valid_eval[hit_col])
            agreement_str = f"{agreement_val:.1f}% (κ={kappa_val:.3f})"
        else:
            model_acc_val = 0.0
            agreement_str = "N/A"
            
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Subset Examples", f"{total_in_subset}")
        c2.metric("Human Accuracy (Maj)", f"{human_acc_val:.1f}%")
        c3.metric("Model Accuracy", f"{model_acc_val:.1f}%" if len(valid_eval) > 0 else "N/A")
        c4.metric("Agreement (Kappa)", agreement_str)
    else:
        st.warning("No trials match the current filters.")
        st.stop()
        
    # Render Selected
    if display_mode == "Single example":
        # Initialize comparison index in session state
        if "comparison_idx" not in st.session_state:
            st.session_state.comparison_idx = 0
            
        # Clamp index
        st.session_state.comparison_idx = max(0, min(st.session_state.comparison_idx, total_in_subset - 1))
        
        # Navigation bar
        nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
        with nav_col1:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.comparison_idx = max(0, st.session_state.comparison_idx - 1)
                st.rerun()
        with nav_col2:
            if total_in_subset > 1:
                st.session_state.comparison_idx = st.slider(
                    "Navigate by Slider",
                    0,
                    total_in_subset - 1,
                    st.session_state.comparison_idx,
                    label_visibility="collapsed"
                )
        with nav_col3:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.comparison_idx = min(total_in_subset - 1, st.session_state.comparison_idx + 1)
                st.rerun()

        # Build list of options for select box
        trial_options = []
        for idx, (_, row) in enumerate(filtered_df.iterrows()):
            speaker_short = row["speaker"]
            ref_preview = row["reference"][:40] + "..." if len(row["reference"]) > 40 else row["reference"]
            h_status = "H:✓" if row["human_is_hit_majority"] else "H:✗"
            m_status = f"M:{'✓' if row[hit_col] else '✗'}" if row[hit_col] is not None else "M:?"
            trial_options.append(f"Ex {idx} | {row['img_name']} | {speaker_short} | {h_status} {m_status} | '{ref_preview}'")
            
        # Selectbox search dropdown
        sel_label = st.selectbox(
            "Search/Select Example Directly:",
            options=trial_options,
            index=st.session_state.comparison_idx
        )
        
        # Sync index from selectbox
        match = re.match(r"^Ex (\d+) \|", sel_label)
        if match:
            selected_idx = int(match.group(1))
            if selected_idx != st.session_state.comparison_idx:
                st.session_state.comparison_idx = selected_idx
                st.rerun()
                
        # Load selected row
        selected_row = filtered_df.iloc[st.session_state.comparison_idx]
        
        render_comparison_example(selected_row, evaluator, show_mouse_move)
    else:
        page_size = st.sidebar.slider("Items per page", 1, 100, 10)
        total_pages = max(1, int(np.ceil(len(filtered_df) / page_size)))
        page_num = st.sidebar.number_input("Page", 1, total_pages, 1)
        start_idx = (page_num - 1) * page_size
        
        for i in range(start_idx, min(start_idx + page_size, len(filtered_df))):
            render_comparison_example(filtered_df.iloc[i], evaluator, show_mouse_move)
            st.divider()
