"""
plot_utils.py

Shared utilities, constants, and data loaders for figure generation scripts.
Deduplicates word counting, dataset extraction, and bivariate confidence ellipse computation.
"""

import json
import gzip
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union, Sequence

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

# Canonical evaluation datasets
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


def get_clean_ref_len(ref: Any) -> int:
    """
    Word count tokenizer matching paper specification.
    Strips code fences, isolates alphanumeric tokens, and discards single-char noise.
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


def extract_dataset(img_name: str) -> str:
    """Map image filename to canonical test dataset partition."""
    for ds in DATASETS:
        if ds in img_name:
            return ds
    return "Other"


def confidence_ellipse_from_cov(
    mean_x: float,
    mean_y: float,
    cov: Optional[np.ndarray],
    n_obs: int,
    ax: plt.Axes,
    n_std: float = 2.447,
    facecolor: str = 'none',
    **kwargs: Any,
) -> Optional[Ellipse]:
    """
    Plots a 2D confidence ellipse for bivariate normal mean using given covariance.
    n_std=2.447 corresponds to 95% confidence region for bivariate normal mean (2 DOF).
    """
    if cov is None or n_obs < 3:
        return None
    cov_mean = cov / n_obs
    eigenvals, eigenvecs = np.linalg.eigh(cov_mean)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvals, 0))
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    ellipse = Ellipse(xy=[mean_x, mean_y], width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)
    return ax.add_patch(ellipse)


def confidence_ellipse(
    x: Union[np.ndarray, List[float], Sequence[float]],
    y: Union[np.ndarray, List[float], Sequence[float]],
    ax: plt.Axes,
    n_std: float = 2.447,
    facecolor: str = 'none',
    is_mean: bool = True,
    **kwargs: Any,
) -> Optional[Ellipse]:
    """Plots a 2D confidence ellipse directly from sample vectors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size or x.size < 3:
        return None
    cov = np.cov(x, y)
    if is_mean:
        cov = cov / len(x)
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    width, height = 2 * n_std * np.sqrt(np.maximum(eigenvals, 0))
    angle = np.degrees(np.arctan2(*eigenvecs[:, 0][::-1]))
    center = [np.mean(x), np.mean(y)]
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)
    return ax.add_patch(ellipse)


def load_human_trials_data(trials_path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads human evaluation interaction trials (handles .json or .json.gz).
    Returns DataFrame with columns ['Model', 'Dataset', 'Length', 'Accuracy'].
    """
    trials_path = Path(trials_path)
    if not trials_path.exists():
        gz_path = trials_path.with_suffix(trials_path.suffix + ".gz")
        if gz_path.exists():
            trials_path = gz_path
        else:
            raise FileNotFoundError(f"Trials file not found: {trials_path}")

    if trials_path.suffix == ".gz":
        with gzip.open(trials_path, "rt", encoding="utf-8") as f:
            trials = json.load(f)
    else:
        with open(trials_path, "r", encoding="utf-8") as f:
            trials = json.load(f)

    rows = []
    for t in trials:
        m = HUMAN_SPEAKER_MAP.get(t.get("speaker"))
        ds = extract_dataset(t.get("img_name", ""))
        if m and ds != "Other":
            l = get_clean_ref_len(t.get("reference", ""))
            acc = 100.0 if t.get("is_hit") else 0.0
            rows.append({"Model": m, "Dataset": ds, "Length": l, "Accuracy": acc})
    return pd.DataFrame(rows)
