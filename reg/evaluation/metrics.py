"""
Evaluation metrics: Spatial grounding accuracy, BBox IoU, distance to target, and language generation metrics.
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from data.processing import is_point_in_bbox, calculate_distance


def compute_bbox_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes (x1, y1, x2, y2).
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(min(x1_1, x2_1), min(x1_2, x2_2))
    inter_y1 = max(min(y1_1, y2_1), min(y1_2, y2_2))
    inter_x2 = min(max(x1_1, x2_1), max(x1_2, x2_2))
    inter_y2 = min(max(y1_1, y2_1), max(y1_2, y2_2))

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = abs((x2_1 - x1_1) * (y2_1 - y1_1))
    area2 = abs((x2_2 - x1_2) * (y2_2 - y1_2))
    union_area = area1 + area2 - inter_area

    if union_area <= 1e-8:
        return 0.0
    return float(inter_area / union_area)


def compute_grounding_accuracy(
    predicted_points: Sequence[Optional[Sequence[float]]],
    target_bboxes: Sequence[Tuple[float, float, float, float]],
) -> float:
    """
    Compute percentage of predictions whose coordinates fall strictly inside target bbox.
    """
    if not predicted_points or not target_bboxes:
        return 0.0
    hits = 0
    total = len(predicted_points)
    for pt, bbox in zip(predicted_points, target_bboxes):
        if pt is not None and bbox is not None:
            if is_point_in_bbox((pt[0], pt[1]), bbox):
                hits += 1
    return float(hits / total)


def compute_mean_distance_to_bbox(
    predicted_points: Sequence[Optional[Sequence[float]]],
    target_bboxes: Sequence[Tuple[float, float, float, float]],
) -> float:
    """
    Compute average Euclidean distance from predicted coordinates to nearest bbox boundary.
    """
    if not predicted_points:
        return 0.0
    distances: List[float] = []
    for pt, bbox in zip(predicted_points, target_bboxes):
        if pt is None or bbox is None:
            continue
        x, y = pt[0], pt[1]
        x1, y1, x2, y2 = bbox
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        cx = min(max(x, min_x), max_x)
        cy = min(max(y, min_y), max_y)
        dist = calculate_distance((x, y), (cx, cy))
        distances.append(dist)
    return float(np.mean(distances)) if distances else 0.0


def compute_mean_sentence_length(sentences: Sequence[str]) -> float:
    """
    Compute average word count across generated sentences.
    """
    if not sentences:
        return 0.0
    lengths = [len(s.split()) for s in sentences]
    return float(np.mean(lengths))


def compute_corpus_bleu(
    hypotheses: Sequence[str],
    references_list: Sequence[Sequence[str]],
) -> Dict[str, float]:
    """
    Compute corpus-level BLEU-1, BLEU-2, BLEU-3, and BLEU-4 scores using NLTK.
    """
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

    tokenized_hyps = [h.lower().split() for h in hypotheses]
    tokenized_refs = [[r.lower().split() for r in refs] for refs in references_list]

    smooth = SmoothingFunction().method1

    weights_dict = {
        "bleu_1": (1.0, 0.0, 0.0, 0.0),
        "bleu_2": (0.5, 0.5, 0.0, 0.0),
        "bleu_3": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0),
        "bleu_4": (0.25, 0.25, 0.25, 0.25),
    }

    results: Dict[str, float] = {}
    for name, weights in weights_dict.items():
        try:
            score = corpus_bleu(tokenized_refs, tokenized_hyps, weights=weights, smoothing_function=smooth)
            results[name] = float(score * 100.0)  # Standard 0-100 percentage scale
        except Exception:
            results[name] = 0.0

    return results


def compute_rouge_l(
    hypotheses: Sequence[str],
    references_list: Sequence[Sequence[str]],
) -> float:
    """
    Compute average ROUGE-L F1 score against multiple gold references.
    """
    def lcs(seq1: List[str], seq2: List[str]) -> int:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if seq1[i] == seq2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
        return dp[m][n]

    scores: List[float] = []
    for hyp, refs in zip(hypotheses, references_list):
        hyp_toks = hyp.lower().split()
        if not hyp_toks:
            scores.append(0.0)
            continue

        best_f1 = 0.0
        for ref in refs:
            ref_toks = ref.lower().split()
            if not ref_toks:
                continue
            lcs_len = lcs(hyp_toks, ref_toks)
            prec = lcs_len / len(hyp_toks)
            rec = lcs_len / len(ref_toks)
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
        scores.append(best_f1)

    return float(np.mean(scores) * 100.0) if scores else 0.0
