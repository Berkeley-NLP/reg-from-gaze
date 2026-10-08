"""
Unified evaluation pipeline: runs speaker generation and listener grounding evaluation.
"""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Union
from tqdm import tqdm

from models.base import BaseSpeaker, BaseListener, SpeakerOutput, ListenerOutput
from data.processing import (
    create_speaker_bbox_overlay,
    create_listener_padded_image,
    normalize_bbox_to_100,
)
from evaluation.metrics import (
    compute_grounding_accuracy,
    compute_mean_distance_to_bbox,
    compute_mean_sentence_length,
    compute_corpus_bleu,
    compute_rouge_l,
)

logger = logging.getLogger(__name__)


def evaluate_dataset(
    speaker: BaseSpeaker,
    listener: BaseListener,
    dataset: Any,
    max_samples: Optional[int] = None,
    output_path: Optional[Union[str, Path]] = None,
    prompt: str = "Briefly describe the object in the red box.",
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Run complete evaluation of speaker referring expression generation and listener grounding.

    Args:
        speaker: Speaker model instance.
        listener: Listener model instance (GazePredictor, MolmoListener, QwenVLListener, etc.).
        dataset: Dataset providing test instances.
        max_samples: Optional limit on the number of samples to evaluate.
        output_path: Optional path to save full evaluation results JSON.
        prompt: Task instruction prompt.
        show_progress: Whether to display tqdm progress bar.

    Returns:
        Dictionary containing summary metrics and per-example details.
    """
    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    logger.info(f"Starting evaluation on {num_samples} samples...")

    # Switch to eval mode
    inner_spk = getattr(speaker, "model", speaker)
    if hasattr(inner_spk, "eval"):
        inner_spk.eval()

    sample_results: List[Dict[str, Any]] = []
    pred_points: List[Optional[Sequence[float]]] = []
    target_bboxes_100: List[Any] = []
    generated_sentences: List[str] = []
    gold_references_list: List[List[str]] = []

    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Evaluating", total=num_samples)

    start_time = time.time()

    for i in iterator:
        item = dataset[i]

        if hasattr(item, "image"):
            img = item.image
            bbox_orig = getattr(item, "bbox_minmax", getattr(item, "bbox", None))
            speaker_img = getattr(item, "speaker_image", None)
            listener_img = getattr(item, "listener_image", None)
            sample_prompt = getattr(item, "prompt", prompt)
            sample_id = getattr(item, "sample_id", f"eval_{i}")
            golds = getattr(item, "gold_references", []) or []
        else:
            img = item.get("image")
            bbox_orig = item.get("bbox", item.get("bbox_minmax"))
            speaker_img = item.get("speaker_image")
            listener_img = item.get("listener_image")
            sample_prompt = item.get("prompt", prompt)
            sample_id = str(item.get("question_id", item.get("sample_id", f"eval_{i}")))
            golds = item.get("gold_references", []) or []

        if img is None or bbox_orig is None:
            continue

        orig_w, orig_h = img.size

        # Prepare images if not already pre-rendered
        if speaker_img is None:
            speaker_img = create_speaker_bbox_overlay(img, bbox_orig)
        if listener_img is None:
            listener_img = create_listener_padded_image(img)

        bbox_100 = normalize_bbox_to_100(bbox_orig, (orig_w, orig_h))

        # 1. Speaker generation
        spk_out: SpeakerOutput = speaker.generate(speaker_img, sample_prompt)
        gen_text = spk_out.text.strip()
        tokens = spk_out.tokens

        # 2. Listener comprehension
        listener_out: ListenerOutput = listener.predict(
            images=[listener_img],
            tokens_list=[tokens],
            bboxes=[bbox_100],
        )[0]

        # Extract final predicted coordinate (or last point in scanpath)
        last_pt = None
        if listener_out.coordinates:
            for pt in reversed(listener_out.coordinates):
                if pt is not None:
                    last_pt = pt[:2]
                    break

        pred_points.append(last_pt)
        target_bboxes_100.append(bbox_100)
        generated_sentences.append(gen_text)
        gold_references_list.append(golds)

        sample_results.append({
            "sample_id": sample_id,
            "generated_text": gen_text,
            "tokens": tokens,
            "target_bbox_100": bbox_100,
            "predicted_point": last_pt,
            "gaze_path": listener_out.coordinates,
            "gold_references": golds,
        })

    elapsed = time.time() - start_time

    # Compute aggregate metrics
    accuracy = compute_grounding_accuracy(pred_points, target_bboxes_100)
    mean_dist = compute_mean_distance_to_bbox(pred_points, target_bboxes_100)
    mean_len = compute_mean_sentence_length(generated_sentences)

    metrics_summary: Dict[str, Any] = {
        "num_evaluated": len(sample_results),
        "grounding_accuracy": float(accuracy),
        "mean_distance_to_bbox": float(mean_dist),
        "mean_length_words": float(mean_len),
        "elapsed_seconds": float(elapsed),
        "seconds_per_sample": float(elapsed / max(1, len(sample_results))),
    }

    # If gold references are present, compute language metrics
    has_golds = any(len(g) > 0 for g in gold_references_list)
    if has_golds:
        bleu_scores = compute_corpus_bleu(generated_sentences, gold_references_list)
        rouge_score = compute_rouge_l(generated_sentences, gold_references_list)
        metrics_summary.update(bleu_scores)
        metrics_summary["rouge_l"] = float(rouge_score)

    full_output = {
        "summary": metrics_summary,
        "results": sample_results,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(full_output, f, indent=2, default=str)
        logger.info(f"Evaluation report written to {out_p}")

    logger.info(
        f"Evaluation finished: "
        f"Acc: {metrics_summary['grounding_accuracy'] * 100:.2f}% | "
        f"Dist: {metrics_summary['mean_distance_to_bbox']:.2f} | "
        f"Len: {metrics_summary['mean_length_words']:.2f} words"
    )

    return full_output
