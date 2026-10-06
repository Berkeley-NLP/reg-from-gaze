#!/usr/bin/env python3
"""
Main evaluation entry point for GazeRL and baseline reference games.

Usage:
    # Evaluate trained checkpoint on validation set
    python scripts/eval.py --checkpoint outputs/checkpoint_final --dataset coco_2014 --split val

    # Evaluate zero-shot speaker baseline
    python scripts/eval.py --speaker molmo --dataset refcoco --split val

    # Instant smoke test
    python scripts/eval.py --smoke-test
"""

import argparse
import logging
from pathlib import Path
import sys

# Ensure project root, reg, and gaze_estimation are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
for p in [str(REPO_ROOT / "gaze_estimation"), str(REPO_ROOT / "reg"), str(REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from configs.base import SpeakerConfig, ListenerConfig
from data.datasets import ReferringExpressionDataset, SyntheticReferringExpressionDataset
from reg.models import get_speaker, get_listener
from reg.training.checkpoint import load_checkpoint
from reg.evaluation.evaluate import evaluate_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gazerl.eval")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GazeRL speaker grounding and language quality.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained speaker checkpoint.")
    parser.add_argument("--speaker", type=str, default="molmo", help="Speaker architecture ('molmo', 'paligemma', 'llava').")
    parser.add_argument("--listener", type=str, default="gaze_predictor", help="Listener model architecture ('gaze_predictor', 'molmo_iterative', 'eval_qwenvl').")
    parser.add_argument("--dataset", type=str, default="coco_2014", help="Dataset to evaluate ('coco_2014', 'coco_2017', 'refcoco').") 
    parser.add_argument("--split", type=str, default="val", help="Dataset split ('val', 'test', etc.).")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of examples to evaluate.")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Path to write output results JSON.")
    parser.add_argument("--prompt", type=str, default="Briefly describe the object in the red box.", help="Speaker generation prompt.")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 4-sample evaluation on synthetic data.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        logger.info("Executing evaluation smoke test on synthetic data...")
        from tests.test_eval_smoke import EvalDummySpeaker, EvalDummyListener

        speaker = EvalDummySpeaker()
        listener = EvalDummyListener()
        dataset = SyntheticReferringExpressionDataset(num_samples=4)

        report = evaluate_dataset(
            speaker=speaker,
            listener=listener,
            dataset=dataset,
            max_samples=4,
            output_path=args.output,
            show_progress=False,
        )
        logger.info("Smoke test passed successfully!")
        return

    # Initialize speaker model
    spk_cfg = SpeakerConfig(architecture=args.speaker)
    logger.info(f"Loading speaker model: {args.speaker}")
    speaker = get_speaker(spk_cfg)

    # If checkpoint is specified, restore trained LoRA adapter weights
    if args.checkpoint:
        logger.info(f"Restoring checkpoint from {args.checkpoint}...")
        load_checkpoint(args.checkpoint, speaker=speaker)

    # Initialize listener model
    lis_cfg = ListenerConfig(listener_type=args.listener)
    logger.info(f"Loading listener model: {args.listener}")
    listener = get_listener(lis_cfg)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} (split: {args.split})...")
    dataset = ReferringExpressionDataset(
        dataset_name=args.dataset,
        split_name=args.split,
        max_samples=args.max_samples,
    )

    # Run evaluation
    results = evaluate_dataset(
        speaker=speaker,
        listener=listener,
        dataset=dataset,
        max_samples=args.max_samples,
        output_path=args.output,
        prompt=args.prompt,
        show_progress=True,
    )

    summary = results["summary"]
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {args.dataset} ({args.split})")
    print("=" * 60)
    print(f"Grounding Accuracy: {summary['grounding_accuracy'] * 100:.2f}%")
    print(f"Mean Distance to Target: {summary['mean_distance_to_bbox']:.2f}")
    print(f"Mean Sentence Length: {summary['mean_length_words']:.2f} words")
    if "bleu_4" in summary:
        print(f"BLEU-4: {summary['bleu_4']:.2f}")
    if "rouge_l" in summary:
        print(f"ROUGE-L: {summary['rouge_l']:.2f}")
    print(f"Total Evaluated: {summary['num_evaluated']}")
    print(f"Results saved to: {args.output}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
