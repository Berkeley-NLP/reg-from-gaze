#!/usr/bin/env python3
"""
Preprocess RefCOCO-Gaze dataset by applying +200ms auditory latency shift.
Usage:
    python scripts/preprocess_data.py \
        --train_in data/refcocogaze_train_correct.json \
        --val_in data/refcocogaze_val_correct.json \
        --timing data/word-timing.json \
        --train_out data/refcocogaze_train_delay.json \
        --val_out data/refcocogaze_val_delay.json
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure package is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gaze_estimation.data.alignment import apply_auditory_latency_delay


def main():
    parser = argparse.ArgumentParser(description="Preprocess RefCOCO-Gaze with auditory latency shift")
    parser.add_argument("--train_in", type=str, default="data/refcocogaze/refcocogaze_train_correct.json")
    parser.add_argument("--val_in", type=str, default="data/refcocogaze/refcocogaze_val_correct.json")
    parser.add_argument("--timing", type=str, default="data/refcocogaze/word-timing.json")
    parser.add_argument("--train_out", type=str, default="data/refcocogaze/refcocogaze_train_delay.json")
    parser.add_argument("--val_out", type=str, default="data/refcocogaze/refcocogaze_val_delay.json")
    parser.add_argument("--latency_ms", type=int, default=200, help="Auditory latency delay in ms (default: 200)")
    args = parser.parse_args()

    print(f"Loading word timing data from {args.timing}...")
    with open(args.timing, "r") as f:
        timing_data = json.load(f)

    # Process training set
    if Path(args.train_in).exists():
        print(f"Processing training data: {args.train_in} -> {args.train_out} (+{args.latency_ms}ms)...")
        with open(args.train_in, "r") as f:
            train_data = json.load(f)
        adjusted_train = apply_auditory_latency_delay(train_data, timing_data, latency_ms=args.latency_ms)
        Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.train_out, "w") as f:
            json.dump(adjusted_train, f, indent=2)
        print(f"  ✓ Saved {len(adjusted_train)} training records to {args.train_out}")

    # Process validation set
    if Path(args.val_in).exists():
        print(f"Processing validation data: {args.val_in} -> {args.val_out} (+{args.latency_ms}ms)...")
        with open(args.val_in, "r") as f:
            val_data = json.load(f)
        adjusted_val = apply_auditory_latency_delay(val_data, timing_data, latency_ms=args.latency_ms)
        Path(args.val_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.val_out, "w") as f:
            json.dump(adjusted_val, f, indent=2)
        print(f"  ✓ Saved {len(adjusted_val)} validation records to {args.val_out}")


if __name__ == "__main__":
    main()
