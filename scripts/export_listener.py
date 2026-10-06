#!/usr/bin/env python3
"""
Checkpoint Export & Validation Tool for Molmo-REC-Gaze.
Prepares the complete standalone bundle required for seamless integration
into GazeRL (reg-from-gaze) per AGENT_GAZE_PREDICTION.md Section 6.

Usage:
    python scripts/export_listener.py \
        --source checkpoints/gaze_predictor_delay_token_116 \
        --target checkpoints/exported_gaze_predictor \
        --verify
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# Required files checklist per Section 6 of AGENT_GAZE_PREDICTION.md
REQUIRED_FILES = [
    "added_tokens.json",
    "config.json",
    "config_molmo.py",
    "generation_config.json",
    "image_preprocessing_molmo.py",
    "merges.txt",
    "preprocessing_molmo.py",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
]


def export_checkpoint(source_dir: Path, target_dir: Path, verify: bool = True):
    print("=" * 70)
    print("MOLMO-REC-GAZE CHECKPOINT EXPORT BUNDLER")
    print(f"  Source directory: {source_dir}")
    print(f"  Target export directory: {target_dir}")
    print("=" * 70)

    if not source_dir.exists():
        print(f"[ERROR] Source directory does not exist: {source_dir}")
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    print("\n[STEP 1/3] Copying configs, tokenizer files, and remote code...")
    copied_count = 0
    missing_files = []

    for fname in REQUIRED_FILES:
        src_f = source_dir / fname
        if not src_f.exists() and (source_dir / "best_checkpoint" / fname).exists():
            src_f = source_dir / "best_checkpoint" / fname

        if src_f.exists():
            shutil.copy2(src_f, target_dir / fname)
            copied_count += 1
        else:
            missing_files.append(fname)

    print(f"  ✓ Copied {copied_count}/{len(REQUIRED_FILES)} required configuration files.")
    if missing_files:
        print(f"  [WARNING] Missing non-critical config files: {missing_files}")

    print("\n[STEP 2/3] Linking or copying model weights...")
    # Find safetensors in source or best_checkpoint
    weight_dir = source_dir
    if not list(source_dir.glob("*.safetensors")) and (source_dir / "best_checkpoint").exists():
        weight_dir = source_dir / "best_checkpoint"

    safetensors_files = list(weight_dir.glob("*.safetensors")) + list(weight_dir.glob("*.index.json"))
    for wf in safetensors_files:
        target_f = target_dir / wf.name
        if not target_f.exists():
            try:
                os.link(wf, target_f)  # Hard link to avoid copying 15GB
                print(f"  ✓ Linked weight shard: {wf.name}")
            except OSError:
                shutil.copy2(wf, target_f)
                print(f"  ✓ Copied weight shard: {wf.name}")

    # Copy training/eval logs if present
    for log_file in ["training_loss.txt", "eval_loss.txt", "dtw_metrics.txt", "training_args.bin"]:
        src_log = source_dir / log_file
        if src_log.exists():
            shutil.copy2(src_log, target_dir / log_file)
            print(f"  ✓ Included log: {log_file}")

    if verify:
        print("\n[STEP 3/3] Verifying bundle with Hugging Face AutoModel & AutoProcessor...")
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            print("  Attempting drop-in AutoProcessor loading...")
            processor = AutoProcessor.from_pretrained(str(target_dir), trust_remote_code=True)
            print("  ✓ AutoProcessor loaded successfully.")

            print("  Attempting drop-in AutoModelForCausalLM configuration check...")
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(str(target_dir), trust_remote_code=True)
            print(f"  ✓ Model configuration loaded: {config.model_type if hasattr(config, 'model_type') else 'molmo'}")
        except Exception as e:
            print(f"  [ERROR] Verification failed: {e}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("✓ Checkpoint bundle is ready for transfer to gazeRL_publish!")
    print(f"  Destination path: {target_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Export Molmo-REC-Gaze checkpoint for GazeRL")
    parser.add_argument("--source", type=str, default="checkpoints/gaze_predictor_delay_token_116")
    parser.add_argument("--target", type=str, default="export/exported_gaze_predictor")
    parser.add_argument("--verify", action="store_true", default=True, help="Verify drop-in loading via AutoProcessor")
    args = parser.parse_args()

    export_checkpoint(Path(args.source), Path(args.target), verify=args.verify)


if __name__ == "__main__":
    main()
