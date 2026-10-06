# GazeRL Publication Cleanup: Handover & Progress Note

**Date**: September 8, 2026  
**Active Python Environment**: `/data/teaywright/conda/envs/gaze/bin/python` (PyTorch 2.7.1+cu126, Transformers 4.50.3)

---

## 1. Decisions & Scope Locked In

- **Speakers**: Molmo-7B (`allenai/Molmo-7B-D-0924`), PaliGemma-3B (`google/paligemma-3b-pt-448`), and LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`).
- **Listeners**:
  - `gaze_predictor_vectorized`: Primary batched gaze listener.
  - `molmo_listener_iterative_vectorized`: Iterative Molmo listener.
  - `molmo_listener_bbox_center` / `binary_last_point`: Single-point REC listener.
  - `models/listeners/eval/`: External evaluation listeners (Qwen-VL, CogVLM).
- **Scope**: RL rollouts and inference only (gaze predictor training scripts will be incorporated later).
- **Preset Naming**: Canonical paper acronyms:
  - `gaze_bfh.yaml` (Gaze Backward First Hit)
  - `gaze_shaping.yaml` (Gaze Distance Shaping)
  - `gaze_seq_any_hit.yaml` (Gaze Sequence Any Hit)
  - `gaze_seq_lp_hit.yaml` (Gaze Last Point Hit)
  - `rec_bfh.yaml` (REC Backward First Hit)
  - `rec_shaping.yaml` (REC Distance Shaping)
  - `rec_seq_any_hit.yaml` (REC Sequence Any Hit)
  - `rec_success.yaml` (Supervised REC Success)

---

## 2. Completed Milestones

### Milestone 1: Packaging, Configs, & Data Preprocessing (COMPLETE ✅)
- **`pyproject.toml`**: Modern PEP 621 package definition (`pip install -e .`).
- **`configs/`**:
  - `configs/constants.py`: Centralized dimensions (512x320 listener, 336 max speaker dim, 0-100 coordinate scale) and default paths.
  - `configs/prompts.py`: Standard detailed, brief, and legacy prompts.
  - `configs/base.py`: Dataclasses for `SpeakerConfig`, `ListenerConfig`, `RewardConfig`, `LoRAConfig`, `DatasetConfig`, `TrainingConfig`, and `ExperimentConfig` with YAML serialization.
  - `configs/presets/*.yaml`: All 8 paper preset YAML configurations.
- **`data/`**:
  - `data/splits/`: Clean verified JSON splits copied (`refcoco_train_ids.json`, `refcoco_val_ids.json`, `coco_2014_train_ids.json`, `coco_2014_val_ids.json`, `coco_2017_train_ids.json`, `coco_2017_val_ids.json`, `split_metadata.json`, `coco_2014_split_metadata.json`).
  - `data/processing.py`: Multi-modal coordinate transforms, red-box speaker overlays, letterboxed listener padding, point parsing (`<x=.. y=..>`), Gaussian parsing (`<x=.. y=.. v=..>`), and Gaussian probability mass calculation.
  - `data/molmo_image_preprocessing.py`: Standalone Molmo image processor.
  - `data/datasets.py`: `DatasetSplitManager`, `ReferringExpressionDataset`, and deterministic `SyntheticReferringExpressionDataset` for testing without downloading massive datasets.
- **`tests/test_data_smoke.py`**: **8/8 unit tests PASSED** in 2.02s.

---

## 3. Current State & Where We Left Off

### Milestone 2: Speaker Models (COMPLETE ✅)
- **`models/base.py`**: Defined `BaseSpeaker` and `BaseListener` abstract base classes, along with `SpeakerOutput` and `ListenerOutput` dataclasses.
- **`models/speakers/`**:
  - `molmo_speaker.py`: Ported and modularized Molmo-7B speaker.
  - `paligemma_speaker.py`: Ported and modularized PaliGemma-3B speaker.
  - `llava_speaker.py`: Ported and modularized LLaVA-1.5-7B speaker.
  - `__init__.py`: Dynamic model loader with `get_speaker(config)`.
- **`tests/test_speakers_smoke.py`**: **5/5 unit tests PASSED** in 6.97s.

### Milestone 3: Listener Models (COMPLETE ✅)
- **`models/listeners/`**:
  - `gaze_predictor.py`: High-performance batched / vectorized gaze predictor listener.
  - `molmo_listener.py`: Unified iterative and single-point REC listener.
  - `eval/qwenvl_listener.py`: Qwen-VL evaluation listener.
  - `eval/cogvlm_listener.py`: CogVLM evaluation listener.
  - `__init__.py`: Factory `get_listener(config)`.
- **`tests/test_listeners_smoke.py`**: **5/5 unit tests PASSED** in 4.59s.

---

## 3. Current State & Where We Are Now

### Milestone 4: RL Rewards & Policy Loss (COMPLETE ✅)
- **`rl/rewards/`**:
  - `base.py`: Abstract `BaseRewardFunction` and standardized `RewardOutput` dataclass.
  - `sparse.py`: `BFHReward` (backward first hit $\gamma=0.9$), `SeqAnyHitReward`, `SeqLPHitReward`, `SupervisedRECReward`.
  - `shaping.py`: `DistanceShapingReward` (continuous delta distance) and `GaussianShapingReward` ($\Delta\Phi$).
  - `penalties.py`: GRPO-style logit norm squared penalty and step-wise length penalty.
  - `composite.py`: Unified `GazeRLReward` with word-to-token projection and punctuation filtering.
  - `__init__.py`: Factory `get_reward_function(config)`.
- **`rl/loss.py`**: Policy gradient loss (REINFORCE + advantage standardization + reference model KL divergence).
- **`rl/rollouts.py`**: Multi-modal trajectory generation and collation (`collect_single_rollout`, `collate_rollouts`).
- **`tests/test_rewards_smoke.py`**: **9/9 unit tests PASSED** in 4.21s.

### Milestone 5: Streamlined Trainer & Infrastructure (COMPLETE ✅)
- **`training/`**:
  - `trainer.py`: Streamlined RL training engine (<280 lines) connecting rollouts, loss, optimizer, and validation.
  - `checkpoint.py`: Checkpoint saving and loading with LoRA adapter support and RNG restoration.
  - `optimization.py`: AdamW optimizer, LR schedulers (constant, cosine, linear), and dropout disabling.
  - `validator.py`: Periodic in-training validation over test splits.
  - `logging.py`: Local JSONL logging and optional WandB integration.
- **`tests/test_train_step_smoke.py`**: Mini-RL training step verification **PASSED**.
- **`tests/test_checkpoint_smoke.py`**: Save and restore state verification **PASSED**.

### Milestone 6: Evaluation Pipeline & Qualitative Visualizer (COMPLETE ✅)
- **`evaluation/`**:
  - `metrics.py`: Spatial grounding accuracy (hit rate), target distance, BBox IoU, sentence length, BLEU-1..4, and ROUGE-L.
  - `evaluate.py`: Unified evaluation runner (`evaluate_dataset`) with summary and per-example JSON reports.
  - `qualitative.py`: Trajectory visualizer (`draw_gaze_trajectory`) drawing scanpaths and status banners.
- **`tests/test_eval_smoke.py`**: **3/3 unit tests PASSED**.
- **`tests/test_viewer_smoke.py`**: **1/1 unit tests PASSED**.

### Milestone 7: Unified CLI Entry Points & Cluster Templates (COMPLETE ✅)
- **`scripts/train.py`**: Single CLI entry point with YAML presets, command-line overrides, and `--smoke-test`.
- **`scripts/eval.py`**: Single evaluation entry point with checkpoint restoration, baselines, and `--smoke-test`.
- **`scripts/sbatch/`**:
  - `submit_train.sbatch`: Single-node GPU training launcher.
  - `submit_eval.sbatch`: Automated batch evaluation launcher.
- **`README.md`**: Publication-ready documentation with architecture highlights, installation, and commands.
- **`LICENSE`**: Apache 2.0.
- **`requirements.txt` & `.gitignore`**: Clean environment specification and repository hygiene.

### Milestone 8: Human Evaluation Dataset, Paper Reproduction & Visualizer (COMPLETE ✅)
- **Human Evaluation Dataset**:
  - `results/human_eval/human_eval_trials_21600.json`: 21,600 human interaction trials with all Prolific participant IDs anonymized (`participant_001` to `participant_572`).
- **Reproduction Pipeline**:
  - `scripts/analysis/reproduce_human_eval.py`: Verified recomputation script matching published numbers for Table 4 (REG Performance), Table 5 (Comprehension Dynamics), and Appendix B.1 (Test split breakdowns).
  - `tests/test_human_eval_reproduce_smoke.py`: Automated test validating reproduction script integrity.
- **Visualizer & Figures**:
  - `scripts/visualize.py`: Interactive Streamlit application for scanpath rollouts and human error analysis.
  - `scripts/plotting/export_qualitative_figure.py`: Camera-ready visual scanpath figure generator.
  - `figures/`: Camera-ready paper figures.
- **Model Evaluation & Canonical Runs**:
  - `results/model_eval/`: Qwen-VL evaluation benchmarks, syntactic stats (Table 14), and RL vs SFT efficiency data.
  - `results/eval_runs/`: 10 canonical paper model runs (21MB total).
  - `results/human_annotations/`: Expert quality ratings (Appendix D.2).
- **Hygiene**:
  - Successfully deleted temporary transfer directory `GAZERL_COLM`.

### Milestone 9: Seamless Listener Training Integration (Molmo-REC-Gaze) (COMPLETE ✅)
- **Native Package Integration**:
  - Integrated the listener training library into the native top-level package `listener/` (`listener.data`, `listener.models`, `listener.training`, `listener.evaluation`, `listener.visualization`).
  - Added backward-compatible alias (`molmo_gaze`) in `listener/__init__.py`.
  - Removed separate `MolmoFinetuning/` directory, symlinks, and archive to eliminate sub-repo artifacts.
- **Unified Directory Layout**:
  - `configs/listener/`: Canonical and ablation training configs (`canonical_molmo_gaze.yaml`, `eval.yaml`, `train.yaml`).
  - `data/refcocogaze/`: Complete RefCOCO-Gaze scanpaths and timings with +200ms latency shift.
  - `results/listener_eval/`: Evaluation tables, scanpath figures, and ablation predictions.
  - `checkpoints/exported_gaze_predictor/`: Drop-in bundle with tokenizer, custom model code, and configuration.
  - `scripts/`: Added `train_listener.py`, `eval_listener.py`, `preprocess_gaze.py`, and `export_listener.py`.
- **Packaging & Dependency Integration**:
  - Added `listener` (`bitsandbytes`, `fastdtw`) and `all` optional dependency groups to root `pyproject.toml` and `requirements.txt`.
  - Installed `gazerl` with native `listener` package in editable mode.
- **Resilience & Compatibility**:
  - Added pure-Python dynamic programming fallback for FastDTW in `listener/evaluation/dtw.py`.
  - Updated `KNOWN_GAZE_PREDICTOR_CANDIDATES` in `configs/constants.py` to auto-detect exported bundles in `checkpoints/exported_gaze_predictor`.
- **Comprehensive Unified Testing**:
  - Moved all 14 listener tests into `tests/` (`test_listener_transforms.py`, `test_listener_parsing.py`, `test_listener_metrics.py`, `test_listener_compatibility.py`).
  - All 74 tests pass out-of-the-box via a single `pytest` invocation.
- **Documentation**:
  - Fully documented the listener training pipeline in `README.md`.

---

## 4. Test Suite Summary

**All 74/74 automated smoke and validation tests PASSED in ~7s**:
```
tests/test_checkpoint_smoke.py              .                             [ 1%]
tests/test_data_smoke.py                    ........                      [12%]
tests/test_eval_smoke.py                    ...                           [16%]
tests/test_human_eval_reproduce_smoke.py    .                             [17%]
tests/test_listener_compatibility.py        ...                           [21%]
tests/test_listener_metrics.py              ....                          [27%]
tests/test_listener_parsing.py              ....                          [32%]
tests/test_listener_transforms.py           ...                           [36%]
tests/test_listeners_smoke.py               .....                         [43%]
tests/test_presets_validation.py            ..........................    [78%]
tests/test_rewards_smoke.py                 .........                     [90%]
tests/test_speakers_smoke.py                .....                         [97%]
tests/test_train_step_smoke.py              .                             [98%]
tests/test_viewer_smoke.py                  .                             [100%]
====================== 74 passed in ~7s ======================
```

---

## 5. Remote Gaze Predictor Retraining Handover
- Created [`AGENT_GAZE_PREDICTION.md`](AGENT_GAZE_PREDICTION.md) (and symlink `AGENT.md`) detailing:
  - Exact shared contracts: $512 \times 320$ letterboxed canvas, $0-100$ coordinate scale, `<x=.. y=..>` syntax.
  - RefCOCO-Gaze preparation: +200ms auditory processing delay, multi-fixation resolution, subword coordinate expansion.
  - Training configuration: Molmo-7B-D-0924, 4 epochs, `paged_adamw_8bit`, `lr=1e-5`, `bf16=True`, effective batch size 8.
  - Export bundle checklist and drop-in integration verification commands for `gazeRL_publish`.

