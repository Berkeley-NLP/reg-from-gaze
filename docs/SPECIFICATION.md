# GazeRL Publication Specification & Triage Plan

> **Target Repository**: `https://github.com/Berkeley-NLP/reg-from-gaze.git`  
> **Goal**: Create a clean, modular, self-contained, and publishable release of **GazeRL** (referring expression generation guided by gaze RL).

---
Author Notes:
1. You can check which configs should be kept by checking in /scratch/ and /data/ to see if results directories with those configs exist. Otherwise, I most likely did not use them.
2. Please err on the side of minimalism
3. Link to paper pdf: https://openreview.net/pdf?id=QiqYKZFAB8

## 1. User Decisions & Configuration Checklist

Please review and fill in your choices for the following core areas:

### 1.1 Supported Speaker Models
Which speaker models should be officially supported in the clean repository?
- [x] **Molmo** (`allenai/Molmo-7B-D-0924`, `allenai/Molmo-7B-O-0924`)
- [x] **PaliGemma** (`google/paligemma-3b-pt-448`, etc.)
- [x] **LLaVA** (`llava-hf/llava-1.5-7b-hf`, etc.)

### 1.2 Supported Listener Models
Which listener architectures should be retained and packaged?
- [x] **Vectorized Gaze Predictor** (primary listener for fast batched rollouts)
- [x] **Molmo Listener (Iterative)** (`molmo_listener_iterative_vectorized.py`)
- [x] **REC Listener (Single Point)** (`binary_last_point` / bbox center)
- [x] **External VLM Listeners for Evaluation** (in `models/listeners/eval/`: Qwen-VL, CogVLM)
- [ ] **Legacy Standard Gaze Predictor** (unvectorized, consolidated/deprecated)

### 1.3 Reward Components & Paper Alignment Renaming
Canonical terminology from the paper:

| Current Internal Config Name | Internal Mechanism | Paper Terminology / Target New Name | Status |
| :--- | :--- | :--- | :--- |
| `sparse_constant_kl_02` | Backward decay (gamma=0.9) with KL=0.02 | `Gaze-BFH` | `[x] Keep & Renamed` |
| `shaping` | Continuous distance-based delta shaping | `Gaze-Shaping` | `[x] Keep & Renamed` |
| `binary` | 1.0 if gaze enters bbox during sequence | `Gaze-SeqAnyHit` | `[x] Keep & Renamed` |
| `binary_last_point` | 1.0 if last point hits bbox | `Gaze-SeqLPHit` | `[x] Keep & Renamed` |
| `supervised` | Supervised bbox center hit reward | `REC-Success` | `[x] Keep & Renamed` |
| `iterative_sparse` | Molmo iterative listener with backward decay | `REC-BFH` | `[x] Keep & Renamed` |
| `iterative_binary` | Molmo iterative listener with sequence hit | `REC-SeqAnyHit` | `[x] Keep & Renamed` |
| `iterative_shaping` | Molmo iterative listener with distance shaping | `REC-Shaping` | `[x] Keep & Renamed` |
| `sparse_constant_kl_0` | Sparse BFH with KL=0 (ablation) | `Gaze-BFH (No-KL)` | `[x] Keep as Ablation` |

### 1.4 Primary Training Configurations
8 Canonical Presets implemented in `configs/presets/`:
- [x] `gaze_bfh.yaml` (Gaze-BFH)
- [x] `gaze_shaping.yaml` (Gaze-Shaping)
- [x] `gaze_seq_any_hit.yaml` (Gaze-SeqAnyHit)
- [x] `gaze_seq_lp_hit.yaml` (Gaze-SeqLPHit)
- [x] `rec_bfh.yaml` (REC-BFH)
- [x] `rec_shaping.yaml` (REC-Shaping)
- [x] `rec_seq_any_hit.yaml` (REC-SeqAnyHit)
- [x] `rec_success.yaml` (REC-Success)
- [x] Architecture ablations: PaliGemma & LLaVA presets

### 1.5 Supported Datasets & Splits
Which dataset formats and splits should be included / documented?
- [ ] **RefCOCO / RefCOCO+ / RefCOCOg**
- [ ] **COCO 2014 / 2017**
- [ ] Curated subset split JSONs in `data_splits/` (specify which ones are essential)

### 1.6 Package & Dependency Management
- [ ] Modern `pyproject.toml` (standard packaging with optional extra dependencies: `[eval]`, `[vllm]`)
- [ ] `requirements.txt` / conda `environment.yml`

### 1.7 Pretrained Checkpoints & Model Hosting
The existing listener models reside at local `/scratch/...` and `/data/...` cluster paths.
- **Where will Gaze Predictor listener weights be hosted for the public?**
  - [ ] HuggingFace Hub (e.g., `Berkeley-NLP/gaze-predictor-molmo`, `Berkeley-NLP/gaze-predictor-paligemma`, etc.)
  - [ ] Google Drive / Zenodo / OSF link with a download script (`scripts/download_models.sh`)
  - [ ] *Other*: `_____________________`
- **Will trained Speaker RL checkpoints (e.g. GazeRL Molmo LoRA) also be released on HuggingFace Hub?**
  - [ ] Yes (allow users to reproduce paper eval without training from scratch)
  - [ ] No / Local only

### 1.8 Scope of Gaze Predictor Code
- [ ] **RL & Inference Only**: Include only the Listener inference & reward extraction logic (simplest, cleanest).
- [ ] **Include Listener Training**: Include training scripts to train the Gaze Predictor from human gaze datasets.

### 1.9 Dataset Setup & Acquisition
How should users acquire the RefCOCO / COCO images?
- [ ] Automatic download / HuggingFace `datasets` streaming where possible
- [ ] Provide a dedicated script: `scripts/download_data.sh`
- [ ] Manual instructions in README (pointing to standard COCO / RefCOCO download URLs)

### 1.10 Hardware Requirements & Quantization
- **Default precision**:
  - [ ] `bfloat16` (standard for modern GPUs: A100 / H100 / RTX 4090 / 3090)
  - [ ] `float16`
- **LoRA / PEFT defaults**:
  - [ ] 16-bit LoRA
  - [ ] 8-bit / 4-bit QLoRA support via bitsandbytes (for lower GPU VRAM)

### 1.11 Logging & Monitoring
- [ ] Local logging by default (TensorBoard / CSV / Console), WandB enabled via optional `--wandb` flag.
- [ ] WandB required by default.

### 1.12 Open Source License & Citation
- **License**:
  - [ ] Apache 2.0 (standard for Berkeley NLP projects)
  - [ ] MIT License
  - [ ] BSD 3-Clause
- **Paper Citation BibTeX**:
  - `[Placeholder for BibTeX citation in README]`

---

## 2. Modularization Plan for Monolithic Files

The existing codebase contains several very large files (e.g. `trainer.py` > 150KB, `training_configs.py` > 70KB, `generate_impl.py` > 85KB). We propose decomposing them into focused, maintainable modules:

```text
gazeRL_publish/
├── configs/
│   ├── base.py               # Structured dataclasses / Pydantic schemas for configs
│   ├── speakers.py           # Speaker hyperparameters and prompt templates
│   ├── listeners.py          # Listener hyperparameters
│   ├── rewards.py            # Reward components and weights
│   └── presets/              # Preset experiment recipes (YAML or Python dataclass instances)
│
├── models/
│   ├── base.py               # Abstract BaseSpeaker and BaseListener classes
│   ├── speakers/             # Modular speaker wrappers (Molmo, PaliGemma, LLaVA)
│   └── listeners/            # Clean GazePredictor & VLM listener wrappers
│
├── rl/
│   ├── rewards/              # Modular reward classes (Sparse, Gaussian, Distance, KL)
│   ├── rollouts.py           # Generation, sampling, and trajectory collation
│   ├── loss.py               # Policy gradient / GRPO loss functions
│   └── memory.py             # Replay / rollout buffer utilities
│
├── training/
│   ├── trainer.py            # Streamlined core RL training loop (<500 lines)
│   ├── checkpoint.py         # Save, load, and resume utilities (with LoRA support)
│   ├── logging.py            # Clean WandB and local logging
│   └── validation.py         # Periodic validation during training
│
├── evaluation/
│   ├── evaluate.py           # Unified entry point for evaluating checkpoints
│   ├── metrics.py            # Spatial accuracy, grounding success, CIDEr/BLEU/Meteor
│   └── qualitative.py        # Qualitative sample generator / visualizer
│
├── data/
│   ├── datasets.py           # Clean PyTorch Dataset classes for RefCOCO / COCO
│   ├── processing.py         # Multi-modal processor / tokenization adapters
│   └── splits/               # Verified split JSONs only
│
├── scripts/
│   ├── train.py              # Main CLI entry point: python scripts/train.py --config ...
│   └── eval.py               # Main CLI evaluation script
│
├── pyproject.toml
└── README.md
```

---

## 3. Tentative File-by-File Triage

Below is a complete inventory of files from `gazeRL/` categorized into **Keep / Refactor**, **Consolidate**, or **Delete / Exclude**. Please review and check/uncheck as needed.

### 3.1 Root Directory Files

| File | Recommendation | Proposed Action / Reason | User Decision |
| :--- | :--- | :--- | :--- |
| `README.md` | **Keep & Refactor** | Write clean, publication-ready documentation with install, train, eval commands | `[x]` Keep |
| `README_CODEBASE.md` | **Consolidate** | Merge relevant architecture notes into main README / docs | `[ ]` Exclude / Docs |
| `episode_viewer.py` | **Refactor** | Streamlit visualizer for RL trajectories and gaze heatmaps (move to `scripts/visualize.py`) | `[x]` Keep |
| `generate_and_evaluate.py` | **Consolidate** | Consolidate into unified `evaluation/evaluate.py` | `[ ]` Delete redundant |
| `load_lora_checkpoint.py` | **Consolidate** | Integrate into `models/` checkpoint loader | `[ ]` Consolidate |
| `utils.py` (top-level) | **Refactor** | Split into appropriate subpackages (`data/`, `rl/`) | `[x]` Refactor |
| `requirements.txt` | **Keep & Refactor** | Clean up dependencies, remove legacy/conflicting packages | `[x]` Keep |
| `requirements_clean.txt` | **Delete** | Merge needed dependencies into single `requirements.txt` / `pyproject.toml` | `[ ]` Delete |
| `environment_cogvlm.yml` | **Exclude / Move** | Move to `docs/` if optional CogVLM eval environment is needed | `[ ]` Optional |
| `cleanup_cache.sh` | **Exclude** | Local maintenance script | `[ ]` Exclude |
| `recreate_env.sh` | **Exclude** | Local setup script | `[ ]` Exclude |
| `setup_ssh_tunnel.sh` | **Exclude** | Local workflow script | `[ ]` Exclude |
| `run_streamlit.sh` | **Consolidate** | Document CLI command in README instead | `[ ]` Exclude |
| Markdown Guides (`GAUSSIAN_GAZE_GUIDE.md`, `LOGIT_PENALTY_GUIDE.md`, etc.) | **Consolidate** | Merge into documentation directory `docs/` | `[x]` Consolidate into `docs/` |
| Plot/PDF/PNG/CSV/Log outputs (`*.png`, `*.pdf`, `*.csv`, `slurm-*.out`) | **Delete / Exclude** | Do not commit experiment output artifacts to git | `[ ]` Exclude |
| Debug JSON files (`debug_gold.json`, `debug_refoi_gold.json`, etc.) | **Delete / Exclude** | Large debug dumps | `[ ]` Exclude |

---

### 3.2 `configs/`

| File | Recommendation | Proposed Action / Reason | User Decision |
| :--- | :--- | :--- | :--- |
| `configs/constants.py` | **Keep & Refactor** | Core paths, special tokens, default constants | `[x]` Keep |
| `configs/prompts.py` | **Keep & Refactor** | Clean prompt templates for different speaker models | `[x]` Keep |
| `configs/speaker_configs.py` | **Keep & Refactor** | Clean speaker configurations | `[x]` Keep |
| `configs/listener_configs.py` | **Keep & Refactor** | Clean listener configurations | `[x]` Keep |
| `configs/reward_configs.py` | **Keep & Refactor** | Modular reward configuration definitions | `[x]` Keep |
| `configs/lora_configs.py` | **Keep & Refactor** | LoRA target modules & PEFT settings | `[x]` Keep |
| `configs/dataset_eval_configs.py` | **Keep & Refactor** | Evaluation dataset parameters | `[x]` Keep |
| `configs/training_configs.py` (70KB) | **Refactor & Trim** | Replace massive dictionary permutations with clear preset dataclasses | `[x]` Refactor |
| `configs/baseline_configs.py` | **Refactor** | Presets for SFT / vanilla baselines | `[x]` Keep |
| `configs/sweep_configs.py` | **Optional** | Hyperparameter sweep configs (keep only if publishing sweeps) | `[ ]` Optional |

---

### 3.3 `models/`

| File | Recommendation | Proposed Action / Reason | User Decision |
| :--- | :--- | :--- | :--- |
| `models/speakers/molmo_speaker.py` | **Keep & Refactor** | Molmo generation, logprob extraction, LoRA integration | `[x]` Keep |
| `models/speakers/paligemma_speaker.py`| **Keep & Refactor** | PaliGemma generation and logprob extraction | `[x]` Keep |
| `models/speakers/llava_speaker.py` | **Keep & Refactor** | LLaVA generation and logprob extraction | `[x]` Keep |
| `models/listeners/gaze_predictor_vectorized.py` | **Keep & Refactor** | High-performance batched gaze predictor listener | `[x]` Keep |
| `models/listeners/gaze_predictor.py` | **Consolidate** | Legacy unvectorized version; merge with vectorized or remove | `[ ]` Consolidate |
| `models/listeners/molmo_listener_vectorized.py` | **Keep & Refactor** | Molmo-based listener for grounding | `[x]` Keep |
| `models/listeners/molmo_listener.py` | **Consolidate** | Unvectorized legacy Molmo listener | `[ ]` Consolidate |

---

### 3.4 `rl_components/`

| File | Recommendation | Proposed Action / Reason | User Decision |
| :--- | :--- | :--- | :--- |
| `rl_components/reward.py` (45KB) | **Keep & Refactor** | Modularize reward computation (sparse, gaussian, distance, penalties) | `[x]` Keep |
| `rl_components/loss.py` | **Keep & Refactor** | Policy loss, advantage normalization, clipping | `[x]` Keep |
| `rl_components/utils.py` | **Keep & Refactor** | Spatial calculations, coordinate transformations | `[x]` Keep |

---

### 3.5 `training/`

| File | Recommendation | Proposed Action / Reason | User Decision |
| :--- | :--- | :--- | :--- |
| `training/trainer.py` (154KB) | **Refactor & Split** | Decompose monolithic engine into clean rollout, optimization, and training loops | `[x]` Refactor |
| `training/base_trainer.py` | **Keep & Refactor** | Abstract base trainer definition | `[x]` Keep |
| `training/checkpoint.py` | **Keep & Refactor** | Checkpoint save/resume with LoRA | `[x]` Keep |
| `training/logging_utils.py` | **Keep & Refactor** | WandB & metrics logging | `[x]` Keep |
| `training/optimization.py` | **Keep & Refactor** | Optimizer, LR schedulers, gradient accumulation | `[x]` Keep |
| `training/validator.py` | **Keep & Refactor** | In-training validation loop | `[x]` Keep |
| `training/memory.py` | **Keep & Refactor** | GPU memory optimization & cleanup | `[x]` Keep |
| `training/baseline.py` | **Keep & Refactor** | Supervised baseline trainer | `[x]` Keep |
| `training/config.py` | **Consolidate** | Merge into unified `configs/` schema | `[ ]` Consolidate |

---

### 3.6 `evaluation/`

| File | Recommendation | Proposed Action / Reason | User Decision |
| :--- | :--- | :--- | :--- |
| `evaluation/evaluate_generated.py` | **Keep & Refactor** | Core evaluation metric computation (grounding + language) | `[x]` Keep |
| `evaluation/generate_impl.py` (85KB) | **Refactor & Clean** | Refactor reference generation pipeline | `[x]` Keep |
| `evaluation/metrics.py` | **Keep & Refactor** | Spatial metrics (hit rate, distance, IoU) | `[x]` Keep |
| `evaluation/qualitative.py` | **Keep & Refactor** | Qualitative visualization and error inspection | `[x]` Keep |
| `evaluation/run.py` / `evaluate.py` | **Consolidate** | Clean single CLI entry point for evaluation | `[x]` Consolidate |
| `evaluation/cogvlm_test.py` / `cogvlm_batched_inference.py` | **Optional** | External CogVLM baseline evaluation scripts | `[ ]` Optional |
| `evaluation/evaluate_gold.py` / `generate_gold.py` | **Consolidate** | Gold reference baseline evaluation | `[x]` Keep/Consolidate |
| `evaluation/analyze.py` | **Consolidate** | Move to analysis tools | `[ ]` Consolidate |

---

### 3.7 `utils/`

| File | Recommendation | Proposed Action / Reason | User Decision |
| :--- | :--- | :--- | :--- |
| `utils/processor.py` | **Keep & Refactor** | Multi-modal processor abstraction | `[x]` Keep |
| `utils/dataset_splits.py` | **Keep & Refactor** | Loading split JSONs and images | `[x]` Keep |
| `utils/dataset_configs.py` | **Keep & Refactor** | Dataset path resolution | `[x]` Keep |
| `utils/image_preprocessing_molmo.py` | **Keep & Refactor** | Image preprocessing routines | `[x]` Keep |

---

### 3.8 `scripts/` (52 Scripts Triage)

Most scripts in `gazeRL/scripts/` are one-off plotters, local sbatch variants, or historical debug routines.

| Category | Files | Recommendation | User Decision |
| :--- | :--- | :--- | :--- |
| **Main Run Scripts** | `train_modular.py`, `run_gazerl.py` | **Consolidate** into `scripts/train.py` and `scripts/eval.py` | `[x]` Consolidate |
| **Clean Sbatch Templates** | `run_parallel_configs_sbatch.sh`, `finetune_refcoco.sbatch` | **Keep 1-2 generic sbatch templates** in `scripts/sbatch/` | `[x]` Keep clean template |
| **Redundant Sbatch Scripts** (10+ variants) | `run_3_instances_*.sh`, `run_restart_debug_sbatch.sh`, etc. | **Delete / Exclude** | `[ ]` Exclude |
| **Plotting & Figure Scripts** (15+ scripts) | `plot_d_neg_barplot.py`, `plot_strategy.py`, `plot_semantic_alignment.py`, `generate_efficiency_*.py`, etc. | **Consolidate** into `scripts/plotting/` or keep only paper figures | `[ ]` Select paper plots |
| **Debug & Comparison Scripts** (15+ scripts) | `compare_processors.py`, `test_wandb_connection.py`, `sample_gaze_paths_refcoco.py`, `find_checkpoint.py`, etc. | **Delete / Exclude** | `[ ]` Exclude |
| **Dataset Prep Scripts** | `split_coco.py`, `split_refcoco.py` | **Keep & Clean** in `scripts/data_prep/` | `[x]` Keep |

---

### 3.9 `data_splits/`

| File | Recommendation | Proposed Action | User Decision |
| :--- | :--- | :--- | :--- |
| `refcoco_train_ids.json`, `refcoco_val_ids.json` | **Keep** | Standard splits for RefCOCO experiments | `[x]` Keep |
| `coco_2014_train_ids.json`, `coco_2014_val_ids.json` | **Keep** | Standard COCO 2014 splits | `[x]` Keep |
| `coco_2017_train_ids.json`, `coco_2017_val_ids.json` | **Keep** | Standard COCO 2017 splits | `[x]` Keep |
| `coco_train_ids_debug*.json`, `*_old.json` | **Delete / Exclude** | Temporary debug splits | `[ ]` Exclude |

---

## 4. Smoke Testing Strategy (Throughout Refactoring)

To guarantee that modularization and refactoring do not break functionality, we will implement lightweight, automated smoke tests at each milestone:

| Milestone / Subsystem | Smoke Test Script / Target | What It Verifies | Execution Time |
| :--- | :--- | :--- | :--- |
| **1. Data & Preprocessing** | `tests/test_data_smoke.py` | RefCOCO / COCO dataset loading, split integrity, bounding box normalization, multi-modal prompt formatting on a dummy 2-sample batch | < 2 seconds |
| **2. Speaker Forward & Rollout** | `tests/test_speakers_smoke.py` | Model instantiation, LoRA attachment, `.generate()` output format, forward logprob extraction on CPU / single GPU | < 10 seconds |
| **3. Listener Grounding** | `tests/test_listeners_smoke.py` | Vectorized gaze predictor & Molmo listener inference on dummy image + text tensor, verifying spatial output format & heatmap dimensions | < 5 seconds |
| **4. Reward Calculations** | `tests/test_rewards_smoke.py` | Unit tests for all renamed reward calculators: binary hit, Gaussian probability mass shaping ($\Delta\Phi$), distance shaping, and logit penalties on synthetic inputs | < 2 seconds |
| **5. Mini-RL Training Step** | `tests/test_train_step_smoke.py` | 1 rollout episode + 1 optimization gradient update step (`--smoke-test`), verifying loss computation, backward pass, optimizer step, and memory cleanup | < 20 seconds |
| **6. Checkpoint Save / Resume** | `tests/test_checkpoint_smoke.py` | Saving LoRA adapter weights, loading into clean model instance, resuming training state deterministically | < 5 seconds |
| **7. Evaluation Pipeline** | `tests/test_eval_smoke.py` | Generating referring expressions on 5 sample images, calculating grounding metrics and language metrics (CIDEr / BLEU) | < 15 seconds |
| **8. Qualitative Visualizer** | `tests/test_viewer_smoke.py` | Running the trajectory viewer / heatmap visualizer on a sample output without UI crashes | < 5 seconds |

---

## 5. Run Scripts Redesign & User Experience (Easy for Everyone)

The original codebase had 50+ divergent scripts with hardcoded paths. The new release will provide a clean, standardized user experience:

### 5.1 Unified Entry Points
Instead of separate scripts per architecture or mode, users only interact with two main entrypoints:
1. **`train.py`**:
   ```bash
   # Quick preset run (YAML configuration)
   python scripts/train.py --config configs/presets/gazerl_molmo_gaussian.yaml

   # Or CLI overrides
   python scripts/train.py --speaker molmo-7b --listener gaze --reward gaussian_shaping --epochs 3

   # Instant verification flag (runs 2 episodes on dummy data to ensure environment works)
   python scripts/train.py --smoke-test
   ```
2. **`evaluate.py`**:
   ```bash
   # Evaluate a trained checkpoint on RefCOCO validation split
   python scripts/evaluate.py --checkpoint checkpoints/best_model --dataset refcoco --split val

   # Evaluate baseline (e.g. gold captions or zero-shot speaker)
   python scripts/evaluate.py --model molmo-7b --mode zero-shot --dataset refcoco
   ```

### 5.2 Streamlined SLURM Templates (`scripts/sbatch/`)
Provide clean, commented, and parameterized SLURM templates:
- `submit_train.sbatch`: Single-node GPU training launcher accepting preset configs or command-line arguments.
- `submit_eval.sbatch`: Batch evaluation launcher.

### 5.3 Path Independence & Portability
- **No hardcoded paths**: All dataset paths, checkpoint directories, and output caches will use environment variables (e.g. `GAZERL_DATA_DIR`, `GAZERL_OUTPUT_DIR`) with sensible defaults (`./data`, `./outputs`).
- **Standardized Packaging**: `pyproject.toml` enables simple one-command installation:
  ```bash
  pip install -e .
  ```

---

## 6. Next Steps

1. **Review & Edit**: Open this file and check/uncheck items or add notes under Section 1 (Checklist), Section 1.3 (Reward Renaming), and Section 3 (File Triage).
2. **Confirm Scope**: Let me know when you've updated the spec or if you want me to proceed with the proposed defaults.
3. **Execution**: We will construct the clean directory structure in `gazeRL_publish/`, refactor each component incrementally, and write smoke tests to ensure complete compatibility.

