# AGENT INSTRUCTION & REPRODUCIBILITY SPECIFICATION: Gaze Prediction Training (Molmo-REC-Gaze)

> **Target Goal**: Train and package the **incremental listener gaze prediction model (`Molmo-REC-Gaze`)** on a remote/clean machine, and export it for seamless integration into the **GazeRL** repository (`reg-from-gaze`).  
> **Paper Reference**: [*Learning to Refer from Estimated Listener Gaze* (arXiv:2609.14207, Wright et al., 2026)](https://arxiv.org/abs/2609.14207)  
> **Target Consumer Repository**: `https://github.com/Berkeley-NLP/reg-from-gaze.git` (`gazeRL_publish`)

---

## 1. Executive Summary & Objective

In GazeRL, speaker models (Molmo, PaliGemma, LLaVA) are trained with reinforcement learning by communicating with an incremental listener model. This listener model, **Molmo-REC-Gaze**, maps an input image $\mathcal{I}$ and an unfolding referring expression sequence $\langle x_1, \dots, x_t \rangle$ to a sequence of visual fixations $\langle g_1, \dots, g_t \rangle$.

You are rebuilding/retraining the **gaze prediction training repository** on another machine. Once training is complete, the resulting model directory will be transferred back to this machine and plugged into `gazeRL_publish`.

This document specifies:
1. **Shared Contract & Data Interfaces**: Exact image sizes, letterbox padding, coordinate frames, point syntax, and prompt schemas expected by `gazeRL_publish`.
2. **Paper Specification & Data Alignment**: Dataset preparation from RefCOCO-Gaze, the 200ms audio latency shift, multi-fixation resolution, and subword label expansion.
3. **Training Hyperparameters**: Exact optimizer, batch size, learning rate, schedule, and precision used in the paper.
4. **Export & Verification Bundle**: Checkpoint structure required for zero-error drop-in loading via Hugging Face `AutoModelForCausalLM` and `AutoProcessor`.

---

## 2. Shared Interface Contract (Between Gaze Predictor and GazeRL)

The GazeRL repository interacts with the gaze predictor via `models/listeners/gaze_predictor.py` ([`GazePredictorVectorized`](file:///accounts/projects/berkeleynlp/teaywright/gazeRL_publish/models/listeners/gaze_predictor.py)). The trained model must adhere strictly to these conventions:

### 2.1 Image Dimensions & Letterboxing
- **Dimensions**: Canvas is strictly **$512 \times 320$** pixels (`width=512, height=320`).
- **Aspect Ratio Padding**: The input image must be scaled to fit within $512 \times 320$ preserving aspect ratio, centered against a solid black background `RGB(0, 0, 0)`:
  ```python
  scale = min(512 / orig_w, 320 / orig_h)
  new_w, new_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
  resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
  padded = Image.new("RGB", (512, 320), color="black")
  pad_x = (512 - new_w) // 2
  pad_y = (320 - new_h) // 2
  padded.paste(resized, (pad_x, pad_y))
  ```
- **No Overlays**: Unlike the speaker policy which sees a red bounding box around the target referent, the listener image **never receives any bounding box annotation or visual hint**.

### 2.2 Coordinate Space ($0 - 100$ Normalized Scale)
- All coordinates $(x, y)$ are represented on a continuous **$0.0$ to $100.0$** normalized scale relative to the $512 \times 320$ canvas:
  $$x_{\text{norm}} = \frac{x_{\text{canvas}}}{512} \times 100.0, \quad y_{\text{norm}} = \frac{y_{\text{canvas}}}{320} \times 100.0$$
- Coordinates on original images must be projected through the letterbox transform:
  $$x_{\text{canvas}} = x_{\text{orig}} \times \text{scale} + \text{pad}_x, \quad y_{\text{canvas}} = y_{\text{orig}} \times \text{scale} + \text{pad}_y$$

### 2.3 Point Coordinate Syntax & Parsing
Standard Molmo outputs XML-like point tags (`<point x="50.0" y="50.0" alt="cat">cat</point>`).  
For GazeRL, the format is simplified to:
- **Point Fixation**: `<x=50.0 y=50.0>` (floating point, 1-2 decimals, single space between $x$ and $y$).
- **Gaussian Fixation** (optional uncertainty ablation): `<x=50.0 y=50.0 v=10.5>` where $v$ is the standard deviation $\sigma \in [0, 100]$ of an isotropic 2D Gaussian.
- **Null / Missing Fixation**: `<x= y=>` (or omitted).
- **Regex expected by GazeRL parser** (`data/processing.py`):
  ```python
  re.search(r'<x=(\d+(?:\.\d+)?)\s+y=(\d+(?:\.\d+)?)>', text, re.IGNORECASE)
  re.search(r'<x=(\d+(?:\.\d+)?)\s+y=(\d+(?:\.\d+)?)\s+v=(\d+(?:\.\d+)?)>', text, re.IGNORECASE)
  ```

### 2.4 Prompt & Autoregressive Sequence Structure
Inference in `GazePredictorVectorized` generates gaze fixations iteratively token-by-token using `processor.process(..., message_format="none")`.

The sequence alternates between referring expression words/tokens and predicted gaze coordinates:
```text
Step 0 (Pre-audio / BOS):
Prompt:  "BOS"
Output:  "<x=50.2 y=48.9>" (or random center fixation)

Step 1 (First word heard):
Prompt:  "BOS <x=50.2 y=48.9> The"
Output:  "<x=48.1 y=45.2>"

Step 2 (Second word heard):
Prompt:  "BOS <x=50.2 y=48.9> The <x=48.1 y=45.2> striped"
Output:  "<x=32.0 y=60.4>"

...

Final Step (EOS):
Prompt:  "BOS <x=50.2 y=48.9> The <x=48.1 y=45.2> striped <x=32.0 y=60.4> cat EOS"
Output:  "<x=35.4 y=65.2>"  <-- Final fixation in target bounding box
```

---

## 3. Dataset & Preprocessing Pipeline (RefCOCO-Gaze)

### 3.1 Data Source
- **Base Dataset**: `RefCOCO-Gaze` (introduced by Mondal et al., 2024, *Look Hear: Gaze Prediction for Speech-Directed Human Attention*).
- **Underlying Images**: RefCOCO images (subset of MS-COCO 2014 train).
- **Dataset Sizes**:
  - **Training set $D_T$**: $16,982$ scanpaths across $1,799$ RefCOCO examples.
  - **Validation set $D_V$**: $869$ scanpaths across $92$ RefCOCO examples.
  *(Note: The test set is withheld / not publicly released in RefCOCO-Gaze; validation set loss determines model checkpoint selection).*

### 3.2 Word & Fixation Temporal Alignment Procedure
RefCOCO-Gaze provides:
1. Audio recordings of human referring expressions with word onset timestamps.
2. Infrared eye-tracker scanpaths of human participants with fixation timestamps $(x, y, t)$.

To construct training pairs $(x^{(i)}, g^{(i)})$ where $|x^{(i)}| = |g^{(i)}|$:
1. **Auditory Latency Delay (200ms)**:
   Add a **+200ms delay** to the timestamp of each spoken word to model human auditory and cognitive processing delay (Kirchner & Thorpe, 2006):
   $$t_{\text{effective}}(w_k) = t_{\text{onset}}(w_k) + 0.200\,\text{seconds}$$
2. **Multi-fixation Resolution**:
   If a participant made multiple fixations during the window between word $w_k$ and word $w_{k+1}$, select the **final fixation** before $t_{\text{effective}}(w_{k+1})$.
3. **Null Fixations**:
   If no fixations occurred during the word window (e.g. very rapid speech), assign a null coordinate $\emptyset$ (represented in string formatting as `<x= y=>` or omitted).
4. **Initial Fixation $g_0$**:
   Paired with `BOS`. Represents participant gaze fixated near the center before audio onset.
5. **Final Fixation $g_T$**:
   Paired with `EOS`. In RefCOCO-Gaze demonstrations, all scanpaths correspond to successful comprehension, so $g_T \in b$ (inside the target referent bounding box).
6. **Subword Token Expansion**:
   If the tokenizer splits word $w_k$ into multiple subwords (e.g. `"striped"` $\to$ `["strip", "ed"]`), **assign each subword token the exact same fixation coordinate label**.

---

## 4. Model Architecture & Training Hyperparameters

The configuration below corresponds to the exact setup used to produce the canonical `gaze_predictor_delay_token_116` checkpoint:

### 4.1 Base Model
- **Identifier**: `allenai/Molmo-7B-D-0924`
- **Architecture**: Single-stream multimodal transformer with vision backbone (ViT) and language model.

### 4.2 Training Hyperparameters (Verified)
| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Base Architecture** | `allenai/Molmo-7B-D-0924` | Hugging Face Transformers |
| **Epochs** | `4` | Full dataset passes |
| **Optimizer** | `paged_adamw_8bit` | Enables 7B fine-tuning on a single GPU |
| **Precision** | `bfloat16` (`bf16=True`) | Standard for Ampere/Hopper (A100/H100) |
| **Learning Rate** | `1e-5` (`1.0e-5`) | Stable fine-tuning LR |
| **LR Scheduler** | `cosine` | Cosine annealing schedule |
| **Warmup Ratio** | `0.03` | 3% linear warmup |
| **Per-Device Batch Size** | `1` | 1 sequence per step |
| **Gradient Accumulation** | `8` | Effective batch size = 8 |
| **Weight Decay** | `0.01` | Regularization |
| **Max Gradient Norm** | `1.0` | Gradient clipping threshold |
| **Eval Strategy** | `steps` every `100` steps | Evaluates on $D_V$ (869 scanpaths) |
| **Model Selection Metric** | `eval_loss` | Save checkpoint with lowest validation loss |
| **Seed** | `42` | Reproducibility |
| **Sequence Length** | `1024` tokens | Sufficient for full scanpath sequences |

### 4.3 Loss Function
Standard autoregressive language modeling cross-entropy loss over the generated coordinate tokens:
$$\mathcal{L}(\phi) = - \sum_{t=1}^T \log \pi_\phi^{\text{G}}(g_t \mid \mathcal{I}, x_{\le t}, g_{< t})$$

---

## 5. Evaluation & Validation Targets (Paper Alignment)

When evaluating the trained checkpoint on the RefCOCO-Gaze validation set ($D_V$), verify alignment against the published metrics in the paper:

| Evaluation Metric | Target / Paper Value | Mondal et al. (ART) Baseline | Description |
| :--- | :--- | :--- | :--- |
| **Path Distance (DTW)** | **~39.9** | 55.9 | Dynamic Time Warping distance to closest human scanpath |
| **REC Accuracy (%)** | **~84.4%** | 66.7% | Percentage of final fixations landing inside target bbox |

*During training, log step metrics to `dtw_metrics.txt` in the format:*
```csv
step,dtw_distance,bbox_accuracy
100,64.17,0.20
...
2100,55.44,0.93
```

---

## 6. Checkpoint Export Bundle & File Checklist

Once training is complete, the exported model directory must contain the complete set of weights, configs, and Molmo processor code files so it can be loaded with zero modifications:

```text
exported_gaze_predictor/
├── added_tokens.json             # Molmo special tokens (<|image|>, <im_start>, etc.)
├── config.json                   # Model architecture configuration
├── config_molmo.py               # Molmo remote model definition
├── generation_config.json        # Default generation parameters
├── image_preprocessing_molmo.py  # Standalone Molmo image transform
├── merges.txt                    # BPE tokenizer merges
├── model.safetensors.index.json  # Safetensors sharding index
├── model-00001-of-00004.safetensors
├── model-00002-of-00004.safetensors
├── model-00003-of-00004.safetensors
├── model-00004-of-00004.safetensors
├── preprocessing_molmo.py        # Tokenizer/processor helper
├── preprocessor_config.json      # Image preprocessor configuration
├── processor_config.json         # Multimodal processor configuration
├── special_tokens_map.json       # Special token definitions
├── tokenizer_config.json         # Tokenizer parameters
├── tokenizer.json                # Complete fast tokenizer
├── training_args.bin             # PyTorch saved training arguments
├── training_loss.txt             # Training loss log
├── eval_loss.txt                 # Validation loss log
├── dtw_metrics.txt               # Evaluation DTW / Accuracy log
└── vocab.json                    # Vocabulary dictionary
```

---

## 7. Integration Back Into GazeRL (`gazeRL_publish`)

Once the directory is copied back to the machine hosting `gazeRL_publish`:

### 7.1 Testing Drop-In Loading
Verify that the model can be loaded by `GazePredictorVectorized`:
```bash
python -c "
from models.listeners.gaze_predictor import GazePredictorVectorized
listener = GazePredictorVectorized(model_path='/path/to/exported_gaze_predictor')
print('Successfully loaded Molmo-REC-Gaze listener!')
"
```

### 7.2 Pointing Presets or Environment Variable
Either update `configs/presets/*.yaml` or set the global environment variable:
```bash
export GAZERL_GAZE_PREDICTOR_PATH="/path/to/exported_gaze_predictor"
```

### 7.3 Verifying GazeRL Test Suite
Run the preflight checks and smoke tests:
```bash
# Run listener unit tests
pytest tests/test_listeners_smoke.py

# Run preset validation (verifies path resolution and configs)
pytest tests/test_presets_validation.py

# Run full test suite (all 59 tests)
pytest tests/
```

### 7.4 Running a Verification Episode
Run a short 10-episode rollout with the newly trained listener:
```bash
python scripts/train.py \
    --config configs/presets/gaze_bfh.yaml \
    --listener-path /path/to/exported_gaze_predictor \
    --episodes 10 \
    --batch-size 2
```

---

## 8. Summary Checklist for the Remote Machine Agent

- [ ] Clone / install environment with PyTorch ($\ge 2.2$), `transformers>=4.45`, `peft`, `accelerate`, `bitsandbytes` (for `paged_adamw_8bit`).
- [ ] Download `allenai/Molmo-7B-D-0924` base weights.
- [ ] Acquire RefCOCO-Gaze dataset annotations and RefCOCO images.
- [ ] Implement data preparation with **+200ms audio latency shift** and letterboxed $512 \times 320$ images.
- [ ] Format prompt sequences as `BOS <x=.. y=..> w1 <x=.. y=..> w2 ...`.
- [ ] Train for 4 epochs using `paged_adamw_8bit`, `lr=1e-5`, `bf16=True`, effective batch size 8.
- [ ] Evaluate every 100 steps; select best checkpoint by `eval_loss` (aiming for DTW $\approx 39.9$, accuracy $\ge 84\%$).
- [ ] Export safetensors and configs as listed in Section 6.
- [ ] Transfer to `gazeRL_publish` and run validation commands in Section 7.
