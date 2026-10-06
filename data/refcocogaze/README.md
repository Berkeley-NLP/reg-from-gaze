# RefCOCO-Gaze Data Specification & Processing

This directory contains the dataset files used to train and evaluate the **Molmo-REC-Gaze** listener model, adapted from **RefCOCO-Gaze** ([Mondal et al., 2024](https://arxiv.org/abs/2403.01166)).

---

## 1. File Inventory

| File | Description | Split | Records |
| :--- | :--- | :--- | :--- |
| `refcocogaze_train_delay.json` | Training scanpaths with **+200ms audio latency shift** | Train | 16,982 scanpaths (1,799 examples) |
| `refcocogaze_val_delay.json` | Validation scanpaths with **+200ms audio latency shift** | Val | 869 scanpaths (92 examples) |
| `refcocogaze_val_correct.json` | Human gold validation scanpaths (used as ground truth for DTW and REC metrics) | Val | 869 scanpaths (92 examples) |
| `word-timing.json` | Spoken referring expression audio onset timestamps per word | Train/Val | Aligned timestamps |
| `art_checkpoint.json` | Pre-computed predictions from the ART baseline model ([Mondal et al., 2024](https://arxiv.org/abs/2403.01166)) | Val | Baseline comparisons |

---

## 2. Audio Latency Processing (+200ms)

To regenerate `refcocogaze_train_delay.json` and `refcocogaze_val_delay.json` from the original RefCOCO-Gaze annotations:

```bash
python scripts/preprocess_data.py \
    --train_in data/refcocogaze_train_correct.json \
    --val_in data/refcocogaze_val_correct.json \
    --timing data/word-timing.json \
    --train_out data/refcocogaze_train_delay.json \
    --val_out data/refcocogaze_val_delay.json \
    --latency_ms 200
```

### Alignment Procedure:
1. **Auditory Latency Delay**: Spoken word timestamps $t_{\text{onset}}(w_k)$ are shifted by $+200\text{ms}$ ($t_{\text{effective}}(w_k) = t_{\text{onset}}(w_k) + 0.200\text{s}$) to model human auditory and cognitive processing delay (Kirchner & Thorpe, 2006).
2. **Multi-Fixation Resolution**: If multiple fixations occur during the window between word $w_k$ and word $w_{k+1}$, the **final fixation** before $t_{\text{effective}}(w_{k+1})$ is selected.
3. **Null Fixations**: Rapid speech without fixations maps to a null coordinate (`<x= y=>` or `None`).
4. **Initial Fixation $g_0$**: Paired with `BOS`, representing pre-audio gaze.
5. **Final Fixation $g_T$**: Paired with `EOS`, landing inside the target referent bounding box ($g_T \in b$).
