# Learning to Refer from Estimated Listener Gaze

Official implementation of the paper:
**Learning to Refer from Estimated Listener Gaze**  
Téa Wright and Alane Suhr (COLM 2026)  
[arXiv:2609.14207](https://arxiv.org/abs/2609.14207)

---

## Overview

REG-from-Gaze trains Vision-Language Model (VLM) speakers to generate referring expressions by optimizing against an incremental listener model (`Molmo-REC-Gaze`). Instead of learning solely from binary communicative success or static text corpora, we convert word-by-word estimated gaze scanpaths into reinforcement learning rewards.

---

## Installation

```bash
git clone [https://github.com/Berkeley-NLP/reg-from-gaze.git](https://github.com/Berkeley-NLP/reg-from-gaze.git)
cd reg-from-gaze

conda create -n gazerl python=3.10 -y
conda activate gazerl
pip install -e ".[all]"
```

---

## Training

### 1. Speaker RL Training

Train a speaker policy (default: `Molmo-7B-D`) using preset configurations:

```bash
# Gaze-BeforeFirstHit
python scripts/train.py --config configs/presets/gaze_bfh.yaml

# Gaze-Shaping
python scripts/train.py --config configs/presets/gaze_shaping.yaml

# Gaze-SeqAnyHit
python scripts/train.py --config configs/presets/gaze_seq_any_hit.yaml

# REC-Success Baseline
python scripts/train.py --config configs/presets/rec_success.yaml
```

Run a fast end-to-end smoke test:
```bash
python scripts/train.py --smoke-test
```

### 2. Listener Fine-Tuning (Molmo-REC-Gaze)

To train the incremental gaze estimation model on RefCOCO-Gaze (+200ms latency alignment):

```bash
# Preprocess alignments
python scripts/preprocess_gaze.py \
    --train_in data/refcocogaze/refcocogaze_train_correct.json \
    --val_in data/refcocogaze/refcocogaze_val_correct.json \
    --timing data/refcocogaze/word-timing.json \
    --train_out data/refcocogaze/refcocogaze_train_delay.json \
    --val_out data/refcocogaze/refcocogaze_val_delay.json

# Fine-tune Molmo-7B listener
python scripts/train_listener.py --config configs/listener/canonical_molmo_gaze.yaml

# Export listener checkpoint for speaker RL
python scripts/export_listener.py \
    --source checkpoints/gaze_predictor_delay_token_116 \
    --target checkpoints/exported_gaze_predictor
```


### Evaluating Models

```bash
# Evaluate a speaker policy
python scripts/eval.py \
    --checkpoint Berkeley-NLP/REG-Molmo-Gaze-SeqAnyHit \
    --dataset coco_2014 \
    --split val

# Evaluate a listener policy (DTW distance & REC accuracy)
python scripts/eval_listener.py --checkpoint Berkeley-NLP/Molmo-REC-Gaze
```


## Citation

```bibtex
@inproceedings{wright2026learning,
  title={Learning to Refer from Estimated Listener Gaze},
  author={Wright, T{\'e}a and Suhr, Alane},
  booktitle={Conference on Language Modeling (COLM)},
  year={2026},
  url={[https://arxiv.org/abs/2609.14207](https://arxiv.org/abs/2609.14207)}
}
```

## License

This project is licensed under the Apache License 2.0.