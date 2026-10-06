# Data Splits & Dataset Management in GazeRL

This document details the exact dataset splits used in [*Learning to Refer from Estimated Listener Gaze*](https://arxiv.org/abs/2609.14207) and explains how to generate custom deterministic splits for new datasets.

---

## 1. Included Canonical Splits

All deterministic split IDs used across training and evaluation are tracked in [`data/splits/`](../data/splits/):

| Split File | Dataset | Image Count | Description |
| :--- | :--- | :---: | :--- |
| `coco_2014_train_ids.json` | COCO 2014 | 68,636 | Main speaker training split (RL & SFT) |
| `coco_2014_val_ids.json` | COCO 2014 | 3,611 | Validation split for checkpoint selection |
| `refcoco_train_ids.json` | RefCOCO | 10,339 | RefCOCO training set partition |
| `refcoco_val_ids.json` | RefCOCO | 543 | RefCOCO validation set partition |
| `coco_2017_train_ids.json` | COCO 2017 | 9,410 | COCO 2017 training benchmark partition |
| `coco_2017_val_ids.json` | COCO 2017 | 494 | COCO 2017 validation benchmark partition |

### Listener Training Dataset (RefCOCO-Gaze)
Located in [`data/refcocogaze/`](../data/refcocogaze/):
- **`refcocogaze_train_correct.json`** & **`refcocogaze_val_correct.json`**: Ground-truth human eye-tracking fixations aligned with spoken audio.
- **`refcocogaze_train_delay.json`** & **`refcocogaze_val_delay.json`**: Audio-latency shifted fixations (+200 ms) used to train the canonical `Molmo-REC-Gaze` listener.
- **`word-timing.json`**: Subword and word boundary acoustic timestamps from forced alignment.

---

## 2. Using Existing Splits in Python

The [`DatasetSplitManager`](../data/datasets.py) handles deterministic partition loading:

```python
from data.datasets import DatasetSplitManager

# Initialize split manager pointing to data/splits directory
split_mgr = DatasetSplitManager("data/splits")

# Load train and validation IDs
train_ids, val_ids = split_mgr.load_splits("coco_2014")
print(f"Loaded {len(train_ids)} train images and {len(val_ids)} val images.")
```

---

## 3. Creating Custom Splits for New Datasets

To split your own custom dataset into reproducible train/validation partitions:

```python
from data.datasets import DatasetSplitManager

split_mgr = DatasetSplitManager("data/splits")

all_image_ids = ["img_001.jpg", "img_002.jpg", "img_003.jpg", ...]

# Generates my_dataset_train_ids.json, my_dataset_val_ids.json, and metadata
train_ids, val_ids = split_mgr.create_custom_split(
    dataset_name="my_dataset",
    all_ids=all_image_ids,
    train_ratio=0.90,  # 90% train, 10% validation
    seed=42,           # Deterministic random seed
)
```

This creates:
- `data/splits/my_dataset_train_ids.json`
- `data/splits/my_dataset_val_ids.json`
- `data/splits/my_dataset_split_metadata.json` (records timestamp, count, seed, and train ratio)
