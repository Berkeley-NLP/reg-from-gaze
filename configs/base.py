"""
Configuration schemas and serialization for GazeRL experiments.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import yaml

from configs.constants import (
    DEFAULT_MOLMO_MODEL,
    DEFAULT_GAZE_PREDICTOR_WEIGHTS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_VALIDATION_BATCH_SIZE,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_NUM_EPISODES,
    DEFAULT_VALIDATION_FREQUENCY,
    DEFAULT_PERIODIC_CHECKPOINT_INTERVAL,
    DEFAULT_KL_COEF,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_OUTPUT_DIR,
)


@dataclass
class SpeakerConfig:
    """Configuration for referring expression generator (speaker)."""
    model_name_or_path: str = DEFAULT_MOLMO_MODEL
    architecture: str = "molmo"  # 'molmo', 'paligemma', 'llava'
    prompt_type: str = "all"      # 'all', 'detailed', 'brief'
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    torch_dtype: str = "bfloat16"


@dataclass
class ListenerConfig:
    """Configuration for listener comprehension model."""
    model_name_or_path: str = DEFAULT_GAZE_PREDICTOR_WEIGHTS
    listener_type: str = "gaze_predictor"  # 'gaze_predictor', 'molmo_iterative', 'rec_single_point', 'eval_qwenvl', 'eval_cogvlm'
    prediction_level: str = "token"        # 'token', 'word'
    strip_punctuation: bool = False
    torch_dtype: str = "bfloat16"


@dataclass
class RewardConfig:
    """Configuration for reward calculation."""
    reward_type: str = "sparse_decay"  # 'sparse_decay', 'distance_shaping', 'binary', 'binary_last_point', 'supervised', 'gaussian_shaping'
    gamma: float = 0.9                 # Backward decay discount factor for sparse reward
    hit_reward: float = 1.0
    miss_reward: float = 0.0
    eos_bonus: float = 0.0
    shaping_scale: float = 1.0
    use_length_penalty: bool = False
    length_penalty_beta: float = 0.01
    logit_penalty_weight: float = 0.0  # GRPO-style logit L2 penalty (0.0 = disabled)
    gaussian_sigma_default: float = 10.0
    zero_reward_for_punctuation: bool = False
    word_reward_mode: str = "all"      # 'all', 'normalize', 'first_only', 'last_only'
    ignore_stop_words: bool = False
    eos_reward_on_success: float = 0.0
    shaping_reward_all_tokens: bool = False
    gaussian_threshold_sparse: Optional[float] = None
    gaussian_use_absolute_prob_mass: bool = False


@dataclass
class LoRAConfig:
    """Configuration for parameter-efficient fine-tuning (LoRA)."""
    enabled: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and split resolution."""
    dataset_name: str = "coco_2014"    # 'refcoco', 'coco_2014', 'coco_2017'
    splits_dir: str = "data/splits"
    use_existing_splits: bool = True
    train_split: str = "train"
    val_split: str = "val"
    max_samples: Optional[int] = None   # Limit dataset size (useful for debugging/smoke test)


@dataclass
class TrainingConfig:
    """Configuration for RL training optimization and runtime."""
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    validation_batch_size: int = DEFAULT_VALIDATION_BATCH_SIZE
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    num_episodes: int = DEFAULT_NUM_EPISODES
    validation_frequency: int = DEFAULT_VALIDATION_FREQUENCY
    periodic_checkpoint_interval: int = DEFAULT_PERIODIC_CHECKPOINT_INTERVAL
    kl_coef: float = DEFAULT_KL_COEF
    entropy_coef: float = DEFAULT_ENTROPY_COEF
    lr_schedule: str = "constant"       # 'constant', 'cosine', 'linear'
    output_dir: str = DEFAULT_OUTPUT_DIR
    seed: int = 42
    wandb: bool = False
    wandb_project: str = "reg-from-gaze"
    wandb_entity: Optional[str] = os.environ.get("WANDB_ENTITY", None)
    wandb_run_name: Optional[str] = None
    smoke_test: bool = False


@dataclass
class ExperimentConfig:
    """Complete experiment specification."""
    name: str = "gaze_bfh"
    description: str = "Default GazeRL experiment"
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    listener: ListenerConfig = field(default_factory=ListenerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        speaker = SpeakerConfig(**data.get("speaker", {}))
        listener = ListenerConfig(**data.get("listener", {}))
        reward = RewardConfig(**data.get("reward", {}))
        lora = LoRAConfig(**data.get("lora", {}))
        dataset = DatasetConfig(**data.get("dataset", {}))
        training = TrainingConfig(**data.get("training", {}))
        return cls(
            name=data.get("name", "experiment"),
            description=data.get("description", ""),
            speaker=speaker,
            listener=listener,
            reward=reward,
            lora=lora,
            dataset=dataset,
            training=training,
        )

    def save_yaml(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
