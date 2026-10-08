"""
Centralized constants and default values for GazeRL.
"""
import os
from pathlib import Path

# =============================================================================
# ENVIRONMENT & PATH CONSTANTS
# =============================================================================
DEFAULT_DATA_DIR = os.environ.get("GAZERL_DATA_DIR", "./data")
DEFAULT_OUTPUT_DIR = os.environ.get("GAZERL_OUTPUT_DIR", "./outputs")
DEFAULT_CACHE_DIR = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# =============================================================================
# IMAGE PROCESSING CONSTANTS
# =============================================================================
# GazePredictor / listener input dimensions (padded to aspect ratio)
LISTENER_IMAGE_WIDTH = 512
LISTENER_IMAGE_HEIGHT = 320

# Speaker image max dimension (resized while preserving aspect ratio)
SPEAKER_MAX_DIM = 336

# Coordinate scale: gaze predictor predictions and bounding boxes use 0-100 scale
COORDINATE_SCALE = 100.0

# =============================================================================
# DEFAULT MODEL IDENTIFIERS
# =============================================================================
DEFAULT_MOLMO_MODEL = "allenai/Molmo-7B-D-0924"
DEFAULT_PALIGEMMA_MODEL = "google/paligemma-3b-pt-448"
DEFAULT_LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"

# Listener checkpoint paths
DEFAULT_HF_GAZE_PREDICTOR_REPO = "Berkeley-NLP/Molmo-REC-Gaze"
DEFAULT_HF_GAZE_SEQANYHIT_REPO = "Berkeley-NLP/REG-Molmo-Gaze-SeqAnyHit"
DEFAULT_HF_GAZE_SHAPING_REPO = "Berkeley-NLP/REG-Molmo-Gaze-Shaping"

_REPO_ROOT = Path(__file__).resolve().parent.parent

KNOWN_GAZE_PREDICTOR_CANDIDATES = [
    str(_REPO_ROOT / "checkpoints" / "exported_gaze_predictor"),
    str(_REPO_ROOT / "checkpoints" / "gaze_predictor_delay_token_116"),
    "/scratch/users/teaywright/gaze_predictor_delay_token_116/best_checkpoint",
    "/scratch/users/teaywright/gaze_predictor_delay_token_116",
    "/data/teaywright/gaze_predictor_delay_token_1215",
    "/data/teaywright/gaze_predictor_delay_1120",
    "/data/teaywright/gaze_predictor_delay",
]

def resolve_gaze_predictor_path(explicit_path: str = None) -> str:
    """
    Resolve the gaze predictor checkpoint path.
    If explicit_path is provided:
      - Returns it if it exists on disk, or is a Hugging Face repo ID
      - If it is a legacy placeholder name, attempts candidate fallback
      - Otherwise, returns explicit_path as-is
    If explicit_path is None:
      - Checks GAZERL_GAZE_PREDICTOR_PATH environment variable
      - Checks known local candidates on cluster/scratch/export
      - Falls back to default Hugging Face repository "Berkeley-NLP/Molmo-REC-Gaze"
    """
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        if "/" in explicit_path and not explicit_path.startswith(("/", "./", "../")):
            if explicit_path not in ("teaywright/gaze-predictor", "checkpoints/exported_gaze_predictor"):
                return explicit_path

    env_path = os.environ.get("GAZERL_GAZE_PREDICTOR_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # First pass: prefer candidates with model weight files
    for cand in KNOWN_GAZE_PREDICTOR_CANDIDATES:
        cand_p = Path(cand)
        if cand_p.exists() and (list(cand_p.glob("*.safetensors")) or list(cand_p.glob("*.bin"))):
            return cand

    # Second pass: accept candidate directory with config/tokenizers
    for cand in KNOWN_GAZE_PREDICTOR_CANDIDATES:
        if os.path.exists(cand):
            return cand

    return DEFAULT_HF_GAZE_PREDICTOR_REPO

DEFAULT_GAZE_PREDICTOR_WEIGHTS = DEFAULT_HF_GAZE_PREDICTOR_REPO

# Molmo listener checkpoint paths (for REC experiments)
DEFAULT_HF_MOLMO_LISTENER_REPO = DEFAULT_MOLMO_MODEL
KNOWN_MOLMO_LISTENER_CANDIDATES = [
    "/scratch/users/teaywright/molmo_listener_bbox_center",
    DEFAULT_MOLMO_MODEL,
]

def resolve_molmo_listener_path(explicit_path: str = None) -> str:
    """
    Resolve Molmo listener checkpoint path.
    """
    if explicit_path:
        if os.path.exists(explicit_path) or "/" not in explicit_path.lstrip("/"):
            return explicit_path
        if "vectorized" not in explicit_path and explicit_path != DEFAULT_HF_MOLMO_LISTENER_REPO:
            return explicit_path

    env_path = os.environ.get("GAZERL_MOLMO_LISTENER_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    for cand in KNOWN_MOLMO_LISTENER_CANDIDATES:
        if os.path.exists(cand):
            return cand

    return explicit_path or DEFAULT_MOLMO_MODEL

DEFAULT_MOLMO_LISTENER_WEIGHTS = resolve_molmo_listener_path()

# =============================================================================
# TRAINING HYPERPARAMETER DEFAULTS
# =============================================================================
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_OPTIMIZER_EPS = 1e-8
DEFAULT_MAX_GRAD_NORM = 3.0
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8
DEFAULT_BATCH_SIZE = 8
DEFAULT_VALIDATION_BATCH_SIZE = 16
DEFAULT_WARMUP_STEPS = 16
DEFAULT_NUM_EPISODES = 64000
DEFAULT_VALIDATION_FREQUENCY = 2000
DEFAULT_PERIODIC_CHECKPOINT_INTERVAL = 1000

# Regularization & Penalty defaults
DEFAULT_KL_COEF = 0.02
DEFAULT_ENTROPY_COEF = 0.01

# Generation defaults
DEFAULT_MAX_NEW_TOKENS = 20
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0

# Bounding box filtering
MIN_BBOX_AREA_FRACTION = 0.0001

# Memory management
HIGH_MEMORY_THRESHOLD_GB = 45.0
MEDIUM_MEMORY_THRESHOLD_GB = 25.0
MEMORY_CLEANUP_FREQUENCY = 100
