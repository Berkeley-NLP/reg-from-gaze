"""
Training loop, Trainer construction, and callbacks for Molmo-REC-Gaze.
"""

from gaze_estimation.training.callbacks import (
    CombinedEvalCallback,
    DTWLoggerCallback,
    BestCheckpointCallback,
)
from gaze_estimation.training.trainer import (
    create_training_arguments,
    create_trainer,
)
