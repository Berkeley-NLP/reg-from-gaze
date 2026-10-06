"""
reg: Referring Expression Generation with Estimated Listener Gaze (GazeRL).
Contains speaker models, listener policy interfaces, RL rewards, trainer, and evaluation.
"""
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from reg import models, rl, training, evaluation

__all__ = ["models", "rl", "training", "evaluation"]
