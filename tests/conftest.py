"""
Global pytest configuration and module path resolution for GazeRL.
Ensures both top-level modules ('reg' and 'gaze_estimation') and their subpackages
are cleanly discoverable during test execution.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for p in [str(REPO_ROOT), str(REPO_ROOT / "reg"), str(REPO_ROOT / "gaze_estimation")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Backward-compatibility alias for legacy imports
import gaze_estimation
sys.modules["listener"] = gaze_estimation
