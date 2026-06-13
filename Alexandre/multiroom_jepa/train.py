"""Training launcher for multiroom-jepa.

Wraps the EB-JEPA ac_video_jepa training loop.

Usage:
    python train.py --fname cfgs/multiroom_train.yaml
    python train.py --fname cfgs/tworooms_train.yaml
"""

import sys
from pathlib import Path

# Expose the cloned eb_jepa as a top-level package
_eb_jepa_root = Path(__file__).parent / "eb_jepa"
if str(_eb_jepa_root) not in sys.path:
    sys.path.insert(0, str(_eb_jepa_root))

# Expose environments/ as a top-level package
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import fire
from examples.ac_video_jepa.main import run

if __name__ == "__main__":
    fire.Fire(run)
