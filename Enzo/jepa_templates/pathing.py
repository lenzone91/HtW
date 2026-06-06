"""Import helpers for the EB-JEPA git submodule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure_eb_jepa_importable(repo_root: str | Path | None = None) -> Path:
    """Add the EB-JEPA submodule to sys.path when it is not installed.

    The root package layout is:
        HtW/
          eb_jepa/
            eb_jepa/
              jepa.py

    Adding HtW/eb_jepa to sys.path makes `import eb_jepa.jepa` work without
    requiring an editable install of the submodule.
    """
    if importlib.util.find_spec("eb_jepa.jepa") is not None:
        return _repo_root(repo_root)

    root = _repo_root(repo_root)
    submodule_path = root / "eb_jepa"
    if submodule_path.exists():
        sys.path.insert(0, str(submodule_path))

    if importlib.util.find_spec("eb_jepa.jepa") is None:
        raise ModuleNotFoundError(
            "Could not import EB-JEPA. Install the submodule or run from the HtW repo root."
        )

    return root


def _repo_root(repo_root: str | Path | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()

    return Path(__file__).resolve().parents[2]

