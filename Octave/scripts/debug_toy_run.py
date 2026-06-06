import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOY_CONFIG = PROJECT_ROOT / "configs" / "runs" / "toy_run_001"

result = subprocess.run(
    [
        sys.executable,
        "-m",
        "Project_1_SepFormer_TSE_HF.idea_1_project_setup.scripts.launch",
        str(TOY_CONFIG),
    ],
    cwd=PROJECT_ROOT.parent.parent,
    text=True,
    encoding="utf-8",
    errors="replace",
    capture_output=True,
)

print(result.stdout)
print(result.stderr)
print(result.returncode)